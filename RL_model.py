
import math
import random
import numpy as np
import os
import sys
# sys.path.append('..')

from collections import namedtuple
import argparse
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
#TODO select env
from RL.env_binary_question import BinaryRecommendEnv
from RL.env_enumerated_question import EnumeratedRecommendEnv
from RL.RL_evaluate import dqn_evaluate
import time
import warnings
warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM: BinaryRecommendEnv,
    LAST_FM_STAR: BinaryRecommendEnv,
    YELP: EnumeratedRecommendEnv,
    YELP_STAR: BinaryRecommendEnv
    }
FeatureDict = {
    LAST_FM: 'feature',
    LAST_FM_STAR: 'feature',
    YELP: 'large_feature',
    YELP_STAR: 'feature'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_space, hidden_size, action_space):
            super(DQN, self).__init__()
            self.state_space = state_space
            self.action_space = action_space
            self.fc1 = nn.Linear(self.state_space, hidden_size)
            self.fc1.weight.data.normal_(0, 0.1)   # initialization
            self.out = nn.Linear(hidden_size, self.action_space)
            self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class Agent(object):
    def __init__(self, device, memory, state_space, hidden_size, action_space, EPS_START = 0.9, EPS_END = 0.05, EPS_DECAY = 200):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.device = device
        self.policy_net = DQN(state_space, hidden_size, action_space).to(device)
        self.target_net = DQN(state_space, hidden_size, action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = memory


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimize_model(self, BATCH_SIZE, GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(n_states)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch


        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.data
    
    def save_model(self, data_name, filename, epoch_user):
        save_rl_agent(dataset=data_name, model=self.policy_net, filename=filename, epoch_user=epoch_user)
    def load_model(self, data_name, filename, epoch_user):
        model_dict = load_rl_agent(dataset=data_name, filename=filename, epoch_user=epoch_user)
        self.policy_net.load_state_dict(model_dict)



def train(args, kg, dataset, filename):


    env = EnvDict[args.data_name](kg, dataset, args.data_name, seed=args.seed, max_turn=args.max_turn,
                       bit_length=args.bit_length, attr_bit_state=args.attr_num, mode='train', command=args.command, ask_num=args.ask_num, entropy_way=args.entropy_method, fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    state_space = env.state_space
    action_space = env.action_space
    memory = ReplayMemory(args.memory_size) #10000
    agent = Agent(device=args.device, memory=memory, state_space=state_space, hidden_size=args.hidden, action_space=action_space)
    tt = time.time()
    # self.reward_dict = {
    #     'ask_suc': 0.1,
    #     'ask_fail': -0.1,
    #     'rec_suc': 1,
    #     'rec_fail': -0.3,
    #     'until_T': -0.3,  # until MAX_Turn
    #     'cand_none': -0.1
    # }
    #ealuation metric  ST@T
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    loss = torch.tensor(0, dtype=torch.float, device=args.device)
    start = time.time()
    #agent load policy parameters
    if args.load_rl_epoch != 0 :
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    for i_episode in range(args.load_rl_epoch+1, args.epochs+1): #args.epochs

        ## =================== Logs about conversations between users and the agent ========================
        blockPrint()  # TODO Block user-agent process output
        ## =================== Logs about conversations between users and the agent ========================

        print('\n================new tuple:{}===================='.format(i_episode))
        state = env.reset()
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
        for t in count(start=1):  # Turn 1 ~ Turn n
            action = agent.select_action(state)
            next_state, reward, done = env.step(action.item())
            next_state = torch.tensor([next_state], device=args.device, dtype=torch.float)
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if done:
                next_state = None

            agent.memory.push(state, action, next_state, reward)
            state = next_state
            newloss = agent.optimize_model(args.batch_size, args.gamma)
            if newloss is not None:
                loss += newloss
            if done and t <= 15:
                if reward.item() == 1:  #recommend successfully
                    if t <= 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t <= 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1
                AvgT += t
                break
            elif t > 15: # Reach maximum turn
                AvgT += 15
                break
        if i_episode % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        enablePrint() # Enable print function

        if i_episode % args.observe_num == 0 and i_episode > 0:
            print('loss : {} in episode {}'.format(loss.item()/args.observe_num, i_episode))
            if i_episode % (args.observe_num * 2) == 0 and i_episode > 0:
                print('save model in episode {}'.format(i_episode))
                save_rl_model_log(dataset=args.data_name, filename=filename, epoch=i_episode, epoch_loss=loss.item()/args.observe_num, train_len=args.observe_num)
                SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT/args.observe_num]
                save_rl_mtric(dataset=args.data_name, filename=filename, epoch=i_episode, SR=SR, spend_time=time.time()-tt)  #save RL metric

            if i_episode % (args.observe_num * 4) == 0 and i_episode > 0:
                agent.save_model(data_name=args.data_name, filename=filename, epoch_user=i_episode) # save RL policy model
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{} Total epoch_uesr:{}'.format(SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT/args.observe_num, i_episode+1))
            print('spend time: {}'.format(time.time()-start))
            SR5, SR10, SR15, AvgT = 0, 0, 0, 0
            loss = torch.tensor(0, dtype=torch.float, device=args.device)
            tt = time.time()

        if i_episode % (args.observe_num * 4) == 0 and i_episode > 0:
            print('Evaluating on Test tuples!')
            dqn_evaluate(args, kg, dataset, agent, filename, i_episode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--target_update', type=int, default=20, help='the number of epochs to update policy parameters')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--hidden', type=int, default=512, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default=LAST_FM_STAR, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    parser.add_argument('--entropy_method', type=str, default='entropy', help='entropy_method is one of {entropy, weight entropy}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--bit_length', type=int, default=20, help='The number of binary bits to record the number of candidate items')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--command', type=int, default=7, help='select state vector')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--observe_num', type=int, default=200, help='the number of epochs to save RL model and metric')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading RL model')

    '''
    # conver_his: Conversation_history;   attr_ent: Entropy of attribute ; bit_length: binary bits to record the number of candidate items 
    # command:1   self.user_embed, self.conver_his, self.attr_ent, self.bit_length
    # command:2   self.attr_ent
    # command:3   self.conver_his
    # command:4   self.bit_length
    # command:5   self.user_embedding
    # command:6   self.conver_his, self.attr_ent, self.bit_length
    # command:7   self.conver_his, self.bit_length
    '''

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    #reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)

    dataset = load_dataset(args.data_name)
    filename = 'train-data-{}-RL-command-{}-ask_method-{}-attr_num-{}-ob-{}'.format(
        args.data_name, args.command, args.entropy_method, args.attr_num, args.observe_num)
    train(args, kg, dataset, filename)

if __name__ == '__main__':
    main()