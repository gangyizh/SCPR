import gym
import math
import random
import numpy as np
import os
import sys
#sys.path.append(os.path.abspath('../DQN_lastfm_large'))
sys.path.append('..')

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

EnvDict = {
    LAST_FM: BinaryRecommendEnv,
    LAST_FM_SMALL: BinaryRecommendEnv,
    YELP: EnumeratedRecommendEnv,
    YELP_SMALL: BinaryRecommendEnv
    }
FeatureDict = {
    LAST_FM: 'feature',
    LAST_FM_SMALL: 'feature',
    YELP: 'large_feature',
    YELP_SMALL: 'feature'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """保存变换"""
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
                # t.max（1）将为每行的列返回最大值。max result的第二列是找到max元素的索引，因此我们选择预期回报较大的操作。
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimize_model(self, BATCH_SIZE, GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # 转置批样本（有关详细说明，请参阅https://stackoverflow.com/a/19343/3343043）。这会将转换的批处理数组转换为批处理数组的转换。
        batch = Transition(*zip(*transitions))

        # 计算非最终状态的掩码并连接批处理元素（最终状态将是模拟结束后的状态）
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(n_states)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 计算Q(s_t, a)-模型计算 Q(s_t)，然后选择所采取行动的列。这些是根据策略网络对每个批处理状态所采取的操作。
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算下一个状态的V(s_{t+1})。非最终状态下一个状态的预期操作值是基于“旧”目标网络计算的；选择max(1)[0]的最佳奖励。这是基于掩码合并的，这样当状态为最终状态时，我们将获得预期状态值或0。
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # 计算期望 Q 值
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # 计算 Huber 损失
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # 优化模型
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
    # env = gym.make('CartPole-v0')
    # env.seed(args.seed)
    # state_space = env.observation_space.shape[0]
    # action_space = env.action_space.n
    # torch.manual_seed(args.seed)
    #-----------------------------------------
    env = EnvDict[args.data_name](kg, dataset, args.data_name, seed=args.seed, max_turn=args.max_turn,
                       cand_len_size=args.cand_len_size, attr_num=args.attr_num, mode='train', command=args.command, ask_num=args.ask_num, entropy_way=args.entropy_method, fm_epoch=args.fm_epoch)
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
    #     'cand_none': -0.1  # have no candidate items
    # }
    #ealuation metric  ST@T
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    loss = torch.tensor(0, dtype=torch.float, device=args.device)
    start = time.time()
    #agent load policy parameters
    #agent.load_model(data_name=args.data_name, filename=filename, epoch_user=50)
    for i_episode in range(args.epochs+1): #args.epochs
        #TODO uncommend this line to print the dialog process
        blockPrint()
        print('\n================new tuple:{}===================='.format(i_episode))
        #episode_reward = 0
        state = env.reset()  # Reset environment and record the starting state
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
        for t in count():   # user  dialog
            action = agent.select_action(state)
            # Step through environment using chosen action
            next_state, reward, done = env.step(action.item())
            # print('reward:{}, done:{}'.format(reward, done))
            #state: uesr_emb + con_his + attr_ent + cand_len 64 + 15 + 20 + 20    1*119
            next_state = torch.tensor([next_state], device=args.device, dtype=torch.float)
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if done:
                next_state = None
            # 在内存中储存当前参数
            agent.memory.push(state, action, next_state, reward)
            # 进入下一状态
            state = next_state
            # 记性一步优化 (在目标网络)
            newloss = agent.optimize_model(args.batch_size, args.gamma)
            if newloss is not None:
                loss += newloss
            #episode_reward += reward
            if done:
                # episode_durations.append(t + 1)
                #update metric SR@T
                if reward.item() == 1:  #recommend successfully
                    if t < 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t < 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1
                AvgT += t
                break
        # 更新目标网络, 复制在 DQN 中的所有权重偏差
        #total_reward.append(episode_reward.item())
        if i_episode % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        enablePrint()
        if i_episode % args.observe_num == 0 and i_episode > 0:
            print('loss : {} in episode {}'.format(loss.item()/args.observe_num, i_episode))
            if i_episode % (args.observe_num * 5) == 0 and i_episode > 0:
                print('save model in episode {}'.format(i_episode))
                save_rl_model_log(dataset=args.data_name, filename=filename, epoch=i_episode, epoch_loss=loss.item()/args.observe_num, train_len=args.observe_num)
                SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT/args.observe_num]
                save_rl_mtric(dataset=args.data_name, filename=filename, epoch=i_episode, SR=SR, spend_time=time.time()-tt)  #save RL SR
            #print('save model and metrics; epoch_user:{}'.format(i_episode + 1))
            if i_episode % (args.observe_num * 10) == 0 and i_episode > 0:
                agent.save_model(data_name=args.data_name, filename=filename, epoch_user=i_episode) # save RL policy model
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{} Total epoch_uesr:{}'.format(SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT/args.observe_num, i_episode+1))
            print('spend time: {}'.format(time.time()-start))
            SR5, SR10, SR15, AvgT = 0, 0, 0, 0
            loss = torch.tensor(0, dtype=torch.float, device=args.device)
            tt = time.time()
        if i_episode % (args.observe_num * 10) == 0 and i_episode > 0:
            print('Evaluating on Test tuples!')
            dqn_evaluate(args, kg, dataset, agent, filename)


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--name', type=str, default='train_agent', help='train agent.')
    parser.add_argument('--seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=100000, help='Max number of epochs.')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--target_update', type=int, default=20, help='epochs num: update policy parameters')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--hidden', type=int, default=512, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default=LAST_FM_SMALL, help='One of {LAST_FM, LAST_FM_SMALL, YELP, YELP_SMALL}.')
    parser.add_argument('--entropy_method', type=str, default='entropy', help='entropy_method one of {entropy, weight entropy}')
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--cand_len_size', type=int, default=20, help='binary state size of candidate items')
    parser.add_argument('--attr_num', type=int, help='entropy state size of topK reachable features')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--command', type=int, default=6, help='select state vector')
    parser.add_argument('--ask_num', type=int, default=1, help='ask feature num')
    parser.add_argument('--observe_num', type=int, default=500, help='the number of epochs to save mode and metric')
    '''
    # command:1   self.user_embed, self.conver_his, self.attr_ent, self.cand_len
    # command:2   self.attr_ent
    # command:3   self.conver_his
    # command:4   self.cond_len
    # command:5   self.user_embedding
    # command:6   self.conver_his, self.attr_ent, self.cand_len
    # command:8   self.conver_his, self.attr_ent, self.cand_len, self.attr_socre
    '''

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    #reset attr_num
    feature_name = FeatureDict[args.data_name]  # 'feature' or 'large_feature'
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)

    dataset = load_dataset(args.data_name)
    filename = 'env-data-{}-RL-command-{}-ask_method-{}-ask_num-{}seed-{}-turn-{}-cand_size-{}-ent_num-{}-ob-{}'.format(
        args.data_name, args.command, args.entropy_method, args.ask_num, args.seed, args.max_turn, args.cand_len_size,
        args.attr_num,
        args.observe_num)
    train(args, kg, dataset, filename)


if __name__ == '__main__':
    main()