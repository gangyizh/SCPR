import time
import argparse
from itertools import count
import torch.nn as nn
import torch
from collections import namedtuple
from utils import *
# #TODO select env
from RL.env_binary_question import BinaryRecommendEnv
from RL.env_enumerated_question import EnumeratedRecommendEnv
EnvDict = {
        LAST_FM: BinaryRecommendEnv,
        LAST_FM_SMALL: BinaryRecommendEnv,
        YELP: EnumeratedRecommendEnv,
        YELP_SMALL: BinaryRecommendEnv
    }

def dqn_evaluate(args, kg, dataset, agent, filename):
    test_env = EnvDict[args.data_name](kg, dataset, args.data_name, seed=args.seed, max_turn=args.max_turn,
                                       cand_len_size=args.cand_len_size, attr_num=args.attr_num, mode='test',
                                       command=args.command, ask_num=args.ask_num, entropy_way=args.entropy_method,
                                       fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    tt = time.time()
    start = tt
    # self.reward_dict = {
    #     'ask_suc': 0.1,
    #     'ask_fail': -0.1,
    #     'rec_suc': 1,
    #     'rec_fail': -0.3,
    #     'until_T': -0.3,  # until MAX_Turn
    #     'cand_none': -0.1  # have no candidate items
    # }
    # ealuation metric  ST@T
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('test ui size:', user_size)
    test_filename = 'test-turn-' + filename   #uncommend
    #test_size = 2500
    #print('random test: {} ui tuple '.format(test_size))
    for user_num in range(user_size):  #test_size + 1
        # TODO uncommend this line to print the dialog process
        blockPrint()
        print('\n================test tuple:{}===================='.format(user_num))
        state = test_env.reset()  # Reset environment and record the starting state
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
        for t in count():  # user  dialog
            action = agent.policy_net(state).max(1)[1].view(1, 1)
            next_state, reward, done = test_env.step(action.item())
            next_state = torch.tensor([next_state], device=args.device, dtype=torch.float)
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if done:
                next_state = None
            state = next_state
            if done:
                enablePrint()
                if reward.item() == 1:  # recommend successfully
                    #t+1 = current_turn
                    SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]  #,  if turn> t then SR[turn] +=1
                    if t < 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t < 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1

                AvgT += t+1
                break
        enablePrint()
        if user_num % args.observe_num == 0 and user_num > 0:
            SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT / args.observe_num]
            SR_TURN = [i/args.observe_num for i in SR_turn_15]
            print('Total epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                       float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num,
                                                AvgT / args.observe_num, user_num + 1))
            # save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=user_num, SR=SR,
            #               spend_time=time.time() - start,
            #               mode='test')  # save RL SR
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT = 0, 0, 0, 0
            SR_turn_15 = [0] * args.max_turn
            tt = time.time()
            if user_num % (args.observe_num*10) == 0 and user_num > 0:
                SR5_mean = np.mean(np.array([item[0] for item in result]))
                SR10_mean = np.mean(np.array([item[1] for item in result]))
                SR15_mean = np.mean(np.array([item[2] for item in result]))
                AvgT_mean = np.mean(np.array([item[3] for item in result]))
                SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean]
                save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=str(user_num)+'-all', SR=SR_all,
                              spend_time=time.time() - start,
                              mode='test')  # save RL SR
                print('save test evaluate successfully!')
                SRturn_all = [0]*args.max_turn
                for i in range(len(SRturn_all)):
                    SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
                print('success turn:{}'.format(SRturn_all))
                PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + test_filename + '.txt'
                with open(PATH, 'a') as f:
                    f.write('===========Test Turn===============\n')
                    f.write('Starting {} user tuples\n'.format(user_num))
                    for i in range(len(SRturn_all)):
                        f.write('training SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
                    f.write('================================\n')


    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean]
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=test_filename, SR=SR_all, spend_time=time.time() - start,
                  mode='test')  # save RL SR
    print('save test evaluate successfully!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='train_agent', help='train agent.')
    parser.add_argument('--seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--target_update', type=int, default=10, help='epochs num: update policy parameters')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    # parser.add_argument('--hidden', type=int, nargs='*', default=[50, 256, 512], help='number of samples')
    parser.add_argument('--hidden', type=int, default=512, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default='yelp', help='data name')
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--cand_len_size', type=int, default=20, help='binary state size of candidate items')
    parser.add_argument('--attr_num', type=int, default=20, help='entropy state size of topK reachable features')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--command', type=int, default=6, help='select state vector')
    parser.add_argument('--ask_num', type=int, default=1, help='ask feature num')
    parser.add_argument('--observe_num', type=int, default=500, help='the number of epochs to save mode and metric')
    '''
    # command:1   self.user_embed, self.conver_his, self.attr_ent, self.cand_len
    # command:2   self.attr_ent
    # command:3   self.conver_his
    # command:4   self.cond_len
    '''
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    kg = load_kg(args.data_name)
    print('data_set:{}'.format(args.data_name))
    feature_length = len(kg.G['feature'].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num', args.attr_num)
    dataset = load_dataset(args.data_name)
    filename = 'v1-{}-RL-command-{}-seed-{}-bs-{}-gamma-{}-tu-{}-lr-{}-hd-{}-turn-{}-cand_size-{}-ask_num-{}-ob-{}'.format(
        args.mode, args.command, args.seed, args.batch_size, args.gamma, args.target_update, args.lr,
        args.hidden, args.max_turn, args.cand_len_size, args.attr_num, args.observe_num)


    env = RecommendEnv(kg, dataset, args.data_name, seed=args.seed, max_turn=args.max_turn,
                       cand_len_size=args.cand_len_size, attr_num=args.attr_num, mode='train', command=args.command, ask_num=args.ask_num)
    set_random_seed(args.seed)
    state_space = env.state_space
    action_space = env.action_space
    memory = ReplayMemory(args.memory_size)  # 10000
    # agent load policy parameters
    agent = Agent(device=args.device, memory=memory, state_space=state_space, hidden_size=args.hidden,
                  action_space=action_space)
    #agent.policy_net = DQN(state_space=state_space, hidden_size=args.hidden, action_space=action_space).to(args.device)
    #TODO change epoch_user
    filename = 'v3-small_yelp-new_data-yelp-valid-RL-command-6-ask_num-1seed-1-turn-16-cand_size-20-ent_num-590-ob-500'
    epoch = 20000
    model_dict = load_rl_agent(dataset=args.data_name, filename=filename, epoch_user=epoch)
    agent.policy_net.load_state_dict(model_dict)
    dqn_evaluate(args, kg, dataset, agent, filename)

if __name__ == '__main__':
    main()