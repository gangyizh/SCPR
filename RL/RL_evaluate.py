import time
import argparse
from itertools import count
import torch.nn as nn
import torch
from collections import namedtuple
from utils import *
from RL.env_binary_question import BinaryRecommendEnv
from RL.env_enumerated_question import EnumeratedRecommendEnv
EnvDict = {
        LAST_FM: BinaryRecommendEnv,
        LAST_FM_STAR: BinaryRecommendEnv,
        YELP: EnumeratedRecommendEnv,
        YELP_STAR: BinaryRecommendEnv
    }

def dqn_evaluate(args, kg, dataset, agent, filename, i_episode):
    test_env = EnvDict[args.data_name](kg, dataset, args.data_name, seed=args.seed, max_turn=args.max_turn,
                                       bit_length=args.bit_length, attr_bit_state=args.attr_num, mode='test',
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
    #     'cand_none': -0.1
    # }
    # ealuation metric  ST@T
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    # SR_turn_15 = [0]* args.max_turn
    SR_turn_15 = np.zeros(args.max_turn, dtype=int)
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    if args.data_name in [LAST_FM_STAR, LAST_FM]:
        test_size = 1000     # Only do 4000 iteration for the sake of time
        user_size = test_size
    if args.data_name in [YELP_STAR, YELP]:
        test_size = 2500     # Only do 2500 iteration for the sake of time
        user_size = test_size
    print('Test size : ', test_size)
    for user_num in range(1, user_size+1):  #user_size
        # TODO uncommend this line to print the dialog process
        blockPrint()
        print('\n================test tuple:{}===================='.format(user_num))
        state = test_env.reset()  #
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
        for t in count(start=1):  # Turn 1 ~ Turn n: user  dialog
            action = agent.policy_net(state).max(1)[1].view(1, 1)
            next_state, reward, done = test_env.step(action.item())
            next_state = torch.tensor([next_state], device=args.device, dtype=torch.float)
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if done:
                next_state = None
            state = next_state
            if done and t <= 15:
                enablePrint()
                if reward.item() == 1:  # recommend successfully
                    # SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]
                    SR_turn_15[(t-1):] += 1  # record SR for each turn
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
            elif t > 15:
                AvgT += 15
                break
        enablePrint()
        if user_num % args.observe_num == 0 and user_num > 0:
            SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT / args.observe_num]
            SR_TURN = [i/args.observe_num for i in SR_turn_15]
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                       float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num,
                                                AvgT / args.observe_num, user_num + 1))
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT = 0, 0, 0, 0
            # SR_turn_15 = [0] * args.max_turn
            SR_turn_15 = np.zeros(args.max_turn, dtype=int)
            tt = time.time()

    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean]
    save_rl_mtric(dataset=args.data_name, filename=filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i+1, SRturn_all[i]))
        f.write('================================\n')

