
import json
import numpy as np
import itertools
import os
import random
from utils import *

from tkinter import _flatten
from collections import Counter
class EnumeratedRecommendEnv(object):
    def __init__(self, kg, dataset, data_name, seed=1, max_turn=15, cand_len_size=20, attr_num=20, mode='train', command=6, ask_num=1, entropy_way='weight entropy', fm_epoch=0):
        #command:1   self.user_embed, self.conver_his, self.attr_ent, self.cand_len
        #command:2   self.attr_ent
        #command:3   self.conver_his
        #command:4   self.cond_len
        self.data_name = data_name
        self.command = command
        self.mode = mode
        self.seed = seed
        self.max_turn = max_turn    #MAX_TURN
        self.attr_state_num = attr_num
        self.cand_len_size = cand_len_size
        self.kg = kg
        self.dataset = dataset
        self.feature_length = getattr(self.dataset, 'large_feature').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'item').value_len

        #TODO action parameters
        self.ask_num = ask_num  #
        self.rec_num = 10
        # TODO entropy  or weight entropy
        self.ent_way = entropy_way

        # user's profile
        self.reachable_feature = []   # user reachable large_feature
        self.user_acc_feature = []  # user accepted large_feature which asked by agent
        self.user_rej_feature = []  # user rejected large_feature which asked by agent
        self.acc_samll_fea = []
        self.rej_samll_fea = []
        self.cand_items = []   # candidate items
        # self.user_item_socre = []  #  user_item  score
        # self.user_feature_score = []  #user_feature score

        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  the number of conversation in current step
        self.cur_node_set = []     #maybe a node or node set  /   normally save large_feature node
        # state veactor
        self.user_embed = None
        self.conver_his = []    #conversation_history
        self.cand_len = []    #the number of candidate items  [category binary]   2**20 items
        self.attr_ent = []  #TOP attr_num  attribute entropy

        self.ui_dict = self.__load_rl_data__(data_name, mode=mode)  # np.array [ u i weight]
        self.user_weight_dict = dict()
        self.user_items_dict = dict()

        #init seed & init user_dict
        set_random_seed(self.seed) # set random seed
        if mode == 'train':
            self.__user_dict_init__() # init self.user_weight_dict  and  self.user_items_dict
        elif mode == 'test':
            self.ui_array = None    # u-i array [ [userID1, itemID1], [userID2, itemID2]]
            self.__test_tuple_generate__()
            self.test_num = 0
        # embeds = {
        #     'ui_emb': ui_emb,
        #     'feature_emb': feature_emb
        # }
        #TODO epoch is important
        embeds = load_embed(data_name, epoch=fm_epoch)
        self.ui_embeds =embeds['ui_emb']
        self.feature_emb = embeds['feature_emb']
        # self.feature_length = self.feature_emb.shape[0]-1

        self.action_space = 2
        # TODO user_emb_size(64) + max_turn + 20 + attr_num(100) + attr_num(100)

        self.state_space_dict = {
            1: self.max_turn + self.cand_len_size + self.attr_state_num + self.ui_embeds.shape[1],
            2: self.attr_state_num,  # attr_ent
            3: self.max_turn,  #conver_his
            4: self.cand_len_size,  #cand_itme
            5: self.ui_embeds.shape[1], # user_embedding
            6: self.cand_len_size + self.attr_state_num + self.max_turn, #attr_ent + conver_his + cand_itme
            8: self.cand_len_size + self.attr_state_num + self.max_turn,

        }
        self.state_space = self.state_space_dict[self.command]
        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,      # recommend fail
            'until_T': -0.3,      #until MAX_Turn
            'cand_none': -0.1    #have no candidate items
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()   # This dict is used to calculate entropy

    def __load_rl_data__(self, data_name, mode):
        if mode == 'train':
            with open(os.path.join(TMP_DIR[data_name], 'review_dict_valid.json'), encoding='utf-8') as f:  #TODO train data in {train, valid}
            #with open(os.path.join(TMP_DIR[data_name], 'review_dict_train.json'), encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
        elif mode == 'test':
            with open(os.path.join(TMP_DIR[data_name], 'review_dict_test.json'), encoding='utf-8') as f:
                mydict = json.load(f)
        # print('load RL {} data'.format(mode))
        return mydict


    def __user_dict_init__(self):   #Calculate the probability of user  and  Calculate items of user interaction
        # users = self.ui_dict[:, 0]
        # items = list(self.ui_dict[:, 1])
        # user_count_dict = dict()
        # for ind, user in enumerate(users):
        #     if user in user_count_dict.keys():
        #         user_count_dict[user] += 1
        #         self.user_items_dict[user].append(items[ind])
        #     else:
        #         user_count_dict[user] = 1
        #         self.user_items_dict[user] = [items[ind]]
        # users_len = len(set(users))
        # for user in user_count_dict.keys():
        #     self.user_weight_dict[user] = user_count_dict[user]/len(users)

        ui_nums = 0
        for items in self.ui_dict.values():
            ui_nums += len(items)
        for user_str in self.ui_dict.keys():
            user_id = int(user_str)
            self.user_weight_dict[user_id] = len(self.ui_dict[user_str])/ui_nums
        print('user_dict init successfully!')

    def __test_tuple_generate__(self):
        ui_list = []
        for user_str, items in self.ui_dict.items():
            user_id = int(user_str)
            for item_id in items:
                ui_list.append([user_id, item_id])
        self.ui_array = np.array(ui_list)
        np.random.shuffle(self.ui_array)



    def reset(self):
        #init  user_id  item_id
        self.cur_conver_step = 0   #reset cur_conversation step
        if self.mode == 'train':
            users = list(self.user_weight_dict.keys())
            #TODO select user by weight?
            #self.user_id = np.random.choice(users, p=list(self.user_weight_dict.values())) # select user  according to user weights
            self.user_id = np.random.choice(users)
            #select item that have interaction with user
            #self.target_item = np.random.choice(self.user_items_dict[self.user_id])
            self.target_item = np.random.choice(self.ui_dict[str(self.user_id)])
        elif self.mode == 'test':
            self.user_id = self.ui_array[self.test_num, 0]
            self.target_item = self.ui_array[self.test_num, 1]
            self.test_num += 1
        # init user's profile
        print('-----------reset state vector------------')
        self.user_acc_feature = []  # user accepted large_feature which asked by agent
        self.user_rej_feature = []  # user rejected large_feature which asked by agent
        self.acc_samll_fea = []
        self.rej_samll_fea = []
        self.cand_items = list(range(self.item_length))  # YELP
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))

        # init  state vector
        self.user_embed = self.ui_embeds[self.user_id].tolist()  # init user_embed   np.array---list
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.cand_len = [self.feature_length >>d & 1 for d in range(self.cand_len_size)][::-1]  #category binary  2**20 items
        self.attr_ent = [0] * self.attr_state_num  # TOP attr_num  attribute entropy

        # init user prefer feature
        self._updata_reachable_feature(start='user')  # self.reachable_feature = []
        self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1

        self._update_cand_items(acc_feature=self.cur_node_set, rej_feature=[])
        self._update_feature_entropy()  # update entropy
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))
        print('reset_reachable_feature: {}'.format(self.reachable_feature))


        # TODO init U-I-A entropy score for yelp
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.ask_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            max_ind_list.append(max_ind)
        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        # fea_score_tuple = list(zip(self.reachable_feature, reach_fea_score))
        # sort_tuple = sorted(fea_score_tuple, key=lambda x: x[1], reverse=True)
        # self.reachable_feature, reach_new_fea_score = zip(*sort_tuple)
        # self.reachable_feature = list(self.reachable_feature)


        #TODO  Pay attention to data_format !
        #self.cand_items = list(range(self.item_length))  # candidate items
        return self._get_state()

    def _get_state(self):
        #TODO change input state here
        if self.command == 1:
            state = [self.user_embed, self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
            print('===state vector:')
            print(' conver_his:', self.conver_his)
            print(' attr_ent:', self.attr_ent)
            print(' cand_len to Decimal num:', len(self.cand_items))
        elif self.command == 2: #attr_ent
            state = self.attr_ent
            state = list(_flatten(state))
            print('===state vector:')
            print(' attr_ent:', self.attr_ent)
            print(' cand_len to Decimal num:', len(self.cand_items))
        elif self.command == 3: #conver_his
            state = self.conver_his
            state = list(_flatten(state))
            print('===state vector:')
            print(' conver_his:', self.conver_his)
            print(' cand_len to Decimal num:', len(self.cand_items))
        elif self.command == 4: #cand_len
            state = self.cand_len
            state = list(_flatten(state))
            print('===state vector:')
            print(' cand_len to Decimal num:', len(self.cand_items))
        elif self.command == 5:  #user_embedding
            state = self.user_embed
            state = list(_flatten(state))
            print('===state vector:')
            print('user embedding:', self.user_embed)
        elif self.command == 6: #attr_ent + conver_his + cand_len
            state = [self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
            print('===state vector:')
            print(' conver_his:', self.conver_his)
            print(' reachable_feature:', self.reachable_feature)
            print(' attr_ent:', self.attr_ent)
            print('YALP_large env3 cand_len to Decimal num:', len(self.cand_items))
        elif self.command == 8:
            state = [self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
            print('===state vector:')
            print(' conver_his:', self.conver_his)
            print(' attr_ent:', self.attr_ent)
            print(' YALP large env3 cand_len to Decimal num:', len(self.cand_items))


        return state

    def step(self, action):   #action:0  ask   action:1  recommend   setp=MAX_TURN  done
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            #update state_vector: conver_his
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif action == 0:   #ask large_feature
            # select topk  reachable features to ask
            print('-->action: ask features')
            ''' update user's profile: user_acc_feature & user_rej_feature & cand_items
            update state_vector:_ask_update(): conver_his, cand_len  
            _update_feature_entropy(): attr_ent(acc or rej)
            '''
            # if cand_items  is empty , done = 1
            reward, done, acc_feature, rej_feature = self._ask_update()  #update user's profile:  user_acc_feature & user_rej_feature & cand_items
            self._update_cand_items(acc_feature, rej_feature)  # update cand_item
            if len(acc_feature):   # can reach new large_feature：  update current node and reachable_feature
                self.cur_node_set = acc_feature
                self._updata_reachable_feature(start='large_feature')  # update user's profile: reachable_feature
                #compute feature_score:  fm_score  or fm_score+ent_score

            # TODO    reachable_feature - reject_feature- acc_feature
            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_rej_feature))
            #YELP need to update entropy
            if self.command in [1, 2, 6, 7]:  # update attr_ent
                self._update_feature_entropy()  #TODO yelp_large_tag need
            if len(self.reachable_feature) != 0:  # if reachable_feature == 0 :cand_item= 1
                reach_fea_score = self._feature_score()  # !!! compute feature score
                max_ind_list = []
                for k in range(self.ask_num):
                    max_score = max(reach_fea_score)
                    max_ind = reach_fea_score.index(max_score)
                    reach_fea_score[max_ind] = 0
                    max_ind_list.append(max_ind)
                max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
                [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
                [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]


        elif action == 1:  #recommend items
            #select topk candidate items to recommend
            cand_item_score = self._item_score()
            item_score_tuple = list(zip(self.cand_items, cand_item_score))
            sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
            self.cand_items, cand_item_score = zip(*sort_tuple)

            '''update state_vector: _recommend_updata(): conver_his, cand_len, 
             _update_feature_entropy(): attr_ent
            '''
            #===================== rec update=========
            reward, done = self._recommend_updata()
            #========================================
            if reward == 1:
                print('-->Recommend successfully!')
            else:
                if self.command in [1, 2, 6, 7]:  # update attr_ent
                    self._update_feature_entropy()
                print('-->Recommend fail !')
                #   sort feature entropy score


        self.cur_conver_step += 1
        return self._get_state(), reward, done

    #TODO   parameters may be different in other dataset
    def _updata_reachable_feature(self, start='large_feature'):
        #TODO  reset_ reachable_feature   !!!!???
        self.reachable_feature = []
        if start == 'user':
            user_like_random_fea = random.choice(self.kg.G['item'][self.target_item]['belong_to_large'])
            self.user_acc_feature.append(user_like_random_fea)  # update user acc_fea
            self.cur_node_set = [user_like_random_fea]

            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = list(self.kg.G['large_feature'][cur_node]['belong_to_large'])  # A-I

                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                print('---> A-I-A item length: {}'.format(len(cand_fea_belong_items)))
                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(list(self.kg.G['item'][item_id]['belong_to_large']))
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))
            self.reachable_feature = next_reachable_feature  # init reachable_feature


        elif start == 'large_feature':
            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = list(self.kg.G['large_feature'][cur_node]['belong_to_large']) # A-I

                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(list(self.kg.G['item'][item_id]['belong_to_large']))
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))

            print('next_reach_fea:{}'.format(next_reachable_feature))
            print('next_reach_fea num:{}'.format(len(next_reachable_feature)))
            print('last_reach_fea:{}'.format(self.reachable_feature))
            print('last_reach_fea num:{}'.format(len(self.reachable_feature)))
            self.reachable_feature = next_reachable_feature
            #self.reachable_feature = list(set(self.reachable_feature) & set(next_reachable_feature))

    def _feature_score(self):
        reach_fea_score = []
        for feature_id in self.reachable_feature:
            score = self.attr_ent[feature_id]
            reach_fea_score.append(score)
        return reach_fea_score

    def _item_score(self):
        cand_item_score = []
        for item_id in self.cand_items:
            item_embed = self.ui_embeds[self.user_length + item_id]
            score = 0
            score += np.inner(np.array(self.user_embed), item_embed)
            prefer_embed = self.feature_emb[self.acc_samll_fea, :]  #np.array (x*64), samll_feature
            for i in range(len(self.acc_samll_fea)):
                score += np.inner(prefer_embed[i], item_embed)
            cand_item_score.append(score)
        return cand_item_score


    def _ask_update(self):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0
        # TODO datafram!     groundTruth == target_item features
        feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to_large']
        #TODO ask new large_feature that not in accpeted set * rejected set !!
        remove_acced_reachable_fea = self.reachable_feature.copy()   # copy reachable_feature

        acc_feature = list(set(remove_acced_reachable_fea[:self.ask_num]) & set(feature_groundtrue))
        rej_feature = list(set(remove_acced_reachable_fea[:self.ask_num]) - set(acc_feature))

        # update user_acc_feature & user_rej_feature
        self.user_acc_feature.append(acc_feature)
        self.user_acc_feature = list(set(_flatten(self.user_acc_feature)))
        self.user_rej_feature.append(rej_feature)
        self.user_rej_feature = list(set(_flatten(self.user_rej_feature)))
        print('=== user profiles list')
        print(' sorted reachable large_feature :', self.reachable_feature)
        print(' large_feature ent:', self.attr_ent)
        print(' sorted reachable large_feature num :', len(self.reachable_feature))
        print(' sorted reachable large_feature top K:', self.reachable_feature[:self.ask_num])
        print(' sorted reachable large_feature top K(remove acc_rej set)', remove_acced_reachable_fea[:self.ask_num])
        print(' accept_feature: ', acc_feature)
        print(' accept_all:', self.user_acc_feature)
        print(' reject_features:', rej_feature)
        print(' reject_all:', self.user_rej_feature)


        #TODO update cand_items    # update state_vector: conver_his  cand_len
        #TODO Datafram !
        if len(acc_feature):
            reward = self.reward_dict['ask_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']   #update conver_his
        else:
            reward = self.reward_dict['ask_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  #update conver_his

        if self.cand_items == []:  #candidate items is empty
            done = 1
            reward = self.reward_dict['cand_none']
        return reward, done, acc_feature, rej_feature

    def _update_cand_items(self, acc_feature, rej_feature):  # TODO： YELP large tag need to modify

        small_feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to']  # TODO small_ground truth
        if len(acc_feature):    #accept large_feature
            print('=== ask acc: update cand_items')
            for feature_id in acc_feature:  #feature_id: large_tag
                feature_small_ids = self.kg.G['large_feature'][feature_id]['link_to_feature']#TODO this is important in YELP： [large_tag--> small_tags]
                for small_id in feature_small_ids:
                    if small_id in small_feature_groundtrue:  # user_accept samll_tag
                        self.acc_samll_fea.append(small_id)
                        feature_items = self.kg.G['feature'][small_id]['belong_to']
                        self.cand_items = set(self.cand_items) & set(feature_items)   #  itersection
                    else:  #uesr reject small_tag
                        self.rej_samll_fea.append(small_id)  #reject no update
                        # neg_feature_items = self.kg.G['feature'][small_id]['belong_to']  # TODO G['small_feature'][small_id]['belong_to']
                        # self.cand_items = set(self.cand_items) - set(neg_feature_items)  # remove items that have neg_feature
            self.cand_items = list(self.cand_items)
        # if len(rej_feature):  #reject large_feature
        #     print('=== ask rej: update cand_items')
        #     for feature_id in rej_feature:
        #         feature_small_ids = self.kg.G['large_feature'][feature_id]['link_to_feature']  # TODO this is important in YELP： [large_tag--> small_tags]
        #         for small_id in feature_small_ids:
        #             self.rej_samll_fea.append(small_id)
        #             neg_feature_items = self.kg.G['feature'][small_id]['belong_to']  #TODO G['small_feature'][small_id]['belong_to']
        #             self.cand_items = set(self.cand_items) - set(neg_feature_items)  # remove items that have neg_feature
        #     self.cand_items = list(self.cand_items)
        # update cand_len
        self.cand_len = [len(self.cand_items) >>d & 1 for d in range(self.cand_len_size)][::-1]  #category binary
        # acc_rej log
        print('-->ask: samll_items')
        print('acc_samll_items:{}'.format(self.acc_samll_fea))
        print('rej_samll_items:{}'.format(self.rej_samll_fea))


    def _recommend_updata(self):
        print('-->action: recommend items')
        recom_items = self.cand_items[: self.rec_num]    # TOP k num to recommend
        if self.target_item in recom_items:
            reward = self.reward_dict['rec_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu'] #update state vector: conver_his
            done = 1
        else:
            reward = self.reward_dict['rec_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  #update state vector: conver_his
            self.cand_items = self.cand_items[self.rec_num:]  #update candidate items
            self.cand_len = [len(self.cand_items) >> d & 1 for d in range(self.cand_len_size)][::-1]  # category binary
            done = 0
        print(' sorted cand_items top K:', self.cand_items[:self.rec_num])
        return reward, done

    def _update_feature_entropy(self):  #YELP need to modify
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            #TODO Dataframe
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))  #NO need to do set operater !
            self.attr_count_dict = dict(Counter(cand_items_fea_list))

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            # note:   self.reachable_feature  is sorted  by large_feature-score in _step()  function
            # TODO ask new large_feature that not in accpeted set * rejected set !!
            real_ask_able_large_fea = self.reachable_feature
            for large_fea_id in real_ask_able_large_fea:
                large_ent = 0
                small_feature = list(self.kg.G['large_feature'][large_fea_id]['link_to_feature'])
                small_feature_in_cand = list(set(small_feature) & set(self.attr_count_dict.keys()))

                for fea_id in small_feature_in_cand:
                    p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                    p2 = 1.0 - p1
                    if p1 == 1:
                        large_ent += 0
                    else:
                        ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                        large_ent += ent
                self.attr_ent[large_fea_id] =large_ent
        elif self.ent_way == 'weight entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            cand_item_score = self._item_score()
            cand_item_score_sig = self.sigmoid(cand_item_score)  # sigmoid(score)
            # TODO Dataframe
            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            # note:   self.reachable_feature  is sorted  by large_feature-score in _step()  function
            # TODO ask new large_feature that not in accpeted set * rejected set !!
            real_ask_able_large_fea = self.reachable_feature
            sum_score_sig = sum(cand_item_score_sig)
            for large_fea_id in real_ask_able_large_fea:
                large_ent = 0
                small_feature = list(self.kg.G['large_feature'][large_fea_id]['link_to_feature'])
                small_feature_in_cand = list(set(small_feature) & set(self.attr_count_dict.keys()))

                for fea_id in small_feature_in_cand:
                    p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                    p2 = 1.0 - p1
                    if p1 == 1 or p1 <= 0:
                        large_ent += 0
                    else:
                        ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                        large_ent += ent
                self.attr_ent[large_fea_id] = large_ent
    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()




