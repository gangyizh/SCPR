
import json
import numpy as np
import os
import random
from utils import *
import itertools
from tkinter import _flatten
from collections import Counter
class BinaryRecommendEnv(object):
    def __init__(self, kg, dataset, data_name, seed=1, max_turn=15, bit_length=20, attr_bit_state=20, mode='train', command=1, ask_num=1, entropy_way='weight entropy', fm_epoch=0):
        self.data_name = data_name
        self.command = command
        self.mode = mode
        self.seed = seed
        self.max_turn = max_turn    #conversation maximum turns
        self.attr_bit_state = attr_bit_state  # The number of binary bits to record the entropy of features
        self.bit_length = bit_length # The number of binary bits to record the number of candidate items
        self.kg = kg
        self.dataset = dataset
        self.feature_length = getattr(self.dataset, 'feature').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'item').value_len

        # action parameters
        self.ask_num = ask_num
        self.rec_num = 10
        #  entropy  or weight entropy
        self.ent_way = entropy_way

        # user's profile
        self.reachable_feature = []   # user reachable feature
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = []   # candidate items

        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  the number of conversation in current step
        self.cur_node_set = []     # maybe a node or a node set  /   normally save feature node
        # state veactor
        self.user_embed = None
        self.conver_his = []    #conversation_history
        self.cand_item_num = []    #the number of candidate items  [binary ]
        self.attr_ent = []  # attribute entropy

        self.ui_dict = self.__load_rl_data__(data_name, mode=mode)  # np.array [ u i weight]
        self.user_weight_dict = dict()
        self.user_items_dict = dict()

        #init seed & init user_dict
        set_random_seed(self.seed) # set random seed
        if mode == 'train':
            self.__user_dict_init__() # init self.user_weight_dict  and  self.user_items_dict
        elif mode == 'test':
            self.ui_array = None    # u-i array [ [userID1, itemID1], ...,[userID2, itemID2]]
            self.__test_tuple_generate__()
            self.test_num = 0
        # === Init feature_map ( used for printing real_name feature path)
        self.__load_feature_map(self.data_name)
        # embeds = {
        #     'ui_emb': ui_emb,
        #     'feature_emb': feature_emb
        # }
        # load fm epoch
        embeds = load_embed(data_name, epoch=fm_epoch)
        self.ui_embeds =embeds['ui_emb']
        self.feature_emb = embeds['feature_emb']
        # self.feature_length = self.feature_emb.shape[0]-1

        self.action_space = 2


        self.state_space_dict = {
            1: self.max_turn + self.bit_length + self.attr_bit_state + self.ui_embeds.shape[1],
            2: self.attr_bit_state,  # attr_ent
            3: self.max_turn,  #conver_his
            4: self.bit_length,  #cand_item
            5: self.ui_embeds.shape[1], # user_embedding
            6: self.bit_length + self.attr_bit_state + self.max_turn, #attr_ent + conver_his + cand_item
            7: self.bit_length + self.max_turn,
        }
        self.state_space = self.state_space_dict[self.command]
        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.3,      # MAX_Turn
            'cand_none': -0.1
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
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_valid.json'), encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
        elif mode == 'test':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_test.json'), encoding='utf-8') as f:
                print('test_data: load RL test data')
                mydict = json.load(f)
        return mydict


    def __user_dict_init__(self):   #Calculate the weight of the number of interactions per user
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
    def __load_feature_map(self, data_name):
        ID_MAP_DIR = DATA_DIR[data_name] + '/ID_map_info'
        FEATURE_FILE_DICT = {
            LAST_FM: 'tag_reverse_map.json',
            LAST_FM_STAR: 'tag_reverse_map.json',
            YELP_STAR: 'second-layer_tag_reverse_map.json',
        }
        with open(os.path.join(ID_MAP_DIR, FEATURE_FILE_DICT[data_name]),
                  encoding='utf-8') as f:
            self.feature_reverse_map = json.load(f)
            print('feature_reverse_map init successfully!')

    def fea_map(self, fea_id):
        return self.feature_reverse_map[str(fea_id)]["real_name"]

    def reset(self):
        # === 1. init  user_id  item_id  cur_step   cur_node_set
        self.cur_conver_step = 0   #reset cur_conversation step
        self.cur_node_set = []   #maybe a (feature) node or a (feature) node set  /  depend on  {args.ask_num & user_accept feature num}
        if self.mode == 'train':
            users = list(self.user_weight_dict.keys())
            # self.user_id = np.random.choice(users, p=list(self.user_weight_dict.values())) # select user  according to user weights
            self.user_id = np.random.choice(users)
            self.target_item = np.random.choice(self.ui_dict[str(self.user_id)])
        elif self.mode == 'test':
            self.user_id = self.ui_array[self.test_num, 0]
            self.target_item = self.ui_array[self.test_num, 1]
            self.test_num += 1

        # init user's profile
        cprint("-----------Reset Conversational Recomendation!------------")
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        self.feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to']
        self.reachable_feature = []  # user reachable feature in cur_step
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = list(range(self.item_length))

        # === 2. init  state vector [Reinforcement learning input]
        self.user_embed = self.ui_embeds[self.user_id].tolist()  # init user_embed   np.array---list
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.cand_item_num = [self.feature_length >> d & 1 for d in range(self.bit_length)][::-1]  #Binary representation of candidate set length
        self.attr_ent = [0] * self.attr_bit_state  # attribute entropy

        # ===============    Transition Stage ===========
        # === 3. [Turn-1 init user prefer feature]
        user_init_fea = self._user_init_prefer_feature(user_init_prefer_fea_num=1)
        # --- 3.1 update user's profile : self.user_acc_feature
        self._update_user_profile(acc_feature=user_init_fea, rej_feature=[])

        # === 4. init Graph path reasoning
        # --- 4.1 init jump path : from user to user's init feature
        self._update_cur_node_set(acc_feature_node=user_init_fea)  # == [latest jump point(set) on the graph] ===

        # --- 4.2 update reachable feature from current feature node
        self._updata_reachable_feature()  # User init prefered feature & update reachable_feature & update self.cur_node_set
        self._remove_asked_reachable_fea()  # remove init user's prefered feature from reachable feature
        print('Number of reachable features: {}'.format(len(self.reachable_feature)))
        # --- 4.3 output jumping log
        self._graph_path_extend(user_start=True, acc_feature_node=self.user_acc_feature.copy())  # Update graph path

        # === 5. update conversational feedback & state_vector(self.conver_his) [RL reward & turn idx]
        # --- 5.1 update RL input : conversational history encode
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1  # the number of conversation in current step
        # --- 5.2  update RL input : the info of candidate items length & update cand_items
        self._update_cand_items(acc_feature=self.user_acc_feature, rej_feature=[])
        print(f'===Number of candidate items: [{len(self.cand_items)}]')
        # === 5.3  update feature_entropy
        self._update_feature_entropy()  #update entropy


        # === 6. score reachable_feature & sort the score
        self._mini_sort_reachable_feature()  # move the top-k (k=ask_num) reachable feature to the front
        return self._get_state()



    def _get_state(self):
        if self.command == 1:
            state = [self.user_embed, self.conver_his, self.attr_ent, self.cand_item_num]
            state = list(_flatten(state))
        elif self.command == 2: #attr_ent
            state = self.attr_ent
            state = list(_flatten(state))
        elif self.command == 3: #conver_his
            state = self.conver_his
            state = list(_flatten(state))
        elif self.command == 4: #cand_len
            state = self.cand_item_num
            state = list(_flatten(state))
        elif self.command == 5:  #user_embedding
            state = self.user_embed
            state = list(_flatten(state))
        elif self.command == 6: #attr_ent + conver_his + cand_len
            state = [self.conver_his, self.attr_ent, self.cand_item_num]
            state = list(_flatten(state))
        elif self.command == 7: #conver_his + cand_len
            state = [self.conver_his, self.cand_item_num]
            state = list(_flatten(state))
        return state

    def step(self, action):   #action:0  ask   action:1  recommend   setp=MAX_TURN  done
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif action == 0:   #ask feature
            print('-->action: ask features')
            reward, done, acc_feature, rej_feature = self._agent_ask_user_response(
                ask_num=self.ask_num)  # update user's profile:  user_acc_feature & user_rej_feature
            if done == 1: #  reachable feature set is empty
                cprint(f'There are no attributes to ask!')
                return self._get_state(), reward, done

            # -------------------------------------------------------
            # == update cand_items & state_vector(self.cand_item_num)
            self._update_cand_items(acc_feature, rej_feature)  # %%% only consider accepted features! See 'CPR' paper! %%%
            # == update user's profile : self.user_acc_feature, self.user_rej_feature
            self._update_user_profile(acc_feature, rej_feature)

            if len(acc_feature):  # can reach new feature：  update current node and reachable_feature
                print(f'--> User accept feature : {acc_feature}{tuple(map(self.fea_map, acc_feature))}')
                self._graph_path_extend(acc_feature_node=acc_feature)  # ==1. Update graph path
                self._update_cur_node_set(acc_feature_node=acc_feature)  # ==2. Update self.cur_node_set
                self._updata_reachable_feature()  # ==3. Update reachable_feature
                self._remove_asked_reachable_fea()  # -- 3.1 remove asked features from reachable feature
                self.item_score_compute_flag = True  # Item scores are calculated when using the weighting entropy method

                if self.command in [1, 2, 6, 7]:  # == update feature's entropy
                    self._update_feature_entropy(item_score_compute=self.item_score_compute_flag)

            elif len(rej_feature):  # update reachable feature
                self._remove_asked_reachable_fea()  # --  remove asked features from reachable feature
                for rej_fea_id in rej_feature:  # update feature's entropy
                    self.attr_ent[rej_fea_id] = 0
                print(f'--> User ignore(reject) feature : {rej_feature}{tuple(map(self.fea_map, rej_feature))}')

            if self.reachable_feature != []:
                self._mini_sort_reachable_feature()  # move the top-k (k=ask_num) reachable feature to the front



        elif action == 1:  #recommend items
            print('-->action: recommend items')
            # ===  Get items which sort by predictional model ===

            if self.conver_his[self.cur_conver_step - 1] in [self.history_dict['rec_fail']]:
                # If the agent's recommendation failed in last turn, there is no need to re-sort items
                pass
            else:
                cand_item_score = self._item_score()
                self.item_score_compute_flag = False  # Item scores has been computed

                item_score_tuple = list(zip(self.cand_items, cand_item_score))
                sort_tuple = sorted(item_score_tuple, key=lambda x: x[1],
                                    reverse=True)  # update cand_items : sorted by score
                self.cand_items, self.cand_item_score = zip(*sort_tuple)

                # -------------------------------------------------------
            # == Get feedback from the user when agent recommend items ==
            reward, done = self._agent_rec_user_response(rec_num=self.rec_num)  # update agent: self.cand_item
            # -------------------------------------------------------

            # == update cand_items & state_vector(self.cand_item_num)
            self._update_cand_items(acc_feature=[], rej_feature=[], rec_suc=bool(done))

            #========================================
            if reward == 1:
                self._graph_path_extend(item_end=True)  # Update graph path
                print('-->Recommend successfully!')
            else:
                if self.command in [1, 2, 6, 7]:  # update attr_ent
                    self._update_feature_entropy(item_score_compute=self.item_score_compute_flag)
                    self._mini_sort_reachable_feature()  # move the top-k (k=ask_num) reachable feature to the front
                print('-->Recommend fail !')

        print('~~~~~~~~ dialog state info ~~~~~~~~~')
        # === print user's profile ===
        print(f"===User accept features: [{self.user_acc_feature}]{tuple(map(self.fea_map, self.user_acc_feature))}")
        print(
            f"===User ignore(weak reject) features: [{self.user_rej_feature}]{tuple(map(self.fea_map, self.user_rej_feature))}")
        # === print agent's info ===
        print(f'===Number of Graph reachable feature: [{len(self.reachable_feature)}]')
        print(f'===Number of candidate items: [{len(self.cand_items)}]')
        self.cur_conver_step += 1
        return self._get_state(), reward, done

    def _graph_path_extend(self, user_start=False, item_end=False, acc_feature_node=None):

        if user_start is True:
            self.graph_path = [f'User node:[{self.user_id}]'] + [f'feature node:{acc_feature_node}{tuple(map(self.fea_map, acc_feature_node))}']
            cprint(f'[Graph Path Jump] User node:[{self.user_id}] -->  feature node:{acc_feature_node}{tuple(map(self.fea_map, acc_feature_node))}')
        elif item_end is True:
            # == perform path jump when user accept item that recommended by agent
            cprint(f'[Graph Path Jump] feature node:{self.cur_node_set}{tuple(map(self.fea_map, self.cur_node_set))} -->  Item node:{self.cand_items}')
            self.graph_path = self.graph_path + [f'Item node:{self.cand_items}']
        else:
            # == perform path jump when user accept new feature that asked by agent
            cprint(f'[Graph Path Jump] feature node:{self.cur_node_set}{tuple(map(self.fea_map, self.cur_node_set))} -->  feature node:{acc_feature_node}{tuple(map(self.fea_map, acc_feature_node))}')
            self.graph_path = self.graph_path + [f'feature node:{acc_feature_node}{tuple(map(self.fea_map, acc_feature_node))}']


    def _update_cur_node_set(self, acc_feature_node):
        # == [latest jump point(set) on the graph] ===
        self.cur_node_set = acc_feature_node  # update current_feature_node set on the graph

    def _user_init_prefer_feature(self, user_init_prefer_fea_num=1):
        # Turn 1: The user initializes the preferred features when start a conversation
        user_like_random_fea = random.sample(self.kg.G['item'][self.target_item]['belong_to'],
                                             k=user_init_prefer_fea_num)
        self.cur_node_set = user_like_random_fea  # update current_feature_node set
        cprint(f'[User init preferred feature] I like [{user_like_random_fea}]{tuple(map(self.fea_map, user_like_random_fea))}!')
        return user_like_random_fea

    def _updata_reachable_feature(self):
        # ===== Graph path reasoning [Turn x : from current node jump to next feature_node in 1-hop]====
        next_reachable_feature = []
        for cur_node in self.cur_node_set:   # update reachable feature ( from current node jump to next feature_node in 1-hop)
            if self.data_name in ['LAST_FM', 'LAST_FM_STAR']:
                # ====[A-U-A  1-hop jump]  co-users: collaborative filtering
                fea_liked_by_users = list(self.kg.G['feature'][cur_node]['like'])  # A-U : users who like cur_feature
                user_friends = self.kg.G['user'][self.user_id]['friends']  #  U-U :  target user's friends
                cand_fea_like_users = list(set(fea_liked_by_users) & set(user_friends))  # co-users: collaborative filtering
                for user_id in cand_fea_like_users:  # A-U-A  # U in [friends & fea_liked_by_user]
                    next_reachable_feature.extend(list(self.kg.G['user'][user_id]['like']))
                next_reachable_feature = list(set(next_reachable_feature))

            # ====[A-I-A  1-hop jump]  co_items: collaborative filtering
            fea_belong_items = list(self.kg.G['feature'][cur_node]['belong_to'])  # A-I : items that have this cur_feature
            cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items)) # co_items: collaborative filtering
            for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items & fea_related_items]
                next_reachable_feature.extend(list(self.kg.G['item'][item_id]['belong_to']))
            next_reachable_feature = list(set(next_reachable_feature))
        # ==== update self.reachable_feature
        self.reachable_feature = next_reachable_feature  # next reachable_feature in 1-hop

    def _mini_sort_reachable_feature(self):
        # Sort reachable features according to the entropy of features
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
            prefer_embed = self.feature_emb[self.user_acc_feature, :]  #np.array (x*64)
            for i in range(len(self.user_acc_feature)):
                score += np.inner(prefer_embed[i], item_embed)
            cand_item_score.append(score)
        return cand_item_score

    def _agent_ask_user_response(self, ask_num):
        '''
        :return:
            RL env feedback: reward, done,
            User response in current turn : acc_feature, rej_feature
        '''
        done = 0
        if len(self.reachable_feature) == 0:  #candidate features is empty
            reward = self.reward_dict['cand_none']
            done = 1
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  # update conver_his
            return reward, done, [], []
        # === 1. Get feature tuple which sort by predictional model
        sort_mini_feas = self.reachable_feature[:ask_num]

        # === 2. User feedback: accept & reject
        acc_feature = list(set(sort_mini_feas) & set(self.feature_groundtrue))
        rej_feature = list(set(sort_mini_feas) - set(acc_feature))


        # === 3. Get action reward & Update state_vector(self.conver_his)
        if len(acc_feature):
            reward = self.reward_dict['ask_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']   #update conver_his
        else:
            reward = self.reward_dict['ask_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  #update conver_his
        return reward, done, acc_feature, rej_feature

    def _remove_asked_reachable_fea(self):
        self.reachable_feature = list(
            set(self.reachable_feature) - set(self.user_acc_feature))  # remove user accept feature
        self.reachable_feature = list(
            set(self.reachable_feature) - set(self.user_rej_feature))  # remove user reject feature

    def _update_user_profile(self, acc_feature, rej_feature, rec_suc=None):
        self.user_acc_feature += acc_feature
        self.user_rej_feature += rej_feature

    def _update_cand_items(self, acc_feature, rej_feature, rec_suc=None):
        """
        ==== CRS takes the attributes accepted by the user as a strong indicator ===
        :param acc_feature:  user accept features when agent asking
        :param rej_feature:  user reject features when agent asking
        :return:
        """
        if rec_suc is None:  # action : Agent asking feature
            if len(acc_feature):    #accept feature
                for feature_id in acc_feature:
                    feature_items = self.kg.G['feature'][feature_id]['belong_to']
                    self.cand_items = set(self.cand_items) & set(feature_items)   #  itersection
                self.cand_items = list(self.cand_items)
            if len(rej_feature):  # Agent only considers all items containing all attributes user accepts
                pass  #
        else: # action : Agent recommend items
            if rec_suc is False:
                self.cand_items = self.cand_items[self.rec_num:]  # update candidate items
                self.cand_item_score = self.cand_item_score[self.rec_num:] # update candidate items score
            elif rec_suc is True:
                self.cand_items = [self.target_item]  # update candidate items
        # update state vector : self.cand_item_num
        self.cand_item_num = [len(self.cand_items) >>d & 1 for d in range(self.bit_length)][::-1]  # binary



    def _agent_rec_user_response(self, rec_num):
        # ===  User feedback: Get action reward & Update state_vector(self.conver_his)
        rec_items = self.cand_items[: rec_num]  # TOP k item to recommend
        print(f'-->Agent recommend items id: {rec_items}')
        # print(f'-->Agent recommend items score: {cand_item_score[: rec_num]}')
        # cprint(f'-->Agent recommend TOP-10 total score:  {sum(cand_item_score[: rec_num])}')
        if self.target_item in rec_items:
            reward = self.reward_dict['rec_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu']  # update state vector: conver_his
            done = 1
        else:
            reward = self.reward_dict['rec_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  # update state vector: conver_his
            done = 0
        if self.cand_items == []:  # candidate items is empty
            done = 1
            reward = self.reward_dict['cand_none']
        return reward, done

    def _update_feature_entropy(self, item_score_compute=True):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))
            self.attr_count_dict = dict(Counter(cand_items_fea_list))
            self.attr_ent = [0] * self.attr_bit_state  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                p2 = 1.0 - p1
                if p1 == 1:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent
        elif self.ent_way == 'weight entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}

            # ====  compute candidate items score ==== high time cost
            if item_score_compute is True:
                self.cand_item_score = self._item_score()
            else:
                pass

            cand_item_score_sig = self.sigmoid(self.cand_item_score)  # sigmoid(score)
            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_bit_state  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            sum_score_sig = sum(cand_item_score_sig)

            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                p2 = 1.0 - p1
                if p1 == 1 or p1 <= 0:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()





