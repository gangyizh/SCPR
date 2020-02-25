import os
import numpy as np
import gzip
import pickle
import pandas as pd
import json
from easydict import EasyDict as edict


class LastFmDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_entities()
        self.load_relations()
    def get_relation(self):
        #Entities
        USER = 'user'
        ITEM = 'item'
        FEATURE = 'feature'

        #Relations
        INTERACT = 'interact'
        FRIEND = 'friends'
        LIKE = 'like'
        BELONG_TO = 'belong_to'
        relation_name = [INTERACT, FRIEND, LIKE, BELONG_TO]

        fm_relation = {
            USER: {
                INTERACT: ITEM,
                FRIEND: USER,
                LIKE: FEATURE,
            },
            ITEM: {
                BELONG_TO: FEATURE,
                INTERACT: USER
            },
            FEATURE: {
                LIKE: USER,
                BELONG_TO: ITEM
            }
        }
        fm_relation_link_entity_type = {
            INTERACT:  [USER, ITEM],
            FRIEND:  [USER, USER],
            LIKE:  [USER, FEATURE],
            BELONG_TO:  [ITEM, FEATURE]
        }
        return fm_relation, relation_name, fm_relation_link_entity_type
    def load_entities(self):
        entity_files = edict(
            user='user_dict.json',
            item='item_dict.json',
            feature='tag_map.json',
        )
        for entity_name in entity_files:
            with open(os.path.join(self.data_dir,entity_files[entity_name]), encoding='utf-8') as f:
                mydict = json.load(f)
            if entity_name == 'feature':
                entity_id = list(mydict.values())
            else:
                entity_id = list(map(int, list(mydict.keys())))
            #entity_value = list(entity_df.iloc[:, 1].values)
            #setattr(self, entity_name, edict(id=entity_id, value=entity_value, value_len=max(entity_id)+1))  #len include the id of 0
            setattr(self, entity_name, edict(id=entity_id, value_len=max(entity_id)+1))  #len include the id of 0
            print('Load', entity_name, 'of size', len(entity_id))
            print(entity_name, 'of max id is', max(entity_id))

    def load_relations(self):
        """
        relation: head entity---> tail entity
        --
        """
        LastFm_relations = edict(
            interact=('user_item.json', self.user, self.item), #(filename, head_entity, tail_entity)
            friends=('user_dict.json', self.user, self.user),
            like=('user_dict.json', self.user, self.feature),
            belong_to=('item_dict.json', self.item, self.feature),
        )
        for name in LastFm_relations:
            #  Save tail_entity
            # 'data' saves list of entity_tail indices   data[1]=[2,3,4]  denote head_entity 1 have relation with tail_entity 2,3,4
            # 'ralation_id' saves tail_entity id
            # 'relation_distrib' saves tail_entity frequency distribution
            relation = edict(
                data=[],
                #relation_id=LastFm_relations[name][2].id,  #xxx tail_entity id
                #relation_distrib=np.zeros(LastFm_relations[name][2].value_len)  # tail_entity frequency distribution
            )
            knowledge = [list([]) for i in range(LastFm_relations[name][1].value_len)]  # empty list  length is the num of head_entity
            # load relation files
            with open(os.path.join(self.data_dir, LastFm_relations[name][0]), encoding='utf-8') as f:
                mydict = json.load(f)
            if name in ['interact']:
                for key, value in mydict.items():
                    head_id = int(key)
                    tail_ids = value
                    knowledge[head_id] = tail_ids
            elif name in ['friends', 'like']:
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str][name]
                    knowledge[head_id] = tail_ids
            elif name in ['belong_to']:
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str]['feature_index']
                    knowledge[head_id] = tail_ids
            relation.data = knowledge
            setattr(self, name, relation)
            tuple_num = 0
            for i in knowledge:
                tuple_num += len(i)
            print('Load', name, 'of size', tuple_num)


# if __name__ == '__main__':
#     from KG_data_generate.utils import *
#     data_dir = '../data/lastfm/'
#     data_name = 'lastfm'
#
#     # save_dir = 'data/train_test/'
#
#     # ----------------------------------------------
#     # Create Dataset for data
#     # ============BEGIN============
#
#     print('Load', data_name, 'dataset from file...')
#     print(TMP_DIR[data_name])
#     if not os.path.isdir(TMP_DIR[data_name]):
#         os.makedirs(TMP_DIR[data_name])
#     dataset = LastFmDataset(data_dir)
#     save_dataset(data_name, dataset)



