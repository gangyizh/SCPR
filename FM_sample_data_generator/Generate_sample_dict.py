import os
import time
from _tkinter import _flatten
from collections import Counter

from sample_utils import *
import numpy as np
import argparse

def generate_sample_dict(data_name, neg2_num=10, mode='train'):
    """
    item_sample_dict : {'item_id': [(prefer_feas, neg2_items), (prefer_feas, neg2_items)...]}
    :return: dict
    For each item, simulate each turn of the conversation, i.e. the attributes the user likes in the conversation and the corresponding second type negative item sample pool
    [the second type of negative items pool (sample_number=neg2_num) is given for specific user preference attributes (Max_entropy simulation strategy) ]
    """
    print(f"Start Generate {data_name} {mode}-sample_dict!")
    data_set = load_dataset(data_name)  #  dataset
    kg = load_kg(data_name)  # KG
    item_sample_dict = dict()
    items_len = getattr(data_set, 'item').value_len
    start = time.time()
    for item_id in range(items_len):
        #print('=======================item:{}=============='.format(item_id))
        if item_id > 1 and item_id % 500 == 0:
            print('--')
            print('Takes {} seconds to finish {}% of this {} task'.format(str(time.time() - start),
                                                                        float(item_id) * 100 / items_len, mode))
        prefer_all = kg.G['item'][item_id]['belong_to']
        prefer_len = len(prefer_all)
        item_sample_dict[str(item_id)] = []
        if mode == 'train':
            for start_fea in prefer_all:  # enumerate all feature
                prefer_part = [start_fea]
                neg_cand_items = list(set(kg.G['feature'][start_fea]['belong_to']) - set([item_id]))  # neg_cand_items: cand_items - item
                if len(neg_cand_items) > neg2_num:  #neg2_num = 10
                    choose_neg_items = list(map(int, np.random.choice(neg_cand_items, size=neg2_num, replace=False)))
                else:   # neg_cand_items < neg2_num  next start_fea
                    choose_neg_items = neg_cand_items
                    mytuple = (prefer_part.copy(), choose_neg_items)
                    item_sample_dict[str(item_id)].append(mytuple)
                    continue

                mytuple = (prefer_part.copy(), choose_neg_items)
                item_sample_dict[str(item_id)].append(mytuple)


                while len(prefer_part) != prefer_len:
                    prefer_fea = list(set(prefer_all) - set(prefer_part))   # remove confirmed preference feature
                    confirm_attr = max_ent_fea_in_cand_items(kg, neg_cand_items, prefer_fea, data_name=data_name)
                    prefer_part.append(confirm_attr)  #update prefer_part
                    neg_cand_items = list(set(neg_cand_items) & set(kg.G['feature'][confirm_attr]['belong_to'])) #update_neg_cand_items
                    #random k neg_cand_items
                    if len(neg_cand_items) > neg2_num:  # neg2_num = 10
                        choose_neg_items = list(
                            map(int, np.random.choice(neg_cand_items, size=neg2_num, replace=False)))
                    elif len(neg_cand_items) == 0:    #no cand_items
                        #print('item id:{}: prefer part:{} has no neg_cand_items'.format(item_id, prefer_part))
                        break
                    else:  # neg_cand_items < neg2_num
                        choose_neg_items = neg_cand_items

                    mytuple = (prefer_part.copy(), choose_neg_items)
                    item_sample_dict[str(item_id)].append(mytuple)
        elif mode in ['valid', 'test']:
            start_fea = np.random.choice(prefer_all)
            prefer_part = [start_fea]
            neg_cand_items = list(set(kg.G['feature'][start_fea]['belong_to']) - set([item_id]))   # neg_cand_items: cand_items - item
            if len(neg_cand_items) > neg2_num:  # neg2_num = 10
                choose_neg_items = list(map(int, np.random.choice(neg_cand_items, size=neg2_num, replace=False)))
            else:  # neg_cand_items < neg2_num  next start_fea
                choose_neg_items = neg_cand_items
                mytuple = (prefer_part.copy(), choose_neg_items)
                item_sample_dict[str(item_id)].append(mytuple)
                continue
            mytuple = (prefer_part.copy(), choose_neg_items)
            item_sample_dict[str(item_id)].append(mytuple)


            while len(prefer_part) != prefer_len:
                prefer_fea = list(set(prefer_all) - set(prefer_part))  # remove confirmed preference feature
                confirm_attr = max_ent_fea_in_cand_items(kg, neg_cand_items, prefer_fea, data_name=data_name)
                prefer_part.append(confirm_attr)  # update prefer_part
                neg_cand_items = list(set(neg_cand_items) & set(kg.G['feature'][confirm_attr]['belong_to']))  # update_neg_cand_items
                # random k neg_cand_items
                if len(neg_cand_items) > neg2_num:  # neg2_num = 300
                    choose_neg_items = list(map(int, np.random.choice(neg_cand_items, size=neg2_num, replace=False)))
                elif len(neg_cand_items) == 0:  # no cand_items
                    # print('item id:{}: prefer part:{} has no neg_cand_items'.format(item_id, prefer_part))
                    break
                else:  # neg_cand_items < neg2_num
                    choose_neg_items = neg_cand_items

                mytuple = (prefer_part.copy(), choose_neg_items)
                item_sample_dict[str(item_id)].append(mytuple)

    save_sample_dict(data_name, item_sample_dict, mode=mode)

def max_ent_fea_in_cand_items(kg, cand_items, prefer_fea, data_name):
    cand_items_fea_list = []
    for item_id in cand_items:
        cand_items_fea_list.append(list(kg.G['item'][item_id]['belong_to']))
    cand_items_fea_list = list(_flatten(cand_items_fea_list))
    attr_count_dict = dict(Counter(cand_items_fea_list))

    ent_list = []
    for fea_id in prefer_fea:
        if fea_id in attr_count_dict.keys():
            p1 = float(attr_count_dict[fea_id])/len(cand_items)
            p2 = 1.0 - p1
            if p1 == 1:
                ent_list.append(0)
            else:
                ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                ent_list.append(ent)
        else:
            ent_list.append(0)

    max_ent_index = ent_list.index(max(ent_list))
    max_ent_fea = prefer_fea[max_ent_index]
    return max_ent_fea




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate DeepFM-BPR Data.")
    parser.add_argument('-start', type=int, default=0, metavar='<start>', dest='start', help='start')
    parser.add_argument('-seed', type=int, default=1, metavar='<seed>', dest='seed', help='seed')
    parser.add_argument('--data_name', type=str, default='YELP', metavar='<data_name>', dest='data_name',
                        choices=[LAST_FM, LAST_FM_STAR, YELP],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP}. '
                             'YELP and YELP_STAR share the same data structure (both use fine-grained attributes)')
    # parser.add_argument('-mode', type=str, default='train', help='the mode in [train, valid]')  # {train}
    args = parser.parse_args()
    set_random_seed(args.seed)
    data_name = args.data_name

    generate_sample_dict(data_name=data_name, neg2_num=10, mode='train')
    generate_sample_dict(data_name=data_name, neg2_num=300, mode='valid')

