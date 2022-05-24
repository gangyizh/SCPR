import pickle
import numpy as np
import random
import torch
import os
import sys
import json
LAST_FM = 'LAST_FM'
LAST_FM_STAR = 'LAST_FM_STAR'
YELP = 'YELP'
YELP_STAR = 'YELP_STAR'

DATA_DIR = {
    LAST_FM: '../data/lastfm',
    YELP: '../data/yelp',
    LAST_FM_STAR: '../data/lastfm_star',
    YELP_STAR: '../data/yelp',
}
TMP_DIR = {
    LAST_FM: '../tmp/last_fm',
    YELP: '../tmp/yelp',
    LAST_FM_STAR: '../tmp/last_fm_star',
    YELP_STAR: '../tmp/yelp_star',
}

def load_ui_data(dataset, mode):
    train_file = DATA_DIR[dataset] + '/UI_Interaction_data/review_dict_{}.json'.format(mode)
    with open(train_file, 'r') as f:
        UI_data = json.load(f)
    return UI_data



def load_dataset(dataset):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj

def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def save_sample_dict(dataset, sample_dict, mode='train'):
    sample_file = TMP_DIR[dataset] + '/FM-sample-data/{}-sample-dict.pickle'.format(mode)
    if not os.path.isdir(TMP_DIR[dataset] + '/FM-sample-data/'):
        os.makedirs(TMP_DIR[dataset] + '/FM-sample-data/')
    pickle.dump(sample_dict, open(sample_file, 'wb'))

def load_sample_dict(dataset, mode='train'):
    sample_file = TMP_DIR[dataset] + '/FM-sample-data/{}-sample-dict.pickle'.format(mode)
    sample_dict = pickle.load(open(sample_file, 'rb'))
    return sample_dict

def save_fm_sample(dataset, sample_data, mode, epoch=0):
    if mode == 'train':
        sample_file = DATA_DIR[dataset] + '/FM_sample_data/sample_fm_data_{}-{}.pkl'.format(mode, epoch)
    if mode == 'valid':
        sample_file = DATA_DIR[dataset] + '/FM_sample_data/sample_fm_data_{}.pkl'.format(mode)
    print('save fm_{}_data in {}'.format(mode, sample_file))
    pickle.dump(sample_data, open(sample_file, 'wb'))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
