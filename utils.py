import pickle
import numpy as np
import random
import torch
import os
import sys
# from knowledge_graph import KnowledgeGraph
# from data_process import LastFmDataset
# from KG_data_generate.lastfm_small_data_process import LastFmSmallDataset
# from KG_data_generate.lastfm_knowledge_graph import KnowledgeGraph
#Dataset names
LAST_FM = 'lastfm'
LAST_FM_SMALL = 'lastfm_small'
YELP = 'yelp'
YELP_SMALL = 'yelp_small'

DATA_DIR = {
    LAST_FM: './data/lastfm',
    YELP: './data/yelp',
    LAST_FM_SMALL: './data/lastfm_small',
    YELP_SMALL: './data/yelp',
}
TMP_DIR = {
    LAST_FM: './tmp/last_fm',
    YELP: './tmp/yelp',
    LAST_FM_SMALL: './tmp/last_fm_small',
    YELP_SMALL: './tmp/yelp',
}
def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var
def save_dataset(dataset, dataset_obj):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)

def load_dataset(dataset):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj

def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))

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

def save_fm_sample(dataset, sample_data, mode):
    sample_file = TMP_DIR[dataset] + '/sample_fm_data_{}.pk1'.format(mode)
    print('save fm_{}_data in {}'.format(mode, sample_file))
    pickle.dump(sample_data, open(sample_file, 'wb'))

def load_fm_sample(dataset, mode):
    sample_file = TMP_DIR[dataset] + '/sample_fm_data_{}.pk1'.format(mode)
    with open(sample_file, 'rb') as f:
        sample_data = pickle.load(f)
    return sample_data

def load_fm_model(dataset, model, filename, epoch):
    model_file = TMP_DIR[dataset] + '/FM-model-merge/' + filename + '-epoch-{}.pt'.format(epoch)
    model_dict = torch.load(model_file)
    print('Model load at {}'.format(model_file))
    return model_dict

def save_fm_model(dataset, model, filename, epoch):
    model_file = TMP_DIR[dataset] + '/FM-model-merge/' + filename + '-epoch-{}.pt'.format(epoch)
    if not os.path.isdir(TMP_DIR[dataset] + '/FM-model-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/FM-model-merge/')
    torch.save(model.state_dict(), model_file)
    print('Model saved at {}'.format(model_file))


def load_embed(dataset, epoch):
    path = TMP_DIR[dataset] + '/FM-model-embeds/' + 'embeds-epoch-{}.pkl'.format(epoch)
    with open(path, 'rb') as f:
        embeds = pickle.load(f)
        print('FM Epoch：{} Embedding load successfully!'.format(epoch))
        return embeds

def save_embed(dataset, embeds, epoch):
    path = TMP_DIR[dataset] + '/FM-model-embeds/' + 'embeds-epoch-{}.pkl'.format(epoch)
    if not os.path.isdir(TMP_DIR[dataset] + '/FM-model-embeds/'):
        os.makedirs(TMP_DIR[dataset] + '/FM-model-embeds/')
    with open(path, 'wb') as f:
        pickle.dump(embeds, f)
        print('Embedding saved successfully!')


def save_fm_model_log(dataset, filename, epoch, epoch_loss, epoch_loss_2, train_len):
    PATH = TMP_DIR[dataset] + '/FM-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/FM-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/FM-log-merge/')
    with open(PATH, 'a') as f:
        f.write('Starting {} epoch\n'.format(epoch))
        f.write('training loss 1: {}\n'.format(epoch_loss / train_len))
        f.write('training loss 2: {}\n'.format(epoch_loss_2 / train_len))
        # f.write('1000 loss: {}\n'.format(loss_1000))


def save_fm_sample_log(dataset, log_data, head_name, mode='train'):
    sample_file = TMP_DIR[dataset] + '/sample_fm_{}_log.txt'.format(mode)
    with open(sample_file, mode='w+', encoding='utf-8') as f:
        for name in head_name:
            f.write(name+'\t')
        f.write('\n')
        np.savetxt(f, log_data, fmt='%s', delimiter='\t')



def load_rl_data(dataset, mode):
    path = TMP_DIR[dataset] + '/rl_data_{}'.format(mode) + '.txt'
    with open(path, mode='r+', encoding='utf-8') as f:
        data = np.loadtxt(path, dtype=np.int64, delimiter=' ', comments='#')
        print('load rl {} data successfully, size is {}'.format(mode, data.shape[0]))
        return data

def save_rl_data(dataset, data, head_name, mode):
    path = TMP_DIR[dataset] + '/rl_data_{}'.format(mode) + '.txt'
    with open(path, mode='w+', encoding='utf-8') as f:
        np.savetxt(path, data, fmt='%d', delimiter=' ', header=' '.join(head_name))


def load_rl_agent(dataset, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    model_dict = torch.load(model_file)
    print('RL policy model load at {}'.format(model_file))
    return model_dict

def save_rl_agent(dataset, model, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-agent/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-agent/')
    torch.save(model.state_dict(), model_file)
    print('RL policy model saved at {}'.format(model_file))



def save_rl_mtric(dataset, filename, epoch, SR, spend_time, mode='train'):
    PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user tuples\n'.format(epoch))
            f.write('training SR@5: {}\n'.format(SR[0]))
            f.write('training SR@10: {}\n'.format(SR[1]))
            f.write('training SR@15: {}\n'.format(SR[2]))
            f.write('training Avg@T: {}\n'.format(SR[3]))
            f.write('Spending time: {}\n'.format(spend_time))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))
    elif mode == 'test':
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR@5: {}\n'.format(SR[0]))
            f.write('Testing SR@10: {}\n'.format(SR[1]))
            f.write('Testing SR@15: {}\n'.format(SR[2]))
            f.write('Testing Avg@T: {}\n'.format(SR[3]))
            f.write('Testing time: {}\n'.format(spend_time))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))

def save_rl_model_log(dataset, filename, epoch, epoch_loss, train_len):
    PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    with open(PATH, 'a') as f:
        f.write('Starting {} epoch\n'.format(epoch))
        f.write('training loss : {}\n'.format(epoch_loss / train_len))
        # f.write('1000 loss: {}\n'.format(loss_1000))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__