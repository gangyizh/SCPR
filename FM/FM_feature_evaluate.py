import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import time
from utils import *
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

import argparse
import sys

# from FM_model import FactorizationMachine



def predict_feature(model,user_output, given_preference, to_test):
    user_emb = model.ui_emb(torch.LongTensor([user_output]))[..., :-1].detach().numpy()
    #gp : 喜好的属性
    gp = model.feature_emb(torch.LongTensor(given_preference))[..., :-1].detach().numpy()
    emb_weight = model.feature_emb.weight[..., :-1].detach().numpy()
    result = list()
    #to_test: 除去喜好属性,剩余的特征
    for test_feature in to_test:
        temp = 0
        #TODO 增加 user*p
        temp += np.inner(user_emb, emb_weight[test_feature])
        for i in range(gp.shape[0]):
            temp += np.inner(gp[i], emb_weight[test_feature])
        result.append(temp)

    return result

def topk(y_true, pred, k):
    y_true_ = y_true[:k]
    pred_ = pred[:k]
    if sum(y_true_) == 0:
        return 0  # TODO: I change it to 0
    else:
        return roc_auc_score(y_true_, pred_)


def rank_by_batch(kg, pickle_file, iter_, bs, pickle_file_length, model, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):
    #TODO  pikle_file[3]---->pickle_file[4]    file is different from EAR
    #TODO  i_neg2_output  === i_cand_neg
    '''
    user_output, item_p_output, i_neg2_output, preference_list = list(), list(), list(), list()
    '''
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    III = pickle_file[3][left:right]
    IV = pickle_file[4][left:right]

    i = 0
    index_none = list()

    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        if i_neg2_output is None or len(i_neg2_output) == 0:
            index_none.append(i)
        i += 1
    #print('Non index length: {}'.format(len(index_none)))

    i = 0
    result_list = list()
    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        # Actually we only need preference list
        if i in index_none:
            i += 1
            continue
        full_feature = kg.G[ITEM][item_p_output][ITEM_FEATURE]
        preference_feature = preference_list
        residual_preference = list(set(full_feature) - set(preference_feature))
        residual_feature_all = list(set(list(range(feature_length - 1))) - set(full_feature))

        if len(residual_preference) == 0:
            continue
        # 就是要看到 residual preference(p3得分高)  在 剩余的全体feature中 被排到了怎样的位置
        to_test = residual_feature_all + residual_preference

        #计算 p*p_u  p=to_test (包括剩余的feature（pn,... | p3,p4)  p_u = p1,p2
        #TODO  得分计算修正  u*p  加上
        predictions = predict_feature(model, user_output, preference_feature, to_test)
        predictions = np.array(predictions)


        mini_gtitems = [residual_preference]
        num_gt = len(mini_gtitems)
        num_neg = len(to_test) - num_gt
        # auc_predictions = np.reshape(predictions[:-num_gt], (1, num_neg))
        # gt_predicitons = np.reshape(predictions[-num_gt:], (num_gt, 1))
        predictions = predictions.reshape((len(to_test), 1)[0])
        y_true = [0] * len(predictions)
        #y_true[-1] = 1
        for i in range(len(residual_preference)):
            y_true[-(i + 1)] = 1
        tmp = list(zip(y_true, predictions))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        y_true, predictions = zip(*tmp)

        icon = []

        for index, item in enumerate(y_true):
            if item > 0:
                icon.append(index)

        #print('icon is: {}, known: {}'.format(icon, len(preference_list)))

        auc = roc_auc_score(y_true, predictions)
        # print('full_feature len:{} preference_list len:{}'.format(len(full_feature), len(preference_list)))
        # print('user_output:{} auc:{}'.format(user_output, auc))
        result_list.append((auc, topk(y_true, predictions, 100), topk(y_true, predictions, 200)
                            , topk(y_true, predictions, 500), topk(y_true, predictions, 1000),
                            topk(y_true, predictions, 2000), len(predictions)))
        # a = topk(y_true, predictions, 10)
        # b = topk(y_true, predictions, 50)
        # c = topk(y_true, predictions, 100)
        # d = topk(y_true, predictions, 200)
        # e = topk(y_true, predictions, 500)
        i += 1
    return result_list


def evaluate_feature(kg, model, epoch, filename, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):
    # TODO add const PAD_IDX1, PAD_IDX2, user_length, data_name, ITEM, ITEM_FEATURE
    model.eval()
    model.cpu()
    tt = time.time()
    pickle_file = load_fm_sample(dataset=data_name, mode='valid')

    print('Open evaluation pickle file: takes {} seconds, evaluation length: {}'.format(time.time() - tt, len(pickle_file[0])))
    pickle_file_length = len(pickle_file[0])

    start = time.time()
    print('Starting {} epoch'.format(epoch))
    bs = 64
    max_iter = int(pickle_file_length / float(bs))


    # TODO: Uncomment this to do the full evaluation
    #max_iter = 100

    result = list()
    print('max_iter-----------', max_iter)
    for iter_ in range(max_iter):
        #print('iter-----', iter_)
        if iter_ > 1 and iter_ % 20 == 0:
            print('--')
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                        float(iter_) * 100 / max_iter))
        result += rank_by_batch(kg, pickle_file, iter_, bs, pickle_file_length, model, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)
        # auc_mean = np.mean(np.array([item[0] for item in result]))
        # auc_mean1 = np.mean(np.array([item[1] for item in result if item[1] > 0]))
        # print('result all auc:{} top100 auc:{}'.format(auc_mean, auc_mean1))

    auc_mean = np.mean(np.array([item[0] for item in result]))
    auc_median = np.median(np.array([item[0] for item in result]))
    print('auc mean: {}'.format(auc_mean), 'auc median: {}'.format(auc_median))

    auc_mean1 = np.mean(np.array([item[1] for item in result if item[1] > 0]))
    auc_median1 = np.median(np.array([item[1] for item in result if item[1] > 0]))
    tot_num1 = sum([1 for item in result if item[6] > 10])
    print('top 100: auc mean: {}'.format(auc_mean1), 'auc median: {}'.format(auc_median1), 'over num {}'.format(tot_num1))

    auc_mean2 = np.mean(np.array([item[2] for item in result if item[2] > 0]))
    auc_median2 = np.median(np.array([item[2] for item in result if item[2] > 0]))
    tot_num2 = sum([1 for item in result if item[6] > 50])
    print('top 200: auc mean: {}'.format(auc_mean2), 'auc median: {}'.format(auc_median2), 'over num {}'.format(tot_num2))

    auc_mean3 = np.mean(np.array([item[3] for item in result if item[3] > 0]))
    auc_median3 = np.median(np.array([item[3] for item in result if item[3] > 0]))
    tot_num3 = sum([1 for item in result if item[6] > 100])
    print('top 500: auc mean: {}'.format(auc_mean3), 'auc median: {}'.format(auc_median3), 'over num {}'.format(tot_num3))

    auc_mean4 = np.mean(np.array([item[4] for item in result if item[4] > 0]))
    auc_median4 = np.median(np.array([item[4] for item in result if item[4] > 0]))
    tot_num4 = sum([1 for item in result if item[6] > 200])
    print('top 1000: auc mean: {}'.format(auc_mean4), 'auc median: {}'.format(auc_median4), 'over num {}'.format(tot_num4))

    auc_mean5 = np.mean(np.array([item[5] for item in result if item[5] > 0]))
    auc_median5 = np.median(np.array([item[5] for item in result if item[5] > 0]))
    tot_num5 = sum([1 for item in result if item[6] > 500])
    print('top 2000: auc mean: {}'.format(auc_mean5), 'auc median: {}'.format(auc_median5), 'over num {}'.format(tot_num5))

    PATH = TMP_DIR[data_name] + '/FM-log-merge/' + filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('validating {} epoch on feature prediction\n'.format(epoch))
        auc_mean = np.mean(np.array([item[0] for item in result]))
        auc_median = np.median(np.array([item[0] for item in result]))
        f.write('auc mean: {}\n'.format(auc_mean))
        f.write('auc median: {}\n'.format(auc_median))
        f.write('top 100 auc mean: {}\n'.format(auc_mean1))
        f.write('top 100 auc median: {}\n'.format(auc_median1))
        f.write('top 200 auc mean: {}\n'.format(auc_mean2))
        f.write('top 200 auc median: {}\n'.format(auc_median2))
        f.write('top 500 auc mean: {}\n'.format(auc_mean3))
        f.write('top 500 auc median: {}\n'.format(auc_median3))
        f.write('top 1000 auc mean: {}\n'.format(auc_mean4))
        f.write('top 1000 auc median: {}\n'.format(auc_median4))
        f.write('top 2000 auc mean: {}\n'.format(auc_mean5))
        f.write('top 2000 auc median: {}\n'.format(auc_median5))
        f.flush()
    model.train()
    cuda_(model)


def main():
    # parser = argparse.ArgumentParser(description="Run DeepFM-BPR.")
    # parser.add_argument('-hs', type=int, metavar='<hs>', dest='hs', help='hs')  # hidden size
    # parser.add_argument('-ip', type=float, metavar='<ip>', dest='ip', help='ip')  # init parameter for hidden
    # parser.add_argument('-dr', type=float, metavar='<dr>', dest='dr', help='dr')  # dropout ratio
    # parser.add_argument('-command', type=int, metavar='<command>', dest='command', help='command')
    # args = parser.parse_args()
    parser = argparse.ArgumentParser(description="Run DeepFM-BPR.")
    parser.add_argument('-lr', type=float, default=0.02, metavar='<lr>', dest='lr', help='lr')
    parser.add_argument('-flr', type=float, default=0.0001, metavar='<flr>', dest='flr',
                        help='flr')  # means Feature update Learning Rate
    parser.add_argument('-reg', type=float, default=0.001, metavar='<reg>', dest='reg',
                        help='reg')  # FM_parameters regular terms
    parser.add_argument('-decay', type=float, default=0.0, metavar='<decay>', dest='decay', help='decay')
    parser.add_argument('-qonly', type=int, default=1, metavar='<qonly>', dest='qonly',
                        help='qonly')  # qonly means we drop 一次项
    parser.add_argument('-bs', type=int, default=64, metavar='<bs>', dest='bs', help='bs')  # batch size
    parser.add_argument('-hs', type=int, default=64, metavar='<hs>', dest='hs',
                        help='hs')  # hidden size & embedding size
    parser.add_argument('-ip', type=float, default=0.01, metavar='<ip>', dest='ip',
                        help='ip')  # init parameter for hidden
    parser.add_argument('-dr', type=float, default=0.5, metavar='<dr>', dest='dr', help='dr')  # dropout ratio
    parser.add_argument('-optim', type=str, default='Ada', metavar='<optim>', dest='optim', help='optim')
    parser.add_argument('-observe', type=int, metavar='<observe>', dest='observe', help='observe')
    parser.add_argument('-pretrain', type=int, default=0, metavar='<pretrain>', dest='pretrain',
                        help='pretrain')  # 可以选择使用哪一种pretrain
    parser.add_argument('-uf', type=int, default=1, metavar='<uf>', dest='uf',
                        help='uf')  # update feature 的缩写  1:update
    parser.add_argument('-rd', type=int, default=0, metavar='<rd>', dest='rd',
                        help='rd')  # remove duplicate 的缩写, 把 preference 和 feature 中相同的 去除
    parser.add_argument('-useremb', type=int, metavar='<useremb>', dest='useremb', help='user embedding')
    parser.add_argument('-freeze', type=int, default=0, metavar='<freeze>', dest='freeze', help='freeze')
    parser.add_argument('-command', type=int, default=8, metavar='<command>', dest='command', help='command')
    parser.add_argument('-seed', type=int, default=0, metavar='<seed>', dest='seed', help='seed')
    args = parser.parse_args()
    global PAD_IDX1, PAD_IDX2
    global user_length, item_length, feature_length
    global data_name
    global ITEM, ITEM_FEATURE
    ITEM = 'item'
    ITEM_FEATURE = 'belong_to'
    data_dir = 'data/lastfm/'
    data_name = LAST_FM
    dataset = load_dataset(data_name)
    kg = load_kg(data_name)
    user_length = int(getattr(dataset, 'user').value_len)
    item_length = int(getattr(dataset, 'item').value_len)
    feature_length = int(getattr(dataset, 'feature').value_len)
    print('user_length:{},item_length:{},feature_length:{}'.format(user_length, item_length, feature_length))
    PAD_IDX1 = user_length + item_length
    PAD_IDX2 = feature_length
    # if args.command == 7:
    #     model = FactorizationMachine(emb_size=64, user_length=user_length, item_length=item_length,
    #                                  feature_length=feature_length, qonly=1, command=args.command, hs=args.hs, ip=args.ip,
    #                                  dr=args.dr, old_new='new')
    #     #fp = '../../data/FM-model-new/v14-test-FM-lr-0.02-reg-0.001-decay-0.0-qonly-1-bs-64-command-7-hs-128-ip-0.01-dr-0.5-epoch-40.pt'
    #     fp = '../../data/FM-model-new/v15-test-FM-lr-0.02-reg-0.001502-decay-0.0-qonly-1-bs-64-command-7-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-newepoch-30.pt'
    #     fp = '../../data/FM-model-new/v15-test-FM-lr-0.02-reg-0.0015-decay-0.0-qonly-1-bs-64-command-7-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-newepoch-40.pt'
    # if args.command == 8:
    #     model = FactorizationMachine(emb_size=64, user_length=user_length, item_length=item_length,
    #                                  feature_length=feature_length, qonly=1, command=args.command, hs=args.hs, ip=args.ip,
    #                                  dr=args.dr, old_new='new')
    #     fp = '../../data/FM-model-new/v13-test-FM-lr-0.02-reg-0.005-decay-0.0-qonly-1-bs-64-command-8epoch-40.pt'
    #     fp = '../../data/FM-model-new/v15-test-FM-lr-0.02-reg-0.0015-decay-0.0-qonly-1-bs-64-command-8-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-newepoch-40.pt'
    #     fp = '../../data/FM-model-latest/v20-test-FM-lr-0.01-flr-0.0001-reg-0.002-decay-0.0-qonly-1-bs-64-command-8-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-new-pretrain-2-uf-1-rd-0-freeze-0epoch-6.pt'
    model = FactorizationMachine(emb_size=args.hs, user_length=user_length, item_length=item_length,
                                 feature_length=feature_length, qonly=args.qonly, hs=args.hs, ip=args.ip, dr=args.dr)
    #fp = './tmp/last_fm/FM-model-merge/v1-test-FM-lr-0.02-flr-0.0001-reg-0.001-decay-0.0-qonly-1-bs-64-command-8-hs-64-ip-0.01-dr-0.5-optim-Ada-pretrain-0-uf-1-rd-0-freeze-0-seed-0-useremb-None-epoch-0.pt'
    fp = './tmp/last_fm/FM-model-merge/v4-FM-lr-0.01-flr-0.001-reg-0.002-decay-0.0-qonly-epoch-0.pt'
    model.load_state_dict(torch.load(fp))
    print('Model loaded successfully!')
    print('Evaluating on feature similarity')
    filename = 'v1-test-FM-lr-{}-flr-{}-reg-{}-decay-{}-qonly-{}-bs-{}-command-{}-hs-{}-ip-{}-dr-{}-optim-{}-pretrain-{}-uf-{}-rd-{}-freeze-{}-seed-{}-useremb-{}'.format(
        args.lr, args.flr, args.reg, args.decay, args.qonly,
        args.bs, args.command, args.hs, args.ip, args.dr, args.optim, args.pretrain, args.uf, args.rd, args.freeze,
        args.seed, args.useremb)
    #filename = 'v4-FM-lr-0.01-flr-0.001-reg-0.002-decay-0.0-qonly'
    epoch = 0
    evaluate_feature(kg, model, epoch, filename, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM,
                     ITEM_FEATURE)

    #evaluate_feature(kg, model, 1, 'lala.txt')


if __name__ == '__main__':
    main()
