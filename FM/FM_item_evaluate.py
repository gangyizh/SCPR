import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import *
import time
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

import argparse
import sys

# from FM_model import FactorizationMachine

def topk(y_true, pred, k):
    y_true_ = y_true[:k]
    pred_ = pred[:k]
    if sum(y_true_) == 0:
        return 0  # TODO: I change it to 0
    else:
        return roc_auc_score(y_true_, pred_)


def rank_by_batch(kg, pickle_file, iter_, bs, pickle_file_length, model, rd, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):
    # TODO  pikle_file[3]---->pickle_file[4]    file is different from EAR
    # TODO  i_neg2_output  === i_cand_neg
    '''
    user_output, item_p_output, i_neg2_output, preference_list = list(), list(), list(), list()
    '''
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    # #TODO EAR
    # III = pickle_file[2][left:right]
    # IV = pickle_file[3][left:right]
    # TODO EKG
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
        if i in index_none:
            i += 1
            continue


        #if i_neg2_output is None:
        #    continue



        # TODO:
        total_list = list(i_neg2_output)[: 1000] + [item_p_output]

        #pool = list(set(range(len(item_dict))) - set(_train_user_to_items[str(user_output)]))
        ## TODO: change this for debug
        #random.shuffle(pool)
        #total_list = pool[: 1000] + [item_p_output]

        user_input = [user_output] * len(total_list)
        #pos_list : user negitem/pos item;    pos_list2: item's feature
        pos_list, pos_list2 = list(), list()
        cumu_length = 0
        for instance in zip(user_input, total_list):
            new_list = list()
            new_list.append(instance[0])
            new_list.append(instance[1] + user_length)
            pos_list.append(torch.LongTensor(new_list))
            f = kg.G[ITEM][instance[1]][ITEM_FEATURE]
            # f = item_dict[str(instance[1])]['feature_index']
            if rd == 1:
                f = list(set(f) - set(preference_list))
            cumu_length += len(f)
            pos_list2.append(torch.LongTensor(f))

        #print('list length(normal): {}'.format(len(total_list)))
        if cumu_length == 0:
            pass
            #continue
            # TODO: means skip it.
            #print('This case, total list length: {}'.format(len(total_list)))
            #for instance in zip(user_input, total_list):
            #    new_list = list()
            #    new_list.append(instance[0])
            #    new_list.append(instance[1] + len(user_list))
            #    pos_list.append(torch.LongTensor(new_list))
            #    f = item_dict[str(instance[1])]['feature_index']
            #    #f = list(set(f) - set(preference_list))
            #    cumu_length += len(f)
            #    pos_list2.append(torch.LongTensor(f))

        pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
        prefer_list = torch.LongTensor(preference_list).expand(len(total_list), len(preference_list))
        # ADD by hc
        if cumu_length != 0:
            #属性多的靠前   补齐元素PAD_IDX2
            pos_list2.sort(key=lambda x: -1 * x.shape[0])
            pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)
        else:
            pos_list2 = torch.LongTensor([PAD_IDX2]).expand(pos_list.shape[0], 1)
            #p2 = []
            #for instance in zip(user_input, total_list):
            #   f = item_dict[str(instance[1])]['feature_index']
            #   #f = list(set(f) - set(preference_list))
            #   p2.append(torch.LongTensor(f))
            #pos_list2 = pad_sequence(p2, batch_first=True, padding_value=PAD_IDX2)


        predictions, _, _ = model(cuda_(pos_list), cuda_(pos_list2), cuda_(prefer_list))
        predictions = predictions.detach().cpu().numpy()

        mini_gtitems = [item_p_output]
        num_gt = len(mini_gtitems)
        num_neg = len(total_list) - num_gt
        # auc_predictions = np.reshape(predictions[:-num_gt], (1, num_neg))
        # gt_predicitons = np.reshape(predictions[-num_gt:], (num_gt, 1))
        predictions = predictions.reshape((num_neg + 1, 1)[0])
        y_true = [0] * len(predictions)
        y_true[-1] = 1
        tmp = list(zip(y_true, predictions))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        y_true, predictions = zip(*tmp)
        auc = roc_auc_score(y_true, predictions)

        result_list.append((auc, topk(y_true, predictions, 10), topk(y_true, predictions, 50)
                            , topk(y_true, predictions, 100), topk(y_true, predictions, 200),
                            topk(y_true, predictions, 500), len(predictions)))
        a = topk(y_true, predictions, 10)
        b = topk(y_true, predictions, 50)
        c = topk(y_true, predictions, 100)
        d = topk(y_true, predictions, 200)
        e = topk(y_true, predictions, 500)
        i += 1
        if a == 0.5:
            pass
    return result_list


def evaluate_item(kg, model, epoch, filename, rd, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):
    #TODO add const PAD_IDX1, PAD_IDX2, user_length, data_name, ITEM, ITEM_FEATURE
    model.eval()
    tt = time.time()
    #TODO EKG
    pickle_file = load_fm_sample(dataset=data_name, mode='valid')
    print('evaluate data:{}'.format(data_name))
    print('Open evaluation pickle file: takes {} seconds, evaluation length: {}'.format(time.time() - tt, len(pickle_file[0])))
    pickle_file_length = len(pickle_file[0])
    print('ui length:{}'.format(pickle_file_length))
    #-----------------------EARS-----------------------------
    #TODO EARS
   # pickle_file_path = './tmp/last_fm/v1-speed-valid-0.pickle'
   #  pickle_file_path = './tmp/last_fm/EAR-EKG-valid-0.pickle'
   #  with open(pickle_file_path, 'rb') as f:
   #      pickle_file = pickle.load(f)
   #  print('Open evaluation pickle file: {} takes {} seconds, evaluation length: {}'.format(pickle_file_path,
   #                                                                                         time.time() - tt,
   #                                                                                         len(pickle_file[0])))
   #  pickle_file_length = len(pickle_file[0])
    # -----------------------EARS-----------------------------
    start = time.time()
    print('Starting {} epoch'.format(epoch))
    bs = 64
    max_iter = int(pickle_file_length / float(bs))


    # TODO: Uncomment this to do the full evaluation
    # max_iter = 100


    result = list()
    #print('max_iter-----------', max_iter)
    for iter_ in range(max_iter):
        #print('iter-----', iter_)
        if iter_ > 1 and iter_ % 50 == 0:
            print('--')
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                        float(iter_) * 100 / max_iter))
        result += rank_by_batch(kg, pickle_file, iter_, bs, pickle_file_length, model, rd, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)
        # auc_mean = np.mean(np.array([item[0] for item in result]))
        # auc_mean1 = np.mean(np.array([item[1] for item in result if item[1] > 0]))
        # print('result all auc:{} top100 auc:{}'.format(auc_mean, auc_mean1))

    auc_mean = np.mean(np.array([item[0] for item in result]))
    auc_median = np.median(np.array([item[0] for item in result]))
    print('auc mean: {}'.format(auc_mean), 'auc median: {}'.format(auc_median),
          'over num {}'.format(len(result)))

    auc_mean1 = np.mean(np.array([item[1] for item in result if item[1] > 0]))
    auc_median1 = np.median(np.array([item[1] for item in result if item[1] > 0]))
    tot_num1 = sum([1 for item in result if item[6] > 10])
    print('top 10: auc mean: {}'.format(auc_mean1), 'auc median: {}'.format(auc_median1), 'over num {}'.format(tot_num1))

    auc_mean2 = np.mean(np.array([item[2] for item in result if item[2] > 0]))
    auc_median2 = np.median(np.array([item[2] for item in result if item[2] > 0]))
    tot_num2 = sum([1 for item in result if item[6] > 50])
    print('top 50: auc mean: {}'.format(auc_mean2), 'auc median: {}'.format(auc_median2), 'over num {}'.format(tot_num2))

    auc_mean3 = np.mean(np.array([item[3] for item in result if item[3] > 0]))
    auc_median3 = np.median(np.array([item[3] for item in result if item[3] > 0]))
    tot_num3 = sum([1 for item in result if item[6] > 100])
    print('top 100: auc mean: {}'.format(auc_mean3), 'auc median: {}'.format(auc_median3), 'over num {}'.format(tot_num3))

    auc_mean4 = np.mean(np.array([item[4] for item in result if item[4] > 0]))
    auc_median4 = np.median(np.array([item[4] for item in result if item[4] > 0]))
    tot_num4 = sum([1 for item in result if item[6] > 200])
    print('top 200: auc mean: {}'.format(auc_mean4), 'auc median: {}'.format(auc_median4), 'over num {}'.format(tot_num4))

    auc_mean5 = np.mean(np.array([item[5] for item in result if item[5] > 0]))
    auc_median5 = np.median(np.array([item[5] for item in result if item[5] > 0]))
    tot_num5 = sum([1 for item in result if item[6] > 500])
    print('top 500: auc mean: {}'.format(auc_mean5), 'auc median: {}'.format(auc_median5), 'over num {}'.format(tot_num5))

    PATH = TMP_DIR[data_name] + '/FM-log-merge/' + filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('validating {} epoch on item prediction\n'.format(epoch))
        auc_mean = np.mean(np.array([item[0] for item in result]))
        auc_median = np.median(np.array([item[0] for item in result]))
        f.write('auc mean: {}\n'.format(auc_mean))
        f.write('auc median: {}\n'.format(auc_median))
        f.write('top 10 auc mean: {}\n'.format(auc_mean1))
        f.write('top 10 auc median: {}\n'.format(auc_median1))
        f.write('top 50 auc mean: {}\n'.format(auc_mean2))
        f.write('top 50 auc median: {}\n'.format(auc_median2))
        f.write('top 100 auc mean: {}\n'.format(auc_mean3))
        f.write('top 100 auc median: {}\n'.format(auc_median3))
        f.write('top 200 auc mean: {}\n'.format(auc_mean4))
        f.write('top 200 auc median: {}\n'.format(auc_median4))
        f.write('top 500 auc mean: {}\n'.format(auc_mean5))
        f.write('top 500 auc median: {}\n'.format(auc_median5))
        f.flush()
    model.train()
    cuda_(model)


def main():
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
    data_name = LAST_FM_SMALL
    dataset = load_dataset(data_name)
    kg = load_kg(data_name)
    user_length = int(getattr(dataset, 'user').value_len)
    item_length = int(getattr(dataset, 'item').value_len)
    feature_length = int(getattr(dataset, 'feature').value_len)
    print('user_length:{},item_length:{},feature_length:{}'.format(user_length, item_length, feature_length))
    PAD_IDX1 = user_length + item_length
    PAD_IDX2 = feature_length
    #model = FactorizationMachine(emb_size=64, user_length=len(user_list), item_length=len(busi_list),
    #                             feature_length=FEATURE_COUNT, qonly=args.qonly, command=args.command, hs=args.hs, ip=args.ip, dr=args.dr)


    # if args.command == 8:
    #     model = FactorizationMachine(emb_size=64, user_length=user_length, item_length=user_length,
    #                                  feature_length=feature_length, qonly=1, hs=args.hs, ip=args.ip,
    #                                  dr=args.dr)
    #     fp = '../../data/FM-model-new/v13-test-FM-lr-0.02-reg-0.005-decay-0.0-qonly-1-bs-64-command-8epoch-40.pt'
    #     fp = '../../data/FM-model-new/v15-test-FM-lr-0.02-reg-0.0015-decay-0.0-qonly-1-bs-64-command-8-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-newepoch-40.pt'
    #     print(fp)
    #     model.load_state_dict(torch.load(fp))
    model = FactorizationMachine(emb_size=args.hs, user_length=user_length, item_length=item_length,
                                 feature_length=feature_length, qonly=args.qonly, hs=args.hs, ip=args.ip, dr=args.dr)
    #fp = './tmp/last_fm/FM-model-merge/v1-test-FM-lr-0.02-flr-0.0001-reg-0.001-decay-0.0-qonly-1-bs-64-command-8-hs-64-ip-0.01-dr-0.5-optim-Ada-pretrain-0-uf-1-rd-0-freeze-0-seed-0-useremb-None-epoch-200.pt'
    #fp = './tmp/last_fm/FM-model-merge/epoch-200.pt'
    #fp = './tmp/last_fm/FM-model-merge/v4-FM-lr-0.01-flr-0.001-reg-0.002-decay-0.0-qonly-epoch-0.pt'
    fp = './tmp/last_fm_small/FM-model-merge/v1-FM-lr-0.02-flr-0.0001-reg-0.001-bs-64-command-8-uf-1-seed-0-epoch-70.pt'
    model.load_state_dict(torch.load(fp))
    print('Model loaded successfully!')
    print('Evaluating on feature similarity')
    # filename = 'v1-test-FM-lr-{}-flr-{}-reg-{}-decay-{}-qonly-{}-bs-{}-command-{}-hs-{}-ip-{}-dr-{}-optim-{}-pretrain-{}-uf-{}-rd-{}-freeze-{}-seed-{}-useremb-{}'.format(
    #     args.lr, args.flr, args.reg, args.decay, args.qonly,
    #     args.bs, args.command, args.hs, args.ip, args.dr, args.optim, args.pretrain, args.uf, args.rd, args.freeze,
    #     args.seed, args.useremb)
    filename = 'v1-FM-lr-{}-flr-{}-reg-{}-bs-{}-command-{}-uf-{}-seed-{}'.format(
        args.lr, args.flr, args.reg, args.bs, args.command, args.uf, args.seed)
    cuda_(model)
    epoch = 70
    #epoch = 200
    #filename = 'lala.txt'
    evaluate_item(kg, model, epoch, filename, 0, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM,
                  ITEM_FEATURE)

if __name__ == '__main__':
    main()