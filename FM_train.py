import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from utils import *
import time
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import argparse
import sys
from FM.FM_model import FactorizationMachine
from FM.FM_feature_evaluate import evaluate_feature
from FM.FM_item_evaluate import evaluate_item


def translate_pickle_to_data(dataset, kg, pickle_file, iter_, bs, pickle_file_length, uf):
    '''
    user_pickle = pickle_file[0]
    item_p_pickle = pickle_file[1]
    i_neg1_pickle = pickle_file[2]
    i_neg2_pickle = pickle_file[3]
    preference_pickle = pickle_file[4]
    '''
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)
    # user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle = zip(*pickle_file[left:right])

    pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_2 = [], [], [], [], [], [], [], []

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    III = pickle_file[2][left:right]
    IV = pickle_file[3][left:right]
    V = pickle_file[4][left:right]

    residual_feature, neg_feature = None, None
    # TODO  last_fm  and Amazon have difference

    if uf == 1:
        feature_range = np.arange(feature_length).tolist()
        # residual_feature： item i feature - like_itemi_feature
           #  list  统一补充 feature_count
        # neg_feature： 每个交互中 除去item i 的属性 其他的负例属性（采样数量为：len（item feature） - len（like_item_feature))
        residual_feature, neg_feature = [], []
        for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
            gt_feature = kg.G[ITEM][item_p_pickle][ITEM_FEATURE]
            #this_residual_feature : user对item i 除去喜欢的feature的list
            # remain_feature： 所有feature 除去item i 的feature
            # this_neg_feature： 从remain_feature 随机选取this_residual_feature的数量
            this_residual_feature = list(set(gt_feature) - set(preference_pickle))
            remain_feature = list(set(feature_range) - set(gt_feature))
            this_neg_feature = np.random.choice(remain_feature, len(this_residual_feature))
            residual_feature.append(torch.LongTensor(this_residual_feature))
            neg_feature.append(torch.LongTensor(this_neg_feature))
        residual_feature = pad_sequence(residual_feature, batch_first=True, padding_value=PAD_IDX2)
        neg_feature = pad_sequence(neg_feature, batch_first=True, padding_value=PAD_IDX2)

    i = 0
    index_none = list()
    for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
        pos_list.append(torch.LongTensor([user_pickle, item_p_pickle + user_length]))
        f = kg.G[ITEM][item_p_pickle][ITEM_FEATURE]
        #f = list(set(f) - set(preference_pickle))
        pos_list2.append(torch.LongTensor(f))
        neg_list.append(torch.LongTensor([user_pickle, i_neg1_pickle + user_length]))
        f = kg.G[ITEM][i_neg1_pickle][ITEM_FEATURE]
        # f = item_dict[str(i_neg1_pickle)]['feature_index']
        #f = list(set(f) - set(preference_pickle))
        neg_list2.append(torch.LongTensor(f))
        preference_list_1.append(torch.LongTensor(preference_pickle))
        if i_neg2_pickle is None:
            index_none.append(i)
        i += 1

    i = 0
    for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
        if i in index_none:
            i += 1
            continue
        new_neg_list.append(torch.LongTensor([user_pickle, i_neg2_pickle + user_length]))
        f = kg.G[ITEM][i_neg2_pickle][ITEM_FEATURE]
        #f = item_dict[str(i_neg2_pickle)]['feature_index']
        #f = list(set(f) - set(preference_pickle))
        new_neg_list2.append(torch.LongTensor(f))
        preference_list_2.append(torch.LongTensor(preference_pickle))
        i += 1

    # print('index none: {}'.format(len(index_none)))
    # print(len(pos_list), len(pos_list2), len(neg_list), len(neg_list2), len(new_neg_list), len(new_neg_list2), len(preference_list_1), len(preference_list_2))
    pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
    pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)
    neg_list = pad_sequence(neg_list, batch_first=True, padding_value=PAD_IDX1)
    neg_list2 = pad_sequence(neg_list2, batch_first=True, padding_value=PAD_IDX2)
    new_neg_list = pad_sequence(new_neg_list, batch_first=True, padding_value=PAD_IDX1)
    new_neg_list2 = pad_sequence(new_neg_list2, batch_first=True, padding_value=PAD_IDX2)
    preference_list_1 = pad_sequence(preference_list_1, batch_first=True, padding_value=PAD_IDX2)
    preference_list_2 = pad_sequence(preference_list_2, batch_first=True, padding_value=PAD_IDX2)

    # return pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_2, index_none
    if uf != 0:
        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2), cuda_(new_neg_list), cuda_(
            new_neg_list2), cuda_(preference_list_1), cuda_(preference_list_2), index_none, cuda_(residual_feature), cuda_(neg_feature)
    else:
        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2), cuda_(new_neg_list), cuda_(
            new_neg_list2), cuda_(preference_list_1), cuda_(preference_list_2), index_none, residual_feature, neg_feature

def train(dataset, kg, model, bs, max_epoch, optimizer1, optimizer2, optimizer3, reg, qonly, observe, command, filename, uf, useremb):
    model.train()
    lsigmoid = nn.LogSigmoid()
    reg_float = float(reg.data.cpu().numpy()[0])

    for epoch in range(max_epoch+1):
        # _______ Do the runtime evaluation _______
        if epoch % observe == 0 and epoch > -1:
        #if epoch == 1 and epoch > -1:
            print('Evaluating on feature similarity')
            evaluate_feature(kg, model, epoch, filename, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)
            print('Evaluating on item similarity')
            evaluate_item(kg, model, epoch, filename, 0, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)

        tt = time.time()
        # Note: 之前是v2
        # pickle_file_path = '../../data/FM-sample-data-new/v1-speed-train-{}.pickle'.format(epoch)
        # if uf == 1:
        #     pickle_file_path = '../../data/FM-sample-data-new/v1-speed-train-{}.pickle'.format(epoch + 30)
        # # pickle_file_path = '../../data/speed-data-random/v3-speed-train-{}.pickle'.format(1)
        # with open(pickle_file_path, 'rb') as f:
        #     pickle_file = pickle.load(f)

        #   pickle_file:  train_fm_data
        pickle_file = load_fm_sample(dataset=data_name, mode='train')

        print('Open pickle file: train_fm_data takes {} seconds'.format(time.time() - tt))
        pickle_file_length = len(pickle_file[0])

        # TODO: model.train 非常重要
        model.train()

        # TODO: 用新的方式来 shuffle, 来 verify 一下 这个的正确性
        # user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle = zip(*pickle_file[left:right])
        mix = list(zip(pickle_file[0], pickle_file[1], pickle_file[2], pickle_file[3], pickle_file[4]))
        random.shuffle(mix)
        I, II, III, IV, V = zip(*mix)
        new_pk_file = [I, II, III, IV, V]

        start = time.time()
        print('Starting {} epoch'.format(epoch))
        epoch_loss = 0
        epoch_loss_2 = 0
        max_iter = int(pickle_file_length / float(bs))

        for iter_ in range(max_iter):
            if iter_ > 1 and iter_ % 1000 == 0:
                print('--')
                print('Takes {} seconds to finish {}% of this epoch'.format(str(time.time() - start),
                                                                            float(iter_) * 100 / max_iter))
                print('loss is: {}'.format(float(epoch_loss) / (bs * iter_)))
                print('iter_:{} Bias grad norm: {}, Static grad norm: {}, Preference grad norm: {}'.format(iter_, torch.norm(model.Bias.grad), torch.norm(model.ui_emb.weight.grad), torch.norm(model.feature_emb.weight.grad)))

            pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_new, index_none, residual_feature, neg_feature \
                = translate_pickle_to_data(dataset, kg, new_pk_file, iter_, bs, pickle_file_length, uf)

            # print(pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_new, index_none, residual_feature, neg_feature)
            # print(pos_list, pos_list2, preference_list_1, neg_list, neg_list2)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            result_pos, feature_bias_matrix_pos, nonzero_matrix_pos = model(pos_list, pos_list2,
                                                                            preference_list_1)  # (bs, 1), (bs, 2, 1), (bs, 2, emb_size)
            # TODO: Debug Mode, delete it later
            if command in [2, 3, 5]:
                result_neg, feature_bias_matrix_neg, nonzero_matrix_neg = model(neg_list, pos_list2, preference_list_1)
                diff = (result_pos - result_neg)
                loss = - lsigmoid(diff).sum(dim=0)  # The Minus is crucial
            if command in [1, 6, 4, 7, 8, 10]:
                result_neg, feature_bias_matrix_neg, nonzero_matrix_neg = model(neg_list, neg_list2, preference_list_1)
                diff = (result_pos - result_neg)
                loss = - lsigmoid(diff).sum(dim=0)  # The Minus is crucial is

            # TODO: change here
            # if command in [7, 8]:
            if command in [7, 8, 11]:
                new_result_neg, new_feature_bias_matrix_neg, new_nonzero_matrix_neg = model(new_neg_list, new_neg_list2,
                                                                                            preference_list_new)
                # print(new_neg_list, new_neg_list2, preference_list_new)
                T = cuda_(torch.tensor([]))
                # i = 0
                for i in range(bs):
                    if i in index_none:
                        # i += 1
                        continue
                    T = torch.cat([T, result_pos[i]], dim=0)
                    # i += 1

                T = T.view(T.shape[0], -1)
                assert T.shape[0] == new_result_neg.shape[0]
                diff = T - new_result_neg
                if loss is not None:
                    loss += - lsigmoid(diff).sum(dim=0)
                else:
                    loss = - lsigmoid(diff).sum(dim=0)


            if command in [9]:  # 按照Wenqiang一开始提出的loss 来算，两个大于号
                # result_pos 是正样本，result_neg 是随机负样本，new_result_neg 是第二种负样本
                result_neg, feature_bias_matrix_neg, nonzero_matrix_neg = model(neg_list, neg_list2, preference_list_1)
                new_result_neg, new_feature_bias_matrix_neg, new_nonzero_matrix_neg = model(new_neg_list, new_neg_list2,
                                                                                            preference_list_new)
                T1, T2 = cuda_(torch.tensor([])), cuda_(torch.tensor([]))
                # i = 0
                for i in range(bs):
                    if i in index_none:
                        # i += 1
                        continue
                    T1 = torch.cat([T1, result_pos[i]], dim=0)
                    T2 = torch.cat([T2, result_neg[i]], dim=0)
                    # i += 1

                T1 = T1.view(T1.shape[0], -1)
                T2 = T2.view(T2.shape[0], -1)
                assert T1.shape[0] == new_result_neg.shape[0]
                assert T2.shape[0] == new_result_neg.shape[0]
                diff1 = T1 - new_result_neg
                loss = - lsigmoid(diff1).sum(dim=0)

                diff2 = new_result_neg - T2
                loss += - lsigmoid(diff2).sum(dim=0)

            '''
            下面这个block 的代码是用来加正则项的
            '''
            if reg_float != 0:
                if qonly != 1:
                    feature_bias_matrix_pos_ = (feature_bias_matrix_pos ** 2).sum(dim=1)  # (bs, 1)
                    feature_bias_matrix_neg_ = (feature_bias_matrix_neg ** 2).sum(dim=1)  # (bs, 1)
                    nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)  # (bs, 1)
                    nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)  # (bs, 1)
                    new_nonzero_matrix_neg_ = (new_nonzero_matrix_neg_ ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    regular_norm = (
                                feature_bias_matrix_pos_ + feature_bias_matrix_neg_ + nonzero_matrix_pos_ + nonzero_matrix_neg_ + new_nonzero_matrix_neg_)
                    # diff = torch.clamp(result_pos - result_neg, -8.0, 1e8)
                    loss += (reg * regular_norm).sum(dim=0)
                else:
                    nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    loss += (reg * nonzero_matrix_pos_).sum(dim=0)
                    loss += (reg * nonzero_matrix_neg_).sum(dim=0)
                    # regular_norm = (nonzero_matrix_pos_ + nonzero_matrix_neg_)
                    # diff = torch.clamp(result_pos - result_neg, -8.0, 1e8)
                    # loss += (reg * regular_norm).sum(dim=0)
            epoch_loss += loss.data
            if iter_ % 1000 == 0 and iter_ > 0:
                print('feature_emb.weight.grad:', model.feature_emb.weight.grad)
                print('1ND iter_:{} preference grad norm: {}'.format(iter_, torch.norm(model.feature_emb.weight.grad)))
                print('1ND loss is: {}'.format(float(epoch_loss) / (bs * iter_)))
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            # TODO: need to specify an optimizer
            if uf == 1:
                A = model.feature_emb(preference_list_1)[..., :-1]
                user_emb = model.ui_emb(pos_list[:, 0])[..., :-1].unsqueeze(dim=1).detach()
                # if use_useremb == 1:
                A = torch.cat([A, user_emb], dim=1)

                B = model.feature_emb(residual_feature)[..., :-1]
                C = model.feature_emb(neg_feature)[..., :-1]

                D = torch.matmul(A, B.transpose(2, 1))
                E = torch.matmul(A, C.transpose(2, 1))

                p_vs_residual = D.view(D.shape[0], -1, 1)
                p_vs_neg = E.view(E.shape[0], -1, 1)

                p_vs_residual = p_vs_residual.sum(dim=1)
                p_vs_neg = p_vs_neg.sum(dim=1)
                diff = (p_vs_residual - p_vs_neg)
                # loss = - lsigmoid(diff).sum(dim=0)  # The Minus is crucial
                temp = - lsigmoid(diff).sum(dim=0)
                # loss += 4 * temp
                loss = temp
                epoch_loss_2 += temp.data

                if iter_ % 1000 == 0 and iter_ > 0:
                    print('2ND iter_:{} preference grad norm: {}'.format(iter_, torch.norm(model.feature_emb.weight.grad)))
                    print('2ND loss is: {}'.format(float(epoch_loss_2) / (bs * iter_)))

                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()

            # These line is to make an alert on console when we meet gradient explosion.
            if iter_ > 0 and iter_ % 1 == 0:
                #TODO uncomment this
                if torch.norm(model.ui_emb.weight.grad) > 200 or torch.norm(model.feature_emb.weight.grad) > 1000:
                #if torch.norm(model.ui_emb.weight.grad) > 100 or torch.norm(model.feature_emb.weight.grad) > 500:
                    print('iter_:{} Bias grad norm: {},F-bias grad norm: {}, '
                          'F-embedding grad norm: {}'.format(iter_, torch.norm(model.Bias.grad),
                                                             torch.norm(model.ui_emb.weight.grad), torch.norm(model.feature_emb.weight.grad)))

            # Uncomment this to use clip gradient norm (but currently we don't need)
            # clip_grad_norm_(model.ui_emb.weight, 5000)
            # clip_grad_norm_(model.feature_emb.weight, 5000)

        print('epoch loss: {}'.format(epoch_loss / pickle_file_length))
        print('epoch loss 2: {}'.format(epoch_loss_2 / pickle_file_length))

        if epoch % 10 == 0:
            save_fm_model(dataset=data_name, model=model, filename=filename, epoch=epoch)
        #save_fm_model(dataset=data_name, model=model, filename=filename, epoch=epoch)


        train_len = len(pickle_file[0])
        save_fm_model_log(dataset=data_name, filename=filename, epoch=epoch, epoch_loss=epoch_loss
                          , epoch_loss_2=epoch_loss_2, train_len=train_len)

def save_embedding(model, filename, epoch):
    model_dict = load_fm_model(data_name, model, filename, epoch)
    model.load_state_dict(model_dict)
    print('Model loaded successfully!')
    ui_emb = model.ui_emb.weight[..., :-1].data.cpu().numpy()
    feature_emb = model.feature_emb.weight[..., :-1].data.cpu().numpy()
    print('ui_size:{}'.format(ui_emb.shape[0]))
    print('fea_size:{}'.format(feature_emb.shape[0]))
    embeds = {
        'ui_emb': ui_emb,
        'feature_emb': feature_emb
    }
    save_embed(data_name, embeds, epoch)




def main():
    parser = argparse.ArgumentParser(description="Run DeepFM-BPR.")
    parser.add_argument('-lr', type=float, default=0.02, metavar='<lr>', dest='lr', help='lr')
    parser.add_argument('-flr', type=float, default=0.0001, metavar='<flr>', dest='flr', help='flr')  # means Feature update Learning Rate
    parser.add_argument('-reg', type=float, default=0.001, metavar='<reg>', dest='reg', help='reg')   #FM_parameters regular terms
    parser.add_argument('-decay', type=float, default=0.0, metavar='<decay>', dest='decay', help='decay')
    parser.add_argument('-qonly', type=int, default=1, metavar='<qonly>', dest='qonly', help='qonly')  # qonly means we drop 一次项
    parser.add_argument('-bs', type=int, default=64, metavar='<bs>', dest='bs', help='bs') #batch size
    parser.add_argument('-hs', type=int, default=64, metavar='<hs>', dest='hs', help='hs')  # hidden size & embedding size
    parser.add_argument('-ip', type=float, default=0.01, metavar='<ip>', dest='ip', help='ip')  # init parameter for hidden
    parser.add_argument('-dr', type=float, default=0.5, metavar='<dr>', dest='dr', help='dr')  # dropout ratio
    parser.add_argument('-optim', type=str, default='Ada', metavar='<optim>', dest='optim', help='optim')
    parser.add_argument('-observe', type=int, default=25, metavar='<observe>', dest='observe', help='observe')
    parser.add_argument('-pretrain', type=int, default=0, metavar='<pretrain>', dest='pretrain', help='pretrain')  # 可以选择使用哪一种pretrain
    parser.add_argument('-uf', type=int, default=1, metavar='<uf>', dest='uf', help='uf')  # update feature 的缩写  1:update
    parser.add_argument('-rd', type=int, default=0, metavar='<rd>', dest='rd', help='rd')  # remove duplicate 的缩写, 把 preference 和 feature 中相同的 去除
    parser.add_argument('-useremb', type=int, metavar='<useremb>', dest='useremb', help='user embedding')
    parser.add_argument('-freeze', type=int, default=0, metavar='<freeze>', dest='freeze', help='freeze')
    parser.add_argument('-command', type=int, default=8, metavar='<command>', dest='command', help='command')
    parser.add_argument('-seed', type=int, default=0, metavar='<seed>', dest='seed', help='seed')
    parser.add_argument('-data_name', type=str, default=LAST_FM_SMALL, metavar='<data_name>', dest='data_name', help='One of {LAST_FM, LAST_FM_SMALL, YELP, YELP_SMALL}.')
    args = parser.parse_args()

    '''
    lr: 学习率
    reg: 正则项系数
    qonly: quadratic only, 忽略了一次项，在FM里也有相应的改变
    observe: 每隔多少个epoch 观察一下（做evaluation）
    The reason we need 'command' is that we need this code to run multiple experiments,
    command == 0: 仅仅使用 user embedding 和 item embedding
    command == 1: 仅仅使用 user embedding, item embedding 以及 feature embedding
    command == 2: user embedding 和 item embedding 和 preference embedding, 负样本的preference 使用 正样本的feature
    但是在evaluate时用了新方法 evaluate
    command == 3: 与2 相同，但是要对preference 加 dropout来训练
    command == 4: 使用2种 负样本，FM模型也需要根据command改变,(即进入 函数forward_1234)
    command == 5: 和2相同，使用旧方法evaluate
    command == 6: 和2相同，在train中加入 dropout, 在evaluate 中也加入 dropout
    note that: 目前的evaluation仅仅支持 command=2
    目前我也暂时把 使用 第二种负样本的代码comment 掉了
    目前能运行的: python FM_split_train.py -lr 0.01 -reg 0.01 -qonly 1 -observe 1 -command 2
    command = 7: 我们的1234 实验
    command = 8: 他们的124 实验，但用了第二种负样本
    '''



    global PAD_IDX1, PAD_IDX2
    global user_length, item_length, feature_length
    global data_name
    global ITEM, ITEM_FEATURE
    ITEM = 'item'
    ITEM_FEATURE = 'belong_to'
    data_name = args.data_name
    dataset = load_dataset(data_name)
    kg = load_kg(data_name)
    user_length = int(getattr(dataset, 'user').value_len)
    item_length = int(getattr(dataset, 'item').value_len)
    feature_length = int(getattr(dataset, 'feature').value_len)
    print('user_length:{},item_length:{},feature_length:{}'.format(user_length, item_length, feature_length))
    PAD_IDX1 = user_length + item_length
    PAD_IDX2 = feature_length

    set_random_seed(args.seed)


    if args.pretrain == 0:  # means no pretrain
        model = FactorizationMachine(emb_size=args.hs, user_length=user_length, item_length=item_length,
                                     feature_length=feature_length, qonly=args.qonly, hs=args.hs, ip=args.ip, dr=args.dr)
    cuda_(model)

    param1, param2 = list(), list()
    param3 = list()

    i = 0
    for name, param in model.named_parameters():
        print(name, param)
        if i == 0:
            param1.append(param)
        else:
            param2.append(param)
        if i == 2:
            param3.append(param)
        i += 1

    print('param1 is: {}, shape:{}\nparam2 is: {}, shape: {}\nparam3 is: {}, shape: {}\n'.format(param1, [param.shape for param in param1], param2, [param.shape for param in param2], param3, [param.shape for param in param3]))
    bs = args.bs
    max_epoch = 250

    if args.optim == 'SGD':
        optimizer1 = torch.optim.SGD(param1, lr=args.lr, weight_decay=0.1)
        optimizer2 = torch.optim.SGD(param2, lr=args.lr)
        optimizer3 = torch.optim.SGD(param3, lr=args.flr)
    if args.optim == 'Ada':
        optimizer1 = torch.optim.Adagrad(param1, lr=args.lr, weight_decay=args.decay)
        optimizer2 = torch.optim.Adagrad(param2, lr=args.lr, weight_decay=args.decay)
        optimizer3 = torch.optim.Adagrad(param3, lr=args.flr, weight_decay=args.decay)

    reg_ = torch.Tensor([args.reg])
    reg_ = torch.autograd.Variable(reg_, requires_grad=False)
    reg_ = cuda_(reg_)

    # file_name = 'v1-test-FM-lr-{}-flr-{}-reg-{}-decay-{}-qonly-{}-bs-{}-command-{}-hs-{}-ip-{}-dr-{}-optim-{}-pretrain-{}-uf-{}-rd-{}-freeze-{}-seed-{}-useremb-{}'.format(args.lr, args.flr, args.reg, args.decay, args.qonly,
    #                                                                                  args.bs, args.command, args.hs, args.ip, args.dr, args.optim, args.pretrain, args.uf, args.rd, args.freeze, args.seed, args.useremb)
    file_name = 'v1-data-{}-FM-lr-{}-flr-{}-reg-{}-bs-{}-command-{}-uf-{}-seed-{}'.format(
        args.data_name, args.lr, args.flr, args.reg, args.bs, args.command, args.uf, args.seed)
    #TODO save embeddings   (uncommend)
    # file_name = 'v4-FM-lr-0.01-flr-0.001-reg-0.002-decay-0.0-qonly'  #lastfm 33
    # file_name = 'yelp-fmdata'  #yelp
    # file_name = 'v1-FM-lr-0.02-flr-0.0001-reg-0.001-bs-64-command-8-uf-1-seed-0' #lastfm_small
    #

    model = train(dataset, kg, model, bs, max_epoch, optimizer1, optimizer2, optimizer3, reg_, args.qonly, args.observe, args.command, file_name, args.uf, args.useremb)
    save_embedding(model, file_name, epoch=max_epoch)

if __name__ == '__main__':
    main()