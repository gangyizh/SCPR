import random
import torch
import torch.nn as nn
import json
import pickle
from utils import *
import time
from torch.nn.utils.rnn import pad_sequence
import argparse
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

    if uf == 1:
        feature_range = np.arange(feature_length).tolist()
        residual_feature, neg_feature = [], []
        for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
            gt_feature = kg.G[ITEM][item_p_pickle][ITEM_FEATURE]
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
        pos_list2.append(torch.LongTensor(f))
        neg_list.append(torch.LongTensor([user_pickle, i_neg1_pickle + user_length]))
        f = kg.G[ITEM][i_neg1_pickle][ITEM_FEATURE]
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
        new_neg_list2.append(torch.LongTensor(f))
        preference_list_2.append(torch.LongTensor(preference_pickle))
        i += 1


    pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
    pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)
    neg_list = pad_sequence(neg_list, batch_first=True, padding_value=PAD_IDX1)
    neg_list2 = pad_sequence(neg_list2, batch_first=True, padding_value=PAD_IDX2)
    new_neg_list = pad_sequence(new_neg_list, batch_first=True, padding_value=PAD_IDX1)
    new_neg_list2 = pad_sequence(new_neg_list2, batch_first=True, padding_value=PAD_IDX2)
    preference_list_1 = pad_sequence(preference_list_1, batch_first=True, padding_value=PAD_IDX2)
    preference_list_2 = pad_sequence(preference_list_2, batch_first=True, padding_value=PAD_IDX2)

    if uf != 0:
        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2), cuda_(new_neg_list), cuda_(
            new_neg_list2), cuda_(preference_list_1), cuda_(preference_list_2), index_none, cuda_(residual_feature), cuda_(neg_feature)
    else:
        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2), cuda_(new_neg_list), cuda_(
            new_neg_list2), cuda_(preference_list_1), cuda_(preference_list_2), index_none, residual_feature, neg_feature

def train(dataset, kg, model, bs, max_epoch, optimizer1, optimizer2, optimizer3, reg, qonly, observe, command, filename, uf, useremb, load_fm_epoch):
    model.train()
    lsigmoid = nn.LogSigmoid()
    reg_float = float(reg.data.cpu().numpy()[0])

    for epoch in range(load_fm_epoch, max_epoch+1):
        # _______ Do the evaluation _______
        if epoch % observe == 0 and epoch > -1:
            print('Evaluating on feature similarity')
            evaluate_feature(kg, model, epoch, filename, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)
            print('Evaluating on item similarity')
            evaluate_item(kg, model, epoch, filename, 0, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)

        tt = time.time()
        pickle_file = load_fm_sample(dataset=data_name, mode='train', epoch=epoch % 50)

        print('Open pickle file: train_fm_data takes {} seconds'.format(time.time() - tt))
        pickle_file_length = len(pickle_file[0])

        model.train()

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

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            result_pos, feature_bias_matrix_pos, nonzero_matrix_pos = model(pos_list, pos_list2,
                                                                            preference_list_1)  # (bs, 1), (bs, 2, 1), (bs, 2, emb_size)

            result_neg, feature_bias_matrix_neg, nonzero_matrix_neg = model(neg_list, neg_list2, preference_list_1)
            diff = (result_pos - result_neg)
            loss = - lsigmoid(diff).sum(dim=0)  # The Minus is crucial is

            if command in [8]:
                # The second type of negative sample
                new_result_neg, new_feature_bias_matrix_neg, new_nonzero_matrix_neg = model(new_neg_list, new_neg_list2,
                                                                                            preference_list_new)
                # Reason for this is that, sometimes the sample is missing, so we have to also omit that in result_pos
                T = cuda_(torch.tensor([]))
                for i in range(bs):
                    if i in index_none:
                        continue
                    T = torch.cat([T, result_pos[i]], dim=0)

                T = T.view(T.shape[0], -1)
                assert T.shape[0] == new_result_neg.shape[0]
                diff = T - new_result_neg
                if loss is not None:
                    loss += - lsigmoid(diff).sum(dim=0)
                else:
                    loss = - lsigmoid(diff).sum(dim=0)

            # regularization
            if reg_float != 0:
                if qonly != 1:
                    feature_bias_matrix_pos_ = (feature_bias_matrix_pos ** 2).sum(dim=1)  # (bs, 1)
                    feature_bias_matrix_neg_ = (feature_bias_matrix_neg ** 2).sum(dim=1)  # (bs, 1)
                    nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)  # (bs, 1)
                    nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)  # (bs, 1)
                    new_nonzero_matrix_neg_ = (new_nonzero_matrix_neg_ ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    regular_norm = (
                                feature_bias_matrix_pos_ + feature_bias_matrix_neg_ + nonzero_matrix_pos_ + nonzero_matrix_neg_ + new_nonzero_matrix_neg_)
                    loss += (reg * regular_norm).sum(dim=0)
                else:
                    nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    loss += (reg * nonzero_matrix_pos_).sum(dim=0)
                    loss += (reg * nonzero_matrix_neg_).sum(dim=0)
            epoch_loss += loss.data
            loss.backward()
            optimizer1.step()
            optimizer2.step()


            if uf == 1:
                # updating feature embedding
                # we try to optimize
                A = model.feature_emb(preference_list_1)[..., :-1]
                user_emb = model.ui_emb(pos_list[:, 0])[..., :-1].unsqueeze(dim=1).detach()
                if useremb == 1:
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
                temp = - lsigmoid(diff).sum(dim=0)
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
                if torch.norm(model.ui_emb.weight.grad) > 200 or torch.norm(model.feature_emb.weight.grad) > 500:
                    print('iter_:{} Bias grad norm: {},F-bias grad norm: {}, '
                          'F-embedding grad norm: {}'.format(iter_, torch.norm(model.Bias.grad),
                                                             torch.norm(model.ui_emb.weight.grad), torch.norm(model.feature_emb.weight.grad)))

            # Uncomment this to use clip gradient norm (but currently we don't need)
            # clip_grad_norm_(model.ui_emb.weight, 5000)
            # clip_grad_norm_(model.feature_emb.weight, 5000)

        print('epoch loss: {}'.format(epoch_loss / pickle_file_length))
        print('epoch loss 2: {}'.format(epoch_loss_2 / pickle_file_length))

        if epoch % 5 == 0 and epoch > 0:
            print('FM Epoch：{} ; start saving FM model.'.format(epoch))
            save_fm_model(dataset=data_name, model=model, filename=filename, epoch=epoch)
            print('FM Epoch：{} ; start saving model embedding.'.format(epoch))
            save_embedding(model, filename, epoch=epoch)


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
    parser.add_argument('-flr', type=float, default=0.0001, metavar='<flr>', dest='flr', help='flr')
    # means the learning rate of feature similarity learning
    parser.add_argument('-reg', type=float, default=0.001, metavar='<reg>', dest='reg', help='reg')
    # regularization
    parser.add_argument('-decay', type=float, default=0.0, metavar='<decay>', dest='decay', help='decay')
    # weight decay
    parser.add_argument('-qonly', type=int, default=1, metavar='<qonly>', dest='qonly', help='qonly')
    # means quadratic form only (letting go other terms in FM equation...)
    parser.add_argument('-bs', type=int, default=64, metavar='<bs>', dest='bs', help='bs')
    #batch size
    parser.add_argument('-hs', type=int, default=64, metavar='<hs>', dest='hs', help='hs')
    # hidden size & embedding size
    parser.add_argument('-ip', type=float, default=0.01, metavar='<ip>', dest='ip', help='ip')
    # init parameter for hidden
    parser.add_argument('-dr', type=float, default=0.5, metavar='<dr>', dest='dr', help='dr')
    # dropout ratio
    parser.add_argument('-optim', type=str, default='Ada', metavar='<optim>', dest='optim', help='optim')
    # optimizer
    parser.add_argument('-observe', type=int, default=25, metavar='<observe>', dest='observe', help='observe')
    # the frequency of doing evaluation
    parser.add_argument('-uf', type=int, default=1, metavar='<uf>', dest='uf', help='uf')
    # update feature
    parser.add_argument('-rd', type=int, default=0, metavar='<rd>', dest='rd', help='rd')
    # remove duplicate, we don;t use this parameter now
    parser.add_argument('-useremb', type=int, default=1, metavar='<useremb>', dest='useremb', help='user embedding')
    # update user embedding during feature similarity
    parser.add_argument('-freeze', type=int, default=0, metavar='<freeze>', dest='freeze', help='freeze')
    # we don't use this param now
    parser.add_argument('-command', type=int, default=8, metavar='<command>', dest='command', help='command')
    # command = 6: normal FM
    # command = 8: with our second type of negative sample
    parser.add_argument('-seed', type=int, default=0, metavar='<seed>', dest='seed', help='seed')
    # random seed
    parser.add_argument('-me', type=int, default=250, metavar='<max_epoch>', dest='max_epoch', help='max_epoch')
    #the number of train epoch
    parser.add_argument('-pretrain', type=int, default=0, metavar='<pretrain>', dest='pretrain', help='pretrain')
    parser.add_argument('-load_fm_epoch', type=int, default=0, metavar='<load_fm_epoch>', dest='load_fm_epoch', help='the epoch of loading FM model')
    # load pretrain model:  Setting pretrain=1 & load_fm_epoch=fm_epoch
    parser.add_argument('--data_name', type=str, default='LAST_FM', metavar='<data_name>', dest='data_name',
                        choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR], help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    args = parser.parse_args()


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
    if args.pretrain == 1:
        model = FactorizationMachine(emb_size=args.hs, user_length=user_length, item_length=item_length,
                                     feature_length=feature_length, qonly=args.qonly, hs=args.hs, ip=args.ip, dr=args.dr)
        file_name = 'v1-data-{}-lr-{}-flr-{}-reg-{}-bs-{}-command-{}-uf-{}-seed-{}'.format(
            args.data_name, args.lr, args.flr, args.reg, args.bs, args.command, args.uf, args.seed)
        model_dict = load_fm_model(data_name, model, file_name, epoch=args.load_fm_epoch)
        model.load_state_dict(model_dict)
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
    max_epoch = args.max_epoch

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

    file_name = 'v1-data-{}-lr-{}-flr-{}-reg-{}-bs-{}-command-{}-uf-{}-seed-{}'.format(
        args.data_name, args.lr, args.flr, args.reg, args.bs, args.command, args.uf, args.seed)


    model = train(dataset, kg, model, bs, max_epoch, optimizer1, optimizer2, optimizer3, reg_, args.qonly, args.observe, args.command, file_name, args.uf, args.useremb, args.load_fm_epoch)
    save_embedding(model, file_name, epoch=max_epoch)

if __name__ == '__main__':
    main()