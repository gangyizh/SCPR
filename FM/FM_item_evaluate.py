
from sklearn.metrics import roc_auc_score
from utils import *
import time
from torch.nn.utils.rnn import pad_sequence


def topk(y_true, pred, k):
    y_true_ = y_true[:k]
    pred_ = pred[:k]
    if sum(y_true_) == 0:
        return 0
    else:
        return roc_auc_score(y_true_, pred_)


def rank_by_batch(kg, pickle_file, iter_, bs, pickle_file_length, model, rd, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):
    '''
    user_output, item_p_output, i_neg2_output, preference_list = list(), list(), list(), list()
    '''
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    III = pickle_file[2][left:right]
    IV = pickle_file[3][left:right]

    i = 0
    index_none = list()

    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        if i_neg2_output is None or len(i_neg2_output) == 0:
            index_none.append(i)
        i += 1

    i = 0
    result_list = list()
    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        if i in index_none:
            i += 1
            continue

        total_list = list(i_neg2_output)[: 1000] + [item_p_output]

        user_input = [user_output] * len(total_list)

        pos_list, pos_list2 = list(), list()
        cumu_length = 0
        for instance in zip(user_input, total_list):
            new_list = list()
            new_list.append(instance[0])
            new_list.append(instance[1] + user_length)
            pos_list.append(torch.LongTensor(new_list))
            f = kg.G[ITEM][instance[1]][ITEM_FEATURE]
            if rd == 1:
                f = list(set(f) - set(preference_list))
            cumu_length += len(f)
            pos_list2.append(torch.LongTensor(f))

        if cumu_length == 0:
            pass


        pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
        prefer_list = torch.LongTensor(preference_list).expand(len(total_list), len(preference_list))

        if cumu_length != 0:
            pos_list2.sort(key=lambda x: -1 * x.shape[0])
            pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)
        else:
            pos_list2 = torch.LongTensor([PAD_IDX2]).expand(pos_list.shape[0], 1)


        predictions, _, _ = model(cuda_(pos_list), cuda_(pos_list2), cuda_(prefer_list))
        predictions = predictions.detach().cpu().numpy()

        mini_gtitems = [item_p_output]
        num_gt = len(mini_gtitems)
        num_neg = len(total_list) - num_gt

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
        i += 1
    return result_list


def evaluate_item(kg, model, epoch, filename, rd, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):
    #TODO add const PAD_IDX1, PAD_IDX2, user_length, data_name, ITEM, ITEM_FEATURE
    model.eval()
    tt = time.time()
    pickle_file = load_fm_sample(dataset=data_name, mode='valid')
    print('evaluate data:{}'.format(data_name))
    print('Open evaluation pickle file: takes {} seconds, evaluation length: {}'.format(time.time() - tt, len(pickle_file[0])))
    pickle_file_length = len(pickle_file[0])
    print('ui length:{}'.format(pickle_file_length))

    start = time.time()
    print('Starting {} epoch'.format(epoch))
    bs = 64
    max_iter = int(pickle_file_length / float(bs))
    # Only do 20 iteration for the sake of time
    max_iter = 20

    result = list()
    for iter_ in range(max_iter):
        if iter_ > 1 and iter_ % 50 == 0:
            print('--')
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                        float(iter_) * 100 / max_iter))
        result += rank_by_batch(kg, pickle_file, iter_, bs, pickle_file_length, model, rd, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)


    auc_mean = np.mean(np.array([item[0] for item in result]))
    auc_median = np.median(np.array([item[0] for item in result]))
    print('auc mean: {}'.format(auc_mean), 'auc median: {}'.format(auc_median),
          'over num {}'.format(len(result)))
    PATH = TMP_DIR[data_name] + '/FM-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[data_name] + '/FM-log-merge/'):
        os.makedirs(TMP_DIR[data_name] + '/FM-log-merge/')
    with open(PATH, 'a') as f:
        f.write('validating {} epoch on item prediction\n'.format(epoch))
        auc_mean = np.mean(np.array([item[0] for item in result]))
        auc_median = np.median(np.array([item[0] for item in result]))
        f.write('auc mean: {}\n'.format(auc_mean))
        f.write('auc median: {}\n'.format(auc_median))

    model.train()
    cuda_(model)
