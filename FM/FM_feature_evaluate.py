
from sklearn.metrics import roc_auc_score
import time
from utils import *



def predict_feature(model,user_output, given_preference, to_test):
    user_emb = model.ui_emb(torch.LongTensor([user_output]))[..., :-1].detach().numpy()
    gp = model.feature_emb(torch.LongTensor(given_preference))[..., :-1].detach().numpy()
    emb_weight = model.feature_emb.weight[..., :-1].detach().numpy()
    result = list()

    for test_feature in to_test:
        temp = 0
        temp += np.inner(user_emb, emb_weight[test_feature])
        for i in range(gp.shape[0]):
            temp += np.inner(gp[i], emb_weight[test_feature])
        result.append(temp)

    return result

def topk(y_true, pred, k):
    y_true_ = y_true[:k]
    pred_ = pred[:k]
    if sum(y_true_) == 0:
        return 0
    else:
        return roc_auc_score(y_true_, pred_)


def rank_by_batch(kg, pickle_file, iter_, bs, pickle_file_length, model, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):
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
        full_feature = kg.G[ITEM][item_p_output][ITEM_FEATURE]
        preference_feature = preference_list
        residual_preference = list(set(full_feature) - set(preference_feature))
        residual_feature_all = list(set(list(range(feature_length - 1))) - set(full_feature))

        if len(residual_preference) == 0:
            continue
        to_test = residual_feature_all + residual_preference

        predictions = predict_feature(model, user_output, preference_feature, to_test)
        predictions = np.array(predictions)


        predictions = predictions.reshape((len(to_test), 1)[0])
        y_true = [0] * len(predictions)
        for i in range(len(residual_preference)):
            y_true[-(i + 1)] = 1
        tmp = list(zip(y_true, predictions))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        y_true, predictions = zip(*tmp)

        icon = []

        for index, item in enumerate(y_true):
            if item > 0:
                icon.append(index)

        auc = roc_auc_score(y_true, predictions)
        result_list.append((auc, topk(y_true, predictions, 10), topk(y_true, predictions, 50)
                            , topk(y_true, predictions, 100), topk(y_true, predictions, 200),
                            topk(y_true, predictions, 500), len(predictions)))
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
    max_iter = 100

    result = list()
    print('max_iter-----------', max_iter)
    for iter_ in range(max_iter):
        if iter_ > 1 and iter_ % 20 == 0:
            print('--')
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                        float(iter_) * 100 / max_iter))
        result += rank_by_batch(kg, pickle_file, iter_, bs, pickle_file_length, model, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)

    auc_mean = np.mean(np.array([item[0] for item in result]))
    auc_median = np.median(np.array([item[0] for item in result]))
    print('auc mean: {}'.format(auc_mean), 'auc median: {}'.format(auc_median))
    PATH = TMP_DIR[data_name] + '/FM-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[data_name] + '/FM-log-merge/'):
        os.makedirs(TMP_DIR[data_name] + '/FM-log-merge/')

    with open(PATH, 'a') as f:
        with open(PATH, 'a') as f:
            f.write('validating {} epoch on feature prediction\n'.format(epoch))
            auc_mean = np.mean(np.array([item[0] for item in result]))
            auc_median = np.median(np.array([item[0] for item in result]))
            f.write('auc mean: {}\n'.format(auc_mean))
            f.write('auc median: {}\n'.format(auc_median))
            f.flush()
    model.train()
    cuda_(model)
