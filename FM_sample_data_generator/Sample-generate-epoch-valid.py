
from sample_utils import *

import time

import argparse



def _get_valid_batch(iter_, bs, valid_list, sample_dict):
    '''
    :param iter_: iteration count
    :param bs: batch size
    :return: 4 tensors:
    (1) user id
    (2) positive item id that has interacted with user
    (3) negative item id that has not interacted with the user in the candidate item set
    (4) preference attribute ids that confirmed by user in current turn
    '''
    left, right = iter_ * bs, min(len(valid_list), (iter_ + 1) * bs)
    user_input, item_p_input = zip(*valid_list[left:right])

    preference_list, i_neg2_output = list(), list()
    for instance in item_p_input:
        # ________neg2 item & prefer features___________
        pool = sample_dict[str(instance)]  # item_sample_dict : {'item_id': [(prefer_feas, neg2_items), (prefer_feas, neg2_items)...]}
        i = 0
        while(True):
            i += 1
            current = random.choice(pool)   # select one turn: (prefer_feas, neg2_items)
            if len(current[1]) >= 1:
                preference_list.append(current[0])
                i_neg2_output.append(current[1])  # select one neg2_item
                break
        if i > 40:
            print('weird')

    assert len(item_p_input) == len(i_neg2_output)

    return user_input, item_p_input, i_neg2_output, preference_list




def main():
    parser = argparse.ArgumentParser(description="Generate DeepFM-BPR Data.")
    parser.add_argument('-start', type=int, default=0, metavar='<start>', dest='start', help='start')
    parser.add_argument('-seed', type=int, default=1, metavar='<seed>', dest='seed', help='seed')
    parser.add_argument('-se', type=int, default=250, metavar='<se>', dest='se', help='sample epoch')
    parser.add_argument('-bs', type=int, default=128, metavar='<bs>', dest='bs', help='batch size')
    parser.add_argument('--data_name', type=str, default='YELP', metavar='<data_name>', dest='data_name',
                        choices=[LAST_FM, LAST_FM_STAR, YELP],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP}.'
                             'YELP and YELP_STAR share the same data structure (both use fine-grained attributes)')
    parser.add_argument('-mode', type=str, default='valid', help='mode{test,valid}')

    args = parser.parse_args()
    set_random_seed(args.seed)
    # Load item-sample-dict (Saving neg2_cand_items )
    sample_dict = load_sample_dict(dataset=args.data_name, mode=args.mode)
    #  Load UI data
    ui_dict = load_ui_data(args.data_name, args.mode)
    ui_list = []
    for user_str, items in ui_dict.items():
        user_id = int(user_str)
        for item_id in items:
            ui_list.append([user_id, item_id])
    ui_array = np.array(ui_list)
    np.random.shuffle(ui_array)
    data = ui_array  # UI data
    valid_list = data.tolist()
    print('Number of UI interaction in {} data: {}'.format(args.mode, data.shape[0]))

    bs = args.bs
    for epoch in range(1):
        # TODO: how about the random seed here?
        kg = load_kg(args.data_name)  # Last_FM KG
        # random.shuffle(train_list)
        print('Have processed {} epochs'.format(epoch))
        max_iter = int(len(valid_list) / float(bs))
        print('max_iter:', max_iter)
        start_time = time.time()
        user_output, item_p_output, i_neg1_output, i_neg2_output, preference_list = list(), list(), list(), list(), list()
        for iter_ in range(max_iter):
            I, II, IV, V = _get_valid_batch(iter_, bs, 1, valid_list, sample_dict)
            user_output += I
            item_p_output += II
            i_neg2_output += IV
            preference_list += V
            if iter_ > 1 and iter_ % 20 == 0:
                print('Takes {} seconds to finish {}% of this epoch'.format(str(time.time() - start_time),
                                                                            float(iter_) * 100 / max_iter))

        #valid list: user_output, item_p_output, i_neg2_output, preference_list
        out = [user_output, item_p_output, i_neg2_output, preference_list]
        save_fm_sample(dataset=args.data_name, sample_data=out, mode=args.mode)


if __name__ == '__main__':
    main()




