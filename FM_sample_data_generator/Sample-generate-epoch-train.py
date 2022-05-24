
from sample_utils import *

import time

import argparse



def _get_train_batch(train_list, kg, iter_, bs, ui_dict, sample_dict):
    '''
    :param iter_: iteration count
    :param bs: batch size
    :return: 5 tensors:
    (1) user id
    (2) positive item id that has interacted with user
    (3) negative item id that has not interacted with user
    (4) negative item id that has not interacted with the user in the candidate item set
    (5) preference attribute ids that confirmed by user in current turn
    '''
    left, right = iter_ * bs, min(len(train_list), (iter_ + 1) * bs)
    user_input, item_p_input = zip(*train_list[left:right])
    user_out, item_out, i_neg1_output, i_neg2_output, preference_list = [], [], [], [], []
    for row in range(len(user_input)):
        user_id = user_input[row]
        item_id = item_p_input[row]

        # _______ negative instances _______
        user_iter_items = set(kg.G['user'][user_id]['interact'])
        items_len = len(kg.G['item'].keys())
        i_neg1 = np.random.choice(items_len)
        while i_neg1 in ui_dict[str(user_id)]:
            i_neg1 = np.random.choice(items_len)
        i_neg1_output.append(i_neg1)

        # ________neg2 item & prefer features___________
        pool = sample_dict[str(item_id)]  # item_sample_dict : {'item_id': [(prefer_feas, neg2_items), (prefer_feas, neg2_items)...]}
        current = random.choice(pool)  # select one turn: (prefer_feas, neg2_items)
        while len(set(current[1]) - set(ui_dict[str(user_id)])) == 0:  # neg2_items can't occur in positve items
            current = random.choice(pool)
        preference_list.append(current[0])

        i_neg2 = None
        if len(current[1]) > 0:
            i_neg2 = random.choice(current[1])  # select one neg2_item
            while i_neg2 in ui_dict[str(user_id)]:
                i_neg2 = random.choice(current[1])
        i_neg2_output.append(i_neg2)

    non_count = len([item for item in i_neg2_output if item is None])

    return user_input, item_p_input, i_neg1_output, i_neg2_output, preference_list



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
    parser.add_argument('-mode', type=str, default='train', help='mode')  # {train}
    args = parser.parse_args()
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
    train_list = data.tolist()

    print('Number of UI interaction in {} data: {}'.format(args.mode, data.shape[0]))
    bs = args.bs
    for epoch in range(args.se):
        # TODO: how about the random seed here ?
        set_random_seed(args.seed)
        kg = load_kg(args.data_name)
        print('Have processed {} epochs'.format(epoch))
        max_iter = int(len(train_list) / float(bs))
        print('max_iter:', max_iter)
        start_time = time.time()
        user_output, item_p_output, i_neg1_output, i_neg2_output, preference_list = list(), list(), list(), list(), list()
        for iter_ in range(max_iter):
            I, II, III, IV, V = _get_train_batch(train_list, kg, iter_, bs, ui_dict, sample_dict)
            user_output += I
            item_p_output += II
            i_neg1_output += III
            i_neg2_output += IV
            preference_list += V
            if iter_ > 1 and iter_ % 2000 == 0:
                print('Takes {} seconds to finish {}% of this epoch'.format(str(time.time() - start_time), float(iter_) * 100 / max_iter))

        out = [user_output, item_p_output, i_neg1_output, i_neg2_output, preference_list]

        save_fm_sample(dataset=args.data_name, sample_data=out, mode='train', epoch=epoch)

if __name__ == '__main__':
    main()
