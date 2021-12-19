import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    def __init__(self, emb_size, user_length, item_length, feature_length, qonly, hs, ip, dr):

        super(FactorizationMachine, self).__init__()

        self.user_length = user_length
        self.item_length  = item_length
        self.feature_length = feature_length

        self.hs = hs
        self.ip = ip
        self.dr = dr

        self.dropout2 = nn.Dropout(p=self.dr)  # dropout ratio
        self.qonly = qonly  # only use quadratic form

        # dimensions
        self.emb_size = emb_size

        print('Feature length is: {}'.format(self.feature_length))

        # _______ User embedding + Item embedding
        self.ui_emb = nn.Embedding(user_length + item_length + 1, emb_size + 1, sparse=False)

        # _______ Feature embedding and Preference embedding are common_______
        self.feature_emb = nn.Embedding(self.feature_length + 1, emb_size + 1, padding_idx=self.feature_length, sparse=False)

        # _______ Scala Bias _______
        self.Bias = nn.Parameter(torch.randn(1).normal_(0, 0.01), requires_grad=True)

        self.init_weight()


    def init_weight(self):
        self.ui_emb.weight.data.normal_(0, 0.01)
        self.feature_emb.weight.data.normal_(0, self.ip)

        # _______ set the padding to zero _______
        self.feature_emb.weight.data[self.feature_length,:] = 0

    def forward(self, ui_pair, feature_index, preference_index):
        '''
        param: a list of user ID and busi ID
        '''
        return self.compute(ui_pair, feature_index, preference_index)

    def compute(self, ui_pair, feature_index, preference_index):
        feature_matrix_ui = self.ui_emb(ui_pair)
        nonzero_matrix_ui = feature_matrix_ui[..., :-1]
        feature_bias_matrix_ui = feature_matrix_ui[..., -1:]

        feature_matrix_preference = self.feature_emb(preference_index)
        # _______ dropout has been done already (when data was passed in) _______
        nonzero_matrix_preference = feature_matrix_preference[..., :-1]  # (bs, 2, emb_size)
        feature_bias_matrix_preference = feature_matrix_preference[..., -1:]  # (bs, 2, 1)

        # _______ concatenate them together ______
        nonzero_matrix = torch.cat((nonzero_matrix_ui, nonzero_matrix_preference), dim=1)
        feature_bias_matrix = torch.cat((feature_bias_matrix_ui, feature_bias_matrix_preference), dim=1)

        # _______ make a clone _______
        nonzero_matrix_clone = nonzero_matrix.clone()
        feature_bias_matrix_clone = feature_bias_matrix.clone()

        # _________ sum_square part _____________
        summed_features_embedding_squared = nonzero_matrix.sum(dim=1, keepdim=True) ** 2  # (bs, 1, emb_size)

        # _________ square_sum part _____________
        squared_sum_features_embedding = (nonzero_matrix * nonzero_matrix).sum(dim=1, keepdim=True)  # (bs, 1, emb_size)

        # ________ FM __________
        FM = 0.5 * (summed_features_embedding_squared - squared_sum_features_embedding)  # (bs, 1, emb_size)

        # Optional: remove the inter-group interaction
        # ***---***

        new_non_zero_2 = nonzero_matrix_preference
        summed_features_embedding_squared_new_2 = new_non_zero_2.sum(dim=1, keepdim=True) ** 2
        squared_sum_features_embedding_new_2 = (new_non_zero_2 * new_non_zero_2).sum(dim=1, keepdim=True)
        newFM_2 = 0.5 * (summed_features_embedding_squared_new_2 - squared_sum_features_embedding_new_2)
        FM = (FM - newFM_2)

        # ***---***

        FM = self.dropout2(FM)  # (bs, 1, emb_size)

        Bilinear = FM.sum(dim=2, keepdim=False)  # (bs, 1)
        result = Bilinear + self.Bias  # (bs, 1)

        return result, feature_bias_matrix_clone, nonzero_matrix_clone
    # end def