import torch.nn as nn
from dataIterator import *
import torch.nn.functional as F

class LightGCNlayer(nn.Module):
    def __init__(self, layers, adj):
        super(LightGCNlayer, self).__init__()
        self.n_layers = layers
        self.adj = adj

    def __dropout_x(self, x, drop):
        keep_prob = 1 - drop
        size = x.size()
        index = x._indices().t()
        values = x._values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size)
        return g

    def bpr_loss(self, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0, batch_size):
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / batch_size
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    def forward(self, users_emb, items_emb):
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        return light_out


class NGCFlayer(nn.Module):
    def __init__(self, layers, adj, embed_size=64, node_drop=0, mess_drop=0.1):
        super(NGCFlayer, self).__init__()

        self.emb_size = embed_size

        self.node_dropout = node_drop
        self.mess_dropout = [mess_drop] * layers

        self.adj = adj

        self.layers = [embed_size] * layers

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.weight_dict = self.init_weight()

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

        return weight_dict

    def bpr_loss(self, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0, batch_size):
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users_emb) ** 2 + torch.norm(pos_emb) ** 2 + torch.norm(neg_emb) ** 2) / 2
        emb_loss = regularizer / batch_size

        return mf_loss, emb_loss

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate

        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, item_emb):
        if self.training and self.node_dropout != 0:
            A_hat = self.sparse_dropout(self.adj, self.node_dropout, self.adj._nnz())
        else:
            A_hat = self.adj

        ego_embeddings = torch.cat([user_emb, item_emb], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = side_embeddings @ self.weight_dict['W_gc_%d' % k] + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = bi_embeddings @ self.weight_dict['W_bi_%d' % k] + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            if self.training and self.mess_dropout[k] != 0:
                ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)

        return all_embeddings
