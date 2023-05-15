from layers import *
from torch_scatter import scatter_mean


class mymodel(nn.Module):
    def __init__(self, G, args):
        super(mymodel, self).__init__()

        self.n_user = G.n_users
        self.n_item = G.n_items

        self.device = torch.device(args.device)
        self.emb_size = args.embed_size
        self.decay = args.decay


        self.user_emb, self.item_emb = self.init_weight(args)
        self.kn_layer = args.kn_layer
        if args.model == 'LightGCN':
            adj = G.create_adj_mat(mode="norm", self_loop=False).to(self.device)
            self.sage_module = LightGCNlayer(args.k_layers, adj)
        elif args.model == 'NGCF':
            adj = G.create_adj_mat(mode="mean", self_loop=True).to(self.device)
            self.sage_module = NGCFlayer(args.k_layers, adj,node_drop=args.node_drop, mess_drop=args.mess_drop)
        else:
            exit(0)

        if self.kn_layer > 0:
            self.a = args.a
            all_id_map, user_kn_G = G.get_user_kn_G(args)
            self.n_kn = user_kn_G.n_items

            all_index = {}
            kns_number = {}
            for kns in all_id_map:
                kn_item = []

                for n in kns:
                    kn_item.append(kns[n])
                if len(kn_item) == 0:
                    continue
                kn_item = np.array(kn_item)
                k = kn_item.shape[1]
                kns_number[k] = len(kns)
                index = kn_item.reshape(-1)
                all_index[k] = torch.from_numpy(index).to(torch.long).to(self.device)
                all_index[k] -= self.n_user
            self.one_index = torch.cat(list(all_index.values()), dim=0)
            self.all_index = all_index
            self.kns_number = kns_number

            if args.kn_model == 'LightGCN':
                user_kn_adj = user_kn_G.create_adj_mat(mode="norm", self_loop=False).to(self.device)
                self.kn_sage_module = LightGCNlayer(args.kn_layer, user_kn_adj)
            elif args.kn_model == 'NGCF':
                user_kn_adj = user_kn_G.create_adj_mat(mode="mean", self_loop=True).to(self.device)
                self.kn_sage_module = NGCFlayer(args.kn_layer, user_kn_adj, node_drop=args.node_drop,
                                                mess_drop=args.mess_drop)
            else:
                exit(0)

    def init_weight(self, args):

        embedding_user = torch.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.emb_size)
        embedding_item = torch.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.emb_size)
        if args.kn_model == "NGCF":
            # xavier init
            initializer = nn.init.xavier_uniform_
            embedding_user = initializer(embedding_user.weight)
            embedding_item = initializer(embedding_item.weight)
        elif args.kn_model == "LightGCN":
            initializer = nn.init.normal_
            embedding_user = initializer(embedding_user.weight, std=0.1)
            embedding_item = initializer(embedding_item.weight, std=0.1)

        return embedding_user, embedding_item

    def item_to_kn(self, item_emb):
        all_kn_embs = []
        for k, index in self.all_index.items():
            kn_emb = item_emb[index]
            kn_emb = kn_emb.reshape(-1, k, self.emb_size)
            kn_emb = torch.mean(kn_emb, dim=1)
            all_kn_embs.append(kn_emb)
        all_kn_emb = torch.cat(all_kn_embs, dim=0)
        return all_kn_emb

    def kn_to_item(self, kn_emb):
        start = 0
        all_kn_embs = []
        for k in self.kns_number:
            end = start + self.kns_number[k]
            emb = torch.repeat_interleave(kn_emb[start:end, :], k, dim=0)
            all_kn_embs.append(emb)
            start = end
        all_kn_emb = torch.cat(all_kn_embs, dim=0)
        out_shape = torch.zeros((self.n_item, kn_emb.shape[1])).to(self.device)
        item = scatter_mean(all_kn_emb, self.one_index, dim=0, out=out_shape)

        return item

    def kn_net_forward(self):

        kn_emb = self.item_to_kn(self.item_emb)

        kn_sage_emb = self.kn_sage_module(self.user_emb, kn_emb)

        user_emb = kn_sage_emb[:self.n_user]
        kn_emb = kn_sage_emb[self.n_user:]
        item_emb = self.kn_to_item(kn_emb)

        item_emb = item_emb * self.a

        out_emb = torch.cat([user_emb, item_emb], dim=0)
        return out_emb

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb = self.forward(users, pos, neg)
        userEmb0, posEmb0, negEmb0 = self.user_emb[users], self.item_emb[pos], self.item_emb[neg]

        loss, reg_loss = self.sage_module.bpr_loss(users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0, len(users))

        reg_loss = reg_loss * self.decay

        return loss + reg_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, user, pos_item, neg_item):

        sage_emb = self.sage_module(self.user_emb, self.item_emb)

        if self.kn_layer > 0:
            kn_sage_emb = self.kn_net_forward()
            out = torch.cat([sage_emb, kn_sage_emb], dim=1)
        else:
            out = sage_emb

        user_emb = out[:self.n_user]
        item_emb = out[self.n_user:]
        return user_emb[user], item_emb[pos_item], item_emb[neg_item]
