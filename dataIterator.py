import os
import pickle as pkl
from utils import _convert_sp_mat_to_sp_tensor
import scipy.sparse as sp
from utils import *


class BipartiteGraph:
    def __init__(self, B, path):
        self.B = B

        self.cache_path = path + "/cache"
        self.user_node = set({n for n, d in self.B.nodes(data=True) if d["bipartite"] == 0})
        self.item_node = {n for n, d in self.B.nodes(data=True) if d["bipartite"] == 1}
        self.n_users = len(self.user_node)
        self.n_items = len(self.item_node)

    def create_adj_mat(self, mode="norm", self_loop=False):

        adj_mat = nx.to_scipy_sparse_array(self.B, dtype=np.float32)

        adj_mat = adj_mat.todok()

        def mean_adj_single(adj):
            # D^-1 * A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)

            return norm_adj.tocoo()

        def normalized_adj_single(adj):

            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        if self_loop:
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        if mode == "norm":
            adj_mat = normalized_adj_single(adj_mat)
        elif mode == "mean":
            adj_mat = mean_adj_single(adj_mat)
        adj_mat = _convert_sp_mat_to_sp_tensor(adj_mat.tocsr())
        adj_mat = adj_mat.coalesce()
        return adj_mat

    def print_n(self):
        print(
            "number of user:" + str(self.n_users) + "\nnumber of item:" + str(self.n_items) + "\nnumber of edge:" + str(
                self.B.number_of_edges()))
        return self.n_users, self.n_items

    def build_user_kn_G(self, node_kn):
        print("building user-kn Graph...")
        user_kn_G = nx.Graph()
        users = self.user_node
        for user in tqdm(users):
            user_kn_G.add_node(user, bipartite=0)
            for item in self.B[user]:
                for kn_node in node_kn[item]:
                    user_kn_G.add_node(kn_node + self.n_users, bipartite=1)
                    user_kn_G.add_edge(kn_node + self.n_users, user)

        return user_kn_G

    def rank_item_sample(self, max_window):
        if os.path.exists(self.cache_path + "/rank_item_sample" + str(max_window) + ".pkl"):
            sample_item = pkl.load(
                open(self.cache_path + "/rank_item_sample" + str(max_window) + ".pkl", "rb"))
            print("using rank samply items cache")
            return sample_item
        else:
            print("samply items by rank")
            if os.path.exists(self.cache_path + "/rank_item_sample" + str(50) + ".pkl"):
                item = pkl.load(open(self.cache_path + "/rank_item_sample" + str(50) + ".pkl", "rb"))
                print("using rank samply items 50 cache")
            else:
                item = self.get_item_G()
                for i in reversed(range(1, 11)):
                    samply_once(item, i * 100)
                for i in reversed(range(5, 10)):
                    samply_once(item, i * 10)
                pkl.dump(item, open(self.cache_path + "/rank_item_sample" + str(50) + ".pkl", "wb"))
            for i in reversed(range(max_window, 50)):
                samply_once(item, i)
            pkl.dump(item, open(self.cache_path + "/rank_item_sample" + str(max_window) + ".pkl", "wb"))
            return item

    def get_item_G(self):
        if os.path.exists(self.cache_path + "/item_G" + ".pkl"):
            item = pkl.load(open(self.cache_path + "/item_G" + ".pkl", "rb"))
        else:
            print("project weight item graph...this step may cost a lot of time")
            item = nx.bipartite.weighted_projected_graph(self.B, self.item_node)
            pkl.dump(item, open(self.cache_path + "/item_G" + ".pkl", "wb"))
        return item

    def get_user_kn_G(self, args):
        if args.samply_mode == "rank":
            samply_G = self.rank_item_sample(args.rank_max_window)
        elif args.samply_mode == "weight":
            samply_G = self.edge_remove(args.num_k)

        if args.rank_min_window > 0:
            item = self.get_item_G()
            assert_min_edge(item, samply_G, args.rank_min_window)
        node_kn, id_map = split_Kn(samply_G, args.num_kn_window, args.max_k, args.min_k, mode='sort')

        user_kn_G = self.build_user_kn_G(node_kn)
        user_kn_G = BipartiteGraph(user_kn_G, args.path + args.dataset)
        user_kn_G.print_n()

        return id_map, user_kn_G

    def edge_remove(self, num_k):
        if os.path.exists(self.cache_path + "/item_sample-K" + str(num_k) + ".pkl"):
            sample_item = pkl.load(open(self.cache_path + "/item_sample-K" + str(num_k) + ".pkl", "rb"))
            print("using samply items cache")
            return sample_item
        else:
            item = self.get_item_G()
            rm_edge_under_weight(item, num_k)
            while count_under_degree_node(item, num_k) - count_under_degree_node(item, 1) > 0:
                rm_edge_under_degree(item, num_k)
            pkl.dump(item, open(self.cache_path + "/item_sample-K" + str(num_k) + ".pkl", "wb"))
            return item

class Data:
    def __init__(self, args):
        self.dataset = args.dataset
        self.path = args.path + args.dataset
        self.cache_path = self.path + "/cache"
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        self.batch_size = args.batch_size
        G = self.build_graph()
        self.G = BipartiteGraph(G, self.path)
        self.n_train = 0
        self.train_set, self.test_set = self.get_train_test_set()

    def build_graph(self):
        if os.path.exists(self.cache_path + "/G.pkl"):
            print("using G cache")
            G = pkl.load(open(self.cache_path + "/G.pkl", "rb"))
            return G
        G = nx.Graph()
        if os.path.exists(self.path + "/test.txt") and os.path.exists(self.path + "/train.txt"):
            print("bulid G from txt")
            ftest = open(self.path + "/test.txt")
            ftrain = open(self.path + "/train.txt")

            for l in ftrain:
                l = l.split(" ")
                G.add_node(eval(l[0]), bipartite=0)

            for l in ftest:
                l = l.split(" ")
                G.add_node(eval(l[0]), bipartite=0)

            n_user = G.number_of_nodes()

            ftrain.seek(0)
            for l in ftrain:
                l = l.split(" ")
                for i in range(1, len(l)):
                    G.add_node(eval(l[i]) + n_user, bipartite=1)
                    G.add_edge(eval(l[0]), eval(l[i]) + n_user)

            ftest.seek(0)
            for l in ftest:
                l = l.split(" ")
                for i in range(1, len(l)):
                    G.add_node(eval(l[i]) + n_user, bipartite=1)

        pkl.dump(G, open(self.cache_path + "/G.pkl", "wb"))

        return G

    def get_train_test_set(self):

        train_set, _, _ = load_data(self.path + "/train.txt")
        test_set, _, _ = load_data(self.path + "/test.txt")

        n_train = 0
        for u in train_set:
            n_train += len(train_set[u])
        self.n_train = n_train
        return train_set, test_set

    def samply_all(self):
        user_num = self.G.n_users
        users = np.random.randint(0, user_num, self.n_train)
        allPos = self.train_set
        S = []

        for i, user in enumerate(users):
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, self.G.n_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])

        return np.array(S)

