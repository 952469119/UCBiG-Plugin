from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import random


# 将scipy.sparse转为torch.sparse.tensor
def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    t = torch.from_numpy(np.stack([coo.row, coo.col]).astype(np.int64))
    v = torch.from_numpy(coo.data).float()
    return torch.sparse_coo_tensor(t, v, coo.shape)


def load_data(file, user_bias=0):
    set = {}
    u_num = 0
    i_num = 0
    with open(file, "r") as f:
        for l in f:
            l = l.strip("\n").split(" ")
            u_num = max(u_num, eval(l[0]))
            if len(l) > 1:
                set[eval(l[0])] = []

                for i in range(1, len(l)):
                    set[eval(l[0])].append(eval(l[i]) + user_bias)
                    i_num = max(i_num, eval(l[i]))

    return set, u_num, i_num


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def rm_edge_under_weight(G, weight):
    print("removing edge by weight...")
    for node in tqdm(G.nodes()):
        rmedge = []
        for to_node in G[node]:
            w = G[node][to_node]['weight']
            if w < weight:
                rmedge.append(to_node)
        for to in rmedge:
            G.remove_edge(node, to)
    return G


def rm_edge_under_degree(G, degree):
    print("removing edge by degree...")
    for n in G.nodes():
        if nx.degree(G, n) < degree:
            rmedge = G[n]
            for to in dict(rmedge).keys():
                G.remove_edge(n, to)
    return G


def count_under_degree_node(G, degree):
    count = 0

    for n, d in nx.degree(G):
        if d < degree:
            count += 1
    return count


def split_Kn(G, window, max_num_k, min_num_k=3, mode="random"):
    visit = {}
    if mode == "random":
        # 这个方法意味着随机排列所有的节点，因此度数大的节点更大概率分到更多的kn
        nodes = list(G.nodes())
        random.shuffle(nodes)
    elif mode == "sort":
        # 这个方法意味着更小的度数的节点会被优先遍历，因此拥有至少一个kn的节点数目会更多
        nodes = nx.degree(G)
        nodes = sorted(nodes, key=lambda x: x[1])
        nodes = [i[0] for i in nodes]
    else:
        # 按顺序访问，编号小的节点拥有更多kn
        nodes = list(G.nodes())
    for n in nodes:
        visit[n] = 0

    # 对于每个顶点，已经构成过kn的邻居的集合
    # 这意味着以该顶点为中心进行搜索时，不允许访问这些邻居，该顶点可以通过访问别的邻居构建新的kn，别的顶点依旧可以与该顶点和其邻居构成新的kn
    node_visit_nbr = {}
    for n in nodes:
        node_visit_nbr[n] = set()

    def find_Kn(all_nbr, exist_n, node, k):
        if visit[node] >= window:
            return False
        # n的邻居
        n_nbr = G[node]
        # 加上n自己本身
        exist_n = exist_n.copy()
        exist_n.add(node)

        if k == 1:
            kn = list(exist_n)
            for exist in kn:
                if visit[exist] >= window:
                    return False

            kn.sort()
            str_kn = ""

            for i in kn:
                str_kn += str(i) + " "
            for j in range(len(kn)):
                node_kn[kn[j]].append(str_kn)
                visit[kn[j]] += 1
                kn_map[str_kn] = 0
                node_visit_nbr[kn[j]].update(set(kn))
            return True

        # n的邻居中去掉已经存在的
        n_nbr = set(n_nbr) - exist_n
        # 取所有的邻居与自身邻居的交集
        nbr = n_nbr & all_nbr

        nbr = nbr.copy()
        all_n = exist_n.copy()
        nbr_list = list(nbr)
        # 共同邻居的数量少于需要构成当前kn的数量，剪枝
        if len(nbr_list) < k - 1:
            return False

        random.shuffle(nbr_list)
        for new_node in nbr_list:
            if visit[new_node] >= window:
                continue
            for exist in exist_n:
                if visit[exist] >= window:
                    return False
            flag = find_Kn(nbr, all_n, new_node, k - 1, )
            if flag:
                return True
        return False

    id_index = 0
    node_kn = {}
    all_kn_map = {}
    id_maps = []
    for i in G.nodes():
        node_kn[i] = []
    for k in reversed(range(min_num_k, max_num_k + 1)):
        kn_map, id_map = {}, {}
        for n in tqdm(nodes):
            if visit[n] >= window or len(G[n]) == 0:
                continue
            loop = True
            while loop:
                allow_nbr = set(G[n]) - node_visit_nbr[n]
                loop = find_Kn(allow_nbr, set(), n, k)
        # 重新编号,并将字典中的字符串转化为numpy数组
        for key in kn_map:
            kn_map[key] = id_index
            kn = key.strip(" ").split(" ")
            id_map[id_index] = np.array([eval(k) for k in kn])
            id_index += 1
        all_kn_map.update(kn_map)
        id_maps.append(id_map)
        print(f"numbers of K{k}:{len(id_map)}")
        # 将node_kn中存的kn信息映射为其的编号
    for node in node_kn:
        for i in range(len(node_kn[node])):
            node_kn[node][i] = all_kn_map[node_kn[node][i]]

    return node_kn, id_maps


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays):
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    return result


def samply_once(item, window):
    nodes = list(item.nodes())
    random.shuffle(nodes)
    for node in tqdm(nodes):
        edge_dict = {}
        w = window
        for to_node in item[node]:
            edge_dict[to_node] = item[node][to_node]['weight']
        edge_dict = sorted(edge_dict.items(), key=lambda x: x[1], reverse=True)
        # 低于这个度数的边都要删除
        if len(edge_dict) <= window:
            continue
        low_degree = edge_dict[window - 1][1]
        # 找到所有低于这个度数的边
        need_del = []
        need_random = []
        count = 0
        for to_node in item[node]:
            if item[node][to_node]['weight'] < low_degree:
                need_del.append(to_node)
            elif item[node][to_node]['weight'] == low_degree:
                need_random.append(to_node)
            else:
                count += 1
        need_del2 = random.sample(need_random, len(need_random) - (w - count))
        need_del.extend(need_del2)
        for to_node in need_del:
            item.remove_edge(node, to_node)


def assert_min_edge(item, samply_item, min_edge):
    need_add_node = []
    for n, d in nx.degree(samply_item):
        if d < 5:
            need_add_node.append(n)
    for node in need_add_node:
        exist_edge = samply_item[node]
        need_random_edge = []
        need_add = min_edge - len(exist_edge)
        if len(exist_edge) == 0:
            need_random_edge = item[node]
        else:
            min_degree = 100000
            for to_node in samply_item[node]:
                min_degree = min(item[node][to_node]['weight'], min_degree)
            for to_node in item[node]:
                if item[node][to_node]['weight'] >= min_degree - 1:
                    need_random_edge.append(to_node)
            need_random_edge = set(need_random_edge) - set(exist_edge)
        need_random_edge = list(need_random_edge)
        if len(need_random_edge) > need_add:
            need_add_edge = random.sample(need_random_edge, need_add)
        else:
            need_add_edge = need_random_edge

        for to_node in need_add_edge:
            samply_item.add_edge(node, to_node, weight=item[node][to_node]['weight'])
