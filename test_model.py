from utils import *


topks = [20]

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def test(model, data, mode=0, rate=1,test_batch_size=8192):
    u_batch_size = test_batch_size
    max_K = 20

    model.eval()
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(data.test_set.keys())

        users_list = []
        rating_list = []
        groundTrue_list = []
        train_set = data.train_set
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(users, batch_size=u_batch_size):

            groundTrue = [data.test_set[u] for u in batch_users]

            item = list(range(data.G.n_items))
            user_item, all_item_emb, _ = model(batch_users, item, [])

            if mode == 1:
                user_item = user_item[:,:64]
                all_item_emb = all_item_emb[:, :64]
            elif mode ==2:
                user_item= user_item[:, 64:]
                all_item_emb = all_item_emb[:, 64:]
            elif mode==3:
                user_item = user_item
                all_item_emb0 = all_item_emb[:,:64]
                all_item_emb1 = all_item_emb[:,64:]

                all_item_emb1 = all_item_emb1*rate

                all_item_emb = torch.cat((all_item_emb0, all_item_emb1), dim=1)

            rating = model.rating(user_item, all_item_emb)

            exclude_index = []
            exclude_items = []
            for i in range(len(batch_users)):
                exclude_index.extend([i] * len(train_set[batch_users[i]]))
                exclude_items.extend(train_set[batch_users[i]])
            exclude_index = np.array(exclude_index)
            exclude_items = np.array(exclude_items)
            rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=max_K)

            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        results['recall'] = float(results['recall'])
        results['precision'] = float(results['precision'])
        results['ndcg'] = float(results['ndcg'])
        return results

