from model import *
from datetime import datetime
from _parser import *
from test_model import *
from time import time

args = parse_args()
device = torch.device(args.device)

def train(model, data_generator, optim):
    model.train()
    n_batch = data_generator.n_train // args.batch_size + 1
    batch = 0

    S = data_generator.samply_all()
    users = torch.Tensor(S[:, 0]).long().to(device)
    posItems = torch.Tensor(S[:, 1]).long().to(device)
    negItems = torch.Tensor(S[:, 2]).long().to(device)

    users, posItems, negItems = shuffle(users, posItems, negItems)

    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            minibatch(users, posItems, negItems, batch_size=args.batch_size)):
        batch_loss = model.bpr_loss(batch_users, batch_pos, batch_neg)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        batch += batch_loss

    log = {'loss': batch.cpu().detach().numpy() / n_batch}
    return log

if __name__ == '__main__':

    start_time = datetime.now().strftime("%m-%d-%H-%M")
    print("[start training time]" + start_time)

    set_seed(args.seed)
    data_generator = Data(args)
    data_generator.G.print_n()
    early_stop_count = 0
    best_log = {"recall": 0, "ndcg": 0}

    model = mymodel(data_generator.G, args).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='max', factor=0.5,
                                                                  patience=args.patience)
    for epoch in range(args.epochs):
        t1 = time()
        log = train(model, data_generator, optim)
        ret = test(model, data_generator)
        log.update(ret)
        t2 = time()
        print(f"[{t2 - t1:.1f}s]epoch={epoch},loss={log['loss']:.4f},recall={log['recall']:.5f},ndcg={log['ndcg']:.5f}")

        reduce_scheduler.step(log['ndcg'])

        if best_log["ndcg"] < log['ndcg'] or best_log["recall"] < log['recall']:
            best_log["ndcg"] = max(best_log["ndcg"], log['ndcg'])
            best_log["recall"] = max(best_log["recall"], log['recall'])
            best_log["epoch"] = epoch
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count == args.early_stop_times:
                print("early stop！")
                break

    print("[best]", best_log)
    end_time = datetime.now().strftime("%m-%d-%H-%M")
    print("[start training time]" + start_time)
    print("[ end  training time]" + end_time)

