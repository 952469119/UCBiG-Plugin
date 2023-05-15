import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RUN MODEL")
    parser.add_argument('--dataset', default="gowalla", help="gowalla,yelp2018,amazon-book")
    parser.add_argument('--path', default="./dataset/")
    # ------------------model------------------
    parser.add_argument('--model', default="LightGCN", help='NGCF,LightGCN')
    parser.add_argument('--kn_model', default="LightGCN", help='NGCF,LightGCN')
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--k_layers', type=int, default=3,help='number of layers in user-item graph')
    parser.add_argument('--kn_layer', type=int, default=2,help='number of layers in user-cliques graph')
    parser.add_argument('--a', type=float, default=0.8, help='alpha rate')
    parser.add_argument('--node_drop', type=float, default=0, help='node drop')
    parser.add_argument('--mess_drop', type=float, default=0.1, help='mess drop')
    # ------------------samply kn--------------------
    parser.add_argument('--num_kn_window', type=int, default=5,help="each item node most has num_kn_window cliques")
    parser.add_argument('--rank_max_window', type=int, default=15,help='most neighbors of item nodes in the item-item graph')
    parser.add_argument('--rank_min_window', type=int, default=0, help='least neighbors of item nodes in the item-item graph')
    parser.add_argument('--max_k', type=int, default=10, help='max k-clique size')
    parser.add_argument('--min_k', type=int, default=3,help='min k-clique size')
    parser.add_argument('--samply_mode', default="rank", help='rank,weight')
    # ------------------train------------------
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--early_stop_times', type=int, default=150)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=1e-4, help='Regularizations.')
    parser.add_argument('--device', default="cuda:0", help='default using gpu,or cpu')

    return parser.parse_args()
