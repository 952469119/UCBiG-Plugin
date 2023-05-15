

# gowalla

LIGHT+

```shell
python main.py --dataset gowalla --model LightGCN --kn_model LightGCN --lr 1e-3 --decay 1e-4  --a 0.8  --k_layers 3 --kn_layer 2  --device cuda:0 --batch_size 1024 --early_stop_times 150 --epochs 1500 --patience 50 --seed 1234 ;
```

NGCF+

```shell
python main.py --dataset gowalla --model NGCF --kn_model NGCF --lr 1e-4 --decay 1e-5 --a 1 --k_layers 3 --kn_layer 2 --device cuda:0 --batch_size 1024 --early_stop_times 70 --epochs 1000 --patience 33 --seed 2023  --node_drop 0 --mess_drop 0.1;
```

# amazon-book

LIGHT+

```shell
python main.py --dataset amazon-book --model LightGCN --kn_model LightGCN --lr 1e-3 --decay 1e-4 --a 1.2 --k_layers 3 --kn_layer 2 --device cuda:0 --batch_size 2048 --early_stop_times 150 --epochs 1500 --patience 50 --seed 1234;
```

NGCF+

```shell
python main.py --dataset amazon-book --model NGCF --kn_model NGCF --lr 5e-4 --decay 1e-5 --a 1 --k_layers 3 --kn_layer 2 --device cuda:0 --batch_size 1024 --seed 2023 --early_stop_times 30 --epochs 300 --seed 1234
```

# yelp 2018

LIGHT+
```shell
python main.py --dataset yelp2018 --model LightGCN --kn_model LightGCN --lr 1e-3 --decay 1e-4 --a 0.3 --k_layers 4 --kn_layer 2 --device cuda:0 --batch_size 1024 --early_stop_times 150 --epochs 1500 --patience 50 --seed 2023 ;
```

NGCF+
```shell
python main.py --dataset yelp2018 --model NGCF --kn_model NGCF --lr 1e-4 --decay 1e-4 --a 1 --k_layers 3 --kn_layer 1 --device cuda:0 --batch_size 1024 --early_stop_times 50 --epochs 500 --patience 20 --node_drop 0 --mess_drop 0.1 --seed 1234;
```
