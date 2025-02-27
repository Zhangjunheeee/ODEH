import torch
from torch import nn, optim
from torch.nn import functional as F

import argparse

from trainer import *
from dataloader import *
from data_utils import *
from eval_utils import *
from trainer import *
from ode import ODEH


ending_time = 1.
burnin_time = 0.0
alpha = 0.98
hidden_size = 128
n_neg_samples = 16

tbptt_len = 20
delta_coef = 0.


def load(name):
    if name in {'wikipedia', 'lastfm', 'reddit', 'mooc'}:
        df, feats = load_jodie_data(f'data/{name}.csv')
    else:
        df, feats = load_recommendation_data(f'data/{name}_5.csv')
    return df, feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='garden')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ttt_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--state_change', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--ttt', action='store_true')
    parser.add_argument('--feat', action='store_true')
    parser.add_argument('--out_dim', type=int, default=256)

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.cuda}')
    df, feats = load(args.dataset)
    if not args.state_change:
        df = df.iloc[:, :3]
    if args.dataset in {'wikipedia', 'lastfm', 'reddit', 'mooc'} and args.feat:
        feat_dim = feats.shape[1]
    else:
        feat_dim = None

    n_users, n_items = df.iloc[:, :2].max() + 1

    train_dl, valid_dl, test_dl = get_dataloaders(df, feats, device, ending_time, burnin_time, alpha)
    model = CoPE(args, n_users, n_items, hidden_size,
                 n_neg_samples=n_neg_samples,
                 feat_dim=feat_dim,
                ).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(1):
        train_one_epoch(model, optimizer, train_dl, delta_coef, tbptt_len, valid_dl, test_dl, fast_eval=args.fast, adaptation=args.ttt, adaptation_lr=args.ttt_lr)
    
    torch.save(model.state_dict(),'model.pkl')
