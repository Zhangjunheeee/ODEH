import time
from copy import deepcopy

import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F

from sklearn.metrics import roc_auc_score

from cope import CoPE
from model_utils import *
from eval_utils import *


def train_one_epoch(model, optimizer, train_dl, delta_coef=1e-5, tbptt_len=20,
                    valid_dl=None, test_dl=None, fast_eval=True, adaptation=False, adaptation_lr=1e-4):
    weight = None
    state_loss_fn = nn.CrossEntropyLoss(weight=weight)
    last_xu, last_xi = model.get_init_states()
    loss_pp = 0.
    loss_norm = 0.
    loss_state = 0.
    optimizer.zero_grad()
    model.train()
    counter = 0
    pbar = tqdm.tqdm(train_dl, ncols=50)
    cum_loss = 0.
    for i, batch in enumerate(pbar):
        t, dt, adj, i2u_adj, u2i_adj, users, items, feats, states = batch
        step_loss, delta_norm, last_xu, last_xi, *_ = model.propagate_update_loss(adj, dt, last_xu, last_xi, feats, i2u_adj, u2i_adj, users, items)
        loss_pp += step_loss
        loss_norm += delta_norm
        if states is not None:
            pred_states = model.predict_state_change(last_xu[users])
            loss_state += state_loss_fn(pred_states, states)
        counter += 1
        if (counter % tbptt_len) == 0 or i == (len(train_dl) - 1):
            total_loss = (loss_pp + loss_norm * delta_coef + loss_state) / counter
            total_loss.backward()
            optimizer.step()
            cum_loss += total_loss.item()
            pbar.set_description(f"Loss={cum_loss/i:.4f}")
            last_xu = last_xu.detach()
            last_xi = last_xi.detach()
            optimizer.zero_grad()
            loss_pp = 0.
            loss_norm = 0.
            loss_state = 0.
            counter = 0
    pbar.close()
    if fast_eval:
        if adaptation:
            rollout_evaluate_fast_ttt(model, delta_coef, adaptation_lr, valid_dl, test_dl, last_xu.detach(), last_xi.detach())
        else:
            rollout_evaluate_fast(model, valid_dl, test_dl, last_xu.detach(), last_xi.detach())
    else:
        if adaptation:
            rollout_evaluate(model, train_dl, valid_dl, test_dl)
        else:
            rollout_evaluate_ttt(model, delta_coef, adaptation_lr, train_dl, valid_dl, test_dl)


def rollout_evaluate_fast(model, valid_dl, test_dl, train_xu, train_xi):
    valid_xu, valid_xi, valid_ranks, auc = rollout(valid_dl, model, train_xu, train_xi)
    print(f"------- Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f} AUC: {auc:.4f}")
    _u, _i, test_ranks, auc = rollout(test_dl, model, valid_xu, valid_xi)
    print(f"=======  Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f} AUC: {auc:.4f}")


def rollout_evaluate_fast_ttt(model, delta_coef, lr, valid_dl, test_dl, train_xu, train_xi):
    model_states = deepcopy(model.state_dict())
    valid_xu, valid_xi, valid_ranks, valid_auc = rollout_with_ttt(valid_dl, model, lr, train_xu, train_xi, delta_coef)
    print(f"------- Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f} AUC: {valid_auc:.4f}")
    _u, _i, test_ranks, test_auc = rollout_with_ttt(test_dl, model, lr, valid_xu, valid_xi, delta_coef)
    print(f"=======  Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f} AUC: {test_auc:.4f}")
    model.load_state_dict(model_states)


def rollout_evaluate(model, train_dl, valid_dl, test_dl):
    train_xu, train_xi, train_ranks, auc = rollout(train_dl, model, *model.get_init_states())
    print(f"Train MRR: {mrr(train_ranks):.4f} Recall@10: {recall_at_k(train_ranks, 10):.4f} AUC: {auc:.4f}")
    valid_xu, valid_xi, valid_ranks, auc = rollout(valid_dl, model, train_xu, train_xi)
    print(f"Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f} AUC: {auc:.4f}")
    _u, _i, test_ranks, auc = rollout(test_dl, model, valid_xu, valid_xi)
    print(f"Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f} AUC: {auc:.4f}")


def rollout_evaluate_ttt(model, delta_coef, lr, train_dl, valid_dl, test_dl):
    model_states = deepcopy(model.state_dict())
    train_xu, train_xi, train_ranks, auc = rollout(train_dl, model, *model.get_init_states())
    print(f"Train MRR: {mrr(train_ranks):.4f} Recall@10: {recall_at_k(train_ranks, 10):.4f} AUC: {auc:.4f}")
    valid_xu, valid_xi, valid_ranks, valid_auc = rollout_with_ttt(valid_dl, model, lr, train_xu, train_xi, delta_coef)
    print(f"Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f} AUC: {valid_auc:.4f}")
    _u, _i, test_ranks, test_auc = rollout_with_ttt(test_dl, model, lr, valid_xu, valid_xi, delta_coef)
    print(f"=======  Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f} AUC: {test_auc:.4f}")
    model.load_state_dict(model_states)


def rollout(dl, model, last_xu, last_xi):
    model.eval()
    ranks = []
    AUC = []
    true_states = []
    pred_states = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dl, position=0, ncols=50):
            t, dt, adj, i2u_adj, u2i_adj, users, items, feats, states = batch
            prop_user, prop_item, last_xu, last_xi = model.propagate_update(adj, dt, last_xu, last_xi, feats, i2u_adj, u2i_adj)
            rs, au = compute_rank(model, prop_user, prop_item, users, items)
            ranks.extend(rs)
            AUC.append(au)
            if states is not None:
                preds = model.predict_state_change(last_xu[users])
                true_states.append(states.cpu().numpy())
                pred_states.append(preds.cpu().numpy()[:, 1])
    if states is not None:
        true_states = np.concatenate(true_states)
        pred_states = np.concatenate(pred_states)
        auc = roc_auc_score(true_states, pred_states)
    else:
        auc = 0.
    auc = auc_avg(AUC)
    return last_xu, last_xi, ranks, auc


def rollout_with_ttt(dl, model, lr, last_xu, last_xi, delta_coef):
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr)
    ranks = []
    AUC = []
    true_states = []
    pred_states = []
    for batch in tqdm.tqdm(dl, position=0, ncols=50):
        optimizer.zero_grad()
        t, dt, adj, i2u_adj, u2i_adj, users, items, feats, states = batch
        step_loss, delta_norm, last_xu, last_xi, prop_user, prop_item = model.propagate_update_loss(
            adj, dt, last_xu, last_xi, feats, i2u_adj, u2i_adj, users, items
        )

        total_loss = step_loss + delta_norm * delta_coef
        total_loss.backward()
        optimizer.step()

        last_xu = last_xu.detach()
        last_xi = last_xi.detach()

        rs, au = compute_rank(model, prop_user, prop_item, users, items)
        ranks.extend(rs)
        AUC.append(au)
        if states is not None:
            preds = model.predict_state_change(last_xu[users])
            true_states.append(states.cpu().numpy())
            pred_states.append(preds.detach().cpu().numpy()[:, 1])

    if states is not None:
        true_states = np.concatenate(true_states)
        pred_states = np.concatenate(pred_states)
        auc = roc_auc_score(true_states, pred_states)
    else:
        auc = 0.
    auc = auc_avg(AUC)
    return last_xu, last_xi, ranks, auc


def compute_rank(model: CoPE, xu, xi, users, items):
    xu = torch.cat([xu, model.user_states], 1)
    xi = torch.cat([xi, model.item_states], 1)
    xxu = F.embedding(users, xu)
    scores = model.compute_pairwise_scores(xxu, xi)
    ranks = []
    AUC = []
    #计算AUC
    pos_u = F.embedding(users, xu)
    pos_i = F.embedding(items, xi)
    pos_scores = model.compute_matched_scores(pos_u, pos_i)
    n_pos = float(len(pos_scores)) #正样本数目
    neg_u_ids = torch.randint(0, model.n_users, size=[model.n_neg_samples // 2], device=users.device)
    neg_i_ids = torch.randint(0, model.n_items, size=[model.n_neg_samples // 2], device=items.device)
    neg_u = F.embedding(neg_u_ids, xu)
    neg_i = F.embedding(neg_i_ids, xi)
    neg_scores = model.compute_matched_scores(neg_u, neg_i)
    n_neg = float(len(neg_scores)) #负样本数目
    pos_neg_scores = torch.cat([pos_scores,neg_scores],0)
    scores_list = pos_neg_scores.tolist()
    scores_list.sort() #排序
    rank = 0
    pos_scores = pos_scores.tolist()
    for i in range(len(pos_scores)):
        index = scores_list.index(pos_scores[i])
        rank = rank + index
    num1 = (n_pos * (n_pos + 1)) / 2
    num2 = n_pos * n_neg
    auc = (rank - num1) / num2
    for line, i in zip(scores, items):
        r = (line >= line[i]).sum().item()
        ranks.append(r)
    return ranks, auc

