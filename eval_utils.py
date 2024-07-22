import numpy as np


def mrr(ranks):
    ranks = np.array(ranks)
    return (1 / ranks).mean()


def recall_at_k(ranks, k):
    ranks = np.array(ranks)
    return (ranks <= k).mean()

def auc_avg(AUC):
    AUC = np.array(AUC)
    return AUC.mean()