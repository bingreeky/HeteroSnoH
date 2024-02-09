import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pdb
import torch.nn.init as init
import math

def prune_adj(oriadj, non_zero_idx, percent):
    
    original_prune_num = int((non_zero_idx / 2) * (percent/100))
    adj = np.copy(oriadj)
    #print("percent:", percent)
    low_adj= np.tril(adj, -1)
    non_zero_low_adj = low_adj[low_adj != 0]
    low_pcen = np.percentile(abs(non_zero_low_adj), percent)
    #print("percentile " + str(low_pcen))
    under_threshold = abs(low_adj) < low_pcen
    before = len(non_zero_low_adj)
    low_adj[under_threshold] = 0
    non_zero_low_adj = low_adj[low_adj != 0]
    after = len(non_zero_low_adj)
    rest_pruned = original_prune_num - (before - after)
    if rest_pruned > 0:
        mask_low_adj = (low_adj != 0)
        low_adj[low_adj == 0] = 2000000
        flat_indices = np.argpartition(low_adj.ravel(), rest_pruned - 1)[:rest_pruned]
        row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
        low_adj = np.multiply(low_adj, mask_low_adj)
        low_adj[row_indices, col_indices] = 0
    adj = low_adj + np.transpose(low_adj)
    adj = np.add(adj, np.identity(adj.shape[0]))

    return adj

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

