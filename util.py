import torch
import yaml
import scipy.sparse as sp
import numpy as np
import re


def read_param(filename):
    with open(filename, 'r') as fd:
        data = yaml.safe_load(fd)
    return data


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    adj = adj + sp.identity(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

