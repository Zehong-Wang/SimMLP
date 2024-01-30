import datetime
from lib2to3.pytree import BasePattern
import numpy as np
import os
import os.path as osp
import random
import logging
from pathlib import Path
import json

import time
import warnings
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

EPS = 1e-6


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_MB(byte):
    return byte / 1024.0 / 1024.0


def combine_dicts(dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    for key, value in result.items():
        result[key] = np.mean(value)

    return result


def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


# Generate random split from the dataset
def get_split(
        num_samples: int,
        train_ratio: float = 0.1,
        test_ratio: float = 0.1,
        ind_ratio: float = 0.0,
):
    # The ind_ratio is used to split the test set

    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * test_ratio)
    test_size = num_samples - train_size - val_size
    trans_size = int(test_size * (1 - ind_ratio))
    indices = torch.randperm(num_samples)

    if ind_ratio == 0:
        return {
            'train': indices[:train_size],
            'valid': indices[train_size: val_size + train_size],
            'test': indices[val_size + train_size:]
        }
    else:
        return {
            'train': indices[:train_size],
            'valid': indices[train_size: val_size + train_size],
            'test': indices[val_size + train_size: trans_size + val_size + train_size],
            'ind_test': indices[trans_size + val_size + train_size:]
        }


# Convert the split to the split with independent test set (inductive setting)
def to_ind_split(split, ind_ratio: float = 0.2):
    test_split = split['test']
    test_size = len(test_split)
    trans_test_size = int(test_size * (1 - ind_ratio))
    split['test'] = test_split[:trans_test_size]
    split['ind_test'] = test_split[trans_test_size:]
    return split


# Split the transductive set and inductive set
def split_data(data, split):
    from torch_geometric.utils import subgraph

    trans_nodes = torch.concat(
        [split['train'],
         split['valid'],
         split['test']]).unique()
    ind_nodes = split['ind_test'].unique()

    x, y, edge_index, edge_attr = data.x, data.y, data.edge_index, data.edge_attr

    trans_edge_index, trans_edge_attr = subgraph(
        trans_nodes,
        edge_index,
        edge_attr,
        relabel_nodes=True,
        num_nodes=x.shape[0],
        return_edge_mask=False
    )

    ind_edge_index, ind_edge_attr = subgraph(
        ind_nodes,
        edge_index,
        edge_attr,
        relabel_nodes=True,
        num_nodes=x.shape[0],
        return_edge_mask=False
    )

    trans_data = data.clone()
    trans_data.x = x[trans_nodes]
    trans_data.y = y[trans_nodes]
    trans_data.edge_index = trans_edge_index
    trans_data.edge_attr = trans_edge_attr

    ind_data = data.clone()
    ind_data.x = x[ind_nodes]
    ind_data.y = y[ind_nodes]
    ind_data.edge_index = ind_edge_index
    ind_data.edge_attr = ind_edge_attr

    return trans_data, ind_data


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def split2mask(split):
    num_nodes = len(split['train']) + len(split['valid']) + len(split['test'])
    return {
        'train': idx2mask(split['train'], num_nodes),
        'valid': idx2mask(split['valid'], num_nodes),
        'test': idx2mask(split['test'], num_nodes)
    }


def mask2idx(mask):
    return torch.where(mask == 1)[0]


def get_split_from_mask(masks):
    if masks.get('valid', None) is None:
        split = [{
            'train': mask2idx(masks['train']),
            'test': mask2idx(masks['test'])
        }]
        num_train = len(split[0]['train'])
        num_valid = int(num_train * 0.2)
        perm = torch.randperm(num_train)
        split[0]['valid'] = split[0]['train'][perm[:num_valid]]
        split[0]['train'] = split[0]['train'][perm[num_valid:]]
        return split

    elif masks['train'].dim() == 1:
        return [{
            'train': mask2idx(masks['train']),
            'valid': mask2idx(masks['valid']),
            'test': mask2idx(masks['test'])
        }]
    elif masks['train'].dim() == 2:
        num_splits = masks['train'].shape[1]
        return [
            {
                'train': mask2idx(masks['train'][:, i]),
                'valid': mask2idx(masks['valid'][:, i]),
                'test': mask2idx(masks['test'][:, i])
            } for i in range(num_splits)
        ]


def get_normalized_cut(data, y):
    A = to_dense_adj(data.edge_index)[0]
    D = torch.diag(torch.sum(A, dim=1))
    normalized_cut = torch.trace(y.T @ A @ y) / torch.trace(y.T @ D @ y)
    return normalized_cut.item()


def get_mad_value(h, mask, target_idx=None):
    # h and mask are in cpu
    from sklearn.metrics import pairwise_distances

    h = h.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()

    dist_arr = pairwise_distances(h, h, metric='cosine')

    mask_dist = np.multiply(dist_arr, mask)

    divide_arr = (mask_dist != 0).sum(1) + 1e-8

    node_dist = mask_dist.sum(1) / divide_arr

    if target_idx is None:
        mad = np.mean(node_dist)
    else:
        node_dist = node_dist * target_idx
        mad = node_dist.sum() / ((node_dist != 0).sum() + 1e-8)

    return mad
