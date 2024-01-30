import torch
from torch_geometric.utils import is_undirected, to_dense_adj

def dropout_edge(edge_index, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def mask_feature(x, p: float = 0.5, mode: str = 'all',
                 fill_value: float = 0.,
                 training: bool = True):

    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 '
                         f'(got {p}')
    if not training or p == 0.0:
        return x, torch.ones_like(x, dtype=torch.bool)
    assert mode in ['row', 'col', 'all']

    if mode == 'row':
        mask = torch.rand(x.size(0), device=x.device) >= p
        mask = mask.view(-1, 1)
    elif mode == 'col':
        mask = torch.rand(x.size(1), device=x.device) >= p
        mask = mask.view(1, -1)
    else:
        mask = torch.rand_like(x) >= p

    x = x.masked_fill(~mask, fill_value)
    return x, mask


def flip_edges(data, p=0.2):
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]

    if is_undirected(data.edge_index):
        num_flip_edges = int(num_edges * p / 2)
    else:
        num_flip_edges = int(num_edges * p)

    adj = to_dense_adj(data.edge_index)[0]

    flipped_edges = torch.randint(0, num_nodes, size=(num_flip_edges, 2))

    for n1, n2 in flipped_edges:
        adj[n1, n2] = 1 - adj[n1, n2]
        adj[n2, n1] = 1 - adj[n2, n1]

    edge_index = adj.to_sparse().coalesce().indices()
    data.edge_index = edge_index
    data.edge_attr = None
    return data


class Augment():
    def __init__(self, edge_mask=0.3, feature_mask=0.3):
        self.edge_mask = edge_mask
        self.feature_mask = feature_mask

    def corrupt(self, x, edge_index, edge_attr = None):
        x, _ = mask_feature(x, p=self.feature_mask)
        edge_index, edge_mask = dropout_edge(edge_index, p=self.edge_mask)
        if edge_attr is not None:
            edge_attr = edge_attr[edge_mask]
          
        return x, edge_index, edge_attr