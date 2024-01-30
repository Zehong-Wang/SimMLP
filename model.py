import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum


def aggr_norm(
    edge_index,
    edge_weight=None,
    num_nodes=None,
    improved=False,
    add_self_loops=True,
    flow="source_to_target",
    dtype=None,
    norm_type="gcn",
):
    fill_value = 2.0 if improved else 1.0

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0, dtype=dtype)

        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)

        deg = sparsesum(adj_t, dim=1)

        deg_inv_sqrt = deg.pow_(-0.5)

        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)

        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))

        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes
            )

            assert tmp_edge_weight is not None

            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]

        idx = col if flow == "source_to_target" else row

        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)

        if norm_type == "gcn":
            deg_inv_sqrt = deg.pow_(-0.5)

            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

            return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        elif norm_type == "row":
            deg_inv = deg.pow_(-1)

            deg_inv.masked_fill_(deg_inv == float("inf"), 0)

            return edge_index, deg_inv[row] * edge_weight

        elif norm_type == "col":
            deg_inv = deg.pow_(-1)

            deg_inv.masked_fill_(deg_inv == float("inf"), 0)

            return edge_index, edge_weight * deg_inv[row]


def get_normalize(hidden_dim, norm="batch"):
    if norm == "batch":
        return nn.BatchNorm1d(hidden_dim)

    elif norm == "layer":
        return nn.LayerNorm(hidden_dim)

    elif norm == "none":
        return nn.Identity()

    else:
        raise NotImplementedError("Do not support this normalization method!")


class GConv_wo_param(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        aggr_norm="gcn",
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")

        super().__init__(**kwargs)

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.improved = improved

        self.add_self_loops = add_self_loops

        self.normalize = normalize

        self.aggr_norm = aggr_norm

    def forward(self, x: Tensor, edge_index, edge_weight) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = aggr_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    self.improved,
                    self.add_self_loops,
                    self.flow,
                    norm_type=self.aggr_norm,
                )

            elif isinstance(edge_index, SparseTensor):
                edge_index = aggr_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    self.improved,
                    self.add_self_loops,
                    self.flow,
                    norm_type=self.aggr_norm,
                )

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        return out

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation,
        num_layers,
        residual=False,
        norm="none",
        dropout=0.0,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        self.activation = activation()

        self.dropout = nn.Dropout(dropout)

        self.residual = residual

        self.norms = nn.ModuleList()

        self.layers = nn.ModuleList()

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))

            self.norms.append(get_normalize(out_dim, norm=norm))

    def forward(self, x, jk=False):
        if self.num_layers == 1:
            return self.layers[0](x)

        if jk:
            zs = []

            # zs.append(x)

        z = x

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h = z

            z = layer(z)

            z = norm(z)

            if i != self.num_layers - 1:
                z = self.activation(z)

                z = self.dropout(z)

            if self.residual and i != 0:
                z = z + h

            if jk:
                zs.append(z)

        z = F.normalize(z, dim=-1, p=2)

        if jk:
            return torch.concat(zs, dim=-1)

        else:
            return z


class GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation,
        num_layers,
        norm="batch",
        dropout=0.5,
        aggr_norm="gcn",
    ):
        super(GNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aggr_norm = aggr_norm
        self.activation = activation()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()  # it is an essential component
        self.dropout = nn.Dropout(dropout)

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(GConv_wo_param(in_dim, out_dim, aggr_norm=aggr_norm))
            self.norms.append(get_normalize(out_dim, norm))

    def forward(self, x, edge_index, edge_attr=None):
        if self.aggr_norm == "id":
            return x

        z = x
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            z = conv(z, edge_index, edge_attr)
            z = self.activation(z)
            z = norm(z)
        return z


class GNN_with_params(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation,
        num_layers,
        norm="batch",
        dropout=0.5,
        aggr_norm="gcn",
    ):
        super(GNN_with_params, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aggr_norm = aggr_norm
        self.activation = activation()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()  # it is an essential component
        self.dropout = nn.Dropout(dropout)

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(GCNConv(in_dim, out_dim))
            self.norms.append(get_normalize(out_dim, norm))

    def forward(self, x, edge_index, edge_attr=None):
        if self.aggr_norm == "id":
            return x

        z = x
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            z = conv(z, edge_index, edge_attr)
            z = self.activation(z)
            z = norm(z)
        return z


class Model(nn.Module):
    def __init__(self, encoder, aggregator, predictor, augmenter, decoder=None):
        super().__init__()

        self.encoder = encoder
        self.aggregator = aggregator
        self.predictor = predictor
        self.augmenter = augmenter
        self.decoder = decoder

    def corrupt(self, x, edge_index, edge_attr):
        x1, edge_index1, edge_attr1 = self.augmenter.corrupt(x, edge_index, edge_attr)

        return x1, edge_index1, edge_attr1

    def encode(self, x):
        z = self.encoder(x)

        return z

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def aggregate(self, z, edge_index, edge_attr) -> torch.Tensor:
        return self.aggregator(z, edge_index, edge_attr)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class SupModel(nn.Module):
    def __init__(
        self, encoder, aggregator, predictor, augmenter, decoder=None, classifier=None
    ):
        super().__init__()

        self.encoder = encoder

        self.aggregator = aggregator

        self.predictor = predictor

        self.augmenter = augmenter

        self.decoder = decoder

        self.classifier = classifier

    def corrupt(self, x, edge_index, edge_attr):
        x1, edge_index1, edge_attr1 = self.augmenter.corrupt(x, edge_index, edge_attr)

        return x1, edge_index1, edge_attr1

    def encode(self, x):
        z = self.encoder(x)

        return z

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def aggregate(self, z, edge_index, edge_attr) -> torch.Tensor:
        return self.aggregator(z, edge_index, edge_attr)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)
