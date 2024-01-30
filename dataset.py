import os.path as osp

from collections import Counter


import torch

import torch.nn.functional as F


from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
    WikiCS,
    TUDataset,
    WikipediaNetwork,
    Actor,
    PPI,
    Reddit2,
    Flickr,
    WebKB,
    HeterophilousGraphDataset,
    EllipticBitcoinDataset,
    DGraphFin,
    Yelp,
)


from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric import transforms as T

from torch_geometric.utils import degree, remove_self_loops, add_self_loops


# Define your own dataset path. The default path is the current directory.

base_path = "./dataset"

# base_path = osp.join('E:\\', 'Python Workspace', 'GraphSSL', 'Dataset')


def get_node_clf_dataset(dataset):
    transform = T.Compose([T.AddSelfLoops(), T.ToUndirected()])

    if dataset == "cora":
        dataset = Planetoid(
            osp.join(base_path, "planetoid"), name="cora", transform=transform
        )

    elif dataset == "citeseer":
        dataset = Planetoid(
            osp.join(base_path, "planetoid"), name="citeseer", transform=transform
        )

    elif dataset == "pubmed":
        dataset = Planetoid(
            osp.join(base_path, "planetoid"), name="pubmed", transform=transform
        )

    elif dataset == "amazon-cs":
        dataset = Amazon(
            osp.join(base_path, "amazon"), name="computers", transform=transform
        )

    elif dataset == "amazon-photo":
        dataset = Amazon(
            osp.join(base_path, "amazon"), name="photo", transform=transform
        )

    elif dataset == "co-cs":
        dataset = Coauthor(
            osp.join(base_path, "coauthor"), name="CS", transform=transform
        )

    elif dataset == "co-phys":
        dataset = Coauthor(
            osp.join(base_path, "coauthor"), name="Physics", transform=transform
        )

    elif dataset == "wiki-cs":
        dataset = WikiCS(
            osp.join(base_path, "wiki"), is_undirected=True, transform=transform
        )

    elif dataset == "flickr":
        dataset = Flickr(osp.join(base_path, "flickr"), transform=transform)

    elif dataset == "ogb-arxiv":
        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv", root=osp.join(base_path, "arxiv"), transform=transform
        )

    elif dataset == "ogb-products":
        dataset = PygNodePropPredDataset(
            name="ogbn-products",
            root=osp.join(base_path, "products"),
            transform=transform,
        )

    elif dataset == "paper100m":
        dataset = PygNodePropPredDataset(
            name="ogbn-papers100M",
            root=osp.join(base_path, "paper100m"),
            transform=transform,
        )

    elif dataset == "reddit":
        dataset = Reddit2(osp.join(base_path, "reddit"), transform=transform)

    elif dataset == "chameleon":
        dataset = WikipediaNetwork(
            osp.join(base_path, "wikipedia"),
            name="chameleon",
            transform=transform,
            geom_gcn_preprocess=True,
        )

    elif dataset == "squirrel":
        dataset = WikipediaNetwork(
            osp.join(base_path, "wikipedia"),
            name="squirrel",
            transform=transform,
            geom_gcn_preprocess=True,
        )

    elif dataset == "actor":
        dataset = Actor(osp.join(base_path, "actor"), transform=transform)

    elif dataset == "cornell":
        dataset = WebKB(osp.join(base_path, "webkb"), name="Cornell")

    elif dataset == "texas":
        dataset = WebKB(osp.join(base_path, "webkb"), name="Texas", transform=transform)

    elif dataset == "wisconsin":
        dataset = WebKB(
            osp.join(base_path, "webkb"), name="Wisconsin", transform=transform
        )

    elif dataset == "roman-empire":
        dataset = HeterophilousGraphDataset(
            osp.join(base_path, "heterophilous"),
            name="roman_empire",
            transform=transform,
        )

    elif dataset == "amazon-ratings":
        dataset = HeterophilousGraphDataset(
            osp.join(base_path, "heterophilous"),
            name="amazon_ratings",
            transform=transform,
        )

    elif dataset == "ppi":
        train_dataset = PPI(
            osp.join(base_path, "ppi"), split="train", transform=transform
        )

        val_dataset = PPI(osp.join(base_path, "ppi"), split="val", transform=transform)

        test_dataset = PPI(
            osp.join(base_path, "ppi"), split="test", transform=transform
        )

        return (train_dataset, val_dataset, test_dataset)

    elif dataset == "elliptic":
        dataset = EllipticBitcoinDataset(
            osp.join(base_path, "aml"), transform=transform
        )

    elif dataset == "dgraph":
        transform = T.Compose(
            [
                T.AddSelfLoops(),
                T.ToUndirected(),
                T.NormalizeFeatures(),
            ]
        )

        dataset = DGraphFin(osp.join(base_path, "dgraph"), transform=transform)

    else:
        raise NotImplementedError("The dataset is not supported!")

    return dataset


def get_graph_clf_dataset(dataset, deg4feat=None):
    if dataset == "collab":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="COLLAB")

    elif dataset == "mutag":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="MUTAG")

    elif dataset == "proteins":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="PROTEINS")

    elif dataset == "dd":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="DD")

    elif dataset == "ptc-mr":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="PTC_MR")

    elif dataset == "enzymes":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="ENZYMES")

    elif dataset == "nci1":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="NCI1")

    elif dataset == "imdb-b":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="IMDB-BINARY")

    elif dataset == "imdb-m":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="IMDB-MULTI")

    elif dataset == "rdt-b":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="REDDIT-BINARY")

    elif dataset == "rdt-m5k":
        dataset = TUDataset(root=osp.join(base_path, "tu"), name="REDDIT-MULTI-5K")

    else:
        raise NotImplementedError("The dataset is not supported!")

    print(dataset[0])

    MAX_DEG = 400

    num_classes = dataset.num_classes

    dataset = list(dataset)

    graph = dataset[0]

    if not deg4feat and graph.x is not None:
        print('Use "attr" as node features.')

    else:
        print("Use degree as node features.")

        feature_dim = 0

        degrees = []

        for g in dataset:
            feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())

            degrees.extend(degree(g.edge_index[0]).tolist())

        oversize = 0

        for d, n in Counter(degrees).items():
            if d > MAX_DEG:
                oversize += n

        feature_dim = min(feature_dim, MAX_DEG)

        feature_dim += 1

        for i, g in enumerate(dataset):
            degrees = degree(g.edge_index[0])

            degrees[degrees > MAX_DEG] = MAX_DEG

            degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])

            feat = F.one_hot(
                degrees.to(torch.long), num_classes=int(feature_dim)
            ).float()

            g.x = feat

            dataset[i] = g

    feature_dim = int(graph.num_features)

    labels = torch.tensor([g.y for g in dataset])

    num_classes = torch.max(labels).item() + 1

    for i, g in enumerate(dataset):
        dataset[i].edge_index = remove_self_loops(dataset[i].edge_index)[0]

        dataset[i].edge_index = add_self_loops(dataset[i].edge_index)[0]

    print(
        f"******* # Graphs: {len(dataset)}, # Features: {feature_dim}, # Classes: {num_classes} *******"
    )

    return dataset, (feature_dim, num_classes)
