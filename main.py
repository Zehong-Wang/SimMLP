#!/usr/bin/env python

# coding: utf-8


import os.path as osp
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj, remove_self_loops

from augment import Augment, flip_edges
from model import GNN, GNN_with_params, MLP, Model
from loss import Bootstrap
from eval import LREvaluator

from utils import (
    seed_everything,
    get_split,
    to_ind_split,
    to_MB,
    combine_dicts,
    get_split_from_mask,
    get_normalized_cut,
    get_mad_value,
    accuracy,
    split_data,
)

from args import get_node_params

from dataset import get_node_clf_dataset


def train(
    encoder_model,
    loss_model,
    data,
    loader,
    optimizer,
    scheduler,
    aug_rounds=1,
    recon_lambda=1,
):
    # If loader is None, use data
    # If loader is not None, use loader

    encoder_model.train()

    if data and not loader:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        loss = 0.0
        for round in range(aug_rounds):
            x1, edge_index1, edge_attr1 = encoder_model.corrupt(
                x, edge_index, edge_attr
            )
            z1 = encoder_model.encode(x1)
            # h1 = encoder_model.aggregate(x1, edge_index1, edge_attr1)
            h1 = encoder_model.aggregate(z1, edge_index1, edge_attr1)
            p1 = encoder_model.predict(z1)

            con_loss = loss_model(p1, h1.detach())
            loss += con_loss

            if recon_lambda != 0:
                recon_loss = F.mse_loss(encoder_model.decode(h1), x)
                loss += recon_lambda * recon_loss

        loss = loss / aug_rounds

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        return loss.item()

    elif loader:
        device = next(encoder_model.parameters()).device

        total_loss = 0

        for batch in loader:
            batch = batch.to(device)

            x = batch.x

            edge_index = batch.edge_index

            edge_attr = batch.edge_attr

            bs = batch.batch_size

            loss = 0

            for round in range(aug_rounds):
                x1, edge_index1, edge_attr1 = encoder_model.corrupt(
                    x, edge_index, edge_attr
                )

                z1 = encoder_model.encode(x1)

                h1 = encoder_model.aggregate(z1, edge_index1, edge_attr1)

                p1 = encoder_model.predict(z1)

                con_loss = loss_model(p1[:bs], h1[:bs].detach())

                loss += con_loss

                if recon_lambda != 0:
                    recon_loss = F.mse_loss(encoder_model.decode(h1[:bs]), x[:bs])

                    loss += recon_lambda * recon_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if scheduler:
            scheduler.step()

        return total_loss / len(loader)


def test_clf_trans(
    encoder_model,
    data,
    loader,
    splits,
    lr_learning_rate=0.01,
    lr_weight_decay=0.0,
    lr_batch_size=0,
    lr_epochs=5000,
    get_ncut=False,
    get_smooth=False,
):
    encoder_model.eval()

    if data and not loader:
        start = time.time()
        z = encoder_model.encode(data.x)
        end = time.time()

        inf_time = end - start
        y = data.y if data.y.dim() == 1 else data.y.squeeze()

    elif loader:
        device = next(encoder_model.parameters()).device

        zs, ys = [], []
        inf_time = 0.0

        for batch in loader:
            batch = batch.to(device)
            bs = batch.batch_size

            start = time.time()
            z = encoder_model.encode(batch.x)
            end = time.time()

            inf_time += end - start
            y = batch.y if batch.y.dim() == 1 else batch.y.squeeze()

            zs.append(z[:bs])
            ys.append(y[:bs])

        z = torch.cat(zs)
        y = torch.cat(ys)

    results = []

    for split in splits:
        result = LREvaluator(
            learning_rate=lr_learning_rate,
            weight_decay=lr_weight_decay,
            batch_size=lr_batch_size,
            num_epochs=lr_epochs,
        ).evaluate(x=z, y=y, split=split, get_preds=get_ncut)

        if get_ncut:
            result, pred = result[0], result[1]
            result["normalized_cut"] = get_normalized_cut(data, pred)

        if get_smooth:
            z = encoder_model.encode(data.x)
            mask = to_dense_adj(remove_self_loops(data.edge_index)[0])[0].to(z.device)
            result["smoothness"] = get_mad_value(z, mask)

        result["inf_time"] = inf_time
        results.append(result)

    return combine_dicts(results)


def save_embedding(encoder_model, data, loader, save_path):
    z = encoder_model.encode(data.x).detach().cpu().numpy()
    np.save(save_path, z)


def test_clf_ind(
    encoder_model,
    data,
    loader,
    split,
    lr_learning_rate=0.01,
    lr_weight_decay=0.0,
    lr_batch_size=0,
    lr_epochs=5000,
):
    encoder_model.eval()

    (trans_data, ind_data) = data

    (trans_loader, ind_loader) = loader

    if trans_data and not trans_loader:
        zs, ys = [], []

        inf_time = 0

        trans_x = trans_data.x

        ind_x = ind_data.x

        start = time.time()

        trans_z = encoder_model.encode(trans_x)

        ind_z = encoder_model.encode(ind_x)

        end = time.time()

        inf_time += end - start

        trans_y = trans_data.y if trans_data.y.dim() == 1 else trans_data.y.squeeze()

        ind_y = ind_data.y if ind_data.y.dim() == 1 else ind_data.y.squeeze()

        z = torch.cat([trans_z, ind_z])

        y = torch.cat([trans_y, ind_y])

    elif trans_loader:
        device = next(encoder_model.parameters()).device

        zs, ys = [], []

        inf_time = 0.0

        for loader in [trans_loader, ind_loader]:
            for batch in loader:
                batch = batch.to(device)

                bs = batch.batch_size

                start = time.time()

                z = encoder_model.encode(batch.x)

                end = time.time()

                inf_time += end - start

                y = batch.y if batch.y.dim() == 1 else batch.y.squeeze()

                zs.append(z[:bs])

                ys.append(y[:bs])

        z = torch.cat(zs)

        y = torch.cat(ys)

    result = LREvaluator(
        learning_rate=lr_learning_rate,
        weight_decay=lr_weight_decay,
        batch_size=lr_batch_size,
        num_epochs=lr_epochs,
    ).evaluate_ind(x=z, y=y, split=split)

    result["inf_time"] = inf_time

    return result


def test_cluster(encoder_model, data, loader):
    from sklearn.cluster import KMeans

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    encoder_model.eval()

    if data and not loader:
        z = encoder_model.encode(data.x)

        y = data.y

    elif loader:
        device = next(encoder_model.parameters()).device

        zs, ys = [], []

        for batch in loader:
            batch = batch.to(device)

            bs = batch.batch_size

            z = encoder_model.encode(batch.x)

            y = batch.y if batch.y.dim() == 1 else batch.y.squeeze()

            zs.append(z[:bs])

            ys.append(y[:bs])

        z = torch.cat(zs)

        y = torch.cat(ys)

    z = z.detach().cpu().numpy()

    y = y.cpu().numpy()

    kmeans = KMeans(n_clusters=y.max() + 1)

    pred = kmeans.fit_predict(z)

    nmi = normalized_mutual_info_score(y, pred)

    ari = adjusted_rand_score(y, pred)

    result = {"nmi": nmi, "ari": ari}

    return result


def main(params):
    device = torch.device(
        f"cuda:{params['device']}" if torch.cuda.is_available() else "cpu"
    )

    dataset = get_node_clf_dataset(params["dataset"])
    data = dataset[0]

    if params["feature_noise"] != 0:
        data.x = (1 - params["feature_noise"]) * data.x + params[
            "feature_noise"
        ] * torch.randn_like(data.x)
        print(
            "Add Gaussian noise on nodes with level {}!".format(params["feature_noise"])
        )

    if params["edge_noise"] != 0:
        data = flip_edges(data, p=params["edge_noise"])
        print("Randomly flip {} edges!".format(params["edge_noise"]))

    if params["setting"] == "trans":
        if params["dataset"] in ["ogb-arxiv", "ogb-products"]:
            split = dataset.get_idx_split()
            splits = [split]

        elif params["dataset"] in [
            "reddit",
            "cornell",
            "texas",
            "wisconsin",
            "roman-empire",
            "amazon-ratings",
        ]:
            masks = {
                "train": data.train_mask,
                "valid": data.val_mask,
                "test": data.test_mask,
            }
            splits = get_split_from_mask(masks)

        elif params["dataset"] in ["elliptic"]:
            masks = {"train": data.train_mask, "test": data.train_mask}

            splits = get_split_from_mask(masks)
            data.x = data.x[:, :94]
        else:
            splits = [
                get_split(
                    num_samples=data.x.size()[0],
                    train_ratio=params["train_ratio"],
                    test_ratio=0.1,
                )
                for _ in range(params["num_splits"])
            ]

        if params["batch_size"] != 0:
            if params["dataset"] in ["ogb-products", "reddit"]:
                num_neighbors = [10] * params["proj_layers"]
            else:
                num_neighbors = [-1] * params["proj_layers"]

            train_loader = NeighborLoader(
                data,
                input_nodes=None,
                num_neighbors=num_neighbors,
                batch_size=params["batch_size"],
                shuffle=True,
            )

            graph_loader = NeighborLoader(
                data,
                input_nodes=None,
                num_neighbors=num_neighbors,
                batch_size=2048,
                shuffle=False,
            )

        else:
            data = data.to(device)

            train_loader = None

            graph_loader = None

    elif params["setting"] == "ind":
        if params["dataset"] in ["ogb-arxiv", "ogb-products"]:
            split = dataset.get_idx_split()

            split = to_ind_split(split, ind_ratio=params["ind_rate"])

        else:
            split = get_split(
                num_samples=data.x.size()[0],
                train_ratio=params["train_ratio"],
                test_ratio=0.1,
                ind_ratio=params["ind_rate"],
            )

        trans_data, ind_data = split_data(data, split)

        if params["batch_size"] != 0:
            if params["dataset"] in ["ogb-arxiv", "ogb-products"]:
                num_neighbors = [10] * params["proj_layers"]

            else:
                num_neighbors = [-1] * params["proj_layers"]

            train_loader = NeighborLoader(
                trans_data,
                input_nodes=None,
                num_neighbors=num_neighbors,
                batch_size=params["batch_size"],
                shuffle=True,
            )

            graph_loader = NeighborLoader(
                trans_data,
                input_nodes=None,
                num_neighbors=num_neighbors,
                batch_size=2048,
                shuffle=False,
            )

            ind_loader = NeighborLoader(
                ind_data,
                input_nodes=None,
                num_neighbors=num_neighbors,
                batch_size=2048,
                shuffle=False,
            )

        else:
            trans_data = trans_data.to(device)
            ind_data = ind_data.to(device)

            train_loader = None
            graph_loader = None
            ind_loader = None

    augmenter = Augment(
        feature_mask=params["feature_mask"], edge_mask=params["edge_mask"]
    )

    mlp_encoder = MLP(
        input_dim=data.num_features,
        hidden_dim=params["hidden_dim"],
        output_dim=params["hidden_dim"],
        activation=nn.PReLU,
        num_layers=params["enc_layers"],
        residual=params["res_enc"],
        norm=params["enc_norm"],
        dropout=params["enc_drop"],
    ).to(device)

    # Non-parametric
    aggregator = GNN(
        input_dim=params["hidden_dim"],
        hidden_dim=params["hidden_dim"],
        output_dim=params["hidden_dim"],
        activation=nn.PReLU,
        num_layers=params["proj_layers"],
        norm=params["proj_norm"],
        dropout=params["proj_drop"],
        aggr_norm=params["aggr_norm"],
    ).to(device)

    # Parametric
    # aggregator = GNN_with_params(
    #     input_dim=data.x.shape[1],
    #     hidden_dim=params["hidden_dim"],
    #     output_dim=params["hidden_dim"],
    #     activation=nn.PReLU,
    #     num_layers=params["proj_layers"],
    #     norm=params["proj_norm"],
    #     dropout=params["proj_drop"],
    #     aggr_norm=params["aggr_norm"],
    # ).to(device)

    predictor = MLP(
        input_dim=params["hidden_dim"],
        hidden_dim=params["pred_dim"],
        output_dim=params["hidden_dim"],
        activation=nn.PReLU,
        num_layers=params["pred_layers"],
        residual=False,
        norm=params["pred_norm"],
        dropout=params["pred_drop"],
    ).to(device)

    decoder = MLP(
        input_dim=params["hidden_dim"],
        hidden_dim=params["hidden_dim"],
        output_dim=data.num_features,
        activation=nn.PReLU,
        num_layers=2,
    ).to(device)

    encoder_model = Model(
        encoder=mlp_encoder,
        aggregator=aggregator,
        predictor=predictor,
        augmenter=augmenter,
        decoder=decoder,
    ).to(device)

    loss_model = Bootstrap(eta=params["eta"], aux_pos_ratio=params["aux_pos_ratio"])

    optimizer = AdamW(
        params=encoder_model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    if params["use_scheduler"]:
        scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / params["epochs"])) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    best_acc = -1
    best_result = {}
    train_time = []
    inf_time = []

    for epoch in range(1, params["epochs"] + 1):
        if params["setting"] == "trans":
            train_data = data

        elif params["setting"] == "ind":
            train_data = trans_data

        start = time.time()

        # train_loader is None if "batch_size" is set to 0.

        loss = train(
            encoder_model=encoder_model,
            loss_model=loss_model,
            data=train_data,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            aug_rounds=params["aug_rounds"],
            recon_lambda=params["recon_lambda"],
        )

        end = time.time()

        train_time.append(end - start)

        # Test model performance

        if epoch % params["verbose"] == 0:
            # save_embedding(encoder_model, data, None, "emb/{}-{}-{}.npy".format(params["dataset"], epoch, params["seed"]))

            if params["setting"] == "trans":
                # if graph_loader is not none, use subgraph loader, else use data

                clf_result = test_clf_trans(
                    encoder_model,
                    data,
                    graph_loader,
                    splits,
                    lr_learning_rate=params["lr_learning_rate"],
                    lr_weight_decay=params["lr_weight_decay"],
                    lr_batch_size=params["lr_batch_size"],
                    lr_epochs=params["lr_epochs"],
                    get_ncut=params["get_ncut"],
                    get_smooth=params["get_smooth"],
                )

                inf_time.append(clf_result["inf_time"])

                result = {
                    "epoch": epoch,
                    "loss": np.round(loss, 6),
                    "val_acc": np.round(clf_result["val_acc"], 4),
                    "test_acc": np.round(clf_result["test_acc"], 4),
                    "default": np.round(clf_result["test_acc"], 4),
                }

                if params["cluster"]:
                    cluster_result = test_cluster(encoder_model, data, graph_loader)
                    result["nmi"] = np.round(cluster_result["nmi"], 4)
                    result["ari"] = np.round(cluster_result["ari"], 4)

                if params["get_ncut"]:
                    result["normalized_cut"] = np.round(clf_result["normalized_cut"], 4)

                if params["get_smooth"]:
                    result["smoothness"] = np.round(clf_result["smoothness"], 4)

                print(result)

                if result["test_acc"] > best_acc:
                    best_result = result

                    best_acc = result["test_acc"]

            if params["setting"] == "ind":
                # if trans_loader and

                clf_result = test_clf_ind(
                    encoder_model,
                    data=(trans_data, ind_data),
                    loader=(graph_loader, ind_loader),
                    split=split,
                    lr_learning_rate=params["lr_learning_rate"],
                    lr_weight_decay=params["lr_weight_decay"],
                    lr_batch_size=params["lr_batch_size"],
                    lr_epochs=params["lr_epochs"],
                )

                inf_time.append(clf_result["inf_time"])

                result = {
                    "epoch": epoch,
                    "loss": np.round(loss, 6),
                    "val_acc": np.round(clf_result["val_acc"], 4),
                    "test_acc": np.round(clf_result["test_acc"], 4),
                    "ind_test_acc": np.round(clf_result["ind_test_acc"], 4),
                    "default": np.round(clf_result["test_acc"], 4),
                }

                print(result)

                if result["test_acc"] > best_acc:
                    best_result = result

                    best_acc = result["test_acc"]

    max_memory_allocated = torch.cuda.max_memory_allocated(device)

    best_result["train_time"] = np.mean(train_time)

    best_result["inf_time"] = np.mean(inf_time)

    best_result["maximum_memory"] = to_MB(max_memory_allocated)

    print("Best results:")

    print(best_result)


if __name__ == "__main__":
    params = get_node_params()
    if params["use_params"]:
        param_path = osp.join("param", "node", f"{params['dataset']}.json")
        with open(param_path, "r") as f:
            default_params = json.load(f)
        params.update(default_params)

    seed_everything(params["seed"])

    main(params)
