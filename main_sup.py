#!/usr/bin/env python

# coding: utf-8

# We only implement our method on transductive setting for node classification task.

import os
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
from model import GNN, MLP, SupModel
from loss import Bootstrap
from eval import LREvaluator
from utils import (
    seed_everything,
    get_split,
    to_MB,
    combine_dicts,
    get_split_from_mask,
    get_normalized_cut,
    get_mad_value,
    accuracy,
)
from args import get_node_params
from dataset import get_node_clf_dataset


def train(
    encoder_model,
    loss_model,
    data,
    optimizer,
    scheduler,
    aug_rounds=1,
    recon_lambda=1,
    clf_lambda=1,
    split=None,
):
    encoder_model.train()

    if data:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        y = data.y if data.y.dim() == 1 else data.y.squeeze()
        loss = 0.0

        for round in range(aug_rounds):
            x1, edge_index1, edge_attr1 = encoder_model.corrupt(
                x, edge_index, edge_attr
            )
            z1 = encoder_model.encode(x1)
            h1 = encoder_model.aggregate(z1, edge_index1, edge_attr1)
            p1 = encoder_model.predict(z1)
            con_loss = loss_model(p1, h1.detach())
            loss += con_loss

            if recon_lambda != 0:
                recon_loss = F.mse_loss(encoder_model.decode(h1), x)
                loss += recon_lambda * recon_loss

        loss = loss / aug_rounds
        z = encoder_model.encode(x)

        logits = encoder_model.classify(z).log_softmax(dim=-1)
        clf_loss = F.cross_entropy(logits[split["train"]], y[split["train"]])

        loss += clf_lambda * clf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        return loss.item()


def test_clf(encoder_model, data, split):
    encoder_model.eval()

    if data:
        start = time.time()
        z = encoder_model.encode(data.x)
        end = time.time()
        inf_time = end - start

        logits = encoder_model.classify(z).log_softmax(dim=-1)
        y = data.y if data.y.dim() == 1 else data.y.squeeze()
        val_acc = accuracy(logits[split["valid"]], y[split["valid"]])
        test_acc = accuracy(logits[split["test"]], y[split["test"]])

    result = {"val_acc": val_acc, "test_acc": test_acc, "inf_time": inf_time}

    return result


def main(base_params):
    device = torch.device(
        f"cuda:{params['device']}" if torch.cuda.is_available() else "cpu"
    )

    dataset = get_node_clf_dataset(params["dataset"])

    data = dataset[0]

    params['num_splits'] = 1
    assert params["num_splits"] == 1

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

    split = splits[0]

    data = data.to(device)

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

    classifier = MLP(
        input_dim=params["hidden_dim"],
        hidden_dim=params["hidden_dim"],
        output_dim=data.y.max().item() + 1,
        activation=nn.PReLU,
        num_layers=1,
    ).to(device)

    encoder_model = SupModel(
        encoder=mlp_encoder,
        aggregator=aggregator,
        predictor=predictor,
        augmenter=augmenter,
        decoder=decoder,
        classifier=classifier,
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
        start = time.time()

        loss = train(
            encoder_model,
            loss_model,
            data,
            optimizer,
            scheduler,
            split=split,
            aug_rounds=params["aug_rounds"],
            recon_lambda=params["recon_lambda"],
            clf_lambda=params["clf_lambda"],
        )

        end = time.time()

        train_time.append(end - start)

        if epoch % params["verbose"] == 0:
            clf_result = test_clf(encoder_model, data=data, split=split)

            inf_time.append(clf_result["inf_time"])

            result = {
                "epoch": epoch,
                "loss": np.round(loss, 6),
                "val_acc": np.round(clf_result["val_acc"], 4),
                "test_acc": np.round(clf_result["test_acc"], 4),
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
