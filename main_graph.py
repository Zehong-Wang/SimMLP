#!/usr/bin/env python

# coding: utf-8

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
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


from augment import Augment
from model import GNN, MLP, Model
from loss import Bootstrap
from eval import SVMEvaluator
from utils import seed_everything, to_MB
from args import get_graph_params
from dataset import get_graph_clf_dataset


def train(
    encoder_model,
    loss_model,
    dataloader,
    optimizer,
    scheduler,
    aug_rounds=1,
    recon_lambda=1,
):
    device = next(encoder_model.parameters()).device

    encoder_model.train()

    total_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)

        x = batch.x

        edge_index = batch.edge_index

        edge_attr = None

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

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if scheduler:
        scheduler.step()

    return total_loss / len(dataloader)


def test(encoder_model, dataloader, pooling="mean", clf="svm"):
    device = next(encoder_model.parameters()).device

    encoder_model.eval()

    x_list = []

    y_list = []

    with torch.no_grad():
        start = time.time()

        for batch in dataloader:
            batch = batch.to(device)

            x, y = batch.x, batch.y

            z = encoder_model.encode(x)

            if pooling == "mean":
                z = global_mean_pool(z, batch.batch)

            elif pooling == "max":
                z = global_max_pool(z, batch.batch)

            elif pooling == "sum":
                z = global_add_pool(z, batch.batch)

            else:
                raise NotImplementedError

            x_list.append(z.cpu().numpy())

            y_list.append(y.cpu().numpy())

        end = time.time()

    x = np.concatenate(x_list, axis=0)

    y = np.concatenate(y_list, axis=0)

    inf_time = end - start

    if clf == "svm":
        result = SVMEvaluator(n_splits=10).evaluate(x=x, y=y)

    else:
        raise NotImplementedError

    result["inf_time"] = inf_time

    result["pooling"] = pooling

    return result


def main(params):
    device = torch.device(
        f"cuda:{params['device']}" if torch.cuda.is_available() else "cpu"
    )

    dataset, (num_features, num_classes) = get_graph_clf_dataset(
        params["dataset"], deg4feat=params["deg4feat"]
    )

    train_loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    eval_loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)

    augmenter = Augment(
        feature_mask=params["feature_mask"], edge_mask=params["edge_mask"]
    )

    mlp_encoder = MLP(
        input_dim=num_features,
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
        output_dim=num_features,
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

    best_micro_f1 = -1

    best_report_result = {}

    train_time = []

    inf_time = []

    for epoch in range(1, params["epochs"] + 1):
        start = time.time()

        loss = train(
            encoder_model,
            loss_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            aug_rounds=params["aug_rounds"],
            recon_lambda=params["recon_lambda"],
        )

        end = time.time()

        train_time.append(end - start)

        # Test model performance

        if epoch % params["verbose"] == 0:
            clf_result = test(
                encoder_model, eval_loader, pooling=params["pooling"], clf="svm"
            )

            inf_time.append(clf_result["inf_time"])

            report_result = {
                "epoch": epoch,
                "loss": np.round(loss, 6),
                "micro_f1": np.round(clf_result["micro_f1"], 4),
                "macro_f1": np.round(clf_result["macro_f1"], 4),
                "micro_f1_std": np.round(clf_result["micro_f1_std"], 4),
                "macro_f1_std": np.round(clf_result["macro_f1_std"], 4),
                "pooling": clf_result["pooling"],
                "default": np.round(clf_result["micro_f1"], 4),
            }

            print(report_result)

            if report_result["micro_f1"] > best_micro_f1:
                best_report_result = report_result

                best_micro_f1 = report_result["micro_f1"]

    max_memory_allocated = torch.cuda.max_memory_allocated(device)

    best_report_result["train_time"] = np.mean(train_time)

    best_report_result["inf_time"] = np.mean(inf_time)

    best_report_result["maximum_memory"] = to_MB(max_memory_allocated)

    print("Best results:")

    print(best_report_result)


if __name__ == "__main__":
    params = get_graph_params()

    if params["use_params"]:
        param_path = osp.join("param", "graph", f"{params['dataset']}.json")

        with open(param_path, "r") as f:
            default_params = json.load(f)

        params.update(default_params)

    seed_everything(params["seed"])

    main(params)
