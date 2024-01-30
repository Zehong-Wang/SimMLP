from typing import Any

from tqdm import tqdm

import numpy as np


import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW


from utils import accuracy


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()

        self.fc = nn.Linear(num_features, num_classes)

        nn.init.xavier_uniform_(self.fc.weight.data)

        nn.init.constant_(self.fc.bias.data, 0.0)

    def forward(self, x):
        z = self.fc(x)

        return z


class MyDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()

        self.data = data

        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LREvaluator:
    def __init__(
        self,
        num_epochs: int = 5000,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        test_interval: int = 20,
        batch_size: int = 0,
    ):
        self.num_epochs = num_epochs

        self.learning_rate = learning_rate

        self.weight_decay = weight_decay

        self.test_interval = test_interval

        self.batch_size = batch_size

    def evaluate(
        self,
        x: torch.FloatTensor,
        y: torch.LongTensor,
        split: dict,
        get_preds: bool = False,
    ):
        device = x.device

        x = x.detach()

        y = y

        input_dim = x.size()[1]

        num_classes = y.max().item() + 1

        if self.batch_size == 0:
            x = x.to(device)

            y = y.to(device)

        else:
            train_dataset = MyDataset(x[split["train"]], y[split["train"]])

            valid_dataset = MyDataset(x[split["valid"]], y[split["valid"]])

            test_dataset = MyDataset(x[split["test"]], y[split["test"]])

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

            valid_loader = DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=False
            )

            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )

        classifier = LogisticRegression(input_dim, num_classes).to(device)

        optimizer = AdamW(
            classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        output_fn = nn.LogSoftmax(dim=-1)

        criterion = nn.NLLLoss()

        best_val_acc = 0

        best_test_acc = 0

        best_epoch = 0

        with tqdm(
            total=self.num_epochs,
            desc="(LR)",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
        ) as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()

                if self.batch_size == 0:
                    output = classifier(x[split["train"]])

                    loss = criterion(output_fn(output), y[split["train"]])

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                else:
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        output = classifier(inputs)

                        loss = criterion(output_fn(output), labels)

                        optimizer.zero_grad()

                        loss.backward()

                        optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()

                    if self.batch_size == 0:
                        y_val = y[split["valid"]]

                        y_test = y[split["test"]]

                        pred_val = classifier(x[split["valid"]])

                        pred_test = classifier(x[split["test"]])

                    else:
                        pred_vals, y_vals = [], []

                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(device), labels.to(device)

                            pred_val = classifier(inputs)

                            pred_vals.append(pred_val.detach())

                            y_vals.append(labels)

                        pred_tests, y_tests = [], []

                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)

                            pred_test = classifier(inputs)

                            pred_tests.append(pred_test.detach())

                            y_tests.append(labels)

                        pred_val = torch.cat(pred_vals)

                        y_val = torch.cat(y_vals)

                        pred_test = torch.cat(pred_tests)

                        y_test = torch.cat(y_tests)

                    val_acc = accuracy(pred_val, y_val)

                    test_acc = accuracy(pred_test, y_test)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                        best_test_acc = test_acc

                        best_epoch = epoch

                        if get_preds:
                            pred_all = classifier(x)

                    pbar.set_postfix(
                        {"best val Acc": best_val_acc, "test Acc": best_test_acc}
                    )

                    pbar.update(self.test_interval)

        result = {"val_acc": best_val_acc, "test_acc": best_test_acc}

        if get_preds:
            return (result, pred_all)

        else:
            return result

    def evaluate_ind(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device

        x = x.detach().to(device)

        input_dim = x.size()[1]

        y = y.to(device)

        num_classes = y.max().item() + 1

        classifier = LogisticRegression(input_dim, num_classes).to(device)

        optimizer = AdamW(
            classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        output_fn = nn.LogSoftmax(dim=-1)

        criterion = nn.NLLLoss()

        best_val_acc = 0

        best_test_acc = 0

        best_ind_test_acc = 0

        best_epoch = 0

        with tqdm(
            total=self.num_epochs,
            desc="(LR)",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
        ) as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()

                output = classifier(x[split["train"]])

                loss = criterion(output_fn(output), y[split["train"]])

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()

                    y_val = y[split["valid"]]

                    y_test = y[split["test"]]

                    y_ind_test = y[split["ind_test"]]

                    pred_val = classifier(x[split["valid"]])

                    pred_test = classifier(x[split["test"]])

                    pred_ind_test = classifier(x[split["ind_test"]])

                    val_acc = accuracy(pred_val, y_val)

                    test_acc = accuracy(pred_test, y_test)

                    ind_test_acc = accuracy(pred_ind_test, y_ind_test)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                        best_test_acc = test_acc

                        best_ind_test_acc = ind_test_acc

                        best_epoch = epoch

                    pbar.set_postfix(
                        {"Trans Acc": best_test_acc, "Ind Acc": best_ind_test_acc}
                    )

                    pbar.update(self.test_interval)

        return {
            "val_acc": best_val_acc,
            "test_acc": best_test_acc,
            "ind_test_acc": best_ind_test_acc,
        }


class SVMEvaluator:
    def __init__(self, n_splits=10):
        self.n_splits = n_splits

    def evaluate(self, x, y):
        from sklearn.metrics import f1_score

        from sklearn.model_selection import StratifiedKFold, GridSearchCV

        from sklearn.svm import SVC

        micro_f1_list = []

        macro_f1_list = []

        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=0)

        for train_idx, test_idx in kf.split(x, y):
            x_train, x_test = x[train_idx], x[test_idx]

            y_train, y_test = y[train_idx], y[test_idx]

            params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}

            svc = SVC(random_state=42)

            # svc = LinearSVC(random_state=42)

            clf = GridSearchCV(svc, params, n_jobs=16)

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            micro_f1 = f1_score(y_test, y_pred, average="micro")

            macro_f1 = f1_score(y_test, y_pred, average="macro")

            micro_f1_list.append(micro_f1)

            macro_f1_list.append(macro_f1)

        micro_f1_mean = np.mean(micro_f1_list)

        macro_f1_mean = np.mean(macro_f1_list)

        micro_f1_std = np.std(micro_f1_list)

        macro_f1_std = np.std(macro_f1_list)

        return {
            "micro_f1": micro_f1_mean,
            "macro_f1": macro_f1_mean,
            "micro_f1_std": micro_f1_std,
            "macro_f1_std": macro_f1_std,
        }
