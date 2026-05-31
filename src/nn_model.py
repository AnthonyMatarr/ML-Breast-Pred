## nn_model.py
from src.config import SEED

import pandas as pd
import numpy as np
import math

import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin


class TabularDataset(Dataset):
    def __init__(self, X_df, y, dtype=torch.float32):
        self.X = torch.tensor(X_df.to_numpy(), dtype=dtype)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dtype = dtype

    def __len__(self):
        return self.X.shape[0]

    def num_feats(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_activation(name, neg_slope=0.01):
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=neg_slope)
    else:
        raise ValueError(f"Unknown activation '{name}'")


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size_list,
        in_dim,
        dropouts,
        activation_list,
        weight_init_scheme="xavier_uniform",
        bias_init=0.0,
    ):
        super().__init__()
        if len(dropouts) != len(hidden_size_list):
            raise ValueError(
                f"dropouts length ({len(dropouts)}) must match "
                f"hidden_size_list length ({len(hidden_size_list)})"
            )
        layers = []
        prev = in_dim
        for h, p, act in zip(hidden_size_list, dropouts, activation_list):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if p and p > 0:
                layers.append(nn.Dropout(p))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights(weight_init_scheme, bias_init)

    def _init_weights(self, init_scheme, bias_init):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_scheme == "xavier_uniform":
                    init.xavier_uniform_(m.weight)
                elif init_scheme == "kaiming_uniform":
                    init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, bias_init)

    def forward(self, x):
        return self.net(x).squeeze(1)


class TorchNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_size_list,
        dropouts,
        activation_name,
        lr=1e-3,
        weight_decay=0,
        optimizer_str="adamw",
        epochs=500,  # ceiling when early_stopping=True
        batch_size=64,
        weight_init_scheme="xavier_uniform",
        bias_init=0.0,
        device="cpu",
        verbose=0,
        seed=SEED,
        neg_slope=0.01,
        # ── Early stopping ────────────────────────────────────────────────────
        early_stopping=True,
        es_patience=20,
        es_min_delta=0.0,
        val_split=0.15,
        monitor="auprc",
        pos_weight=1,
    ):
        self.hidden_size_list = hidden_size_list
        self.dropouts = dropouts
        self.activation_name = activation_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_str = optimizer_str
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_init_scheme = weight_init_scheme
        self.bias_init = bias_init
        self.device = device
        self.verbose = verbose
        self.seed = seed
        self.neg_slope = neg_slope
        self.pos_weight = pos_weight
        self.early_stopping = early_stopping
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.val_split = val_split
        if monitor not in ["auroc", "auprc", "loss"]:
            raise ValueError('monitor must be one of ["auroc", "auprc", "loss"]')
        self.monitor = monitor
        self.model_ = None

    def _make_loader(self, X, y, shuffle):
        ds = TabularDataset(X, y)
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=g if shuffle else None,
        )

    def _eval_val(self, model, val_loader, criterion):
        """Returns (mean_val_loss, metric | None)."""
        model.eval()
        val_losses, all_logits, all_targets = [], [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                val_losses.append(criterion(logits, target).item())
                all_logits.append(logits.detach().cpu())
                all_targets.append(target.detach().cpu())

        mean_loss = float(np.mean(val_losses))
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        proba = torch.sigmoid(all_logits).numpy()
        targets_np = all_targets.numpy()

        if self.monitor == "auroc":
            metric = (
                roc_auc_score(targets_np, proba)
                if len(np.unique(targets_np)) > 1
                else -mean_loss
            )
        elif self.monitor == "auprc":
            metric = (
                average_precision_score(targets_np, proba)
                if len(np.unique(targets_np)) > 1
                else -mean_loss
            )
        else:  # loss
            metric = -mean_loss

        return mean_loss, metric

    def fit(self, X, y, eval_set=None):
        torch.set_num_threads(1)  # prevents oversubscription w/ joblib workers
        torch.manual_seed(self.seed)
        self.feature_names_in_ = np.array(X.columns)
        in_dim = X.shape[1]

        acts = [
            get_activation(self.activation_name, self.neg_slope)
            for _ in self.hidden_size_list
        ]
        model = MLP(
            self.hidden_size_list,
            in_dim,
            self.dropouts,
            acts,
            self.weight_init_scheme,
            self.bias_init,
        ).to(self.device)

        ## Class-prior bias init
        pos_rate = np.clip(float(np.mean(y > 0)), 1e-7, 1 - 1e-7)
        log_odds = math.log(pos_rate / (1.0 - pos_rate))
        output_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.constant_(output_linear.bias, log_odds)

        ## Class weighted for loss
        pos_weight_tensor = torch.tensor(
            [self.pos_weight], dtype=torch.float32, device=self.device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        optimizer = (
            optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            if self.optimizer_str == "adamw"
            else optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        )

        # ── EARLY STOPPING
        if self.early_stopping:
            if eval_set is not None:
                X_val_es, y_val_es = eval_set[0]
                X_tr, y_tr = X, y
            else:
                X_tr, X_val_es, y_tr, y_val_es = train_test_split(
                    X,
                    y,
                    test_size=self.val_split,
                    random_state=self.seed,
                    stratify=y,
                )
            train_loader = self._make_loader(X_tr, y_tr, shuffle=True)
            val_loader = self._make_loader(X_val_es, y_val_es, shuffle=False)
        else:
            train_loader = self._make_loader(X, y, shuffle=True)
            val_loader = None

        # ── Train loop ────────────────────────────────────────────────────────
        best_metric = None
        best_state_dict = None
        best_epoch = None
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                criterion(model(data), target).backward()
                optimizer.step()

            if not self.early_stopping:
                continue

            ### EARLY STOPPING MONITOR ###
            val_loss, current_metric = self._eval_val(model, val_loader, criterion)

            improved = best_metric is None or current_metric > (
                best_metric + (abs(best_metric) * self.es_min_delta)
            )
            if improved:
                best_metric = current_metric
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"val_loss={val_loss:.4f} | "
                    f"{self.monitor}={current_metric:.4f} | "
                    f"patience={patience_counter}/{self.es_patience}"
                )

            if patience_counter >= self.es_patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}.")
                break

        if self.early_stopping and best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        self.model_ = model
        self.classes_ = np.unique(y)

        if self.early_stopping:
            if best_epoch is not None:
                ## Get optimal epoch # (1-indexed)
                self.best_iteration_ = best_epoch + 1
            else:
                ## Fall back to self.epochs if early stopping never triggered (or never improved)
                self.best_iteration_ = self.epochs
        else:
            ## If early stopping not used
            self.best_iteration_ = None
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Call fit() before predict_proba()")
        self.model_.eval()
        with torch.no_grad():
            X_tensor = (
                torch.tensor(X.values, dtype=torch.float32)
                if isinstance(X, pd.DataFrame)
                else torch.tensor(X, dtype=torch.float32)
            ).to(self.device)
            proba = torch.sigmoid(self.model_(X_tensor))
            return np.column_stack((1 - proba.cpu().numpy(), proba.cpu().numpy()))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return average_precision_score(y, self.predict_proba(X)[:, 1])


def load_nn_clf(data_path, in_dim, device):
    """
    Load an INFERENCE ONLY saved nn model from memory
    """
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    h_params = data["h_params"]
    state_dict = data["state_dict"]
    feature_names_in_ = data["feature_names_in_"]

    hidden_size_list = [h_params["hl_1"]]
    dropouts = [h_params["dr_1"]]

    for layer_size in ["2", "3"]:
        if f"hl_{layer_size}" in h_params:
            hidden_size_list.append(h_params[f"hl_{layer_size}"])
            dropouts.append(h_params[f"dr_{layer_size}"])

    clf = TorchNNClassifier(
        hidden_size_list=hidden_size_list,
        dropouts=dropouts,
        activation_name=h_params["act_func_str"],
        epochs=data.get("epochs", 500),
        # lr=h_params["lr"],
        # weight_decay=h_params["weight_decay"],
        # batch_size=h_params["batch_size"],
        neg_slope=h_params.get("neg_slope", 0.01),
        device=device,
        early_stopping=False,  # weights already trained; skip fit logic on load
    )

    acts = [
        get_activation(clf.activation_name, clf.neg_slope) for _ in clf.hidden_size_list
    ]
    assert in_dim == len(feature_names_in_)
    clf.model_ = MLP(
        clf.hidden_size_list,
        in_dim,
        clf.dropouts,
        acts,
        clf.weight_init_scheme,
        clf.bias_init,
    ).to(device)
    clf.model_.load_state_dict(state_dict)
    clf.model_.eval()
    clf.classes_ = np.array([0, 1])
    clf.feature_names_in_ = feature_names_in_
    return clf
