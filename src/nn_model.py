from src.config import SEED

import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


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
        x = self.X[idx]
        y = self.y[idx]
        return x, y


def get_activation(name, trial=None):
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        slope = trial.suggest_float("neg_slope", 1e-3, 1e1, log=True) if trial else 0.01
        return nn.LeakyReLU(negative_slope=slope)
    else:
        raise ValueError(f"Unknown activation {name}")


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
                f"Expected dropouts to have same length as hidden states ({len(hidden_size_list)}), got {len(dropouts)} instead"
            )

        layers = []
        prev = in_dim
        ########## Build Skeleton #############
        for h, p, act in zip(hidden_size_list, dropouts, activation_list):
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if p and p > 0:
                layers.append(nn.Dropout(p))
            prev = h
        # Output node
        layers.append(nn.Linear(prev, 1))
        # Build backbone
        self.net = nn.Sequential(*layers)
        self._init_weights(init_scheme=weight_init_scheme, bias_init=bias_init)

    def _init_weights(
        self, init_scheme: str = "xavier_uniform", bias_init: float = 0.0
    ):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                ### Weight initialization ###
                if init_scheme == "xavier_uniform":
                    # Good default when using ReLU/ELU/Tanh with fan_avg scaling
                    init.xavier_uniform_(m.weight)
                elif init_scheme == "kaiming_uniform":
                    # Good with ReLU-like activations (uses fan_in scaling)
                    init.kaiming_uniform_(m.weight, nonlinearity="relu")
                ### Bias initialization ###
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
        optimizer_str="adam",
        epochs=30,
        batch_size=64,
        weight_init_scheme="xavier_uniform",
        bias_init=0.0,
        device="cpu",
        verbose=0,
        seed=SEED,
        # -- Early stopping --
        early_stopping=False,
        es_patience=10,
        es_min_delta=0.0,
        val_split=0.2,
        monitor="auc",
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
        # -- early stopping --
        self.early_stopping = early_stopping
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.val_split = val_split
        if monitor not in ("auc", "loss"):
            raise ValueError("monitor must be 'auc' or 'loss'")
        self.monitor = monitor

        self.model_ = None
        self._fit_X = None
        self._fit_y = None

    def _make_optimizer(self, model):
        if self.optimizer_str.lower() == "adamw":
            return optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_str.lower() == "adam":
            return optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer_str: {self.optimizer_str}")

    def fit(self, X, y):
        self._fit_X = X.copy()
        self._fit_y = y.copy()
        self.feature_names_in_ = np.array(X.columns)
        in_dim = X.shape[1]

        # Instantiate new activation for each layer
        acts = [get_activation(self.activation_name) for _ in self.hidden_size_list]

        # Build model and send to device
        model = MLP(
            self.hidden_size_list,
            in_dim,
            self.dropouts,
            acts,
            self.weight_init_scheme,
            self.bias_init,
        ).to(self.device)

        ## only need this if using >1 GPU
        if (
            isinstance(self.device, str)
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            # Use ALL visible GPUs in this process
            model = nn.DataParallel(model)

        optimizer = self._make_optimizer(model)
        criterion = nn.BCEWithLogitsLoss()

        # -- train/val split for early stopping --
        if self.early_stopping and self.val_split > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.val_split,
                random_state=self.seed,
                stratify=y,
            )
            ds_train = TabularDataset(X_train, y_train)
            ds_val = TabularDataset(X_val, y_val)

            g = torch.Generator()
            g.manual_seed(self.seed)
            train_loader = DataLoader(
                ds_train,
                batch_size=self.batch_size,
                shuffle=True,
                generator=g,
                pin_memory=True if "cuda" in str(self.device) else False,
                # num_workers=5, #only for tuning
                num_workers=0,
            )
            val_loader = DataLoader(
                ds_val,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True if "cuda" in str(self.device) else False,
                # num_workers=5, #only for tuning
                num_workers=0,
            )
        else:
            ## no early stopping
            # Dataset
            ds = TabularDataset(X, y)

            g = torch.Generator()
            g.manual_seed(self.seed)
            train_loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                generator=g,
                # only useful if using GPU
                pin_memory=True if "cuda" in str(self.device) else False,
                # num_workers=5, #only for tuning
                num_workers=0,
            )
            val_loader = None  # not used

        # =====================> Train loop
        best_metric = None
        best_state_dict = None
        patience_counter = 0

        for epoch in range(self.epochs):
            ## Train
            model.train()
            for data, target in train_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            # =========> Early stopping
            if self.early_stopping and val_loader is not None:
                model.eval()
                val_losses = []
                all_logits = []
                all_targets = []
                with torch.no_grad():
                    for data, target in val_loader:
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)
                        logits = model(data)
                        v_loss = criterion(logits, target)
                        val_losses.append(v_loss.item())

                        all_logits.append(logits.detach().cpu())
                        all_targets.append(target.detach().cpu())

                mean_val_loss = float(np.mean(val_losses))
                if self.monitor == "loss":
                    current_metric = -mean_val_loss  # higher is better
                else:  # monitor == "auc"
                    all_logits = torch.cat(all_logits)
                    all_targets = torch.cat(all_targets)
                    val_proba = torch.sigmoid(all_logits).numpy()
                    val_targets_np = all_targets.numpy()
                    # guard against degenerate cases
                    if len(np.unique(val_targets_np)) > 1:
                        current_metric = roc_auc_score(val_targets_np, val_proba)
                    else:
                        current_metric = -mean_val_loss  # fallback
                # check improvement
                if (
                    best_metric is None
                    or current_metric > best_metric + self.es_min_delta
                ):
                    best_metric = current_metric
                    best_state_dict = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.verbose:
                    print(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"val_loss={mean_val_loss:.4f}, "
                        f"{self.monitor}={current_metric:.4f}, "
                        f"patience={patience_counter}/{self.es_patience}"
                    )

                if patience_counter >= self.es_patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break  # stop training loop
        # restore best weights if stopped early
        if self.early_stopping and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        self.model_ = model
        self.classes_ = np.unique(y)  # Needed for sklearn ClassifierMixin
        self._fit_X = None
        self._fit_y = None
        return self

    def predict_proba(self, X):
        self.model_.eval()  # type: ignore
        with torch.no_grad():
            if isinstance(X, pd.DataFrame):
                X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model_(X_tensor)  # type: ignore
            proba = torch.sigmoid(logits)
            return np.column_stack((1 - proba.cpu().numpy(), proba.cpu().numpy()))

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)


def load_nn_clf(data_path, in_dim, device):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    h_params = data["h_params"]
    state_dict = data["state_dict"]
    feature_names_in_ = data["feature_names_in_"]
    hidden_size_list = [h_params["hl_1"], h_params["hl_2"]]
    dropouts = [h_params["dr_1"], h_params["dr_2"]]
    if "hl_3" in h_params and "dr_3" in h_params:
        hidden_size_list.append(h_params["hl_3"])
        dropouts.append(h_params["dr_3"])
    if "hl_4" in h_params and "dr_4" in h_params:
        hidden_size_list.append(h_params["hl_4"])
        dropouts.append(h_params["dr_4"])
    activation_name = h_params["act_func_str"]
    num_epochs = h_params["num_epochs"]
    lr = h_params["lr"]
    weight_decay = h_params["weight_decay"]
    batch_size = h_params["batch_size"]

    clf = TorchNNClassifier(
        hidden_size_list=hidden_size_list,
        dropouts=dropouts,
        activation_name=activation_name,
        epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
    )

    acts = [get_activation(clf.activation_name) for _ in clf.hidden_size_list]
    clf.model_ = MLP(
        hidden_size_list=clf.hidden_size_list,
        in_dim=in_dim,
        dropouts=clf.dropouts,
        activation_list=acts,
        weight_init_scheme=clf.weight_init_scheme,
        bias_init=clf.bias_init,
    ).to(device)

    clf.model_.load_state_dict(state_dict)
    clf.model_.to(device)
    clf.model_.eval()
    clf.classes_ = np.array([0, 1])
    clf.feature_names_in_ = feature_names_in_
    return clf
