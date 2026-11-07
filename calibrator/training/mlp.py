"""The calibrator MLP (PyTorch implementation matching the paper exactly).

Paper appendix B.2: three hidden layers (128, 64, 32), each followed by batch
normalization, ReLU and dropout 0.3; Adam with learning rate 5e-4 and weight
decay 1e-4; binary cross-entropy; early stopping (patience 10) on a 10%
validation split. At inference time the batch-norm layers are folded into the
preceding linear layers, so ``export`` produces pure-NumPy weights that the MCP
server can evaluate without a torch dependency.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


class _CalibrationNet(nn.Module):
    """Linear -> BatchNorm1d -> ReLU -> Dropout, repeated, then a linear head."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 64, 32), dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPCalibrator:
    """Thin trainer around the PyTorch network; ``fit`` / ``predict_proba`` keep the old API."""

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64, 32),
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        max_epochs: int = 2000,
        patience: int = 10,
        seed: int = 42,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.in_dim: int | None = None
        self.model: _CalibrationNet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPCalibrator":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        y_int = y.astype(int)
        self.in_dim = X.shape[1]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y_int, test_size=0.1, random_state=self.seed, stratify=y_int
        )

        model = _CalibrationNet(self.in_dim, self.hidden, self.dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        X_fit_t = torch.from_numpy(X_fit.astype(np.float32))
        y_fit_t = torch.from_numpy(y_fit.astype(np.float32))
        X_val_t = torch.from_numpy(X_val.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.float32))

        best_val = float("inf")
        best_state: dict | None = None
        no_improve = 0
        for _ in range(self.max_epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(X_fit_t), y_fit_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_val_t), y_val_t))
            if val_loss < best_val - 1e-8:
                best_val, no_improve = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        self.model = _CalibrationNet(self.in_dim, self.hidden, self.dropout)
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "call fit before predict_proba"
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X))
            return torch.sigmoid(logits).numpy()

    def export(self) -> dict:
        """Fold batch-norm into the linear layers and return NumPy inference weights.

        Returns ``{"hidden": [(W, b), ...], "out": (W_out, b_out)}`` where every
        ``W`` has shape ``(out_features, in_features)``, matching torch's layout.
        """
        assert self.model is not None, "call fit before export"
        sd = self.model.state_dict()
        eps = 1e-5  # torch BatchNorm1d default
        folded = []
        for i, _ in enumerate(self.hidden):
            W = sd[f"net.{4 * i}.weight"].cpu().numpy()
            b = sd[f"net.{4 * i}.bias"].cpu().numpy()
            gamma = sd[f"net.{4 * i + 1}.weight"].cpu().numpy()
            beta = sd[f"net.{4 * i + 1}.bias"].cpu().numpy()
            rm = sd[f"net.{4 * i + 1}.running_mean"].cpu().numpy()
            rv = sd[f"net.{4 * i + 1}.running_var"].cpu().numpy()
            scale = gamma / np.sqrt(rv + eps)
            folded.append((W * scale[:, None], (b - rm) * scale + beta))
        out_idx = 4 * len(self.hidden)
        W_out = sd[f"net.{out_idx}.weight"].cpu().numpy()
        b_out = sd[f"net.{out_idx}.bias"].cpu().numpy()
        return {"hidden": folded, "out": (W_out, b_out)}


def folded_predict(exported: dict, X: np.ndarray) -> np.ndarray:
    """Evaluate exported folded weights in pure NumPy (no torch)."""
    a = np.asarray(X, dtype=float)
    for W, b in exported["hidden"]:
        a = np.maximum(a @ W.T + b, 0.0)
    W_out, b_out = exported["out"]
    logit = (a @ W_out.T + b_out).ravel()
    return 1.0 / (1.0 + np.exp(-logit))


def load_folded(path) -> "callable":
    """Load a ``*.npz`` exported by ``calibrator.training`` into a callable."""
    z = np.load(path)
    n_hidden = int(z["n_hidden"])
    hidden = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n_hidden)]
    out = (z["W_out"], z["b_out"])

    def predict(X: np.ndarray) -> np.ndarray:
        return folded_predict({"hidden": hidden, "out": out}, X)

    return predict
