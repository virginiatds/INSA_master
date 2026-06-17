"""
conformal_methods.py
Neural-network conformal methods used in the final experiments.

Methods:
1. PointNN + Split Conformal
2. QuantileNN + Conformalized Quantile Regression (CQR)
3. PointNN + Locally Adaptive Split Conformal

Important design choice:
- No Optuna is used in the final version.
- A small fixed hyperparameter grid is tested for all three methods.
- The grid is selected only using an internal validation split from the proper training set.
- Calibration is used only to compute Q. Test is used only for final evaluation.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from conformal_core import (
    ALPHA,
    CQR_Q_HI,
    CQR_Q_LO,
    DEVICE,
    HYPERPARAMETER_GRID,
    RANDOM_STATE,
    TARGET_COVERAGE,
    cfg_with_hp,
    clip_interval,
    conformal_quantile,
    global_summary,
    group_metrics,
    inv_y,
    mondrian_q,
    set_seed,
)

METHOD_LABELS = {
    "point": "PointNN + Split Conformal",
    "cqr": "QuantileNN + CQR",
    "local": "PointNN + Local SC",
}


# ---------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------
class MLP(nn.Module):
    """Simple multilayer perceptron for point regression."""

    def __init__(self, n_features: int, hidden=(128, 64), dropout=0.10, out_dim=1):
        super().__init__()
        layers = []
        d = n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class QuantileMLP(MLP):
    """MLP that directly predicts q05 and q95 for CQR."""

    def __init__(self, n_features: int, hidden=(128, 64), dropout=0.10):
        super().__init__(n_features, hidden, dropout, out_dim=2)


def make_loader(X, y, batch_size=512, shuffle=True):
    """Create a PyTorch loader from numpy arrays."""
    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Predict using a PyTorch model and return numpy output."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return model(x).detach().cpu().numpy()


def pinball_loss(pred, y, q_lo=CQR_Q_LO, q_hi=CQR_Q_HI):
    """Pinball loss for q05/q95 plus a small quantile-crossing penalty."""
    y = y.view(-1, 1)
    lo, hi = pred[:, 0:1], pred[:, 1:2]
    e_lo = y - lo
    e_hi = y - hi
    loss_lo = torch.maximum(q_lo * e_lo, (q_lo - 1.0) * e_lo).mean()
    loss_hi = torch.maximum(q_hi * e_hi, (q_hi - 1.0) * e_hi).mean()
    crossing = torch.clamp(lo - hi, min=0).mean()
    return loss_lo + loss_hi + 0.10 * crossing


# ---------------------------------------------------------------------
# Internal training helpers
# ---------------------------------------------------------------------
def internal_train_val_split(X, y):
    """Same internal validation split for all hyperparameter candidates."""
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)


def train_point_once(X_train, y_train, X_val, y_val, cfg, loss_name="mse"):
    """Train one point MLP candidate and return the best validation model."""
    set_seed(RANDOM_STATE)
    model = MLP(X_train.shape[1], cfg.hidden, cfg.dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss() if loss_name == "mse" else nn.L1Loss()
    loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)

    Xv = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    best_score, best_state, best_epoch, wait = np.inf, None, 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            score = float(loss_fn(model(Xv), yv).item())

        if score < best_score - 1e-7:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, {"val_score": best_score, "best_epoch": best_epoch}


def train_quantile_once(X_train, y_train, X_val, y_val, cfg):
    """
    Train one CQR candidate.
    Model selection is not based only on pinball loss: raw validation coverage and width are also used.
    """
    set_seed(RANDOM_STATE)
    model = QuantileMLP(X_train.shape[1], cfg.hidden, cfg.dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)

    Xv = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    best_key, best_state, best_info, wait = None, None, {}, 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = pinball_loss(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xv)
            loss = float(pinball_loss(pred, yv).item())
            lo = torch.minimum(pred[:, 0], pred[:, 1])
            hi = torch.maximum(pred[:, 0], pred[:, 1])
            raw_cov = float(((yv >= lo) & (yv <= hi)).float().mean().item())
            raw_width = float((hi - lo).mean().item())

        # Prefer candidates with target raw coverage; among them, prefer shorter intervals.
        key = (0, raw_width, abs(raw_cov - TARGET_COVERAGE), loss) if raw_cov >= TARGET_COVERAGE else (1, TARGET_COVERAGE - raw_cov, raw_width, loss)
        if best_key is None or key < best_key:
            best_key = key
            best_state = deepcopy(model.state_dict())
            best_info = {
                "best_epoch": epoch + 1,
                "pinball_val": loss,
                "raw_val_coverage_%": raw_cov * 100,
                "raw_val_width_scaled": raw_width,
                "selection_key": best_key,
            }
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, best_info


# ---------------------------------------------------------------------
# Hyperparameter grid selection
# ---------------------------------------------------------------------
def select_point_model(X, y, cfg, loss_name="mse"):
    """Select PointNN hyperparameters using only the internal validation split."""
    Xtr, Xvl, ytr, yvl = internal_train_val_split(X, y)
    candidates = HYPERPARAMETER_GRID if cfg.use_grid else [{"hidden": cfg.hidden, "lr": cfg.lr, "dropout": cfg.dropout, "weight_decay": cfg.weight_decay}]

    best = None
    for hp in candidates:
        hp_cfg = cfg_with_hp(cfg, hp)
        model, info = train_point_once(Xtr, ytr, Xvl, yvl, hp_cfg, loss_name=loss_name)
        score = float(info["val_score"])
        if best is None or score < best["score"]:
            best = {"score": score, "model": model, "hp": hp, "info": info}

    return best["model"], {"selected_hp": best["hp"], **best["info"]}


def select_quantile_model(X, y, cfg):
    """Select QuantileNN/CQR hyperparameters using only the internal validation split."""
    Xtr, Xvl, ytr, yvl = internal_train_val_split(X, y)
    candidates = HYPERPARAMETER_GRID if cfg.use_grid else [{"hidden": cfg.hidden, "lr": cfg.lr, "dropout": cfg.dropout, "weight_decay": cfg.weight_decay}]

    best = None
    for hp in candidates:
        hp_cfg = cfg_with_hp(cfg, hp)
        model, info = train_quantile_once(Xtr, ytr, Xvl, yvl, hp_cfg)
        key = info["selection_key"]
        if best is None or key < best["key"]:
            best = {"key": key, "model": model, "hp": hp, "info": info}

    info = dict(best["info"])
    info.pop("selection_key", None)
    return best["model"], {"selected_hp": best["hp"], **info}


# ---------------------------------------------------------------------
# Final conformal calibration
# ---------------------------------------------------------------------
def finalize_result(method_key, label, scores_cal, data, cfg, train_s, interval_function, extra=None):
    """Apply Global or Mondrian conformal calibration and build standardized outputs."""
    y_test = data["y_test_kg"]
    q_global = conformal_quantile(scores_cal, ALPHA)

    if cfg.strategy == "global":
        strategy_name = "global"
        group_strategy = f"global_by_{cfg.group_col}"
        q_dict = {g: q_global for g in cfg.group_order}
        q_test = q_global
    else:
        strategy_name = f"mondrian_{cfg.group_col}"
        group_strategy = strategy_name
        q_dict = mondrian_q(scores_cal, data["cal_aux"], cfg.group_col, cfg.group_order)
        test_groups = data["test_aux"][cfg.group_col].astype(str).values
        q_test = np.array([q_dict[g] for g in test_groups], dtype=float)

    lo, hi = interval_function(q_test)
    lo, hi = clip_interval(lo, hi)

    summary = pd.DataFrame([
        global_summary(lo, hi, y_test, method_key, label, strategy_name, np.mean(list(q_dict.values())), train_s)
    ])
    groups = group_metrics(
        lo, hi, y_test, data["test_aux"], data["cal_aux"], cfg.group_col, cfg.group_order,
        q_dict, method_key, label, group_strategy, q_global, train_s,
    )
    q_table = pd.DataFrame([
        {
            "method_key": method_key,
            "method": label,
            "strategy": strategy_name,
            "group_col": cfg.group_col if cfg.strategy == "mondrian" else None,
            "group": group if cfg.strategy == "mondrian" else "ALL",
            "Q_kg": q,
        }
        for group, q in (q_dict.items() if cfg.strategy == "mondrian" else {"ALL": q_global}.items())
    ])

    out = {
        "method_key": method_key,
        "method": label,
        "strategy": cfg.strategy,
        "summary_all": summary,
        "group_metrics_all": groups,
        "q_table_all": q_table,
        "intervals": {strategy_name: {"lo": lo, "hi": hi}},
        "Q_global": q_global,
        "q_dict": q_dict,
        "train_s": round(float(train_s), 1),
        "alpha": ALPHA,
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------
# The three methods
# ---------------------------------------------------------------------
def run_point_sc(data: dict, cfg) -> dict:
    """PointNN + Split Conformal: final interval = [mu(x) - Q, mu(x) + Q]."""
    t0 = time.perf_counter()
    model, selection = select_point_model(data["X_proper"], data["y_proper"], cfg, loss_name="mse")
    train_s = time.perf_counter() - t0

    mu_cal = inv_y(predict(model, data["X_cal"]), data["y_scaler"])
    mu_test = inv_y(predict(model, data["X_test"]), data["y_scaler"])
    scores_cal = np.abs(data["y_cal_kg"] - mu_cal)

    return finalize_result(
        "point", METHOD_LABELS["point"], scores_cal, data, cfg, train_s,
        interval_function=lambda q: (mu_test - q, mu_test + q),
        extra={"model_selection": selection},
    )


def run_cqr(data: dict, cfg) -> dict:
    """QuantileNN + CQR: learn q05/q95 directly, then conformalize them."""
    t0 = time.perf_counter()
    model, selection = select_quantile_model(data["X_proper"], data["y_proper"], cfg)
    train_s = time.perf_counter() - t0

    cal_pred = predict(model, data["X_cal"])
    test_pred = predict(model, data["X_test"])

    lo_cal = inv_y(np.minimum(cal_pred[:, 0], cal_pred[:, 1]), data["y_scaler"])
    hi_cal = inv_y(np.maximum(cal_pred[:, 0], cal_pred[:, 1]), data["y_scaler"])
    lo_test_raw = inv_y(np.minimum(test_pred[:, 0], test_pred[:, 1]), data["y_scaler"])
    hi_test_raw = inv_y(np.maximum(test_pred[:, 0], test_pred[:, 1]), data["y_scaler"])

    scores_cal = np.maximum(lo_cal - data["y_cal_kg"], data["y_cal_kg"] - hi_cal)

    return finalize_result(
        "cqr", METHOD_LABELS["cqr"], scores_cal, data, cfg, train_s,
        interval_function=lambda q: (lo_test_raw - q, hi_test_raw + q),
        extra={"model_selection": selection, "quantiles": {"q_lo": CQR_Q_LO, "q_hi": CQR_Q_HI}},
    )


def run_local_sc(data: dict, cfg) -> dict:
    """Locally Adaptive SC: final interval = [mu(x) - Q*sigma(x), mu(x) + Q*sigma(x)]."""
    t0 = time.perf_counter()

    # Select the same grid using the mean model validation score.
    mean_model, selection = select_point_model(data["X_proper"], data["y_proper"], cfg, loss_name="mse")

    # Train a scale model on residuals from the proper internal validation split.
    Xtr, Xvl, ytr, yvl = internal_train_val_split(data["X_proper"], data["y_proper"])
    hp_cfg = cfg_with_hp(cfg, selection["selected_hp"])

    # Refit the mean model on the same internal split used to create residual targets.
    mean_model, _ = train_point_once(Xtr, ytr, Xvl, yvl, hp_cfg, loss_name="mse")
    mu_v_kg = inv_y(predict(mean_model, Xvl), data["y_scaler"])
    yv_kg = inv_y(yvl, data["y_scaler"])
    log_resid = np.log(np.abs(yv_kg - mu_v_kg) + 1e-3)

    Xs_tr, Xs_vl, ys_tr, ys_vl = internal_train_val_split(Xvl, log_resid)
    sigma_model, sigma_info = train_point_once(Xs_tr, ys_tr, Xs_vl, ys_vl, hp_cfg, loss_name="l1")
    train_s = time.perf_counter() - t0

    mu_cal = inv_y(predict(mean_model, data["X_cal"]), data["y_scaler"])
    mu_test = inv_y(predict(mean_model, data["X_test"]), data["y_scaler"])
    sigma_cal = np.maximum(np.exp(predict(sigma_model, data["X_cal"])), 1e-3)
    sigma_test = np.maximum(np.exp(predict(sigma_model, data["X_test"])), 1e-3)

    scores_cal = np.abs(data["y_cal_kg"] - mu_cal) / sigma_cal

    return finalize_result(
        "local", METHOD_LABELS["local"], scores_cal, data, cfg, train_s,
        interval_function=lambda q: (mu_test - q * sigma_test, mu_test + q * sigma_test),
        extra={"model_selection": selection, "sigma_model_selection": sigma_info},
    )


def run_three_methods(data: dict, cfg) -> Dict[str, dict]:
    """Run the three neural conformal methods with the same split and same grid."""
    return {
        "point": run_point_sc(data, cfg),
        "cqr": run_cqr(data, cfg),
        "local": run_local_sc(data, cfg),
    }
