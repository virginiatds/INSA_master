"""
conformal_core.py
Shared utilities for the final conformal prediction experiments.

This file defines:
- the two research modes (biomedical vs methodological);
- the feature sets for each mode;
- the conformal alpha and CQR quantiles;
- data loading, preprocessing and split creation;
- global and subgroup metrics;
- Mondrian group-specific conformal quantiles.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Main conformal settings
# ---------------------------------------------------------------------
ALPHA = 0.10                         # final conformal error level: 90% coverage
TARGET_COVERAGE = 1.0 - ALPHA
TARGET_COVERAGE_PCT = 100.0 * TARGET_COVERAGE

CQR_Q_LO = 0.05                       # direct lower quantile for CQR
CQR_Q_HI = 0.95                       # direct upper quantile for CQR

TARGET_COL = "birth_weight_kg"
DEFAULT_CLEAN_CSV = "data/natality2023_clean_smoking_birthweight_1000k.csv"
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LOWER_KG = 0.30
CLIP_UPPER_KG = 7.00

# Same small hyperparameter grid for the three neural-network methods.
# The selection criterion is method-specific, but the candidate grid is identical.
HYPERPARAMETER_GRID = [
    {"hidden": (64, 64), "lr": 1e-3, "dropout": 0.10, "weight_decay": 1e-5},
    {"hidden": (128, 64), "lr": 1e-3, "dropout": 0.10, "weight_decay": 1e-5},
    {"hidden": (128, 64), "lr": 5e-4, "dropout": 0.10, "weight_decay": 1e-5},
    {"hidden": (128, 128), "lr": 5e-4, "dropout": 0.20, "weight_decay": 1e-5},
]

RENAME_MAP = {
    "baby_poids_kg": "birth_weight_kg",
    "cig_intensity_group": "smoking_intensity",
}

SMOKE_MAP = {
    "Non-smoker (0 cig/day)": "non-smoker",
    "Low smoker (1-5 cig/day)": "low",
    "Moderate smoker (6-10 cig/day)": "moderate",
    "Heavy smoker (>10 cig/day)": "heavy",
    "non-smoker": "non-smoker",
    "low": "low",
    "moderate": "moderate",
    "heavy": "heavy",
}

# ---------------------------------------------------------------------
# Two possible research modes
# ---------------------------------------------------------------------
# Mode 1: biomedical/applied.
# Do not include variables known only after birth/delivery.
BIOMEDICAL_FEATURES = [
    # Maternal characteristics available before/during pregnancy
    "MAGER", "MEDUC", "mere_taille_cm", "prepreg_weight_kg", "bmi_calculado",

    # Smoking history: before pregnancy and during pregnancy
    "CIG_0", "smoke_before_preg_any",
    "CIG_1", "CIG_2", "CIG_3", "avg_cig_preg", "smoke_preg_any",

    # Obstetric history known before the current delivery
    "PRIORLIVE", "PRIORDEAD", "PRIORTERM", "LBO_REC", "TBO_REC",
    "has_prior_live", "has_prior_dead", "has_prior_term",
    "prior_pregnancy_history_count", "is_first_live_birth", "is_first_total_birth",

    # Clinical risk factors and fetal sex
    "RF_PDIAB_bin", "RF_GDIAB_bin", "RF_PHYPE_bin", "RF_GHYPE_bin", "RF_EHYPE_bin", "RF_PPTERM_bin",
    "sex_male",
]

# Mode 2: methodological.
# Uses biomedical data to compare interval methods; may include post-birth/gestational variables.
METHODOLOGICAL_FEATURES = BIOMEDICAL_FEATURES + ["COMBGEST", "preterm_current"]

GROUPS = {
    "biomedical": {
        "group_col": "smoking_intensity",
        "order": ["non-smoker", "low", "moderate", "heavy"],
    },
    "methodological": {
        "group_col": "preterm_x_smoking_intensity",
        "order": [
            "preterm + non-smoker", "preterm + low", "preterm + moderate", "preterm + heavy",
            "term + non-smoker", "term + low", "term + moderate", "term + heavy",
        ],
    },
}


@dataclass
class ExperimentConfig:
    csv_file: str
    dataset_name: str
    model_mode: str             # "biomedical" or "methodological"
    strategy: str               # "global" or "mondrian"
    out_dir: str
    epochs: int = 250
    patience: int = 25
    batch_size: int = 512
    hidden: Tuple[int, ...] = (128, 64)
    dropout: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    use_grid: bool = True

    @property
    def group_col(self) -> str:
        return GROUPS[self.model_mode]["group_col"]

    @property
    def group_order(self) -> List[str]:
        return GROUPS[self.model_mode]["order"]

    @property
    def features(self) -> List[str]:
        return BIOMEDICAL_FEATURES if self.model_mode == "biomedical" else METHODOLOGICAL_FEATURES


def cfg_with_hp(cfg: ExperimentConfig, hp: Dict) -> ExperimentConfig:
    """Return a copy of cfg with one hyperparameter candidate applied."""
    return replace(
        cfg,
        hidden=tuple(hp["hidden"]),
        lr=float(hp["lr"]),
        dropout=float(hp["dropout"]),
        weight_decay=float(hp["weight_decay"]),
    )


# ---------------------------------------------------------------------
# Reproducibility and split handling
# ---------------------------------------------------------------------
def set_seed(seed: int = RANDOM_STATE) -> None:
    """Fix random seeds for NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def split_hash(split: Dict[str, list]) -> str:
    """Create a short hash to verify that the same split was used."""
    clean = {k: [int(x) for x in split[k]] for k in ["proper", "cal", "test"]}
    return hashlib.md5(json.dumps(clean, sort_keys=True).encode()).hexdigest()


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create auxiliary variables expected by the experiments.

    The cleaning script already creates most of these columns. This function keeps
    the experiment robust if the CSV was produced by an older version.
    """
    df = df.rename(columns=RENAME_MAP).copy()

    # Ensure maternal BMI is available.
    if "bmi_calculado" not in df.columns and all(c in df.columns for c in ["prepreg_weight_kg", "mere_taille_cm"]):
        df["bmi_calculado"] = df["prepreg_weight_kg"] / ((df["mere_taille_cm"] / 100.0) ** 2)

    # Ensure smoking history variables are available.
    if all(c in df.columns for c in ["CIG_1", "CIG_2", "CIG_3"]):
        df["avg_cig_preg"] = df[["CIG_1", "CIG_2", "CIG_3"]].mean(axis=1, skipna=True)
    if "CIG_0" in df.columns and "smoke_before_preg_any" not in df.columns:
        df["smoke_before_preg_any"] = (pd.to_numeric(df["CIG_0"], errors="coerce").fillna(0) > 0).astype(int)
    if "avg_cig_preg" in df.columns and "smoke_preg_any" not in df.columns:
        df["smoke_preg_any"] = (pd.to_numeric(df["avg_cig_preg"], errors="coerce").fillna(0) > 0).astype(int)

    if "smoking_intensity" not in df.columns:
        raise ValueError("The CSV must contain smoking_intensity or cig_intensity_group.")

    raw_smoke = df["smoking_intensity"].astype(str)
    df["smoking_intensity"] = raw_smoke.map(SMOKE_MAP).fillna(raw_smoke.str.lower())
    df["smoking_status"] = np.where(df["smoking_intensity"].eq("non-smoker"), "non-smoker", "smoker")

    if "preterm_current" in df.columns:
        df["preterm_status"] = df["preterm_current"].map({0: "term", 1: "preterm"}).fillna(df["preterm_current"].astype(str))
        df["preterm_x_smoking_intensity"] = df["preterm_status"].astype(str) + " + " + df["smoking_intensity"].astype(str)

    return df


def load_dataset(csv_file: str) -> pd.DataFrame:
    """Load a semicolon/decimal-comma CSV and standardize group columns."""
    df = pd.read_csv(csv_file, sep=";", decimal=",", low_memory=False)
    return add_group_columns(df)


def check_required_columns(df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    """Fail early if the selected mode requires columns missing from the CSV."""
    missing = [c for c in cfg.features + [TARGET_COL, cfg.group_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for mode={cfg.model_mode}: {missing}")


def get_or_create_split(df: pd.DataFrame, cfg: ExperimentConfig) -> Dict[str, list]:
    """
    Create one proper/calibration/test split: 40%/40%/20%.
    The split is saved in the output folder for reproducibility.
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    split_file = Path(cfg.out_dir) / f"split_{cfg.dataset_name}_{cfg.model_mode}.pkl"

    if split_file.exists():
        with open(split_file, "rb") as f:
            split = pickle.load(f)
        print(f"Loaded split: {split_file.name} | hash={split_hash(split)}")
        return split

    strat_col = cfg.group_col if cfg.group_col in df.columns else "smoking_status"
    strat = df[strat_col].astype(str)
    strat = strat if strat.value_counts().min() >= 2 else None

    idx_train_cal, idx_test = train_test_split(
        df.index, test_size=0.20, random_state=RANDOM_STATE, stratify=strat,
    )

    strat2 = df.loc[idx_train_cal, strat_col].astype(str)
    strat2 = strat2 if strat2.value_counts().min() >= 2 else None

    idx_proper, idx_cal = train_test_split(
        idx_train_cal, test_size=0.50, random_state=RANDOM_STATE, stratify=strat2,
    )

    split = {"proper": idx_proper.tolist(), "cal": idx_cal.tolist(), "test": idx_test.tolist()}
    with open(split_file, "wb") as f:
        pickle.dump(split, f)

    print(f"Created split: {split_file.name} | hash={split_hash(split)}")
    return split


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
def prepare_data(df: pd.DataFrame, cfg: ExperimentConfig, split: Dict[str, list]) -> dict:
    """
    Fit imputer/scalers only on the proper training set.
    Calibration and test sets are transformed with these fitted objects.
    """
    check_required_columns(df, cfg)

    proper = df.loc[split["proper"]].copy()
    cal = df.loc[split["cal"]].copy()
    test = df.loc[split["test"]].copy()

    def x_numeric(part: pd.DataFrame) -> pd.DataFrame:
        return part[cfg.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_proper = x_scaler.fit_transform(imputer.fit_transform(x_numeric(proper)))
    X_cal = x_scaler.transform(imputer.transform(x_numeric(cal)))
    X_test = x_scaler.transform(imputer.transform(x_numeric(test)))

    y_proper = y_scaler.fit_transform(proper[[TARGET_COL]].astype(float)).ravel()
    y_cal_kg = cal[TARGET_COL].astype(float).values
    y_test_kg = test[TARGET_COL].astype(float).values

    aux_cols = [
        c for c in ["smoking_status", "smoking_intensity", "preterm_status", "preterm_x_smoking_intensity"]
        if c in df.columns
    ]

    return {
        "X_proper": X_proper,
        "y_proper": y_proper,
        "X_cal": X_cal,
        "X_test": X_test,
        "y_cal_kg": y_cal_kg,
        "y_test_kg": y_test_kg,
        "cal_aux": cal[aux_cols].copy(),
        "test_aux": test[aux_cols].copy(),
        "y_scaler": y_scaler,
        "features": cfg.features,
        "split_hash": split_hash(split),
    }


def inv_y(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-transform a scaled target vector back to kg."""
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


# ---------------------------------------------------------------------
# Conformal utilities and metrics
# ---------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Finite-sample conformal quantile: k = ceil((n + 1) * (1 - alpha))."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        raise ValueError("Empty conformal scores.")
    k = min(max(int(np.ceil((len(s) + 1) * (1.0 - alpha))), 1), len(s))
    return float(np.sort(s)[k - 1])


def clip_interval(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clip unrealistic birth-weight intervals and preserve lower <= upper."""
    lo = np.clip(lo, CLIP_LOWER_KG, CLIP_UPPER_KG)
    hi = np.clip(hi, CLIP_LOWER_KG, CLIP_UPPER_KG)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def global_summary(lo, hi, y, method_key: str, method_label: str, strategy: str, q_value: float, train_s: float) -> dict:
    """Compute global test metrics."""
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    mid = (lo + hi) / 2
    err = y - mid
    return {
        "method_key": method_key,
        "method": method_label,
        "strategy": strategy,
        "n_test": int(len(y)),
        "coverage_%": round(float(covered.mean() * 100), 3),
        "avg_width_kg": round(float(width.mean()), 4),
        "mean_lower_kg": round(float(np.mean(lo)), 4),
        "mean_upper_kg": round(float(np.mean(hi)), 4),
        "MAE_kg": round(float(np.abs(err).mean()), 4),
        "RMSE_kg": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "Q_kg": round(float(q_value), 4),
        "train_s": round(float(train_s), 1),
    }


def group_metrics(lo, hi, y, test_aux, cal_aux, group_col, order, q_dict, method_key, method_label, strategy, q_global, train_s) -> pd.DataFrame:
    """Compute subgroup test metrics for Global-by-group and Mondrian analyses."""
    rows = []
    gt = test_aux[group_col].astype(str).values
    gc = cal_aux[group_col].astype(str).values
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    mid = (lo + hi) / 2
    err = y - mid

    for group in order:
        mt = gt == group
        mc = gc == group
        row = {
            "method_key": method_key,
            "method": method_label,
            "strategy": strategy,
            "group_col": group_col,
            "group": group,
            "n_cal": int(mc.sum()),
            "n_test": int(mt.sum()),
            "Q_kg": round(float(q_dict.get(group, q_global)), 4),
            "train_s": round(float(train_s), 1),
        }
        if mt.sum() > 0:
            row.update({
                "coverage_%": round(float(covered[mt].mean() * 100), 3),
                "avg_width_kg": round(float(width[mt].mean()), 4),
                "MAE_kg": round(float(np.abs(err[mt]).mean()), 4),
                "RMSE_kg": round(float(np.sqrt(np.mean(err[mt] ** 2))), 4),
            })
        rows.append(row)

    return pd.DataFrame(rows)


def mondrian_q(scores_cal, cal_aux, group_col, order) -> Dict[str, float]:
    """
    Compute one conformal Q per Mondrian group.
    No fallback to global Q is used because the datasets are large.
    """
    groups = cal_aux[group_col].astype(str).values
    q_by_group = {}

    for group in order:
        mask = groups == str(group)
        if int(mask.sum()) == 0:
            counts = pd.Series(groups).value_counts().to_dict()
            raise ValueError(
                f"No calibration sample found for group='{group}' in group_col='{group_col}'. "
                f"Available calibration counts: {counts}"
            )
        q_by_group[group] = conformal_quantile(scores_cal[mask], ALPHA)

    return q_by_group
