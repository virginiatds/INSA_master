"""
prepare_cdc_natality.py
Create the cleaned CDC Natality CSV used by the conformal experiments.

Default output:
    data/natality2023_clean_smoking_birthweight_1000k.csv

Example:
    python prepare_cdc_natality.py --raw data/Birth2023_1000k.txt --out data/natality2023_clean_smoking_birthweight_1000k.csv
    python prepare_cdc_natality.py --raw data/Birth2022_1000k.txt --out data/natality2022_clean_smoking_birthweight_1000k.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# CDC fixed-width positions used in this project.
COLSPECS = [
    (74, 76), (78, 79), (123, 124),
    (170, 172), (172, 174), (174, 176), (178, 179), (181, 182),
    (252, 254), (254, 256), (256, 258), (258, 260),
    (279, 281), (291, 294),
    (312, 313), (313, 314), (314, 315), (315, 316), (316, 317), (317, 318),
    (453, 454), (474, 475), (489, 491), (503, 507),
]
NAMES = [
    "MAGER", "MAGER9", "MEDUC",
    "PRIORLIVE", "PRIORDEAD", "PRIORTERM", "LBO_REC", "TBO_REC",
    "CIG_0", "CIG_1", "CIG_2", "CIG_3",
    "M_Ht_In", "PWgt_R",
    "RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE", "RF_EHYPE", "RF_PPTERM",
    "DPLURAL", "SEX", "COMBGEST", "DBWT",
]

RISK_COLS = ["RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE", "RF_EHYPE", "RF_PPTERM"]
PREG_CIG_COLS = ["CIG_1", "CIG_2", "CIG_3"]

BMI_MIN, BMI_MAX = 10.0, 80.0
DBWT_MIN, DBWT_MAX = 227, 8165
MAGER_MIN = 10
HT_MIN_IN = 24


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/Birth2023(1000k).txt", help="Raw CDC fixed-width file.")
    parser.add_argument("--out", default="data/natality2023_clean_smoking_birthweight_1000k.csv", help="Clean CSV output.")
    return parser.parse_args()


def read_raw_birth_file(path: str) -> pd.DataFrame:
    """Read selected columns from the CDC fixed-width file."""
    df = pd.read_fwf(path, colspecs=COLSPECS, names=NAMES, dtype=str)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def to_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric CDC fields and keep Y/N risk fields as strings."""
    numeric_cols = [
        "MAGER", "MAGER9", "MEDUC", "PRIORLIVE", "PRIORDEAD", "PRIORTERM",
        "LBO_REC", "TBO_REC", "CIG_0", "CIG_1", "CIG_2", "CIG_3",
        "M_Ht_In", "PWgt_R", "DPLURAL", "COMBGEST", "DBWT",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def replace_cdc_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace CDC missing/unknown codes by NaN."""
    for col in ["CIG_0", "CIG_1", "CIG_2", "CIG_3"]:
        df.loc[df[col] == 99, col] = np.nan
    for col in ["PRIORLIVE", "PRIORDEAD", "PRIORTERM"]:
        df.loc[df[col] == 99, col] = np.nan
    for col in ["LBO_REC", "TBO_REC"]:
        df.loc[df[col] == 9, col] = np.nan

    df.loc[df["PWgt_R"] == 999, "PWgt_R"] = np.nan
    df.loc[df["M_Ht_In"] == 99, "M_Ht_In"] = np.nan
    df.loc[df["DBWT"] == 9999, "DBWT"] = np.nan
    df.loc[df["COMBGEST"] == 99, "COMBGEST"] = np.nan
    df.loc[df["MEDUC"] == 9, "MEDUC"] = np.nan
    return df


def basic_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply conservative filters aimed at obvious recording errors."""
    bmi_filter = (df["PWgt_R"] * 0.453592) / ((df["M_Ht_In"] * 2.54 / 100.0) ** 2)
    df = df.assign(_bmi_filter=bmi_filter)

    steps = []

    def keep(data: pd.DataFrame, mask, label: str) -> pd.DataFrame:
        out = data.loc[mask].copy()
        steps.append({"step": label, "rows_remaining": len(out), "rows_removed": len(data) - len(out)})
        return out

    steps.append({"step": "raw", "rows_remaining": len(df), "rows_removed": 0})
    df = keep(df, df["DPLURAL"].eq(1), "singletons only")
    df = keep(df, df["DBWT"].between(DBWT_MIN, DBWT_MAX), "valid birth weight range")
    df = keep(df, df["MAGER"].ge(MAGER_MIN), "maternal age >= 10")
    df = keep(df, df["M_Ht_In"].ge(HT_MIN_IN), "maternal height >= 24 in")
    df = keep(df, df["SEX"].isin(["M", "F"]), "known fetal sex")
    df = keep(df, df["COMBGEST"].notna(), "known gestational age")
    df = keep(df, df["_bmi_filter"].between(BMI_MIN, BMI_MAX), "maternal BMI 10-80")

    return df.drop(columns=["_bmi_filter"]), pd.DataFrame(steps)


def encode_education(df: pd.DataFrame) -> pd.DataFrame:
    """Create education labels and low/medium/high groups."""
    labels = {
        1: "8th grade or less", 2: "9th through 12th grade with no diploma",
        3: "High school graduate or GED completed", 4: "Some college credit, but not a degree",
        5: "Associate degree (AA, AS)", 6: "Bachelor's degree (BA, AB, BS)",
        7: "Master's degree", 8: "Doctorate or Professional Degree",
    }
    df["MEDUC_label"] = df["MEDUC"].map(labels).fillna("Unknown or not stated")
    df["MEDUC_group"] = df["MEDUC"].apply(lambda x: "unknown" if pd.isna(x) else "low" if x <= 3 else "medium" if x <= 5 else "high")
    return df


def impute_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features, including CIG_0 and maternal BMI."""
    df["SEX"] = df["SEX"].astype(str).str.upper().where(df["SEX"].astype(str).str.upper().isin(["M", "F"]), np.nan)
    df["sex_male"] = df["SEX"].map({"M": 1, "F": 0})

    for col in ["CIG_0", "CIG_1", "CIG_2", "CIG_3"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df["avg_cig_preg_available"] = df[PREG_CIG_COLS].mean(axis=1, skipna=True)
    for col in PREG_CIG_COLS:
        df[col] = df[col].fillna(df["avg_cig_preg_available"]).fillna(0)
    df["CIG_0"] = df["CIG_0"].fillna(0)
    for col in ["CIG_0", "CIG_1", "CIG_2", "CIG_3"]:
        df[col] = df[col].clip(lower=0, upper=97)

    for col in RISK_COLS:
        df[col] = df[col].astype(str).str.strip().str.upper().where(lambda s: s.isin(["Y", "N", "U"]), np.nan)
        df[f"{col}_bin"] = df[col].map({"Y": 1, "N": 0})
        df[f"{col}_missing"] = df[f"{col}_bin"].isna().astype(int)
        df[f"{col}_bin"] = df[f"{col}_bin"].fillna(0).astype(int)

    for col in ["PRIORLIVE", "PRIORDEAD", "PRIORTERM", "LBO_REC", "TBO_REC"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    for col in ["PRIORLIVE", "PRIORDEAD", "PRIORTERM"]:
        df[col] = df[col].fillna(0).clip(lower=0, upper=30)
    for col in ["LBO_REC", "TBO_REC"]:
        df[col] = df[col].fillna(df[col].median()).fillna(1).clip(lower=1, upper=8)

    df["baby_poids_kg"] = (df["DBWT"] / 1000).round(3)
    df["mere_taille_cm"] = (df["M_Ht_In"] * 2.54).round(2)
    df["prepreg_weight_kg"] = (df["PWgt_R"] * 0.453592).round(2)
    df["bmi_calculado"] = (df["prepreg_weight_kg"] / ((df["mere_taille_cm"] / 100.0) ** 2)).round(2)
    df["LBW"] = (df["baby_poids_kg"] < 2.5).astype(int)
    df["preterm_current"] = (df["COMBGEST"] < 37).astype(int)

    df["avg_cig_preg"] = df[PREG_CIG_COLS].mean(axis=1).round(2)
    df["max_cig_preg"] = df[PREG_CIG_COLS].max(axis=1)
    df["smoke_preg_any"] = (df["max_cig_preg"] > 0).astype(int)
    df["smoke_before_preg_any"] = (df["CIG_0"] > 0).astype(int)

    df["has_prior_live"] = (df["PRIORLIVE"] > 0).astype(int)
    df["has_prior_dead"] = (df["PRIORDEAD"] > 0).astype(int)
    df["has_prior_term"] = (df["PRIORTERM"] > 0).astype(int)
    df["prior_pregnancy_history_count"] = (df["PRIORLIVE"] + df["PRIORDEAD"] + df["PRIORTERM"]).astype(int)
    df["is_first_live_birth"] = (df["LBO_REC"] == 1).astype(int)
    df["is_first_total_birth"] = (df["TBO_REC"] == 1).astype(int)

    bins = [-0.001, 0, 5, 10, 200]
    labels = ["Non-smoker (0 cig/day)", "Low smoker (1-5 cig/day)", "Moderate smoker (6-10 cig/day)", "Heavy smoker (>10 cig/day)"]
    df["cig_intensity_group"] = pd.cut(df["avg_cig_preg"], bins=bins, labels=labels)
    return df


def final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only model inputs, target, groups and diagnostics used downstream."""
    cols = [
        "baby_poids_kg", "LBW",
        "MAGER", "MAGER9", "MEDUC", "MEDUC_label", "MEDUC_group",
        "mere_taille_cm", "prepreg_weight_kg", "bmi_calculado",
        "SEX", "sex_male",
        "CIG_0", "CIG_1", "CIG_2", "CIG_3",
        "CIG_0_missing", "CIG_1_missing", "CIG_2_missing", "CIG_3_missing",
        "avg_cig_preg_available", "avg_cig_preg", "max_cig_preg",
        "cig_intensity_group", "smoke_before_preg_any", "smoke_preg_any",
        "COMBGEST", "preterm_current",
        "PRIORLIVE", "PRIORDEAD", "PRIORTERM", "LBO_REC", "TBO_REC",
        "PRIORLIVE_missing", "PRIORDEAD_missing", "PRIORTERM_missing", "LBO_REC_missing", "TBO_REC_missing",
        "has_prior_live", "has_prior_dead", "has_prior_term",
        "prior_pregnancy_history_count", "is_first_live_birth", "is_first_total_birth",
        "RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE", "RF_EHYPE", "RF_PPTERM",
        "RF_PDIAB_bin", "RF_GDIAB_bin", "RF_PHYPE_bin", "RF_GHYPE_bin", "RF_EHYPE_bin", "RF_PPTERM_bin",
        "RF_PDIAB_missing", "RF_GDIAB_missing", "RF_PHYPE_missing", "RF_GHYPE_missing", "RF_EHYPE_missing", "RF_PPTERM_missing",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def main():
    args = parse_args()
    raw_path = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw file: {raw_path}")
    df = read_raw_birth_file(str(raw_path))
    df = to_numeric_columns(df)
    df = replace_cdc_missing_codes(df)
    df, filter_log = basic_filters(df)
    df = encode_education(df)
    df = impute_and_derive(df)
    final = final_columns(df)

    final.to_csv(out_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    filter_log.to_csv(out_path.with_name(out_path.stem + "_filter_log.csv"), index=False)

    print(f"Saved clean CSV: {out_path}")
    print(f"Shape: {final.shape}")
    print("Key features available: CIG_0, smoke_before_preg_any, bmi_calculado, cig_intensity_group")


if __name__ == "__main__":
    main()
