#!/usr/bin/env python3
"""Generate feature table for Global AI Challenge cooling load model.

Input: processed/all_buildings_clean.parquet (from data_prep step0)
Output: processed/all_buildings_features.parquet

Features added per record:
    · hour, dayofweek, month, is_weekend
    · sin_hour, cos_hour  (cyclic encoding)
    · sin_doy,  cos_doy   (day-of-year cyclic)
    · lag_1 .. lag_24     (previous hour loads)
    · roll_mean_168       (7-day moving average load)
    · roll_std_168        (7-day moving std)

All lag/rolling features are computed within each building group to avoid leakage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("processed")
INPUT_FILE = PROCESSED_DIR / "all_buildings_clean.parquet"
OUTPUT_FILE = PROCESSED_DIR / "all_buildings_features.parquet"

LAGS = list(range(1, 25))  # 1-24 h
ROLL_WINDOW = 24 * 7  # 168 h (7 days)

# Hong Kong public holidays for 2024-2025 relevant to dataset
HK_HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13",
    "2024-03-29", "2024-03-30", "2024-04-01", "2024-04-04", "2024-05-01",
    "2024-05-15", "2024-06-10", "2024-07-01", "2024-09-18", "2024-10-01",
    "2024-10-11", "2024-12-25", "2024-12-26",
    # 2025 (up to Jan 30)
    "2025-01-01", "2025-01-29", "2025-01-30",
}

# Pre-compute date objects for holiday and long-weekend detection
_HDATES = {datetime.strptime(d, "%Y-%m-%d").date() for d in HK_HOLIDAYS}
_LONG_WEEKEND = set()
for d in _HDATES:
    _LONG_WEEKEND.add(d - timedelta(days=1))
    _LONG_WEEKEND.add(d + timedelta(days=1))

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek  # Monday=0
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # cyclic encodings
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    doy = ts.dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)
    # holiday
    df["is_holiday"] = ts.dt.date.astype(str).isin(HK_HOLIDAYS).astype(int)
    # working hour flag: 8-18 on weekday & not holiday
    df["working_hour"] = ((df["hour"].between(8, 18)) & (df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)
    # quarter cyclic encoding
    quarter = ts.dt.quarter
    df["sin_quarter"] = np.sin(2 * np.pi * quarter / 4)
    df["cos_quarter"] = np.cos(2 * np.pi * quarter / 4)
    # long weekend flag (day before/after a holiday)
    df["is_long_weekend"] = ts.dt.date.isin(_LONG_WEEKEND).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate engineered features for cooling load prediction")
    parser.add_argument("--input", type=Path, default=INPUT_FILE, help="Path to merged clean parquet file")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Path to save features parquet")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_parquet(args.input)
    # ensure timestamp is datetime with timezone
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["building", "timestamp"]).reset_index(drop=True)

    df = add_time_features(df)

    # compute lag & rolling per building to avoid leakage
    df_list: list[pd.DataFrame] = []
    for b, g in df.groupby("building", sort=False):
        g = g.sort_values("timestamp").copy()
        for lag in LAGS:
            g[f"lag_{lag}"] = g["load_kw"].shift(lag)
        g[f"roll_mean_{ROLL_WINDOW}"] = g["load_kw"].rolling(ROLL_WINDOW, min_periods=1).mean()
        g[f"roll_std_{ROLL_WINDOW}"] = g["load_kw"].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
        df_list.append(g)

    feat_df = pd.concat(df_list).reset_index(drop=True)

    # drop rows with any NaN in lag features (first 24 h per building)
    cols_to_check = [f"lag_{lag}" for lag in LAGS]
    feat_df = feat_df.dropna(subset=cols_to_check)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(args.output)
    print("Saved engineered features to", args.output, "with", len(feat_df), "rows")


if __name__ == "__main__":
    main() 