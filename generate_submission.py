#!/usr/bin/env python3
"""Generate submission.csv using trained LightGBM model with recursive forecasting."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta

PROCESSED_DIR = Path("processed")
FEATURE_FILE = PROCESSED_DIR / "all_buildings_features.parquet"
TEST_FILE = Path("global-ai-challenge-building-e-m-facilities-acad-25 (1)/test.csv")
SUBMIT_OUT = Path("submission.csv")

CAT_FEATURES = ["building", "dayofweek", "month", "is_weekend", "is_holiday", "working_hour", "is_long_weekend"]
TARGET = "load_kw"
LAGS = list(range(1, 25))
ROLL_WINDOW = 168

HK_HOLIDAYS = {
    "2024-01-01","2024-02-10","2024-02-11","2024-02-12","2024-02-13","2024-03-29","2024-03-30","2024-04-01","2024-04-04","2024-05-01","2024-05-15","2024-06-10","2024-07-01","2024-09-18","2024-10-01","2024-10-11","2024-12-25","2024-12-26","2025-01-01","2025-01-29","2025-01-30",
}

# precompute long weekend date set
_HDATES = {datetime.strptime(d, "%Y-%m-%d").date() for d in HK_HOLIDAYS}
LONG_WEEKEND = {d + timedelta(days=off) for d in _HDATES for off in (-1, 1)}

def train_full_model() -> LGBMRegressor:
    df = pd.read_parquet(FEATURE_FILE)
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    X = df.drop(columns=[TARGET, "timestamp"])
    y = df[TARGET]
    model = LGBMRegressor(n_estimators=1500, learning_rate=0.05, num_leaves=64, objective="regression", subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(X, y, categorical_feature=CAT_FEATURES)
    return model


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"]       = ts.dt.hour
    df["dayofweek"]  = ts.dt.dayofweek
    df["month"]      = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["sin_hour"]   = np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"]   = np.cos(2*np.pi*df["hour"]/24)
    doy               = ts.dt.dayofyear
    df["sin_doy"]    = np.sin(2*np.pi*doy/365)
    df["cos_doy"]    = np.cos(2*np.pi*doy/365)
    df["is_holiday"] = ts.dt.date.astype(str).isin(HK_HOLIDAYS).astype(int)
    df["working_hour"] = ((df["hour"].between(8,18)) & (df["is_weekend"]==0) & (df["is_holiday"]==0)).astype(int)
    quarter = ts.dt.quarter
    df["sin_quarter"] = np.sin(2*np.pi*quarter/4)
    df["cos_quarter"] = np.cos(2*np.pi*quarter/4)
    # long weekend flag
    long_weekend_flag = ts.dt.date.isin(LONG_WEEKEND)
    df["is_long_weekend"] = long_weekend_flag.astype(int)
    return df


def recursive_forecast(model: LGBMRegressor, history: pd.Series, future_times: List[pd.Timestamp], building: str) -> List[float]:
    preds = []
    hist = history.copy()
    for ts in future_times:
        feat = {"timestamp": ts, "building": building}
        df_row = pd.DataFrame([feat])
        df_row = add_static_features(df_row)
        # Convert all categorical features to category dtype
        for c in CAT_FEATURES:
            if c in df_row.columns:
                df_row[c] = df_row[c].astype("category")
        for lag in LAGS:
            df_row[f"lag_{lag}"] = hist.get(ts - pd.Timedelta(hours=lag), np.nan)
        w_start = ts - pd.Timedelta(hours=ROLL_WINDOW)
        window = hist.loc[w_start: ts - pd.Timedelta(hours=1)]
        df_row[f"roll_mean_{ROLL_WINDOW}"] = window.mean() if len(window)>0 else 0.0
        df_row[f"roll_std_{ROLL_WINDOW}"]  = window.std(ddof=0) if len(window)>1 else 0.0
        # fill remaining NaNs with roll_mean (skip categorical columns)
        roll_mean_val = df_row[f"roll_mean_{ROLL_WINDOW}"].iloc[0]
        numeric_cols = df_row.select_dtypes(include=[np.number]).columns
        df_row[numeric_cols] = df_row[numeric_cols].fillna(roll_mean_val)
        X = df_row.drop(columns=["timestamp"])
        pred = float(model.predict(X)[0])
        preds.append(pred)
        hist.loc[ts] = pred
    return preds


def main():
    model = train_full_model()
    test_df = pd.read_csv(TEST_FILE)
    test_df["timestamp"] = pd.to_datetime(test_df["prediction_time"], dayfirst=True, errors="raise")
    test_df["timestamp"] = test_df["timestamp"].dt.tz_localize("Asia/Hong_Kong", nonexistent="shift_forward")
    test_df["building"] = test_df["building_id"].str.extract(r"Building([A-D])_", expand=False)

    # history per building
    train = pd.read_parquet(FEATURE_FILE)[["timestamp", "building", TARGET]]
    hist_dict: Dict[str, pd.Series] = {}
    for b, g in train.groupby("building"):
        s = pd.Series(g[TARGET].values, index=g["timestamp"])
        s.sort_index(inplace=True)
        hist_dict[b] = s

    preds_col: List[float] = [0.0]*len(test_df)

    # group times per building in original order list map
    building_times_map: Dict[str, List[tuple[int, pd.Timestamp]]] = {}
    for idx, (bld, ts) in enumerate(zip(test_df["building"], test_df["timestamp"])):
        building_times_map.setdefault(bld, []).append((idx, ts))

    for bld, idx_ts_list in building_times_map.items():
        idx_ts_list_sorted = sorted(idx_ts_list, key=lambda x: x[1])
        times = [t for _, t in idx_ts_list_sorted]
        preds = recursive_forecast(model, hist_dict[bld], times, bld)
        # assign back
        for (orig_idx, _), pred in zip(idx_ts_list_sorted, preds):
            preds_col[orig_idx] = pred

    test_df["predicted_load"] = preds_col
    test_df[["building_id", "prediction_time", "predicted_load"]].to_csv(SUBMIT_OUT, index=False)
    print("Saved submission to", SUBMIT_OUT)


if __name__ == "__main__":
    main() 