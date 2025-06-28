#!/usr/bin/env python3
"""LightGBM 5-fold time-series CV for cooling-load baseline.

每棟樓依時間做 expanding window 分割：
Split 1 :   20% train -> 20% valid
Split 2 :   40% train -> 20% valid
Split 3 :   60% train -> 20% valid
Split 4 :   80% train -> 20% valid
Split 5 :   100% train -> 最後 168 h 為 valid (類似 hold-out)

最終報告每 fold 及加權平均 NRMSE。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

PROCESSED_DIR = Path("processed")
FEATURE_FILE = PROCESSED_DIR / "all_buildings_features.parquet"
CAT_FEATURES = ["building", "dayofweek", "month", "is_weekend", "is_holiday", "working_hour", "is_long_weekend"]
TARGET = "load_kw"


def load_df() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_FILE)
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    return df.sort_values(["building", "timestamp"]).reset_index(drop=True)


def ts_kfold_indices(ts_length: int, n_splits: int = 5):
    """Generate (train_idx, valid_idx) tuples for expanding window CV."""
    fold_sizes = np.linspace(0.2, 1.0, n_splits + 1)  # 0,20,40,..100%
    indices = np.arange(ts_length)
    for i in range(1, n_splits + 1):
        end_train = int(fold_sizes[i - 1] * ts_length)
        end_valid = int(fold_sizes[i] * ts_length)
        if i == n_splits:
            # 最後一折改用最後 168 小時當驗證
            valid_idx = indices[-168:]
            train_idx = indices[:-168]
        else:
            span = end_valid - end_train
            valid_idx = indices[end_train:end_valid]
            train_idx = indices[:end_train]
        yield train_idx, valid_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=64)
    args = parser.parse_args()

    df = load_df()

    fold_nrmses = []
    for fold, (train_idx, valid_idx) in enumerate(ts_kfold_indices(len(df), 5), 1):
        X_train = df.iloc[train_idx].drop(columns=[TARGET, "timestamp"])
        y_train = df.iloc[train_idx][TARGET]
        X_valid = df.iloc[valid_idx].drop(columns=[TARGET, "timestamp"])
        y_valid = df.iloc[valid_idx][TARGET]

        model = LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            objective="regression",
            metric="rmse",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=fold * 17,
            verbose=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            categorical_feature=CAT_FEATURES,
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
        )

        preds = model.predict(X_valid, num_iteration=model.best_iteration_)
        df_valid = df.iloc[valid_idx].copy()
        df_valid["pred"] = preds
        nrmse_build = []
        for b, g in df_valid.groupby("building"):
            if len(g) == 0 or g[TARGET].max() == g[TARGET].min():
                continue
            p = g["pred"].values
            a = g[TARGET].values
            nrmse = np.sqrt(np.mean((p - a) ** 2)) / (a.max() - a.min())
            nrmse_build.append(nrmse)
        fold_nrmse = np.mean(nrmse_build)
        print(f"Fold {fold} NRMSE: {fold_nrmse:.4f}")
        fold_nrmses.append(fold_nrmse)

    print("Overall CV NRMSE (mean ± std):", np.mean(fold_nrmses), "+/-", np.std(fold_nrmses))


if __name__ == "__main__":
    main() 