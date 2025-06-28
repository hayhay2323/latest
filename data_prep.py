#!/usr/bin/env python3
"""Data preprocessing script for Global AI Challenge Cooling Load task.

Steps
-----
1. Load raw 15-min chiller sensor CSVs for Buildings A-D.
2. Compute instantaneous cooling load (kW) for every available chiller:
       Q_i = 4.19 * FR_i * (T_return_i - T_supply_i)
   where FR is Chilled Water Flow Rate (L/s) and temperatures are in °C.
   If FR or temperatures are missing/zero, resulting Q_i is set to 0.
3. Sum all available chiller loads to obtain "total_cooling_load".
4. Resample from 15-min to 1-hour granularity using the mean (equivalent
   to energy-preserving average power method).
5. Persist hourly data per building as Parquet and merged CSV for modelling.

Usage
-----
$ python data_prep.py --data-dir "global-ai-challenge-building-e-m-facilities-acad-25 (1)/Buildings_Datasets/Buildings_Datasets" --out-dir "processed"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

# Constants
SPECIFIC_HEAT_KJ_PER_KG_C = 4.19  # kJ/kg·°C, equals kW·s per (L/s·°C).
HK_TZ = "Asia/Hong_Kong"
CHILLER_PATTERN = re.compile(r"CHR-(\d{2})-(KW|CHWSWT|CHWRWT|CHWFWR)")


def find_chiller_ids(columns: list[str]) -> list[str]:
    """Extract unique two-digit chiller IDs (e.g., '01', '02', '03')."""
    ids = set()
    for col in columns:
        m = CHILLER_PATTERN.match(col)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def compute_instantaneous_cooling_load(df: pd.DataFrame) -> pd.Series:
    """Compute total cooling load (kW) for every timestamp in df."""
    chiller_ids = find_chiller_ids(df.columns)
    total_cl = pd.Series(0.0, index=df.index)
    for cid in chiller_ids:
        fr = df.get(f"CHR-{cid}-CHWFWR", 0)  # L/s
        tsh = df.get(f"CHR-{cid}-CHWSWT", 0)  # °C supply
        tr = df.get(f"CHR-{cid}-CHWRWT", 0)  # °C return
        # delta T; negative values (sensor glitch) clipped to 0
        delta_t = (tr - tsh).clip(lower=0)
        qi = SPECIFIC_HEAT_KJ_PER_KG_C * fr * delta_t  # kW
        # Replace NaNs with 0 (missing sensor)
        qi = qi.fillna(0)
        total_cl += qi
    return total_cl


def process_file(path: Path) -> pd.DataFrame:
    print(f"Processing {path.name} …", file=sys.stderr)
    df = pd.read_csv(path, parse_dates=["record_timestamp"], dayfirst=True)
    df["record_timestamp"] = df["record_timestamp"].dt.tz_localize(HK_TZ, nonexistent="shift_forward")
    df = df.set_index("record_timestamp").sort_index()

    df["total_cooling_load_kw"] = compute_instantaneous_cooling_load(df)

    # Resample to hourly average power (kW)
    hourly = df["total_cooling_load_kw"].resample("1H").mean()
    return hourly.to_frame()


def main():
    parser = argparse.ArgumentParser(description="Preprocess chiller datasets and compute hourly cooling load.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing Building_*.csv files")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to save processed outputs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    hourly_all = []
    for csv_file in sorted(args.data_dir.glob("Building_*_summary_table.csv")):
        building_id = csv_file.stem.split("_")[1]  # e.g., 'A'
        hourly_df = process_file(csv_file)
        hourly_df["building"] = building_id
        # Save per building
        hourly_df.to_parquet(args.out_dir / f"{building_id}_hourly.parquet")
        hourly_all.append(hourly_df)

    merged = pd.concat(hourly_all).reset_index().rename(columns={"record_timestamp": "timestamp"})
    merged.to_csv(args.out_dir / "all_buildings_hourly.csv", index=False)
    print("Saved processed data to", args.out_dir)


if __name__ == "__main__":
    main() 