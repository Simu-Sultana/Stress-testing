import argparse
import os
import pickle
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Check PhysioNet LOS units (days vs hours vs minutes)")
parser.add_argument("--data_dir", type=str, required=True, help="Folder containing physionet_2012.pkl")
parser.add_argument("--n_show", type=int, default=10, help="How many example rows to print")
args = parser.parse_args()

pkl_path = os.path.join(args.data_dir.rstrip("/"), "physionet_2012.pkl")

with open(pkl_path, "rb") as f:
    data, oc, train_ids, val_ids, test_ids = pickle.load(f)

print("\n==================== INPUT ====================")
print("PKL:", pkl_path)
print("data type:", type(data), "| oc type:", type(oc))
print("data shape:", getattr(data, "shape", None), "| columns:", list(data.columns) if isinstance(data, pd.DataFrame) else "N/A")
print("oc   shape:", getattr(oc, "shape", None),   "| columns:", list(oc.columns) if isinstance(oc, pd.DataFrame) else "N/A")

# Basic requirements
if not isinstance(data, pd.DataFrame):
    raise ValueError("❌ This script expects `data` to be a pandas DataFrame.")
if not isinstance(oc, pd.DataFrame):
    raise ValueError("❌ This script expects `oc` to be a pandas DataFrame.")
if "ts_id" not in data.columns or "minute" not in data.columns:
    raise ValueError(f"❌ data must contain columns ['ts_id','minute']. Found: {list(data.columns)}")
if "ts_id" not in oc.columns:
    raise ValueError(f"❌ oc must contain 'ts_id'. Found: {list(oc.columns)}")
if "length_of_stay" not in oc.columns:
    raise ValueError(f"❌ oc must contain 'length_of_stay'. Found: {list(oc.columns)}")

# Clean / coerce
tmp = data[["ts_id", "minute"]].dropna().copy()
tmp["minute"] = pd.to_numeric(tmp["minute"], errors="coerce")
tmp = tmp.dropna(subset=["minute"])

# Convert ts_id types to be consistent for grouping/mapping
# (PhysioNet often has int ts_id; handle float safely)
if tmp["ts_id"].dtype.kind == "f":
    frac = (tmp["ts_id"] - np.round(tmp["ts_id"])).abs().max()
    if frac > 1e-9:
        print(f"\n⚠ WARNING: data.ts_id has fractional part up to {frac}. Rounding before int.")
    tmp["ts_id"] = np.round(tmp["ts_id"]).astype(np.int64)
else:
    tmp["ts_id"] = pd.to_numeric(tmp["ts_id"], errors="coerce").astype("Int64")

oc2 = oc[["ts_id", "length_of_stay"]].copy()
oc2["length_of_stay"] = pd.to_numeric(oc2["length_of_stay"], errors="coerce")
oc2 = oc2.dropna(subset=["ts_id", "length_of_stay"])

# Compute max minute per ts_id
print("\nComputing max(minute) per ts_id ...")
max_minute = tmp.groupby("ts_id", sort=False)["minute"].max()

# Map max_minute onto oc rows
oc2["ts_id_int"] = pd.to_numeric(oc2["ts_id"], errors="coerce").astype(np.int64)
oc2["max_minute"] = oc2["ts_id_int"].map(max_minute)

# Drop rows where mapping failed
paired = oc2.dropna(subset=["max_minute"]).copy()

print("\n==================== QUICK STATS ====================")
print("oc length_of_stay describe:")
print(paired["length_of_stay"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())

print("\nmax(minute) describe:")
print(paired["max_minute"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())

# Infer unit by ratio: max_minute / length_of_stay
# If LOS is in days -> ratio ~1440
# If LOS is in hours -> ratio ~60
# If LOS is in minutes -> ratio ~1
ratio = paired["max_minute"] / paired["length_of_stay"]
ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

med = float(ratio.median())
print("\n==================== UNIT INFERENCE ====================")
print(f"Median ratio = median(max_minute / length_of_stay) = {med:.6f}")

def closeness(x, target):
    return abs(x - target)

candidates = {
    "minutes": 1.0,
    "hours": 60.0,
    "days": 1440.0,
}
best_unit = min(candidates.keys(), key=lambda k: closeness(med, candidates[k]))

print(f"Most likely unit for oc['length_of_stay']: {best_unit.upper()} (because median ratio closest to {candidates[best_unit]})")

# Show a few example rows
n = args.n_show
ex = paired[["ts_id", "length_of_stay", "max_minute"]].head(n).copy()
ex["ratio"] = ex["max_minute"] / ex["length_of_stay"]
print(f"\n==================== EXAMPLES (first {n}) ====================")
print(ex.to_string(index=False))

print("\nDONE.")
