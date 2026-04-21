import argparse
import os
import pandas as pd
import pickle
import numpy as np

# --------------------------------------------------------
# 1. Parse command-line arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser(description="Create MIMIC-III unbalanced subsets (LOS target) for STRaTS")
parser.add_argument("--data_dir", type=str, required=True, help="Path to MIMIC-III processed folder (contains mimic_iii.pkl)")
parser.add_argument("--out_dir", type=str, required=True, help="Where to save subset PKL files")
parser.add_argument("--pct", type=int, required=True, help="Percentage of TRAIN admissions to be labeled 1 (length_of_stay > threshold)")
args = parser.parse_args()

RAW_DATA_PATH = args.data_dir.rstrip("/")
OUT_DIR = args.out_dir.rstrip("/")
os.makedirs(OUT_DIR, exist_ok=True)

print("\n==========================================")
print(" RAW_DATA_PATH =", RAW_DATA_PATH)
print(" OUT_DIR       =", OUT_DIR)
print("==========================================\n")

# --------------------------------------------------------
# 2. Read pickle file
# --------------------------------------------------------
filepath = os.path.join(RAW_DATA_PATH, "mimic_iii.pkl")
with open(filepath, "rb") as f:
    data, oc, train_ids, val_ids, test_ids = pickle.load(f)

pct = args.pct

# --------------------------------------------------------
# 3. Build length_of_stay (days) from time-series data
#    Your data has: ['ts_id','minute','variable','value','TABLE']
#    We define LOS_days(ts_id) = max(minute) / 1440.
# --------------------------------------------------------
if "minute" not in data.columns or "ts_id" not in data.columns:
    raise ValueError(f"❌ data must contain columns ['ts_id','minute']. Found: {list(data.columns)}")

# drop NaNs for safety
tmp = data[["ts_id", "minute"]].dropna()

# data.ts_id is float like 200001.0 -> convert to int (safe since all .0)
ts = tmp["ts_id"].values
frac = np.nanmax(np.abs(ts - np.round(ts)))
if frac > 1e-9:
    print(f"⚠ WARNING: data.ts_id has fractional part up to {frac}. Rounding before int conversion.")
    tmp["ts_id"] = np.round(tmp["ts_id"]).astype(np.int64)
else:
    tmp["ts_id"] = tmp["ts_id"].astype(np.int64)

# groupby max minute per ts_id
print("Computing LOS from data: max(minute)/1440 per ts_id ...")
max_minute = tmp.groupby("ts_id", sort=False)["minute"].max()
los_days_by_id = max_minute / (60 * 24)  # 1440
los_days_by_id.name = "length_of_stay"

# attach to oc
if "ts_id" not in oc.columns:
    raise ValueError(f"❌ oc must contain 'ts_id'. Found: {list(oc.columns)}")

oc = oc.copy()
oc["length_of_stay"] = oc["ts_id"].map(los_days_by_id)

missing = oc["length_of_stay"].isna().sum()
if missing > 0:
    print(f"⚠ WARNING: {missing} oc rows have missing length_of_stay (ts_id not found in data). They will remain NaN.")

# --------------------------------------------------------
# 4. Compute threshold T (days) so that pct% of TRAIN admissions are class 1
#    (KEEP EXACTLY SAME AS PHYSIONET SCRIPT)
# --------------------------------------------------------
lengths = oc[oc.ts_id.isin(train_ids)]["length_of_stay"].values
lengths = lengths[~np.isnan(lengths)]  # small safety to avoid NaN percentile crash

if len(lengths) == 0:
    raise ValueError("❌ No valid TRAIN length_of_stay values found (maybe ts_id mismatch).")

T = np.percentile(lengths, 100 - pct)
oc["unbalanced"] = (oc["length_of_stay"] > T).astype(int)

print(f"Threshold length of stay: {T:.2f} days")

for split_name, split_ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_oc = oc[oc.ts_id.isin(split_ids)]
    counts = split_oc["unbalanced"].value_counts(normalize=True) * 100
    print(f"{split_name.upper()} split:")
    print(f"  Class 0: {counts.get(0,0):.2f}%, Class 1: {counts.get(1,0):.2f}%")

# --------------------------------------------------------
# 5. Save unbalanced LOS threshold to a separate CSV (same as PhysioNet)
# --------------------------------------------------------
los_csv_path = os.path.join(OUT_DIR, "unbalanced_los_days.csv")

los_row = pd.DataFrame([{
    "dataset": "mimic_iii",
    "perturbation": "unbalanced",
    "pct": int(pct),
    "los_threshold_days": float(T)
}])

los_row.to_csv(
    los_csv_path,
    mode="a",
    header=not os.path.exists(los_csv_path),
    index=False
)

print(f"   ✔ Saved LOS days summary: {los_csv_path}")

# --------------------------------------------------------
# 6. Save PKL (unchanged structure)
# --------------------------------------------------------
out_path = os.path.join(OUT_DIR, f"mimic_iii_unbalanced_{pct}.pkl")

with open(out_path, "wb") as f:
    pickle.dump([data, oc, train_ids, val_ids, test_ids], f)

print(f"   ✔ Saved: {out_path}")
