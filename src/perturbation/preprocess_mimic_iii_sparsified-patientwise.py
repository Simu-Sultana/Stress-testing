import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import pickle
import numpy as np
import pandas as pd


# --------------------------------------------------------
# 1) Parse arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser(
    description="MIMIC-III patient-wise sparsification (GRU-D safe: keep labels + ensure non-empty delta)"
)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--pct", type=int, required=True)
args = parser.parse_args()

RAW_DATA_PATH = args.data_dir.rstrip("/")
OUT_DIR = args.out_dir.rstrip("/")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.RandomState(args.seed)

print("\n==========================================")
print(" PATIENT-WISE SPARSIFICATION (GRU-D SAFE)")
print(" RAW_DATA_PATH =", RAW_DATA_PATH)
print(" OUT_DIR       =", OUT_DIR)
print(" PCT           =", args.pct)
print(" SEED          =", args.seed)
print("==========================================\n")


# --------------------------------------------------------
# 2) Helpers
# --------------------------------------------------------
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint, keep semantics."""
    if "ts_id" in df.columns:
        df["ts_id"] = pd.to_numeric(df["ts_id"], errors="coerce", downcast="integer")
    if "minute" in df.columns:
        df["minute"] = pd.to_numeric(df["minute"], errors="coerce", downcast="integer")
    if "variable" in df.columns and df["variable"].dtype == "object":
        df["variable"] = df["variable"].astype("category")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce", downcast="float")
    return df


# --------------------------------------------------------
# 3) Load original processed dataset
# --------------------------------------------------------
with open(os.path.join(RAW_DATA_PATH, "mimic_iii.pkl"), "rb") as f:
    data, oc, train_ids, val_ids, test_ids = pickle.load(f)

data = optimize_dtypes(data)
# IMPORTANT: keep oc as-is because it contains labels like in_hospital_mortality


# --------------------------------------------------------
# 4) Separate static vs temporal variables
# --------------------------------------------------------
demo_features = [
    "Age", "Gender", "Height",
    "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"
]

demo = data[data["variable"].isin(demo_features)].copy()
ts   = data[~data["variable"].isin(demo_features)].copy()
ts = ts.sort_values(["ts_id", "minute"])


# --------------------------------------------------------
# 5) Patient-wise sparsification with GRU-D safety
#    Key constraints (to avoid BAD DELTA / delta=[]):
#      - only keep rows with minute < MAX_MINUTE (matches Dataset cropping)
#      - keep >= 2 distinct minutes per ts_id in that window
# --------------------------------------------------------
MAX_MINUTE = 880  # should match args.max_timesteps used in your pipeline

stats_records = []
keep_idx_parts = []
dropped_ts_ids = []

for ts_id, group in ts.groupby("ts_id", sort=False):
    group = group.sort_values("minute")

    # keep only rows in the window used downstream
    group_win = group[group["minute"] < MAX_MINUTE]

    rows_before = len(group)
    rows_in_win = len(group_win)

    if rows_in_win == 0:
        dropped_ts_ids.append(int(ts_id))
        stats_records.append({
            "ts_id": int(ts_id),
            "rows_before": int(rows_before),
            "rows_in_window": 0,
            "pct": int(args.pct),
            "rows_kept": 0,
            "rows_removed": int(rows_before),
            "dropped_reason": "no_rows_before_MAX_MINUTE",
        })
        continue

    uniq_minutes = group_win["minute"].unique()
    if len(uniq_minutes) < 2:
        dropped_ts_ids.append(int(ts_id))
        stats_records.append({
            "ts_id": int(ts_id),
            "rows_before": int(rows_before),
            "rows_in_window": int(rows_in_win),
            "pct": int(args.pct),
            "rows_kept": 0,
            "rows_removed": int(rows_before),
            "dropped_reason": "only_1_unique_minute_in_window",
        })
        continue

    # sample within window
    rows_kept_target = int(np.ceil(rows_in_win * args.pct / 100.0))
    rows_kept = max(2, rows_kept_target)

    take = min(rows_kept, rows_in_win)
    sampled_idx = group_win.sample(n=take, random_state=args.seed).index.to_numpy()

    keep_idx_parts.append(sampled_idx)
    stats_records.append({
        "ts_id": int(ts_id),
        "rows_before": int(rows_before),
        "rows_in_window": int(rows_in_win),
        "pct": int(args.pct),
        "rows_kept": int(len(sampled_idx)),
        "rows_removed": int(rows_before - len(sampled_idx)),
        "dropped_reason": "",
    })

keep_idx = np.concatenate(keep_idx_parts) if keep_idx_parts else np.array([], dtype=np.int64)
ts_sub = ts.loc[keep_idx].copy()

# Collapse duplicates and avoid huge categorical cartesian products
ts_sub = (
    ts_sub.groupby(["ts_id", "minute", "variable"], as_index=False, observed=True)["value"]
    .mean()
)

# enforce window after groupby (safety)
ts_sub = ts_sub[ts_sub["minute"] < MAX_MINUTE]
ts_sub = optimize_dtypes(ts_sub)

# Ensure still >=2 unique minutes per ts_id after collapsing
mins_per_id = ts_sub.groupby("ts_id", observed=True)["minute"].nunique()
good_ids = mins_per_id[mins_per_id >= 2].index.to_numpy()
ts_sub = ts_sub[ts_sub["ts_id"].isin(good_ids)]

valid_ts_ids = ts_sub["ts_id"].unique()


# --------------------------------------------------------
# 6) Filter demo/oc/splits (keep labels!)
# --------------------------------------------------------
demo_sub = demo[demo["ts_id"].isin(valid_ts_ids)].copy()
demo_sub = optimize_dtypes(demo_sub)

# keep oc labels (do NOT rebuild oc from ts_sub)
oc_sub = oc[oc["ts_id"].isin(valid_ts_ids)].copy()
oc_sub = optimize_dtypes(oc_sub)

train_sub = np.intersect1d(train_ids, valid_ts_ids)
val_sub   = np.intersect1d(val_ids, valid_ts_ids)
test_sub  = np.intersect1d(test_ids, valid_ts_ids)


# --------------------------------------------------------
# 7) Merge final dataset
# --------------------------------------------------------
data_sub = pd.concat([demo_sub, ts_sub], axis=0, copy=False, ignore_index=False)
data_sub = data_sub.sort_values(by=["ts_id", "minute"])


# --------------------------------------------------------
# 8) Save stats CSV + pickle
# --------------------------------------------------------
stats_df = pd.DataFrame(stats_records)
stats_csv = os.path.join(OUT_DIR, f"mimic_iii_patientwise_stats_{args.pct}_{args.seed}.csv")
stats_df.to_csv(stats_csv, index=False)
print(f"✔ Saved stats CSV: {stats_csv}")

out_path = os.path.join(OUT_DIR, f"mimic_iii_sparsified-patientwise_{args.pct}_{args.seed}.pkl")
with open(out_path, "wb") as f:
    pickle.dump([data_sub, oc_sub, train_sub, val_sub, test_sub], f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✔ Saved dataset: {out_path}")
print(f"✔ Dropped ts_ids: {len(dropped_ts_ids)} (no-window or <2 minutes)")
print(f"✔ Final valid ts_ids: {len(valid_ts_ids)}")
print("✔ Uses observed=True in groupby to avoid huge memory allocation")