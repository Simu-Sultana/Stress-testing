import argparse
import os
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# --------------------------------------------------------
# 1) Parse arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser(
    description="MIMIC-III sparsification grouped by (ts_id, variable) (GRU-D safe: enforce >=2 minutes after windowing)"
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

print("\n======================================================")
print(" SPARSIFICATION: GROUP BY (ts_id, variable) (GRU-D SAFE)")
print(" DATASET       = MIMIC-III")
print(" RAW_DATA_PATH =", RAW_DATA_PATH)
print(" OUT_DIR       =", OUT_DIR)
print(" PCT           =", args.pct)
print(" SEED          =", args.seed)
print("======================================================\n")

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

def compute_stats(df: pd.DataFrame):
    tp = df.groupby("ts_id", observed=True)["minute"].nunique()
    meas = df.groupby("ts_id", observed=True).size()
    return float(tp.mean()), float(meas.mean())

with open(os.path.join(RAW_DATA_PATH, "mimic_iii.pkl"), "rb") as f:
    data, oc, train_ids, val_ids, test_ids = pickle.load(f)

data = optimize_dtypes(data)

demo_features = [
    "Age", "Gender", "Height",
    "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"
]

demo = data[data["variable"].isin(demo_features)].copy()
ts   = data[~data["variable"].isin(demo_features)].copy()
ts = ts.sort_values(["ts_id", "minute"])

MAX_MINUTE = 880
ts_win = ts[ts["minute"] < MAX_MINUTE].copy()

avg_tp_before, avg_meas_before = compute_stats(ts_win)
print(
    f"BEFORE (windowed) sparsification: "
    f"avg_timepoints={avg_tp_before:.2f}, "
    f"avg_measurements={avg_meas_before:.2f}"
)

mins_per_id = ts_win.groupby("ts_id", observed=True)["minute"].nunique()
eligible_ids = mins_per_id[mins_per_id >= 2].index.to_numpy()

ts_win = ts_win[ts_win["ts_id"].isin(eligible_ids)].copy()

stats_records = []
keep_idx_parts = []
dropped_ts_ids = set()

for ts_id, g_id in ts_win.groupby("ts_id", sort=False):
    minutes_all = np.sort(g_id["minute"].unique())
    if minutes_all.size < 2:
        dropped_ts_ids.add(int(ts_id))
        continue

    local_keep = []

    for var, g in g_id.groupby("variable", sort=False, observed=True):
        rows_before = len(g)

        # pct-based target per (ts_id,var)
        target = int(np.ceil(rows_before * args.pct / 100.0))
        take = max(1, target)
        take = min(take, rows_before)

        # sample indices
        sampled_idx = g.sample(n=take, random_state=args.seed).index.to_numpy()
        local_keep.append(sampled_idx)

        stats_records.append({
            "ts_id": int(ts_id),
            "variable": str(var),
            "rows_before": int(rows_before),
            "rows_in_window": int(rows_before),
            "pct": int(args.pct),
            "rows_kept": int(len(sampled_idx)),
            "rows_removed": int(rows_before - len(sampled_idx)),
            "dropped_reason": "",
        })

    if not local_keep:
        dropped_ts_ids.add(int(ts_id))
        continue

    local_keep_idx = np.concatenate(local_keep)
    tmp = ts_win.loc[local_keep_idx]

    
    tmp2 = (
        tmp.groupby(["ts_id", "minute", "variable"], as_index=False, observed=True)["value"]
        .mean()
    )

    if tmp2["minute"].nunique() < 2:
        dropped_ts_ids.add(int(ts_id))
        continue

    
    keep_idx_parts.append(local_keep_idx)

if not keep_idx_parts:
    raise RuntimeError("No samples kept. Check MAX_MINUTE / pct / seed / dataset.")

keep_idx = np.concatenate(keep_idx_parts)
ts_sub_raw = ts_win.loc[keep_idx].copy()

ts_sub = (
    ts_sub_raw.groupby(["ts_id", "minute", "variable"], as_index=False, observed=True)["value"]
    .mean()
)

ts_sub = ts_sub[ts_sub["minute"] < MAX_MINUTE].copy()
ts_sub = optimize_dtypes(ts_sub)

mins_after = ts_sub.groupby("ts_id", observed=True)["minute"].nunique()
good_ids = mins_after[mins_after >= 2].index.to_numpy()
ts_sub = ts_sub[ts_sub["ts_id"].isin(good_ids)].copy()

valid_ts_ids = ts_sub["ts_id"].unique()

avg_tp_after, avg_meas_after = compute_stats(ts_sub)
print(
    f"AFTER (windowed) sparsification:  "
    f"avg_timepoints={avg_tp_after:.2f}, "
    f"avg_measurements={avg_meas_after:.2f}"
)

stats_df = pd.DataFrame(stats_records)
csv_path = os.path.join(
    OUT_DIR,
    f"mimic_iii_patient-variable_stats_{args.pct}_{args.seed}.csv"
)
stats_df.to_csv(csv_path, index=False)
print(f"\n✔ Saved per-(ts_id, variable) stats CSV:\n  {csv_path}")

demo_sub = demo[demo["ts_id"].isin(valid_ts_ids)].copy()
demo_sub = optimize_dtypes(demo_sub)

oc_sub = oc[oc["ts_id"].isin(valid_ts_ids)].copy()
oc_sub = optimize_dtypes(oc_sub)

train_sub = np.intersect1d(train_ids, valid_ts_ids)
val_sub   = np.intersect1d(val_ids, valid_ts_ids)
test_sub  = np.intersect1d(test_ids, valid_ts_ids)

data_sub = pd.concat([demo_sub, ts_sub], axis=0, copy=False, ignore_index=False)
data_sub = data_sub.sort_values(by=["ts_id", "minute"])

out_path = os.path.join(
    OUT_DIR,
    f"mimic_iii_sparsified-tsid-varid_{args.pct}_{args.seed}.pkl"
)

with open(out_path, "wb") as f:
    pickle.dump([data_sub, oc_sub, train_sub, val_sub, test_sub], f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\n✔ Saved sparsified dataset:\n  {out_path}")
print(f"✔ Dropped ts_ids during sampling safety: {len(dropped_ts_ids)}")
print(f"✔ Final valid ts_ids: {len(valid_ts_ids)}")
print("✔ Uses minute<MAX_MINUTE and >=2 unique minutes per ts_id (GRU-D safe)")
print("✔ Uses observed=True in groupby to avoid huge memory allocation")