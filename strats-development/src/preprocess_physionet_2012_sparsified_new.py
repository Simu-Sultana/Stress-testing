import argparse
import os
import pandas as pd
import pickle
import numpy as np

# --------------------------------------------------------
# 1. Parse arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser(
    description="PhysioNet-2012 patient-wise sparsification with CSV logging"
)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--pct", type=int, required=True)
args = parser.parse_args()

RAW_DATA_PATH = args.data_dir.rstrip("/")
OUT_DIR = args.out_dir.rstrip("/")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(args.seed)

print("\n==========================================")
print(" PATIENT-WISE SPARSIFICATION (CSV LOGGING)")
print(" RAW_DATA_PATH =", RAW_DATA_PATH)
print(" OUT_DIR       =", OUT_DIR)
print(" PCT           =", args.pct)
print(" SEED          =", args.seed)
print("==========================================\n")

# --------------------------------------------------------
# 2. Helper: compute dataset statistics
# --------------------------------------------------------
def compute_stats(df):
    tp = df.groupby("ts_id")["minute"].nunique()
    meas = df.groupby("ts_id").size()
    return tp.mean(), meas.mean()

# --------------------------------------------------------
# 3. Load dataset
# --------------------------------------------------------
with open(os.path.join(RAW_DATA_PATH, "physionet_2012.pkl"), "rb") as f:
    data, oc, train_ids, val_ids, test_ids = pickle.load(f)

# --------------------------------------------------------
# 4. Separate static vs temporal variables
# --------------------------------------------------------
demo_features = [
    "Age", "Gender", "Height",
    "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"
]

demo = data[data["variable"].isin(demo_features)]
ts = data[~data["variable"].isin(demo_features)]

# --------------------------------------------------------
# 5. Global statistics BEFORE
# --------------------------------------------------------
avg_tp_before, avg_meas_before = compute_stats(ts)
print(
    f"BEFORE sparsification: "
    f"avg_timepoints={avg_tp_before:.2f}, "
    f"avg_measurements={avg_meas_before:.2f}"
)

# --------------------------------------------------------
# 6. Patient-wise sparsification + logging
# --------------------------------------------------------
stats_records = []
ts_sub_parts = []

for ts_id, group in ts.groupby("ts_id"):
    rows_before = len(group)
    rows_kept = max(1, int(np.ceil(rows_before * args.pct / 100)))
    rows_removed = rows_before - rows_kept

    # sample rows
    sampled = group.sample(n=rows_kept, random_state=args.seed)

    ts_sub_parts.append(sampled)

    # log stats
    stats_records.append({
        "ts_id": ts_id,
        "rows_before": rows_before,
        "pct": args.pct,
        "rows_kept": rows_kept,
        "rows_removed": rows_removed
    })

# concatenate sparsified time series
ts_sub = pd.concat(ts_sub_parts, ignore_index=True)

# --------------------------------------------------------
# 7. Global statistics AFTER
# --------------------------------------------------------
avg_tp_after, avg_meas_after = compute_stats(ts_sub)
print(
    f"AFTER sparsification:  "
    f"avg_timepoints={avg_tp_after:.2f}, "
    f"avg_measurements={avg_meas_after:.2f}"
)

# --------------------------------------------------------
# 8. Save per-patient CSV
# --------------------------------------------------------
stats_df = pd.DataFrame(stats_records)

csv_path = os.path.join(
    OUT_DIR,
    f"physionet_2012_patientwise_stats_{args.pct}_{args.seed}.csv"
)
stats_df.to_csv(csv_path, index=False)

print(f"\n✔ Saved per-patient stats CSV:")
print(f"  {csv_path}")

# --------------------------------------------------------
# 9. Keep only valid admissions
# --------------------------------------------------------
valid_ts_ids = ts_sub.ts_id.unique()

demo = demo[demo.ts_id.isin(valid_ts_ids)]
oc_sub = oc[oc.ts_id.isin(valid_ts_ids)]

train_sub = np.intersect1d(train_ids, valid_ts_ids)
val_sub   = np.intersect1d(val_ids, valid_ts_ids)
test_sub  = np.intersect1d(test_ids, valid_ts_ids)

# --------------------------------------------------------
# 10. Merge final dataset
# --------------------------------------------------------
data_sub = (
    pd.concat([demo, ts_sub])
    .sort_values(by=["ts_id", "minute"])
    .reset_index(drop=True)
)

# --------------------------------------------------------
# 11. Save sparsified dataset
# --------------------------------------------------------
out_path = os.path.join(
    OUT_DIR,
    f"physionet_2012_sparsified_patientwise_{args.pct}_{args.seed}.pkl"
)

with open(out_path, "wb") as f:
    pickle.dump(
        [data_sub, oc_sub, train_sub, val_sub, test_sub],
        f
    )

print(f"\n✔ Saved sparsified dataset:")
print(f"  {out_path}")
