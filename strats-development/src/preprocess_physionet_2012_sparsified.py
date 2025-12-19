import argparse
from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np

# --------------------------------------------------------
# 1. Parse command-line arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser(description="Create PhysioNet 2012 subsets (downsampling train+val) for STRaTS")
parser.add_argument("--data_dir", type=str, required=True, help="Path to PhysioNet-2012 extracted folder")
parser.add_argument("--out_dir", type=str, required=True, help="Where to save subset PKL files")
parser.add_argument("--seed", type=int, required=True, help="Random seed")
parser.add_argument("--pct", type=int, required=True, help="percentage of time points to keep")
args = parser.parse_args()

RAW_DATA_PATH = args.data_dir.rstrip("/")
OUT_DIR = args.out_dir.rstrip("/")
os.makedirs(OUT_DIR, exist_ok=True)

print("\n==========================================")
print(" RAW_DATA_PATH =", RAW_DATA_PATH)
print(" OUT_DIR       =", OUT_DIR)
print("==========================================\n")


def compute_stats(df):
    # Number of time points per ts_id
    timepoints_per_ts = df.groupby("ts_id")["minute"].nunique()
    avg_timepoints = timepoints_per_ts.mean()
    
    # Total measurements per ts_id
    measurements_per_ts = df.groupby("ts_id").size()
    avg_measurements = measurements_per_ts.mean()
    
    return avg_timepoints, avg_measurements

# --------------------------------------------------------
# 2. Read pickle file
# --------------------------------------------------------
filepath = os.path.join(RAW_DATA_PATH, "physionet_2012.pkl")
with open(filepath, "rb") as f:
    data, oc, train_ids, val_ids, test_ids = pickle.load(f)

np.random.seed(args.seed)

pct = args.pct

# do not subsample static variables
demo_features = ['Age', 'Gender', 'Height', 'ICUType_1', 'ICUType_2', 'ICUType_3', 'ICUType_4']

demo = data[data["variable"].isin(demo_features)]
ts = data[~data["variable"].isin(demo_features)]

ts_sub = ts.sample(frac=pct/100, random_state=args.seed)

# --------------------------------------------------------
# 3. Print stats
# --------------------------------------------------------

# before subsampling
avg_tp_before, avg_meas_before = compute_stats(ts)
print(f"Before subsampling: avg time points per ts_id = {avg_tp_before:.2f}, "
        f"avg measurements per ts_id = {avg_meas_before:.2f}")

# After subsampling
avg_tp_after, avg_meas_after = compute_stats(ts_sub)
print(f"After subsampling:  avg time points per ts_id = {avg_tp_after:.2f}, "
        f"avg measurements per ts_id = {avg_meas_after:.2f}")


# retain only admissions with at least one temporal measurement
demo = demo[demo.ts_id.isin(ts_sub.ts_id)]
oc_sub = oc[oc.ts_id.isin(ts_sub.ts_id)]

train_sub = np.intersect1d(train_ids, ts_sub.ts_id)
val_sub   = np.intersect1d(val_ids, ts_sub.ts_id)
test_sub  = np.intersect1d(test_ids, ts_sub.ts_id)

data_sub = pd.concat([demo, ts_sub]).sort_values(by="ts_id").reset_index(drop=True)

# save
out_path = os.path.join(OUT_DIR, f"physionet_2012_sparsified_{pct}_{args.seed}.pkl")

with open(out_path, "wb") as f:
    pickle.dump(
        [
            data_sub,
            oc_sub,
            train_sub,
            val_sub,
            test_sub
        ],
        f
    )

print(f"   ✔ Saved: {out_path}")