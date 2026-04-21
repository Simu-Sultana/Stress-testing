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
parser.add_argument("--pct", type=int, required=True, help="Percentage of train+val to keep, e.g., 0.9")
args = parser.parse_args()

RAW_DATA_PATH = args.data_dir.rstrip("/")
OUT_DIR = args.out_dir.rstrip("/")
os.makedirs(OUT_DIR, exist_ok=True)

print("\n==========================================")
print(" RAW_DATA_PATH =", RAW_DATA_PATH)
print(" OUT_DIR       =", OUT_DIR)
print("==========================================\n")


# READ PICKLE FILE
filepath = os.path.join(RAW_DATA_PATH, "physionet_2012.pkl")
ts, oc, train_ids, val_ids, test_ids = pickle.load(open(filepath, "rb"))

np.random.seed(args.seed)

pct = args.pct
r = pct / 100

#for seed in range(100)
#ratios = [i/10 for i in range(9, 0, -1)]   # 0.9, 0.8, ..., 0.1
#print("\nGenerating subsets (train+val downsampled, test fixed):", ratios)
#    for r in ratios:

######## SUBSAMPLE TRAIN AND VALIDATION SEPARATELY

# Number of training+validation samples to keep
train_sub = np.random.choice(train_ids, int(len(train_ids) * r), replace=False) 
val_sub   = np.random.choice(val_ids,   int(len(val_ids)   * r), replace=False)
keep_ids  = np.concatenate([train_sub, val_sub])

print(f"\n➡ Creating physio_{pct}.pkl | keeping {len(keep_ids)}/{len(train_ids) + len(val_ids)} train+val")

# Filter time-series + outcomes
ts_sub = ts[ts.ts_id.isin(np.concatenate([keep_ids, test_ids]))].reset_index(drop=True)
oc_sub = oc[oc.ts_id.isin(np.concatenate([keep_ids, test_ids]))].reset_index(drop=True)

# ---------------------------------------------
# REPORT HOW MANY SAMPLES USED IN THIS SUBSET
# ---------------------------------------------
orig_train = len(train_ids)
orig_valid = len(val_ids)
orig_test  = len(test_ids)

new_train = len(train_sub)
new_valid = len(val_sub)
new_test  = orig_test

print("\n📊 Subset composition:")
print(f"   Original train: {orig_train} → Subset train: {new_train}")
print(f"   Original valid: {orig_valid} → Subset valid: {new_valid}")
print(f"   Original test : {orig_test}  → Subset test:  {new_test} (unchanged)")
print(f"   TOTAL in PKL  : {new_train + new_valid + new_test}")


out_path = os.path.join(OUT_DIR, f"physionet_2012_subsampled_{pct}_{args.seed}.pkl")

with open(out_path, "wb") as f:
    pickle.dump(
        [
            ts_sub,
            oc_sub,
            train_sub,
            val_sub,
            test_ids
        ],
        f
    )

print(f"   ✔ Saved: {out_path}")