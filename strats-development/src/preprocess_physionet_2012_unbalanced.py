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
parser.add_argument("--pct", type=int, required=True, help="Percentage of admissions to be labeled 1 (length_of_stay > threshold)")
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
filepath = os.path.join(RAW_DATA_PATH, "physionet_2012.pkl")
with open(filepath, "rb") as f:
    data, oc, train_ids, val_ids, test_ids = pickle.load(f)

pct = args.pct

# compute threshold for length of stay such that pct% admissions have

lengths = oc[oc.ts_id.isin(train_ids)]["length_of_stay"].values
T = np.percentile(lengths, 100 - pct)
oc["unbalanced"] = (oc["length_of_stay"] > T).astype("int")

print(f"Threshold length of stay: {T:.2f} days")

for split_name, split_ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_oc = oc[oc.ts_id.isin(split_ids)]
    counts = split_oc["unbalanced"].value_counts(normalize=True) * 100
    print(f"{split_name.upper()} split:")
    print(f"  Class 0: {counts.get(0,0):.2f}%, Class 1: {counts.get(1,0):.2f}%")

# save
out_path = os.path.join(OUT_DIR, f"physionet_2012_unbalanced_{pct}.pkl")

with open(out_path, "wb") as f:
    pickle.dump(
        [
            data,
            oc,
            train_ids,
            val_ids,
            test_ids
        ],
        f
    )

print(f"   ✔ Saved: {out_path}")