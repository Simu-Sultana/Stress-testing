import argparse
import os
import pandas as pd
import pickle
import numpy as np

# --------------------------------------------------------
# 1. Parse command-line arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser(description="Create MIMIC-III unbalanced subsets (LOS target) for STRaTS")
parser.add_argument("--data_dir", type=str, required=True, help="Path to MIMIC-III processed folder")
parser.add_argument("--out_dir", type=str, required=True, help="Where to save output PKL + CSV files")
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
# 3. Check LOS exists (must be inside oc)
# --------------------------------------------------------
if "length_of_stay" not in oc.columns:
    raise ValueError(
        "❌ 'length_of_stay' column not found in oc.\n"
        "This mimic_iii.pkl does not contain LOS, so unbalanced LOS target cannot be created.\n"
        "Fix: add/compute LOS and save it into oc as column 'length_of_stay'.\n"
        f"oc columns found: {list(oc.columns)[:20]} ..."
    )

# --------------------------------------------------------
# 4. Compute threshold T (days) so that pct% of TRAIN admissions are class 1
# --------------------------------------------------------
lengths = oc[oc.ts_id.isin(train_ids)]["length_of_stay"].values
T = np.percentile(lengths, 100 - pct)
oc["unbalanced"] = (oc["length_of_stay"] > T).astype(int)

print(f"Threshold length of stay: {T:.2f} days")

for split_name, split_ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
    split_oc = oc[oc.ts_id.isin(split_ids)]
    counts = split_oc["unbalanced"].value_counts(normalize=True) * 100
    print(f"{split_name.upper()} split:")
    print(f"  Class 0: {counts.get(0,0):.2f}%, Class 1: {counts.get(1,0):.2f}%")

# --------------------------------------------------------
# 5. Save ONE CSV: the full oc table
# --------------------------------------------------------
oc_csv_path = os.path.join(OUT_DIR, f"mimic_iii_unbalanced_{pct}_oc.csv")
oc.to_csv(oc_csv_path, index=False)
print(f"   ✔ Saved oc CSV: {oc_csv_path}")

# --------------------------------------------------------
# 6. Save ONE PKL: unchanged structure
# --------------------------------------------------------
out_path = os.path.join(OUT_DIR, f"mimic_iii_unbalanced_{pct}.pkl")

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
