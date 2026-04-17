# --------------------------------------------------------
# Create MIMIC-III sparsified PKLs (drop % of observations)
# Follows EXACT format used for PhysioNet sparsified script.
#
# Output PKL format:
#   [data_sparse, oc, train_ids, valid_ids, test_ids]
# --------------------------------------------------------

import argparse
import os
import pickle
import numpy as np
import pandas as pd

# --------------------------------------------------------
# 1. Parse command-line arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser(description="Create sparsified MIMIC-III PKLs (STRaTS format)")
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing mimic_iii.pkl")
parser.add_argument("--out_dir", type=str, required=True, help="Directory to save sparsified PKLs")
parser.add_argument("--ratio", type=float, required=True,
                    help="Fraction of OBSERVATIONS to KEEP (e.g., 0.1 → keep 10% rows)")
parser.add_argument("--seed", type=int, required=True, help="Random seed")
args = parser.parse_args()

DATA_DIR = args.data_dir.rstrip("/")
OUT_DIR = args.out_dir.rstrip("/")
os.makedirs(OUT_DIR, exist_ok=True)

print("\n==========================================")
print(" RAW_DATA_PATH =", DATA_DIR)
print(" OUT_DIR       =", OUT_DIR)
print(" SPARSITY KEEP RATIO =", args.ratio)
print(" SEED          =", args.seed)
print("==========================================\n")

# --------------------------------------------------------
# 2. Load original MIMIC-III PKL
# --------------------------------------------------------
pkl_path = os.path.join(DATA_DIR, "mimic_iii.pkl")
print("Loading:", pkl_path)

data, oc, train_ids, valid_ids, test_ids = pickle.load(open(pkl_path, "rb"))

print("\n✔ Loaded mimic_iii.pkl:")
print(f"   data rows   : {len(data)}")
print(f"   oc rows     : {len(oc)}")
print(f"   train_ids   : {len(train_ids)}")
print(f"   valid_ids   : {len(valid_ids)}")
print(f"   test_ids    : {len(test_ids)}\n")

# --------------------------------------------------------
# 3. Sparsify time-series
# --------------------------------------------------------
np.random.seed(args.seed)
keep_ratio = args.ratio
pct = int(keep_ratio * 100)

print(f"➡ Creating sparsified dataset: mimic_iii_sparsified_{pct}_{args.seed}.pkl")
print("   Each patient time-series is reduced independently.\n")

data_sparse_list = []

# Group by time-series id (ts_id)
for ts_id, group in data.groupby("ts_id"):
    n = len(group)
    k = max(1, int(n * keep_ratio))  # keep at least 1 row
    
    idx_keep = np.random.choice(group.index, k, replace=False)
    data_sparse_list.append(group.loc[idx_keep])

data_sparse = pd.concat(data_sparse_list).reset_index(drop=True)

# --------------------------------------------------------
# 4. Print summary
# --------------------------------------------------------
print("📊 Sparsification summary:")
print(f"   Original rows : {len(data)}")
print(f"   Sparsified rows: {len(data_sparse)}")
print(f"   ≈ {pct}% rows kept\n")

# --------------------------------------------------------
# 5. Save output PKL
# --------------------------------------------------------
out_file = os.path.join(OUT_DIR, f"mimic_iii_sparsified_{pct}_{args.seed}.pkl")

pickle.dump(
    [
        data_sparse,
        oc,          # unchanged
        train_ids,   # unchanged
        valid_ids,   # unchanged
        test_ids     # unchanged
    ],
    open(out_file, "wb")
)

print("✔ Saved sparsified PKL →", out_file)
print("Done!\n")
