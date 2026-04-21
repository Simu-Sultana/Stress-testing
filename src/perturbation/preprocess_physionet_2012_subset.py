import argparse
from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np

# --------------------------------------------------------
# 1. Parse command-line arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser(description="Create PhysioNet 2012 subsets (90%→10%) for STRaTS")
parser.add_argument("--data_dir", type=str, required=True, help="Path to PhysioNet-2012 extracted folder")
parser.add_argument("--out_dir", type=str, required=True, help="Where to save subset PKL files")
args = parser.parse_args()

RAW_DATA_PATH = args.data_dir.rstrip("/")
OUT_DIR = args.out_dir.rstrip("/")
os.makedirs(OUT_DIR, exist_ok=True)

print("\n==========================================")
print(" RAW_DATA_PATH =", RAW_DATA_PATH)
print(" OUT_DIR       =", OUT_DIR)
print("==========================================\n")


# --------------------------------------------------------
# 2. Original PhysioNet functions
# --------------------------------------------------------
def read_ts(raw_data_path, set_name):
    ts = []
    folder = f"{raw_data_path}/set-{set_name}"

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    pbar = tqdm(os.listdir(folder), desc=f"Reading time series set {set_name}")
    for f in pbar:
        df = pd.read_csv(f"{folder}/{f}").iloc[1:]
        df = df.loc[df.Parameter.notna()]
        if len(df) <= 5:
            continue
        df = df.loc[df.Value >= 0]   # remove negative → missing
        df["RecordID"] = f[:-4]
        ts.append(df)

    ts = pd.concat(ts, ignore_index=True)

    # convert HH:MM → minutes from admission
    ts["Time"] = ts["Time"].astype(str)
    ts["minute"] = ts["Time"].apply(lambda x: int(x[:2])*60 + int(x[3:]))

    ts.rename(columns={
        "Parameter": "variable",
        "Value": "value",
        "RecordID": "ts_id"
    }, inplace=True)

    return ts[["ts_id", "minute", "variable", "value"]]


def read_outcomes(raw_data_path, set_name):
    oc = pd.read_csv(
        f"{raw_data_path}/Outcomes-{set_name}.txt",
        usecols=["RecordID", "Length_of_stay", "In-hospital_death"]
    )
    oc["subset"] = set_name
    oc["ts_id"] = oc.RecordID.astype(str)

    oc.rename(columns={
        "Length_of_stay": "length_of_stay",
        "In-hospital_death": "in_hospital_mortality"
    }, inplace=True)

    return oc[["ts_id", "length_of_stay", "in_hospital_mortality", "subset"]]


# --------------------------------------------------------
# 3. Load full dataset
# --------------------------------------------------------
print("Loading full dataset...\n")

ts = pd.concat([read_ts(RAW_DATA_PATH, s) for s in ["a", "b", "c"]], ignore_index=True)
oc = pd.concat([read_outcomes(RAW_DATA_PATH, s) for s in ["a", "b", "c"]], ignore_index=True)

# remove outcomes without a time-series file
valid_ids = ts.ts_id.unique()
oc = oc.loc[oc.ts_id.isin(valid_ids)]

ts.drop_duplicates(inplace=True)

# --------------------------------------------------------
# 4. Convert ICUType categorical to one-hot style
# --------------------------------------------------------
mask = ts.variable == "ICUType"
for v in [4, 3, 2, 1]:
    idx = mask & (ts.value == v)
    ts.loc[idx, "variable"] = f"ICUType_{v}"

ts.loc[mask, "value"] = 1


# --------------------------------------------------------
# 5. Generate STRaTS-style splits
# --------------------------------------------------------
train_valid_ids = list(oc.loc[oc.subset != "a"].ts_id)  # B + C sets
test_ids = list(oc.loc[oc.subset == "a"].ts_id)         # A set

np.random.seed(123)
np.random.shuffle(train_valid_ids)

bp = int(0.8 * len(train_valid_ids))
full_train_ids = train_valid_ids[:bp]
valid_ids = train_valid_ids[bp:]

oc = oc.drop(columns="subset")   # remove helper column

# ---------------------------------------------------
# CREATE SUBSET PKLS — SAMPLE ONLY TRAIN + VAL
# Test set stays fixed (as in STRaTS paper)
# ---------------------------------------------------

ratios = [i/10 for i in range(9, 0, -1)]   # 0.9, 0.8, ..., 0.1
print("\nGenerating subsets (train+val downsampled, test fixed):", ratios)

np.random.seed(123)

# Convert lists to numpy arrays
train_ids = np.array(full_train_ids)   # FIXED
valid_ids = np.array(valid_ids)
test_ids  = np.array(test_ids)         # fixed test split (subset A)

# Combine train + val
full_trainval = np.concatenate([train_ids, valid_ids])
total_tv = len(full_trainval)


for r in ratios:
    pct = int(r * 100)

    # Number of training+validation samples to keep
    keep_n = int(total_tv * r)
    keep_ids = np.random.choice(full_trainval, keep_n, replace=False)

    print(f"\n➡ Creating physio_{pct}.pkl  |  keeping {keep_n}/{total_tv} train+val stays")

    # Filter time-series + outcomes
    ts_sub = ts[ts.ts_id.isin(np.concatenate([keep_ids, test_ids]))].reset_index(drop=True)
    oc_sub = oc[oc.ts_id.isin(np.concatenate([keep_ids, test_ids]))].reset_index(drop=True)

    # Split back into train/val while keeping the same logic
    train_sub = np.array([i for i in train_ids if i in keep_ids])
    val_sub   = np.array([i for i in valid_ids if i in keep_ids])
    test_sub  = test_ids.copy()   # test is unchanged

    # ---------------------------------------------
    # REPORT HOW MANY SAMPLES USED IN THIS SUBSET
    # ---------------------------------------------
    orig_train = len(train_ids)
    orig_valid = len(valid_ids)
    orig_test  = len(test_ids)

    new_train = len(train_sub)
    new_valid = len(val_sub)
    new_test  = len(test_sub)

    print("\n📊 Subset composition:")
    print(f"   Original train: {orig_train} → Subset train: {new_train}")
    print(f"   Original valid: {orig_valid} → Subset valid: {new_valid}")
    print(f"   Original test : {orig_test}  → Subset test:  {new_test} (unchanged)")
    print(f"   TOTAL in PKL  : {new_train + new_valid + new_test}")


    out_path = f"../data/processed/physio_{pct}.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(
            [
                ts_sub,
                oc_sub,
                train_sub,
                val_sub,
                test_sub
            ],
            f
        )

    print(f"   ✔ Saved: {out_path}")