import argparse
import os
import pickle
import pandas as pd
import numpy as np

def short_type(x):
    return f"{type(x).__module__}.{type(x).__name__}"

def print_df_info(df: pd.DataFrame, name="DataFrame", max_cols=40):
    print(f"\n--- {name} ---")
    print("type:", short_type(df))
    print("shape:", df.shape)
    cols = list(df.columns)
    print("columns (first {}):".format(min(max_cols, len(cols))), cols[:max_cols])
    # show a few dtypes
    dtypes = df.dtypes.astype(str).to_dict()
    sample = list(dtypes.items())[:20]
    print("dtypes (first 20):", sample)
    # show a small head without flooding
    print("head(3):")
    print(df.head(3).to_string(index=False))

def inspect_obj(obj, prefix="obj", depth=0, max_depth=3):
    """
    Lightweight recursive inspection.
    """
    indent = "  " * depth
    print(f"{indent}{prefix}: {short_type(obj)}", end="")

    # Basic sizes
    if isinstance(obj, (list, tuple)):
        print(f" (len={len(obj)})")
        if depth < max_depth:
            for i, v in enumerate(obj[:10]):
                inspect_obj(v, prefix=f"{prefix}[{i}]", depth=depth+1, max_depth=max_depth)
            if len(obj) > 10:
                print(f"{indent}  ... (only first 10 shown)")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f" (keys={len(keys)})")
        # show first keys
        print(f"{indent}  keys (first 20): {keys[:20]}")
        if depth < max_depth:
            # inspect first few values
            for k in keys[:10]:
                inspect_obj(obj[k], prefix=f"{prefix}[{repr(k)}]", depth=depth+1, max_depth=max_depth)
            if len(keys) > 10:
                print(f"{indent}  ... (only first 10 key-values inspected)")
        return

    if isinstance(obj, pd.DataFrame):
        print(" [DataFrame]")
        return

    if isinstance(obj, np.ndarray):
        print(f" (shape={obj.shape}, dtype={obj.dtype})")
        return

    # fallback
    print("")

def main():
    parser = argparse.ArgumentParser(description="Inspect the contents of mimic_iii.pkl (no changes made).")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing mimic_iii.pkl")
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip("/")
    pkl_path = os.path.join(data_dir, "mimic_iii.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"❌ Not found: {pkl_path}")

    print("\n================= LOADING =================")
    print("PKL:", pkl_path)

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    print("\n================= TOP LEVEL =================")
    print("Top-level type:", short_type(obj))

    # Print shallow structure
    print("\n================= STRUCTURE (SHALLOW) =================")
    inspect_obj(obj, prefix="pkl", max_depth=2)

    # If classic structure: [data, oc, train_ids, val_ids, test_ids]
    if isinstance(obj, (list, tuple)) and len(obj) >= 5:
        print("\n================= DETECTED CLASSIC FORMAT =================")
        data = obj[0]
        oc = obj[1]
        train_ids = obj[2]
        val_ids = obj[3]
        test_ids = obj[4]

        print("data type:", short_type(data))
        print("oc type:", short_type(oc))
        print("train_ids type:", short_type(train_ids), "len:", len(train_ids) if hasattr(train_ids, "__len__") else "NA")
        print("val_ids type:", short_type(val_ids), "len:", len(val_ids) if hasattr(val_ids, "__len__") else "NA")
        print("test_ids type:", short_type(test_ids), "len:", len(test_ids) if hasattr(test_ids, "__len__") else "NA")

        if isinstance(oc, pd.DataFrame):
            print_df_info(oc, name="oc (outcomes/metadata)")

            # common id columns
            id_candidates = ["ts_id", "stay_id", "icustay_id", "hadm_id", "subject_id"]
            found_ids = [c for c in id_candidates if c in oc.columns]
            print("\nID columns found in oc:", found_ids if found_ids else "None")

            # LOS-related columns
            los_candidates = [
                "length_of_stay", "los", "los_days", "stay_length",
                "icu_los", "hospital_los", "los_icu", "los_hospital",
                "admittime", "dischtime", "intime", "outtime"
            ]
            found_los = [c for c in los_candidates if c in oc.columns]
            print("LOS/timestamp columns found in oc:", found_los if found_los else "None")

        # Inspect `data` for minute presence (common for fallback LOS)
        print("\n================= DATA QUICK CHECK =================")
        if isinstance(data, pd.DataFrame):
            print_df_info(data, name="data (time-series?)")
            print("\nDoes data have 'minute'? ->", "minute" in data.columns)
        elif isinstance(data, dict):
            keys = list(data.keys())
            print("data is dict with keys:", len(keys))
            # check first few entries
            checked = 0
            has_minute = 0
            for k in keys[:20]:
                v = data[k]
                checked += 1
                if isinstance(v, pd.DataFrame) and "minute" in v.columns:
                    has_minute += 1
            print(f"Checked first {checked} dict entries: {has_minute} had a DataFrame with 'minute' column.")
        else:
            print("data is not a DataFrame or dict. type:", short_type(data))

    else:
        print("\n================= NOTE =================")
        print("This PKL is NOT in the classic [data, oc, train_ids, val_ids, test_ids] format.")
        print("The shallow structure above should help identify where oc/data/splits are stored.")

if __name__ == "__main__":
    main()
