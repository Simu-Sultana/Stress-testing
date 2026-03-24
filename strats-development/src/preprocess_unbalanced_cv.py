import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def build_input_filename(dataset: str, fold: int | None) -> str:
    # Keep PhysioNet behavior unchanged.
    # For MIMIC-III, always use the full pickle and build contiguous folds here.
    if dataset == "mimic_iii":
        return f"{dataset}.pkl"

    if fold is None:
        return f"{dataset}.pkl"
    return f"{dataset}_fold_{fold}.pkl"


def build_output_stem(dataset: str, fold: int | None, pct: int) -> str:
    if fold is None:
        return f"{dataset}_unbalanced_{pct}"
    return f"{dataset}_fold_{fold}_unbalanced_{pct}"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def compute_mimic_length_of_stay_from_data(data: pd.DataFrame, oc: pd.DataFrame) -> pd.DataFrame:
    if "minute" not in data.columns or "ts_id" not in data.columns:
        raise ValueError(
            f"❌ MIMIC data must contain columns ['ts_id', 'minute']. Found: {list(data.columns)}"
        )

    if "ts_id" not in oc.columns:
        raise ValueError(f"❌ oc must contain 'ts_id'. Found: {list(oc.columns)}")

    tmp = data[["ts_id", "minute"]].dropna().copy()

    ts = tmp["ts_id"].values
    frac = np.nanmax(np.abs(ts - np.round(ts)))
    if frac > 1e-9:
        print(f"⚠ WARNING: data.ts_id has fractional part up to {frac}. Rounding before int conversion.")
        tmp["ts_id"] = np.round(tmp["ts_id"]).astype(np.int64)
    else:
        tmp["ts_id"] = tmp["ts_id"].astype(np.int64)

    print("Computing LOS from data: max(minute)/1440 per ts_id ...")
    max_minute = tmp.groupby("ts_id", sort=False)["minute"].max()
    los_days_by_id = max_minute / (60 * 24)
    los_days_by_id.name = "length_of_stay"

    oc = oc.copy()
    oc["length_of_stay"] = oc["ts_id"].map(los_days_by_id)

    missing = oc["length_of_stay"].isna().sum()
    if missing > 0:
        print(f"⚠ WARNING: {missing} oc rows have missing length_of_stay (ts_id not found in data).")

    return oc


def ensure_physionet_length_of_stay(oc: pd.DataFrame) -> pd.DataFrame:
    if "length_of_stay" not in oc.columns:
        raise ValueError(
            f"❌ PhysioNet oc must already contain 'length_of_stay'. Found: {list(oc.columns)}"
        )
    return oc.copy()


def normalize_ids(ids):
    return np.array([str(x) for x in ids], dtype=str)


def rebuild_splits_from_pickle_order(
    oc: pd.DataFrame,
    fold: int,
    train_frac: float = 0.8,
    seed: int = 123,
    stratify_col: str = "in_hospital_mortality",
):
    """
    Contiguous split by current pickle row order:
      fold 0 -> first 33% test
      fold 1 -> middle 33% test
      fold 2 -> last 33% test
    Remaining 67% -> stratified 80/20 train/val
    Returns ts_id strings.
    """
    if fold not in [0, 1, 2]:
        raise ValueError(f"❌ fold must be 0, 1, or 2. Got: {fold}")

    oc = oc.reset_index(drop=True).copy()
    ids = oc["ts_id"].astype(str).values
    y = oc[stratify_col].values
    n = len(oc)

    bounds = np.linspace(0, n, 4, dtype=int)
    start = bounds[fold]
    end = bounds[fold + 1]

    test_idx = np.arange(start, end)
    train_val_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])

    test_ids = ids[test_idx]
    train_val_ids = ids[train_val_idx]
    y_train_val = y[train_val_idx]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx_inner, val_idx_inner = next(sss.split(train_val_ids, y_train_val))

    train_ids = train_val_ids[train_idx_inner]
    val_ids = train_val_ids[val_idx_inner]

    if (set(train_ids) & set(val_ids)) or (set(train_ids) & set(test_ids)) or (set(val_ids) & set(test_ids)):
        raise ValueError("❌ Overlap detected between train/val/test splits.")

    print(f"Using fold {fold} from current pickle order")
    print(f"  Test rows kept aside: [{start}:{end}) out of {n}")
    print(f"  Test size: {len(test_ids)}")
    print(f"  Remaining for train/val: {len(train_val_ids)}")

    return normalize_ids(train_ids), normalize_ids(val_ids), normalize_ids(test_ids)


def create_unbalanced_labels(oc: pd.DataFrame, train_ids, pct: int):
    if pct <= 0:
        raise ValueError("❌ pct must be >= 1.")

    train_ids = set(normalize_ids(train_ids))
    oc = oc.copy()
    oc_ts = oc["ts_id"].astype(str)

    lengths = oc.loc[oc_ts.isin(train_ids), "length_of_stay"].values
    lengths = lengths[~np.isnan(lengths)]

    if len(lengths) == 0:
        raise ValueError("❌ No valid TRAIN length_of_stay values found.")

    T = np.percentile(lengths, 100 - pct)
    oc["unbalanced"] = (oc["length_of_stay"] > T).astype(int)

    return oc, float(T)


def print_split_stats(oc: pd.DataFrame, train_ids, val_ids, test_ids):
    oc_ts = oc["ts_id"].astype(str)

    for split_name, split_ids in zip(
        ["train", "val", "test"],
        [train_ids, val_ids, test_ids]
    ):
        split_ids = set(normalize_ids(split_ids))
        split_oc = oc.loc[oc_ts.isin(split_ids)]
        counts = split_oc["unbalanced"].value_counts(normalize=True) * 100

        print(f"{split_name.upper()} split:")
        print(f"  n = {len(split_oc)}")
        print(f"  Class 0: {counts.get(0,0):.2f}%, Class 1: {counts.get(1,0):.2f}%")


def append_threshold_csv(out_dir: Path, dataset: str, fold: int | None, pct: int, threshold_days: float):
    csv_path = out_dir / "unbalanced_los_days.csv"

    row = pd.DataFrame([{
        "dataset": dataset,
        "fold": "full" if fold is None else f"fold_{fold}",
        "perturbation": "unbalanced",
        "pct": int(pct),
        "los_threshold_days": float(threshold_days)
    }])

    row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    print(f"   ✔ Saved LOS days summary: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create unbalanced PKLs with custom contiguous 3-fold CV."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["physionet_2012", "mimic_iii"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pct", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True, help="0, 1, or 2")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset
    pct = args.pct
    fold = args.fold

    in_file = build_input_filename(dataset, fold)
    out_stem = build_output_stem(dataset, fold, pct)

    in_path = data_dir / in_file
    out_path = out_dir / f"{out_stem}.pkl"

    print("\n==========================================")
    print(" DATASET       =", dataset)
    print(" DATA_DIR      =", data_dir)
    print(" OUT_DIR       =", out_dir)
    print(" FOLD          =", fold)
    print(" PCT           =", pct)
    print(" INPUT FILE    =", in_file)
    print(" OUTPUT FILE   =", out_path.name)
    print("==========================================\n")

    if not in_path.exists():
        raise FileNotFoundError(f"❌ Input pickle not found: {in_path}")

    data, oc, _train_ids_old, _val_ids_old, _test_ids_old = load_pickle(in_path)

    if dataset == "mimic_iii":
        oc = compute_mimic_length_of_stay_from_data(data, oc)
    else:
        oc = ensure_physionet_length_of_stay(oc)

    train_ids, val_ids, test_ids = rebuild_splits_from_pickle_order(
        oc=oc,
        fold=fold,
        train_frac=args.train_frac,
        seed=args.seed,
        stratify_col="in_hospital_mortality",
    )

    oc, threshold_days = create_unbalanced_labels(oc, train_ids, pct)

    print(f"Threshold length of stay: {threshold_days:.4f} days")
    print_split_stats(oc, train_ids, val_ids, test_ids)
    append_threshold_csv(out_dir, dataset, fold, pct, threshold_days)

    save_pickle([data, oc, train_ids, val_ids, test_ids], out_path)
    print(f"   ✔ Saved: {out_path}")


if __name__ == "__main__":
    main()
    