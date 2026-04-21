import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ── MIMIC-III only ────────────────────────────────────────────────────────────

def coerce_mimic_ts_ids(data: pd.DataFrame, oc: pd.DataFrame):
    """
    Round and cast ts_id to plain numpy int64 in both data and oc.
    Using plain int64 (not pandas nullable Int64) so that after a pickle
    round-trip dataset.py's data.ts_id.isin(train_ids) always type-matches.
    PhysioNet never calls this.
    """
    def _to_int64(series: pd.Series, label: str) -> pd.Series:
        s = series.dropna()
        frac = np.nanmax(np.abs(s.values.astype(float) - np.round(s.values.astype(float)))) if len(s) else 0.0
        if frac > 1e-9:
            print(
                f"WARNING: {label}.ts_id has fractional part up to {frac:.2e}. "
                "Rounding before int conversion."
            )
        # Use float->round->int64 to handle both float64 and object columns.
        return pd.to_numeric(series, errors="coerce").round().astype("int64")

    data = data.copy()
    oc = oc.copy()
    data["ts_id"] = _to_int64(data["ts_id"], "data")
    oc["ts_id"]   = _to_int64(oc["ts_id"],   "oc")
    return data, oc


def compute_mimic_length_of_stay_from_data(data: pd.DataFrame, oc: pd.DataFrame) -> pd.DataFrame:
    if "minute" not in data.columns or "ts_id" not in data.columns:
        raise ValueError(
            f"MIMIC data must contain columns ['ts_id', 'minute']. Found: {list(data.columns)}"
        )
    if "ts_id" not in oc.columns:
        raise ValueError(f"oc must contain 'ts_id'. Found: {list(oc.columns)}")

    print("Computing LOS from data: max(minute) / (60*24) per ts_id ...")
    max_minute = (
        data[["ts_id", "minute"]]
        .dropna()
        .groupby("ts_id", sort=False)["minute"]
        .max()
    )
    los_days_by_id = max_minute / (60 * 24)
    los_days_by_id.name = "length_of_stay"

    oc = oc.copy()
    oc["length_of_stay"] = oc["ts_id"].map(los_days_by_id)

    missing = oc["length_of_stay"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} oc rows have missing length_of_stay (ts_id not found in data).")

    return oc


def load_mimic_pickle(path: Path):
    """
    Accept any of:
      [data, oc]
      [data, oc, train_ids, val_ids, test_ids]
      {"data": ..., "oc": ...}
    Returns (data, oc). Old split ids are discarded — we rebuild them.
    """
    raw = load_pickle(path)

    if isinstance(raw, dict):
        if "data" not in raw or "oc" not in raw:
            raise ValueError(f"MIMIC pickle dict must have keys 'data' and 'oc'. Got: {list(raw.keys())}")
        return raw["data"], raw["oc"]

    if isinstance(raw, (list, tuple)):
        if len(raw) >= 2:
            return raw[0], raw[1]
        raise ValueError(f"MIMIC pickle list/tuple must have at least 2 elements. Got: {len(raw)}")

    raise ValueError(f"Unrecognised MIMIC pickle type: {type(raw)}")


# ── PhysioNet only ────────────────────────────────────────────────────────────

def ensure_physionet_length_of_stay(oc: pd.DataFrame) -> pd.DataFrame:
    if "length_of_stay" not in oc.columns:
        raise ValueError(
            f"PhysioNet oc must already contain 'length_of_stay'. Found: {list(oc.columns)}"
        )
    return oc.copy()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _ids_as_str(ids) -> set:
    """String-convert ids for overlap checks only — never for saving."""
    return set(str(x) for x in ids)


def rebuild_splits_from_pickle_order(
    oc: pd.DataFrame,
    fold: int,
    train_frac: float = 0.8,
    seed: int = 123,
    stratify_col: str = "in_hospital_mortality",
):
    """
    Contiguous split by current pickle row order:
      fold 0 -> first  ~33% test
      fold 1 -> middle ~33% test
      fold 2 -> last   ~33% test
    Remaining ~67% -> stratified 80/20 train/val.

    Returns plain numpy int64 arrays for MIMIC-III (after coerce_mimic_ts_ids)
    or whatever dtype oc["ts_id"] has for PhysioNet — either way a plain numpy
    array that survives pickle round-trips and works with pandas isin().
    """
    if fold not in [0, 1, 2]:
        raise ValueError(f"fold must be 0, 1, or 2. Got: {fold}")

    oc = oc.reset_index(drop=True).copy()

    # Pull ids as a plain numpy array regardless of source dtype.
    # For MIMIC-III ts_id is already plain int64 after coerce_mimic_ts_ids().
    # np.asarray() on a pandas Series always gives a numpy array.
    ids = np.asarray(oc["ts_id"])
    y   = np.asarray(oc[stratify_col])
    n   = len(oc)

    bounds = np.linspace(0, n, 4, dtype=int)
    start  = bounds[fold]
    end    = bounds[fold + 1]

    test_idx      = np.arange(start, end)
    train_val_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])

    test_ids      = ids[test_idx]
    train_val_ids = ids[train_val_idx]
    y_train_val   = y[train_val_idx]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx_inner, val_idx_inner = next(sss.split(train_val_ids, y_train_val))

    train_ids = train_val_ids[train_idx_inner]
    val_ids   = train_val_ids[val_idx_inner]

    # Overlap check using string conversion (safe for any dtype).
    if (_ids_as_str(train_ids) & _ids_as_str(val_ids)) or \
       (_ids_as_str(train_ids) & _ids_as_str(test_ids)) or \
       (_ids_as_str(val_ids)   & _ids_as_str(test_ids)):
        raise ValueError("Overlap detected between train/val/test splits.")

    print(f"Using fold {fold} from current pickle order")
    print(f"  Test rows: [{start}:{end}) out of {n}")
    print(f"  Test size:      {len(test_ids)}")
    print(f"  Train size:     {len(train_ids)}")
    print(f"  Val size:       {len(val_ids)}")

    # Return plain numpy arrays — guaranteed to survive pickle round-trips and
    # work correctly with dataset.py's data.ts_id.isin(train_ids).
    return train_ids, val_ids, test_ids


def create_unbalanced_labels(oc: pd.DataFrame, train_ids, pct: int):
    if pct <= 0:
        raise ValueError("pct must be >= 1.")

    oc = oc.copy()
    lengths = oc.loc[oc["ts_id"].isin(train_ids), "length_of_stay"].values
    lengths = lengths[~np.isnan(lengths.astype(float))]

    if len(lengths) == 0:
        raise ValueError("No valid TRAIN length_of_stay values found.")

    T = np.percentile(lengths, 100 - pct)
    oc["unbalanced"] = (oc["length_of_stay"] > T).astype(int)

    return oc, float(T)


def print_split_stats(oc: pd.DataFrame, train_ids, val_ids, test_ids):
    for split_name, split_ids in zip(
        ["train", "val", "test"],
        [train_ids, val_ids, test_ids],
    ):
        split_oc = oc.loc[oc["ts_id"].isin(split_ids)]
        counts   = split_oc["unbalanced"].value_counts(normalize=True) * 100
        print(f"{split_name.upper()} split:")
        print(f"  n = {len(split_oc)}")
        print(f"  Class 0: {counts.get(0, 0):.2f}%,  Class 1: {counts.get(1, 0):.2f}%")


def append_threshold_csv(out_dir: Path, dataset: str, fold: int | None, pct: int, threshold_days: float):
    csv_path = out_dir / "unbalanced_los_days.csv"
    row = pd.DataFrame([{
        "dataset":           dataset,
        "fold":              "full" if fold is None else f"fold_{fold}",
        "perturbation":      "unbalanced",
        "pct":               int(pct),
        "los_threshold_days": float(threshold_days),
    }])
    row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    print(f"   Saved LOS threshold summary: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create unbalanced PKLs with custom contiguous 3-fold CV."
    )
    parser.add_argument("--dataset",    type=str,   required=True, choices=["physionet_2012", "mimic_iii"])
    parser.add_argument("--data_dir",   type=str,   required=True)
    parser.add_argument("--out_dir",    type=str,   required=True)
    parser.add_argument("--pct",        type=int,   required=True)
    parser.add_argument("--fold",       type=int,   required=True, help="0, 1, or 2")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed",       type=int,   default=123)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset
    pct     = args.pct
    fold    = args.fold

    in_file  = build_input_filename(dataset, fold)
    out_stem = build_output_stem(dataset, fold, pct)
    in_path  = data_dir / in_file
    out_path = out_dir  / f"{out_stem}.pkl"

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
        raise FileNotFoundError(f"Input pickle not found: {in_path}")

    # ── Load & prepare ────────────────────────────────────────────────────────
    if dataset == "mimic_iii":
        data, oc = load_mimic_pickle(in_path)
        # Coerce ts_id to plain int64 in both frames before anything else runs.
        # This ensures type consistency through all downstream isin() calls and
        # across the pickle round-trip into dataset.py.
        data, oc = coerce_mimic_ts_ids(data, oc)
        # Apply the same 24-hour filter that dataset.py applies at runtime.
        # Without this, patients whose temporal observations all fall outside
        # [0, 1440] minutes survive into the split lists but have zero rows
        # after dataset.py filters them, producing empty delta arrays and
        # crashing get_batch_grud() with "Bad delta shape".
        # Static variables (Age, Gender) have minute=0 so they are kept.
        data = data.loc[(data["minute"] >= 0) & (data["minute"] <= 24 * 60)].copy()
        print(f"After 24h filter: {data['ts_id'].nunique()} patients with data remaining")
        # FIX: exclude patients who have ONLY static variables (Age, Gender) and
        # no temporal observations. These patients pass the 24h filter because
        # static variables have minute=0, but they produce an empty delta array
        # in get_batch_grud() and crash with "Bad delta shape".
        static_vars = ["Age", "Gender"]
        temporal_ids = data.loc[~data["variable"].isin(static_vars), "ts_id"].unique()
        data = data.loc[data["ts_id"].isin(temporal_ids)].copy()
        oc   = oc.loc[oc["ts_id"].isin(temporal_ids)].copy()
        print(f"After temporal filter: {data['ts_id'].nunique()} patients with temporal data")
        oc = compute_mimic_length_of_stay_from_data(data, oc)
    else:
        # PhysioNet: unchanged from original.
        data, oc, _train_ids_old, _val_ids_old, _test_ids_old = load_pickle(in_path)
        oc = ensure_physionet_length_of_stay(oc)

    # ── Splits ────────────────────────────────────────────────────────────────
    train_ids, val_ids, test_ids = rebuild_splits_from_pickle_order(
        oc=oc,
        fold=fold,
        train_frac=args.train_frac,
        seed=args.seed,
        stratify_col="in_hospital_mortality",
    )

    # Quick sanity check — print ts_id dtype so any future mismatch is obvious.
    print(f"  data.ts_id dtype:  {data['ts_id'].dtype}")
    print(f"  train_ids dtype:   {train_ids.dtype}")
    print(f"  Spot-check isin: {data['ts_id'].isin(train_ids[:5]).sum()} rows match first 5 train ids")

    # ── Labels ────────────────────────────────────────────────────────────────
    oc, threshold_days = create_unbalanced_labels(oc, train_ids, pct)

    print(f"Threshold length of stay: {threshold_days:.4f} days")
    print_split_stats(oc, train_ids, val_ids, test_ids)
    append_threshold_csv(out_dir, dataset, fold, pct, threshold_days)

    save_pickle([data, oc, train_ids, val_ids, test_ids], out_path)
    print(f"   Saved: {out_path}")


if __name__ == "__main__":
    main()