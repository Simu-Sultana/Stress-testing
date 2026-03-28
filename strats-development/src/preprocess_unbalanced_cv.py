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


# ── MIMIC-III: normalise ts_id to clean integers in-place ────────────────────
# FIX: centralise all ts_id coercion here so every downstream function sees
#      consistent integer ts_ids.  PhysioNet never calls this.
def coerce_mimic_ts_ids(data: pd.DataFrame, oc: pd.DataFrame):
    """
    Round-then-int-cast ts_id in both `data` and `oc` (in place on copies).
    Returns (data_fixed, oc_fixed).
    """
    def _to_int64(series: pd.Series, label: str) -> pd.Series:
        s = series.dropna()
        frac = np.nanmax(np.abs(s.values - np.round(s.values))) if len(s) else 0.0
        if frac > 1e-9:
            print(
                f"⚠ WARNING: {label}.ts_id has fractional part up to {frac:.2e}. "
                "Rounding before int conversion."
            )
        return np.round(series).astype("int64")  # plain numpy int64 — survives pickle round-trip

    data = data.copy()
    oc = oc.copy()
    data["ts_id"] = _to_int64(data["ts_id"], "data")
    oc["ts_id"]   = _to_int64(oc["ts_id"],   "oc")
    return data, oc


def compute_mimic_length_of_stay_from_data(data: pd.DataFrame, oc: pd.DataFrame) -> pd.DataFrame:
    if "minute" not in data.columns or "ts_id" not in data.columns:
        raise ValueError(
            f"❌ MIMIC data must contain columns ['ts_id', 'minute']. Found: {list(data.columns)}"
        )
    if "ts_id" not in oc.columns:
        raise ValueError(f"❌ oc must contain 'ts_id'. Found: {list(oc.columns)}")

    # FIX: ts_ids are already coerced integers (done in main before this call),
    #      so the groupby key and the oc.ts_id map key are always the same type.
    print("Computing LOS from data: max(minute)/1440 per ts_id ...")
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
        print(f"⚠ WARNING: {missing} oc rows have missing length_of_stay (ts_id not found in data).")

    return oc


def ensure_physionet_length_of_stay(oc: pd.DataFrame) -> pd.DataFrame:
    if "length_of_stay" not in oc.columns:
        raise ValueError(
            f"❌ PhysioNet oc must already contain 'length_of_stay'. Found: {list(oc.columns)}"
        )
    return oc.copy()


def normalize_ids(ids):
    # Used ONLY for internal overlap checks — never for saved output.
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
    Remaining 67% -> stratified 80/20 train/val.

    Returns ts_id arrays in the SAME dtype as oc["ts_id"] so that
    dataset.py's data.ts_id.isin(train_ids) always gets a type match.
    """
    if fold not in [0, 1, 2]:
        raise ValueError(f"❌ fold must be 0, 1, or 2. Got: {fold}")

    oc = oc.reset_index(drop=True).copy()

    # Preserve the original dtype (Int64 for MIMIC-III, str/int for PhysioNet).
    # Do NOT convert to strings here — dataset.py matches ids against data.ts_id
    # using isin(), which is dtype-sensitive.
    ids = oc["ts_id"].values
    y = oc[stratify_col].values
    n = len(oc)

    bounds = np.linspace(0, n, 4, dtype=int)
    start = bounds[fold]
    end = bounds[fold + 1]

    test_idx = np.arange(start, end)
    train_val_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])

    test_ids     = ids[test_idx]
    train_val_ids = ids[train_val_idx]
    y_train_val  = y[train_val_idx]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx_inner, val_idx_inner = next(sss.split(train_val_ids, y_train_val))

    train_ids = train_val_ids[train_idx_inner]
    val_ids   = train_val_ids[val_idx_inner]

    # Overlap check — use string conversion only here so Int64 NA values don't
    # cause set-comparison issues.
    if (set(normalize_ids(train_ids)) & set(normalize_ids(val_ids))) or \
       (set(normalize_ids(train_ids)) & set(normalize_ids(test_ids))) or \
       (set(normalize_ids(val_ids))   & set(normalize_ids(test_ids))):
        raise ValueError("❌ Overlap detected between train/val/test splits.")

    print(f"Using fold {fold} from current pickle order")
    print(f"  Test rows kept aside: [{start}:{end}) out of {n}")
    print(f"  Test size: {len(test_ids)}")
    print(f"  Remaining for train/val: {len(train_val_ids)}")

    # Return arrays in the native ts_id dtype — this is what gets saved to the
    # pickle and what dataset.py will call .isin() against.
    return train_ids, val_ids, test_ids


def create_unbalanced_labels(oc: pd.DataFrame, train_ids, pct: int):
    if pct <= 0:
        raise ValueError("❌ pct must be >= 1.")

    oc = oc.copy()
    # Use isin() directly — train_ids is already the same dtype as oc["ts_id"]
    # (Int64 for MIMIC-III, str/int for PhysioNet) so the match is type-safe.
    lengths = oc.loc[oc["ts_id"].isin(train_ids), "length_of_stay"].values
    lengths = lengths[~np.isnan(lengths.astype(float))]

    if len(lengths) == 0:
        raise ValueError("❌ No valid TRAIN length_of_stay values found.")

    T = np.percentile(lengths, 100 - pct)
    oc["unbalanced"] = (oc["length_of_stay"] > T).astype(int)

    return oc, float(T)


def print_split_stats(oc: pd.DataFrame, train_ids, val_ids, test_ids):
    # Use isin() directly — ids are in native ts_id dtype, no string conversion needed.
    for split_name, split_ids in zip(
        ["train", "val", "test"],
        [train_ids, val_ids, test_ids]
    ):
        split_oc = oc.loc[oc["ts_id"].isin(split_ids)]
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


# ── MIMIC-III: flexible pickle loading ───────────────────────────────────────
# FIX: the original code blindly unpacked 5 elements, which crashes if the
#      MIMIC-III full pickle is [data, oc] or any other shape.
def load_mimic_pickle(path: Path):
    """
    Accept any of:
      [data, oc]
      [data, oc, train_ids, val_ids, test_ids]
      {"data": ..., "oc": ...}   (dict form)
    Returns (data, oc).  Old split ids are discarded — we rebuild them.
    """
    raw = load_pickle(path)

    if isinstance(raw, dict):
        if "data" not in raw or "oc" not in raw:
            raise ValueError(f"❌ MIMIC pickle dict must have keys 'data' and 'oc'. Got: {list(raw.keys())}")
        return raw["data"], raw["oc"]

    if isinstance(raw, (list, tuple)):
        if len(raw) >= 2:
            return raw[0], raw[1]
        raise ValueError(f"❌ MIMIC pickle list/tuple must have at least 2 elements. Got: {len(raw)}")

    raise ValueError(f"❌ Unrecognised MIMIC pickle type: {type(raw)}")


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

    # ── Load ─────────────────────────────────────────────────────────────────
    if dataset == "mimic_iii":
        # FIX: use flexible loader instead of hardcoded 5-tuple unpack
        data, oc = load_mimic_pickle(in_path)
        # FIX: coerce ts_ids to clean integers in both frames before anything
        #      else touches them — this fixes the "1234.0" vs "1234" mismatch
        #      that silently emptied splits and LOS lookups.
        data, oc = coerce_mimic_ts_ids(data, oc)
        oc = compute_mimic_length_of_stay_from_data(data, oc)
    else:
        # PhysioNet: unchanged
        data, oc, _train_ids_old, _val_ids_old, _test_ids_old = load_pickle(in_path)
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