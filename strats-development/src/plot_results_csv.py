from pathlib import Path
import csv
import re
import pandas as pd


REPO_ROOT    = Path(__file__).resolve().parent.parent
OUTDIR       = REPO_ROOT / "plots_metrics_cv"
OUTDIR.mkdir(exist_ok=True)

RESULTS_ROOT = REPO_ROOT / "results_cv"

csv_paths = list(RESULTS_ROOT.rglob("*.csv"))
print("RESULTS_ROOT:", RESULTS_ROOT)
print("Found CSVs  :", len(csv_paths))
if not csv_paths:
    raise SystemExit("No CSVs found under results_cv/.")


BASE_COLS = [
    "dataset", "target", "model", "perturbation",
    "file", "pct", "seed", "fold", "start_time", "end_time",
]

# Cols to read but exclude from the final output
DROP_COLS = {"output_dir", "output_dir_prefix", "load_ckpt_path"}

EXPECTED_PCTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]


def parse_path_metadata(path: Path) -> dict:
    parts = path.parts
    meta  = {}
    fold_idx = next(
        (i for i, p in enumerate(parts) if re.fullmatch(r"fold_\d+", p)), None
    )
    if fold_idx is not None:
        meta["fold"] = int(parts[fold_idx].split("_")[1])
        if fold_idx >= 4:
            meta["dataset"]      = parts[fold_idx - 4]
            meta["perturbation"] = parts[fold_idx - 3]
            meta["model"]        = parts[fold_idx - 2]
    fname_match = re.search(r"fold_(\d+)_([^_]+(?:_[^_]+)*)_(\d+)$", path.stem)
    if fname_match:
        meta.setdefault("fold", int(fname_match.group(1)))
        meta["target"] = fname_match.group(2)
        meta["pct"]    = int(fname_match.group(3))
    return meta


def safe_extract_csv(path: Path) -> pd.DataFrame | None:
    try:
        path_meta = parse_path_metadata(path)
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return None
        header    = [h.strip() for h in rows[0]]
        test_cols = [c for c in header if c.startswith("test_")]
        if not test_cols:
            return None
        extra  = ["train_frac"] + list(DROP_COLS)
        wanted = [c for c in BASE_COLS + extra if c in header] + test_cols
        col_idx = {col: idx for idx, col in enumerate(header)}
        extracted = []
        for row in rows[1:]:
            if not row:
                continue
            rec      = {}
            has_data = False
            for col in wanted:
                idx = col_idx.get(col)
                val = row[idx].strip() if idx is not None and idx < len(row) else None
                rec[col] = val
                if val not in (None, ""):
                    has_data = True
            if not has_data:
                continue
            for k, v in path_meta.items():
                if k not in rec or rec[k] in (None, ""):
                    rec[k] = v
            extracted.append(rec)
        return pd.DataFrame(extracted) if extracted else None
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


# ── Load & clean ───────────────────────────────────────────────────────────────
dfs = [df for p in csv_paths if (df := safe_extract_csv(p)) is not None and not df.empty]
if not dfs:
    raise SystemExit("No usable CSVs found.")

data = pd.concat(dfs, ignore_index=True)

if "fold" not in data.columns:
    data["fold"] = None

required = {"dataset", "target", "model", "perturbation", "pct"}
if missing := required - set(data.columns):
    raise SystemExit(f"Missing required columns: {missing}")

metric_cols = sorted(
    c for c in data.columns
    if c.startswith("test_") and pd.to_numeric(data[c], errors="coerce").notna().sum() > 0
)
print("Detected metric columns:", metric_cols)
if not metric_cols:
    raise SystemExit("No usable test_ metric columns found.")

# Filter to train_frac == 1.0 only
if "train_frac" in data.columns:
    data["train_frac"] = pd.to_numeric(data["train_frac"], errors="coerce")
    before = len(data)
    data = data[data["train_frac"] == 1.0].copy()
    print(f"Kept train_frac=1.0: {len(data)} rows (dropped {before - len(data)})")
else:
    print("[WARN] train_frac column not found — no filtering applied.")

# Keep only relevant columns, dropping train_frac and the three output cols
keep_cols = [c for c in BASE_COLS if c in data.columns and c not in DROP_COLS] + metric_cols
data = data[keep_cols].copy()

data["pct"]  = pd.to_numeric(data["pct"],  errors="coerce")
data["fold"] = pd.to_numeric(data["fold"], errors="coerce")
if "seed" in data.columns:
    data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
for c in metric_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
for tc in ("end_time", "start_time"):
    if tc in data.columns:
        data[tc] = pd.to_datetime(data[tc], errors="coerce")

data = data[data["pct"].isin(EXPECTED_PCTS)].copy()

missing_mask = data[metric_cols].isna().all(axis=1)
print(f"Rows with no test metrics (filled with 0.0): {missing_mask.sum()}")
data[metric_cols] = data[metric_cols].fillna(0.0)

# ── Save ───────────────────────────────────────────────────────────────────────
out_csv = OUTDIR / "all_results.csv"
data.to_csv(out_csv, index=False)
print(f"\nDone. CSV saved to: {out_csv} ({len(data)} rows)")