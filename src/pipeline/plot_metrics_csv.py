import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import csv
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
OUTDIR = REPO_ROOT / "aggregated"
OUTDIR.mkdir(exist_ok=True)

csv_paths = list(RESULTS_ROOT.rglob("*.csv"))
print("RESULTS_ROOT:", RESULTS_ROOT)
print("Found CSVs:", len(csv_paths))
if not csv_paths:
    raise SystemExit("No CSVs found under results/. Check your results folder path.")


BASE_COLS = [
    "dataset", "target", "model", "perturbation",
    "file", "pct", "seed", "start_time", "end_time"
]

EXPECTED_PCTS = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def safe_extract_csv(path: Path) -> pd.DataFrame | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return None

        header = [h.strip() for h in rows[0]]
        if not header:
            return None

        wanted_cols = [c for c in BASE_COLS if c in header]
        test_cols = [c for c in header if c.startswith("test_")]

        if not test_cols:
            return None

        wanted_cols += test_cols

        col_to_idx = {}
        for idx, col in enumerate(header):
            if col not in col_to_idx:
                col_to_idx[col] = idx

        extracted_rows = []
        for row in rows[1:]:
            if not row:
                continue

            record = {}
            has_any_data = False

            for col in wanted_cols:
                idx = col_to_idx[col]
                val = row[idx].strip() if idx < len(row) else None
                record[col] = val
                if val not in (None, ""):
                    has_any_data = True

            if not has_any_data:
                continue

            record["_source_csv"] = str(path)
            extracted_rows.append(record)

        if not extracted_rows:
            return None

        return pd.DataFrame(extracted_rows)

    except Exception as e:
        print(f"[WARN] Failed to read CSV: {path} ({e})")
        return None


dfs = []
for p in csv_paths:
    df = safe_extract_csv(p)
    if df is not None and not df.empty:
        dfs.append(df)

if not dfs:
    raise SystemExit("No usable CSVs with test metrics were found.")

data = pd.concat(dfs, ignore_index=True)

required = {"dataset", "target", "model", "perturbation", "file", "pct"}
missing = required - set(data.columns)
if missing:
    raise SystemExit(f"Missing required columns in extracted data: {missing}")

# Skip sparsified patientwise perturbations
before_skip = len(data)
data = data[data["perturbation"].str.lower().str.strip() != "sparsified-patientwise"].copy()
print(f"Skipped sparsified patientwise rows: {before_skip - len(data)}")

metric_cols = []
for c in data.columns:
    if c.startswith("test_"):
        s = pd.to_numeric(data[c], errors="coerce")
        if s.notna().sum() > 0:
            metric_cols.append(c)

metric_cols = sorted(metric_cols)
print("Detected metric columns:", metric_cols)

if not metric_cols:
    raise SystemExit("No usable test_ metric columns found.")

keep_cols = [c for c in BASE_COLS if c in data.columns] + metric_cols
data = data[keep_cols].copy()

# Convert types
data["pct"] = pd.to_numeric(data["pct"], errors="coerce")
if "seed" in data.columns:
    data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
for c in metric_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
if "end_time" in data.columns:
    data["end_time"] = pd.to_datetime(data["end_time"], errors="coerce")
if "start_time" in data.columns:
    data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")

# Keep only expected pct values
data = data[data["pct"].isin(EXPECTED_PCTS)].copy()

# Drop rows with no usable metrics
before_metric_filter = len(data)
data = data.dropna(subset=metric_cols, how="all").copy()
print(f"Discarded rows with no usable test metrics: {before_metric_filter - len(data)}")

# Only drop truly duplicate runs (same dataset, target, model, perturbation, pct, seed, file)
group_cols = ["dataset", "target", "model", "perturbation", "pct", "seed", "file"]
existing_group_cols = [c for c in group_cols if c in data.columns]

before = len(data)
if "end_time" in data.columns:
    data = data.sort_values("end_time")
data = data.drop_duplicates(subset=existing_group_cols, keep="last")
print(f"Removed truly duplicate runs: {before - len(data)}")

outpath = OUTDIR / "aggregated_results.csv"
data.to_csv(outpath, index=False)
print(f"Done. Saved CSV to: {outpath}")
print(f"Shape: {data.shape}")