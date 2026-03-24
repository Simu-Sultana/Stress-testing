from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "plots_per_duration"
OUTDIR.mkdir(exist_ok=True)

RESULTS_ROOT = REPO_ROOT / "results"

csv_paths = list(RESULTS_ROOT.rglob("*.csv"))
print("RESULTS_ROOT:", RESULTS_ROOT)
print("Found CSVs:", len(csv_paths))
if not csv_paths:
    raise SystemExit("No CSVs found under results/. Check your results folder path.")


BASE_COLS = [
    "dataset", "target", "model", "perturbation",
    "file", "pct", "seed", "device",
    "start_time", "end_time"
]

EXPECTED_PCTS = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

DURATION_CANDIDATES = [
    "duration_in_seconds",
    "duration_seconds",
    "duration_sec",
    "duration",
    "runtime_in_seconds",
    "runtime_seconds",
    "runtime_sec",
    "runtime"
]


def format_duration(seconds: float) -> str:
    """Convert seconds into a readable short label."""
    if pd.isna(seconds):
        return ""
    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


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
        found_duration_cols = [c for c in DURATION_CANDIDATES if c in header]

        has_timestamps = ("start_time" in header and "end_time" in header)
        if not found_duration_cols and not has_timestamps:
            return None

        wanted_cols += found_duration_cols[:1]

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
    raise SystemExit("No usable CSVs with duration or timestamp information were found.")

data = pd.concat(dfs, ignore_index=True)

required = {"dataset", "target", "model", "perturbation", "file", "pct"}
missing = required - set(data.columns)
if missing:
    raise SystemExit(f"Missing required columns in extracted data: {missing}")

duration_col = None
for c in DURATION_CANDIDATES:
    if c in data.columns:
        duration_col = c
        break

if duration_col is not None:
    data["runtime_seconds"] = pd.to_numeric(data[duration_col], errors="coerce")
    print(f"Using duration column: {duration_col}")
else:
    data["runtime_seconds"] = pd.NA
    print("No explicit duration column found. Will compute from end_time - start_time.")

if "start_time" in data.columns:
    data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")
if "end_time" in data.columns:
    data["end_time"] = pd.to_datetime(data["end_time"], errors="coerce")

if "start_time" in data.columns and "end_time" in data.columns:
    missing_runtime = data["runtime_seconds"].isna()
    data.loc[missing_runtime, "runtime_seconds"] = (
        data.loc[missing_runtime, "end_time"] - data.loc[missing_runtime, "start_time"]
    ).dt.total_seconds()

data["pct"] = pd.to_numeric(data["pct"], errors="coerce")
if "seed" in data.columns:
    data["seed"] = pd.to_numeric(data["seed"], errors="coerce")

data["runtime_seconds"] = pd.to_numeric(data["runtime_seconds"], errors="coerce")
data = data[data["pct"].isin(EXPECTED_PCTS)].copy()

if "device" in data.columns:
    before_device = len(data)
    data["device"] = data["device"].astype(str).str.strip().str.lower()
    cuda_mask = data["device"].str.contains("cuda", na=False)
    if cuda_mask.any():
        data = data[cuda_mask].copy()
        print(f"Filtered to CUDA runs: {before_device} -> {len(data)} rows")
    else:
        print("[INFO] No rows explicitly marked as CUDA. Keeping all rows.")
else:
    data["device"] = "cuda"
    print("[INFO] No device column found. Assuming all runs used CUDA.")

before_runtime_filter = len(data)
data = data.dropna(subset=["runtime_seconds"]).copy()
after_runtime_filter = len(data)
print(f"Discarded rows with no usable runtime: {before_runtime_filter - after_runtime_filter}")

before_positive_filter = len(data)
data = data[data["runtime_seconds"] > 0].copy()
after_positive_filter = len(data)
print(f"Discarded rows with non-positive runtime: {before_positive_filter - after_positive_filter}")

if data.empty:
    raise SystemExit("No usable runtime rows remain after filtering.")

run_id_cols = ["dataset", "target", "model", "perturbation", "file", "pct", "seed"]
existing_run_id_cols = [c for c in run_id_cols if c in data.columns]

if existing_run_id_cols:
    before = len(data)
    if "end_time" in data.columns:
        data = data.sort_values("end_time")
        data = data.drop_duplicates(subset=existing_run_id_cols, keep="last")
    else:
        data = data.drop_duplicates(subset=existing_run_id_cols, keep="last")
    after = len(data)
    print(f"Removed duplicate run rows: {before - after}")

data["_x"] = data["pct"]
data["_x_name"] = "% of data"

# convert to hours for plotting
data["runtime_hours"] = data["runtime_seconds"] / 3600.0


def agg_mean_std_count(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "_x"]).agg(
        mean_hours=("runtime_hours", "mean"),
        std_hours=("runtime_hours", "std"),
        count=("runtime_hours", "count"),
        mean_seconds=("runtime_seconds", "mean")
    ).reset_index()

    g["std_hours"] = g["std_hours"].fillna(0.0)
    return g


for (ds, tgt, pert), dsub in data.groupby(["dataset", "target", "perturbation"]):
    dsub = dsub.dropna(subset=["_x", "runtime_hours"]).copy()

    if dsub.empty:
        continue

    present_x = [x for x in EXPECTED_PCTS if x in set(dsub["_x"].tolist())]
    if len(present_x) < 2:
        print(f"[INFO] Skipping {ds} | {tgt} | {pert} because fewer than 2 valid pct values were found.")
        continue

    subdir = OUTDIR / ds / tgt / pert
    subdir.mkdir(parents=True, exist_ok=True)

    g = agg_mean_std_count(dsub)
    if g.empty:
        continue

    g = g[g["_x"].isin(EXPECTED_PCTS)].copy()
    g["_x"] = pd.Categorical(g["_x"], categories=EXPECTED_PCTS, ordered=True)
    g = g.sort_values("_x")
    g["_x"] = g["_x"].astype(float)

    print(f"\n[{ds} | {tgt} | {pert} | runtime_hours]")
    print(g[["model", "_x", "count", "mean_hours", "std_hours"]].to_string(index=False))

    plt.figure(figsize=(9, 5))

    for model_name in sorted(g["model"].dropna().unique()):
        m = g[g["model"] == model_name].sort_values("_x")

        plt.plot(m["_x"], m["mean_hours"], marker="o", label=model_name)
        plt.fill_between(
            m["_x"],
            m["mean_hours"] - m["std_hours"],
            m["mean_hours"] + m["std_hours"],
            alpha=0.2
        )

        # annotate each point with readable duration
        for _, row in m.iterrows():
            label = format_duration(row["mean_seconds"])
            plt.annotate(
                label,
                (row["_x"], row["mean_hours"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8
            )

    plt.xlabel("% of data")
    plt.ylabel("Duration (hours)")
    plt.xticks(EXPECTED_PCTS, EXPECTED_PCTS)

    plt.figtext(
        0.5,
        -0.03,
        f"{ds} | {tgt} | {pert} | device=cuda",
        ha="center",
        fontsize=10
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    outpath = subdir / "runtime_hours.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

print("Done. Saved plots under:", OUTDIR)