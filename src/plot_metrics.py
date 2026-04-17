from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent   # src/.. = repo root
OUTDIR = REPO_ROOT / "plots_per_metric"
OUTDIR.mkdir(exist_ok=True)

RESULTS_ROOT = REPO_ROOT / "results"

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
    """
    Read CSV row-by-row.
    Keep only:
      - base columns
      - all columns starting with test_
    Skip config-only CSVs with no test_ columns.
    """
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

keep_cols = [c for c in BASE_COLS if c in data.columns] + metric_cols + ["_source_csv"]
data = data[keep_cols].copy()

# Convert numeric columns
data["pct"] = pd.to_numeric(data["pct"], errors="coerce")
if "seed" in data.columns:
    data["seed"] = pd.to_numeric(data["seed"], errors="coerce")

for c in metric_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

if "end_time" in data.columns:
    data["end_time"] = pd.to_datetime(data["end_time"], errors="coerce")
if "start_time" in data.columns:
    data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")

# Keep only your intended percentages
data = data[data["pct"].isin(EXPECTED_PCTS)].copy()

# Drop rows that have no usable test metric at all
before_metric_filter = len(data)
data = data.dropna(subset=metric_cols, how="all").copy()
after_metric_filter = len(data)
print(f"Discarded rows with no usable test metrics: {before_metric_filter - after_metric_filter}")

# Keep latest duplicate run
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

# X-axis stays strictly pct
data["_x"] = data["pct"]
data["_x_name"] = "% of data"


def agg_mean_std_count(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["model", "_x"])[metric].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


for (ds, tgt, pert), dsub in data.groupby(["dataset", "target", "perturbation"]):
    dsub = dsub.dropna(subset=["_x"])

    if dsub.empty:
        continue

    present_x = [x for x in EXPECTED_PCTS if x in set(dsub["_x"].tolist())]
    if len(present_x) < 2:
        print(f"[INFO] Skipping {ds} | {tgt} | {pert} because fewer than 2 valid pct values were found.")
        continue

    subdir = OUTDIR / ds / tgt / pert
    subdir.mkdir(parents=True, exist_ok=True)

    for metric in metric_cols:
        dplot = dsub.dropna(subset=[metric]).copy()
        if dplot.empty:
            continue

        g = agg_mean_std_count(dplot, metric)
        if g.empty:
            continue

        g = g[g["_x"].isin(EXPECTED_PCTS)].copy()
        g["_x"] = pd.Categorical(g["_x"], categories=EXPECTED_PCTS, ordered=True)
        g = g.sort_values("_x")
        g["_x"] = g["_x"].astype(float)

        print(f"\n[{ds} | {tgt} | {pert} | {metric}]")
        print(g[["model", "_x", "count"]].to_string(index=False))

        plt.figure(figsize=(6, 4))

        for model_name in sorted(g["model"].dropna().unique()):
            m = g[g["model"] == model_name].sort_values("_x")
            plt.plot(m["_x"], m["mean"], marker="o", label=model_name)
            plt.fill_between(
                m["_x"],
                m["mean"] - m["std"],
                m["mean"] + m["std"],
                alpha=0.2
            )

        ylabel = metric.replace("test_", "").upper()

        plt.xlabel("% of data")
        plt.ylabel(ylabel)
        plt.xticks(EXPECTED_PCTS, EXPECTED_PCTS)
        plt.figtext(
            0.5,
            -0.02,
            f"{ds} | {tgt} | {pert} | {metric}",
            ha="center",
            fontsize=10
        )

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        safe_metric = metric.replace("/", "_").replace("@", "_at_")
        outpath = subdir / f"{safe_metric}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()

print("Done. Saved plots under:", OUTDIR)