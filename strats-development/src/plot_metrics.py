from pathlib import Path
import re
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

dfs = []
for p in csv_paths:
    try:
        df = pd.read_csv(p)
        df["_source_csv"] = str(p)
        dfs.append(df)
    except Exception as e:
        print(f"[WARN] Skipping unreadable CSV: {p} ({e})")

if not dfs:
    raise SystemExit("All CSVs failed to read. Check permissions/corruption.")

data = pd.concat(dfs, ignore_index=True)

required = {"dataset", "target", "model", "perturbation", "file"}
missing = required - set(data.columns)
if missing:
    raise SystemExit(f"Missing required columns in CSVs: {missing}")

metric_cols = []
for c in data.columns:
    if c.startswith("test_"):
        s = pd.to_numeric(data[c], errors="coerce")
        if s.notna().sum() > 0:
            metric_cols.append(c)

metric_cols = sorted(metric_cols)
print("Detected metric columns:", len(metric_cols))
if not metric_cols:
    raise SystemExit("No metric columns detected (expected columns starting with test_ or val_).")

def add_x(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pct" in df.columns:
        df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
        if df["pct"].notna().sum() > 0 and df["pct"].nunique(dropna=True) > 1:
            df["_x"] = df["pct"]
            df["_x_name"] = "% of data"
            return df

    
    def extract_last_num(s):
        if not isinstance(s, str):
            return None
        nums = re.findall(r"(\d+)", s)
        return int(nums[-1]) if nums else None

    df["_x"] = pd.to_numeric(df["file"].apply(extract_last_num), errors="coerce")
    df["_x_name"] = "level"
    return df

def agg_mean_std(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["model", "_x"])[metric].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g

for (ds, tgt, pert), dsub in data.groupby(["dataset", "target", "perturbation"]):
    dsub = add_x(dsub)
    dsub = dsub.dropna(subset=["_x"])

   
    if dsub["_x"].nunique() < 2:
        continue

    subdir = OUTDIR / ds / tgt / pert
    subdir.mkdir(parents=True, exist_ok=True)

    for metric in metric_cols:
        if metric not in dsub.columns:
            continue

        y = pd.to_numeric(dsub[metric], errors="coerce")
        if y.notna().sum() == 0:
            continue

        dplot = dsub.copy()
        dplot[metric] = y
        dplot = dplot.dropna(subset=[metric])
        if dplot.empty:
            continue

        g = agg_mean_std(dplot, metric)

        plt.figure(figsize=(6, 4))
        for model_name in sorted(g["model"].unique()):
            m = g[g["model"] == model_name].sort_values("_x")
            plt.plot(m["_x"], m["mean"], marker="o", label=model_name)
            plt.fill_between(m["_x"], m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.2)

        xlabel = str(dplot["_x_name"].iloc[0])
        ylabel = metric.replace("test_", "").replace("val_", "").upper()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.figtext(
            0.5,                # x position (center)
            -0.02,              # y position (slightly below plot)
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
