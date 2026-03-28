from pathlib import Path
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR    = REPO_ROOT / "plots_metrics_cv"
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

EXPECTED_PCTS  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
EXPECTED_FOLDS = [0, 1, 2]

# One colour per model — used consistently in every panel
MODEL_COLORS = [
    "#E63946",  # red
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
]


# ── Path / filename parsing ────────────────────────────────────────────────────

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
        wanted  = [c for c in BASE_COLS if c in header] + test_cols
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
            rec["_source_csv"] = str(path)
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

keep_cols = [c for c in BASE_COLS if c in data.columns] + metric_cols + ["_source_csv"]
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

before = len(data)
data   = data.dropna(subset=metric_cols, how="all").copy()
print(f"Discarded rows with no test metrics: {before - len(data)}")

run_id   = ["dataset", "target", "model", "perturbation", "file", "pct", "seed", "fold"]
existing = [c for c in run_id if c in data.columns]
if existing:
    before = len(data)
    if "end_time" in data.columns:
        data = data.sort_values("end_time")
    data = data.drop_duplicates(subset=existing, keep="last")
    print(f"Removed duplicate rows: {before - len(data)}")

data["_x"] = data["pct"]


# ── Main plot loop ─────────────────────────────────────────────────────────────
for (ds, tgt, pert), dsub in data.groupby(["dataset", "target", "perturbation"]):
    dsub = dsub.dropna(subset=["_x"])
    if dsub.empty:
        continue

    present_x = [x for x in EXPECTED_PCTS if x in set(dsub["_x"])]
    if len(present_x) < 2:
        print(f"[INFO] Skipping {ds}|{tgt}|{pert} — fewer than 2 pct values.")
        continue

    model_names   = sorted(dsub["model"].dropna().unique())
    folds_present = sorted(dsub["fold"].dropna().unique())
    n_models      = len(model_names)
    n_folds       = len(folds_present)
    color_map     = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(model_names)}

    subdir = OUTDIR / ds / tgt / pert
    subdir.mkdir(parents=True, exist_ok=True)

    for metric in metric_cols:
        dplot = dsub.dropna(subset=[metric]).copy()
        if dplot.empty:
            continue

        ylabel = metric.replace("test_", "").upper()

        # ── Layout ────────────────────────────────────────────────────────────
        #
        # Row 0           : mean ± std across all folds  (summary)
        # Rows 1..n_folds : one row per fold  ← small multiples
        #
        # Every row has the same axes limits and the same lines (one per model),
        # so the reader can scan top-to-bottom and directly compare folds.

        n_rows  = 1 + n_folds          # summary row + one row per fold
        fig_h   = 3.2 * n_rows
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(10, fig_h),
            sharex=True, sharey=True,   # identical axes across all panels
        )
        if n_rows == 1:
            axes = [axes]

        # Global y-limits so every panel is directly comparable
        y_all  = dplot[metric].dropna()
        y_min  = y_all.min()
        y_max  = y_all.max()
        y_pad  = (y_max - y_min) * 0.10 or 0.05
        y_lim  = (y_min - y_pad, y_max + y_pad)

        # ── Row 0: summary — mean ± std across all folds ──────────────────────
        ax = axes[0]
        for model_name in model_names:
            m = (
                dplot[dplot["model"] == model_name]
                .groupby("_x")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            m["std"] = m["std"].fillna(0.0)
            m = m[m["_x"].isin(EXPECTED_PCTS)].sort_values("_x")
            color = color_map[model_name]
            ax.plot(m["_x"], m["mean"], marker="o", color=color,
                    linewidth=2.0, markersize=5, label=model_name)
            ax.fill_between(m["_x"],
                            m["mean"] - m["std"],
                            m["mean"] + m["std"],
                            color=color, alpha=0.18)

        ax.set_title("All folds — mean ± std", fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(y_lim)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_facecolor("#F0F4FA")   # slightly different bg to mark the summary

        # Legend once, in the summary panel
        handles = [mpatches.Patch(color=color_map[m], label=m) for m in model_names]
        ax.legend(handles=handles, title="Model", fontsize=8,
                  title_fontsize=8, framealpha=0.92, loc="best")

        # ── Rows 1..n_folds: one panel per fold ───────────────────────────────
        for fi, fold_val in enumerate(folds_present):
            ax = axes[fi + 1]
            fold_data = dplot[dplot["fold"] == fold_val]

            for model_name in model_names:
                mf = (
                    fold_data[fold_data["model"] == model_name]
                    .groupby("_x")[metric].mean()
                    .reset_index().sort_values("_x")
                )
                mf = mf[mf["_x"].isin(EXPECTED_PCTS)]
                if mf.empty:
                    continue
                ax.plot(mf["_x"], mf[metric],
                        marker="o", color=color_map[model_name],
                        linewidth=1.8, markersize=5)

            ax.set_title(f"Fold {int(fold_val)}", fontsize=10,
                         fontweight="bold", loc="left")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_ylim(y_lim)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.set_facecolor("#FAFAFA")

        # X-axis label only on the bottom panel
        axes[-1].set_xlabel("% of data", fontsize=9)
        axes[-1].set_xticks(EXPECTED_PCTS)
        axes[-1].set_xticklabels(EXPECTED_PCTS, rotation=45, fontsize=7)

        fig.suptitle(
            f"{ds}  |  target: {tgt}  |  perturbation: {pert}  |  metric: {metric}",
            fontsize=11, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        safe_metric = metric.replace("/", "_").replace("@", "_at_")
        outpath = subdir / f"{safe_metric}.png"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {outpath}")

print("\nDone. Saved plots under:", OUTDIR)