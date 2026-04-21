import pandas as pd
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH  = REPO_ROOT / "plots_metrics_cv" / "all_results.csv"

df = pd.read_csv(CSV_PATH)

DATASETS = ["physionet_2012", "mimic_iii"]
MODELS   = ["gru", "grud", "tcn", "sand", "strats"]
FOLDS    = [0, 1, 2]
PCTS     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]

all_combos = pd.DataFrame(
    list(product(DATASETS, MODELS, FOLDS, PCTS)),
    columns=["dataset", "model", "fold", "pct"]
)

df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
df["pct"]  = pd.to_numeric(df["pct"],  errors="coerce")

present = df[["dataset", "model", "fold", "pct"]].drop_duplicates()

missing = all_combos.merge(present, on=["dataset", "model", "fold", "pct"], how="left", indicator=True)
missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")

print(f"\nExpected : {len(all_combos)}")
print(f"Present  : {len(present)}")
print(f"Missing  : {len(missing)}")

print("\n── Missing combinations ──────────────────────────────")
print(missing.sort_values(["dataset", "model", "fold", "pct"]).to_string(index=False))

print("\n── Missing by dataset/model ──────────────────────────")
print(missing.groupby(["dataset", "model"]).size().reset_index(name="missing_count").to_string(index=False))