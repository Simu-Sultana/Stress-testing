from pathlib import Path
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = REPO_ROOT / "results_cv"

DATASETS = ["physionet_2012", "mimic_iii"]
TARGET = "unbalanced"   # this is now part of the folder path
MODELS = ["gru", "grud", "tcn", "sand", "strats"]

# only one perturbation/folder name in your new structure
PERTURBATION = "unbalanced"

# folds from your structure: fold_0, fold_1, fold_2
FOLDS = [0, 1, 2]

# percentages seen in your screenshot / runs
PCTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]


# =========================================================
# BUILD EXPECTED RUN NAME
# Example:
#   physionet_2012_fold_0_unbalanced_10
# CSV:
#   physionet_2012_fold_0_unbalanced_10.csv
# =========================================================
def make_run_name(dataset, fold, pct):
    return f"{dataset}_fold_{fold}_unbalanced_{pct}"


# =========================================================
# CHECK MISSING FILES
# =========================================================
missing = []
found = []

for dataset in DATASETS:
    for model in MODELS:
        for fold in FOLDS:
            for pct in PCTS:
                run_name = make_run_name(dataset, fold, pct)

                csv_path = (
                    RESULTS_ROOT
                    / dataset
                    / TARGET
                    / model
                    / PERTURBATION
                    / f"fold_{fold}"
                    / run_name
                    / f"{run_name}.csv"
                )

                item = (dataset, TARGET, model, PERTURBATION, fold, pct, csv_path)

                if csv_path.exists():
                    found.append(item)
                else:
                    missing.append(item)


# =========================================================
# PRINT SUMMARY
# =========================================================
print("=" * 100)
print(f"RESULTS_ROOT         : {RESULTS_ROOT}")
print(f"Found combinations   : {len(found)}")
print(f"Missing combinations : {len(missing)}")
print("=" * 100)


# =========================================================
# GROUPED SUMMARY
# =========================================================
grouped = defaultdict(list)
for dataset, target, model, perturbation, fold, pct, csv_path in missing:
    grouped[(dataset, target, model, perturbation)].append((fold, pct))

print("\n" + "=" * 100)
print("GROUPED SUMMARY")
print("=" * 100)

for key in sorted(grouped):
    dataset, target, model, perturbation = key
    pairs = sorted(grouped[key])
    print(f"\n{dataset} | {target} | {model} | {perturbation}")
    print(f"Missing count: {len(pairs)}")
    print(", ".join([f"(fold={fold}, pct={pct})" for fold, pct in pairs]))


# =========================================================
# SAVE REPORT
# =========================================================
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
out_file = RESULTS_ROOT / "missing_csv_report_cv.txt"

with open(out_file, "w") as f:
    f.write(f"RESULTS_ROOT         : {RESULTS_ROOT}\n")
    f.write(f"Found combinations   : {len(found)}\n")
    f.write(f"Missing combinations : {len(missing)}\n\n")

    f.write("MISSING COMBINATIONS\n")
    f.write("=" * 100 + "\n")
    for dataset, target, model, perturbation, fold, pct, csv_path in missing:
        f.write(
            f"{dataset} | {target} | {model} | {perturbation} | "
            f"fold={fold} | pct={pct} | expected={csv_path}\n"
        )

    f.write("\nGROUPED SUMMARY\n")
    f.write("=" * 100 + "\n")
    for key in sorted(grouped):
        dataset, target, model, perturbation = key
        pairs = sorted(grouped[key])
        f.write(f"\n{dataset} | {target} | {model} | {perturbation}\n")
        f.write(f"Missing count: {len(pairs)}\n")
        f.write(", ".join([f"(fold={fold}, pct={pct})" for fold, pct in pairs]) + "\n")

print(f"\nSaved report to: {out_file}")
