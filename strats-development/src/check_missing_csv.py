from pathlib import Path
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = REPO_ROOT / "results"

DATASETS = ["physionet_2012", "mimic_iii"]
TARGET = "in_hospital_mortality"
MODELS = ["gru", "grud", "tcn", "sand", "strats"]
PERTURBATIONS = ["subsampled", "sparsified-tsid-varid"]
PCTS = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEEDS = list(range(10))


# =========================================================
# BUILD EXPECTED RUN NAME
# Assumption:
#   run folder name = "<dataset>_<perturbation>_<pct>_<seed>"
#   csv file name    = "<dataset>_<perturbation>_<pct>_<seed>.csv"
#
# Example:
#   mimic_iii_subsampled_70_5/mimic_iii_subsampled_70_5.csv
# =========================================================
def make_run_name(dataset, perturbation, pct, seed):
    return f"{dataset}_{perturbation}_{pct}_{seed}"


# =========================================================
# CHECK MISSING FILES
# =========================================================
missing = []
found = []

for dataset in DATASETS:
    for model in MODELS:
        for perturbation in PERTURBATIONS:
            for pct in PCTS:
                for seed in SEEDS:
                    run_name = make_run_name(dataset, perturbation, pct, seed)

                    csv_path = (
                        RESULTS_ROOT
                        / dataset
                        / TARGET
                        / model
                        / perturbation
                        / run_name
                        / f"{run_name}.csv"
                    )

                    item = (dataset, TARGET, model, perturbation, pct, seed, csv_path)

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
for dataset, target, model, perturbation, pct, seed, csv_path in missing:
    grouped[(dataset, target, model, perturbation)].append((pct, seed))

print("\n" + "=" * 100)
print("GROUPED SUMMARY")
print("=" * 100)

for key in sorted(grouped):
    dataset, target, model, perturbation = key
    pairs = sorted(grouped[key])
    print(f"\n{dataset} | {target} | {model} | {perturbation}")
    print(f"Missing count: {len(pairs)}")
    print(", ".join([f"({pct},{seed})" for pct, seed in pairs]))

# =========================================================
# SAVE REPORT
# =========================================================
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
out_file = RESULTS_ROOT / "missing_csv_report.txt"

with open(out_file, "w") as f:
    f.write(f"RESULTS_ROOT         : {RESULTS_ROOT}\n")
    f.write(f"Found combinations   : {len(found)}\n")
    f.write(f"Missing combinations : {len(missing)}\n\n")

    f.write("MISSING COMBINATIONS\n")
    f.write("=" * 100 + "\n")
    for dataset, target, model, perturbation, pct, seed, csv_path in missing:
        f.write(
            f"{dataset} | {target} | {model} | {perturbation} | "
            f"pct={pct} | seed={seed} | expected={csv_path}\n"
        )

    f.write("\nGROUPED SUMMARY\n")
    f.write("=" * 100 + "\n")
    for key in sorted(grouped):
        dataset, target, model, perturbation = key
        pairs = sorted(grouped[key])
        f.write(f"\n{dataset} | {target} | {model} | {perturbation}\n")
        f.write(f"Missing count: {len(pairs)}\n")
        f.write(", ".join([f"({pct},{seed})" for pct, seed in pairs]) + "\n")

print(f"\nSaved report to: {out_file}")