# Stress-testing Temporal Deep Learning Models on Sparse and Irregular Clinical Time Series

This repository contains code to preprocess clinical time-series datasets, generate perturbed `.pkl` files, train multiple temporal deep learning models, and analyze their robustness under controlled perturbation settings.

The project focuses on stress-testing temporal models under the following perturbations:
- **Subsampled**
- **Sparsified (patientwise)**
- **Sparsified (tsid-varid)**
- **Unbalanced**

### Supported datasets
- **MIMIC-III**
- **PhysioNet 2012**

### Supported models
- **GRU**
- **GRU-D**
- **TCN**
- **SaND**
- **STraTS**

---

## 1. Environment Setup

### Recommended Python version
- Python 3.10.9

### Main dependencies
This project uses:
- PyTorch with CUDA 11.7
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn
- Transformers

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Install PyTorch first:

```bash
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

Then install the remaining packages:

```bash
pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 pytz==2023.3.post1 tqdm==4.66.1 matplotlib==3.7.5 scikit-learn==1.2.2 transformers==4.35.2
```

You can also keep the dependency list in `requirement.txt`.

---

## 2. Repository Structure

The repository is organized around three main output locations:

```text
.
├── data/
│   └── processed/
├── results/
├── plots_per_metric/
├── src/
├── requirement.txt
├── folder_structure.txt
└── .gitignore
```

### Main folder purpose
- `data/processed/` stores the base dataset `.pkl` files and all generated perturbed `.pkl` files
- `results/` stores experiment outputs such as CSV files, logs, checkpoints, and run-specific results
- `plots_per_metric/` stores metric-wise plots generated from the experiment results
- `src/` contains preprocessing scripts, training code, model implementations, and plotting utilities

---

## 3. Dataset Orientation and Where to Put the `.pkl` Files

Before running anything, place the **base processed dataset files** inside:

```text
data/processed/
```

Expected base files:

```text
data/processed/mimic_iii.pkl
data/processed/physionet_2012.pkl
```

### Important note
These two files act as the **starting point** for preprocessing.

All perturbation scripts use:
- `--data_dir data/processed`
- `--out_dir data/processed`

This means:
1. the scripts **read the original/base `.pkl` files from `data/processed/`**
2. they also **write the newly generated perturbed `.pkl` files back into `data/processed/`**

### In simple words
- Put `mimic_iii.pkl` and `physionet_2012.pkl` in `data/processed/`
- Run the preprocessing scripts
- The generated perturbation-specific `.pkl` files will also appear in `data/processed/`
- Training will then load the required file by name using the `--file` argument

---

## 4. Preprocessing: Generate Perturbed `.pkl` Files

The repository includes separate preprocessing scripts for each dataset and perturbation type.

### 4.1 Example commands for MIMIC-III

```bash
python3 src/preprocess_mimic_iii_subsampled.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 2 \
  --pct 20

python3 src/preprocess_mimic_iii_sparsified-patientwise.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 0 \
  --pct 40

python3 src/preprocess_mimic_iii_sparsified-tsid-varid.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 2 \
  --pct 80

python3 src/preprocess_mimic_iii_unbalanced.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --pct 80
```

### 4.2 Example commands for PhysioNet 2012

```bash
python3 src/preprocess_physionet_2012_subsampled.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 0 \
  --pct 20

python3 src/preprocess_physionet_2012_sparsified-patientwise.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 2 \
  --pct 40

python3 src/preprocess_physionet_2012_sparsified-tsid-varid.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 2 \
  --pct 80

python3 src/preprocess_physionet_2012_unbalanced.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --pct 80
```

---

## 5. Generate All Perturbed `.pkl` Files Automatically

For the perturbations:
- `subsampled`
- `sparsified-patientwise`
- `sparsified-tsid-varid`

The preprocessing is run for:
- percentages: `10, 20, 30, ..., 90`
- seeds: `0` and `2`

For:
- `unbalanced`

The preprocessing is run for:
- percentages: `10, 20, 30, ..., 90`
- no seed

### 5.1 Full loop for MIMIC-III

```bash
PCTS=(10 20 30 40 50 60 70 80 90)
SEEDS=(0 2)
PERTS_WITH_SEED=("subsampled" "sparsified-patientwise" "sparsified-tsid-varid")

for pert in "${PERTS_WITH_SEED[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for pct in "${PCTS[@]}"; do
      python3 "src/preprocess_mimic_iii_${pert}.py" \
        --data_dir data/processed \
        --out_dir data/processed \
        --seed "$seed" \
        --pct "$pct"
    done
  done
done

for pct in "${PCTS[@]}"; do
  python3 src/preprocess_mimic_iii_unbalanced.py \
    --data_dir data/processed \
    --out_dir data/processed \
    --pct "$pct"
done
```

### 5.2 Full loop for PhysioNet 2012

```bash
PCTS=(10 20 30 40 50 60 70 80 90)
SEEDS=(0 2)
PERTS_WITH_SEED=("subsampled" "sparsified-patientwise" "sparsified-tsid-varid")

for pert in "${PERTS_WITH_SEED[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for pct in "${PCTS[@]}"; do
      python3 "src/preprocess_physionet_2012_${pert}.py" \
        --data_dir data/processed \
        --out_dir data/processed \
        --seed "$seed" \
        --pct "$pct"
    done
  done
done

for pct in "${PCTS[@]}"; do
  python3 src/preprocess_physionet_2012_unbalanced.py \
    --data_dir data/processed \
    --out_dir data/processed \
    --pct "$pct"
done
```

---

## 6. Naming Convention of Generated `.pkl` Files

The training script passes file names such as:
- `mimic_iii_subsampled_20_0`
- `physionet_2012_sparsified-patientwise_40_2`
- `mimic_iii_unbalanced_80`

So the perturbed `.pkl` files generated during preprocessing should follow the same naming logic.

### Interpretation

#### Example 1
`mimic_iii_subsampled_20_0`
- dataset: `mimic_iii`
- perturbation: `subsampled`
- percentage: `20`
- seed: `0`

#### Example 2
`physionet_2012_sparsified-patientwise_40_2`
- dataset: `physionet_2012`
- perturbation: `sparsified-patientwise`
- percentage: `40`
- seed: `2`

#### Example 3
`mimic_iii_unbalanced_80`
- dataset: `mimic_iii`
- perturbation: `unbalanced`
- percentage: `80`
- no seed

---

## 7. Run Training for All Models

Training is executed from the **repository root** using `src/main.py`.

The training configuration uses:
- datasets: `physionet_2012`, `mimic_iii`
- models: `gru`, `grud`, `tcn`, `sand`, `strats`
- percentages: `10` to `90`
- seeds: `0`, `2`
- perturbations: `subsampled`, `sparsified-patientwise`, `sparsified-tsid-varid`
- plus a separate `unbalanced` loop

### Full training script

```bash
# Run from repo root

DATA_DIR="data/processed"
RESULTS_ROOT="results"
MAX_EPOCHS=50
TRAIN_FRAC=1.0

DATASETS=(physionet_2012 mimic_iii)
MODELS=(gru grud tcn sand strats)
PCTS=(10 20 30 40 50 60 70 80 90)
SEEDS=(0 2)
PERTS=(subsampled sparsified-patientwise sparsified-tsid-varid)

hparams () {
  local d="$1" m="$2"
  if [[ "$d" == physionet_2012 ]]; then
    case "$m" in
      gru)    echo "43 0.2 0.2 0.0001 2 4 4 24 12" ;;
      grud)   echo "49 0.2 0.2 0.0001 2 4 4 24 12" ;;
      tcn)    echo "64 0.1 0.2 0.0005 6 4 4 24 12" ;;
      sand)   echo "64 0.3 0.3 0.0005 4 2 4 24 12" ;;
      strats) echo "50 0.2 0.2 0.0005 2 4 4 24 2"  ;;
    esac
  else # mimic_iii
    case "$m" in
      gru)    echo "50 0.2 0.2 0.0001 2 4 4 24 12" ;;
      grud)   echo "60 0.2 0.2 0.0001 2 4 4 24 12" ;;
      tcn)    echo "128 0.1 0.2 0.0001 4 4 4 24 12" ;;
      sand)   echo "64 0.3 0.3 0.0005 4 2 4 24 12" ;;
      strats) echo "50 0.2 0.2 0.0005 2 4 4 24 2"  ;;
    esac
  fi
}

run_main () {
  local dataset="$1" target="$2" model="$3" pert="$4" file="$5"
  read -r HID DROPOUT ATTN LR NL NH KS R M <<<"$(hparams "$dataset" "$model")"
  RUN_DIR="${RESULTS_ROOT}/${dataset}/${target}/${model}/${pert}/${file}"
  mkdir -p "$RUN_DIR"
  python3 src/main.py \
    --dataset "$dataset" --target "$target" --model_type "$model" \
    --file "$file" --output_dir "$RUN_DIR" \
    --train_frac "$TRAIN_FRAC" --max_epochs "$MAX_EPOCHS" \
    --hid_dim "$HID" --dropout "$DROPOUT" --attention_dropout "$ATTN" \
    --lr "$LR" --num_layers "$NL" --num_heads "$NH" --kernel_size "$KS" --r "$R" --M "$M"
}

# ---- 3 perturbations (with seed) ----
TARGET="in_hospital_mortality"
for d in "${DATASETS[@]}"; do
  for p in "${PERTS[@]}"; do
    for m in "${MODELS[@]}"; do
      for s in "${SEEDS[@]}"; do
        for pct in "${PCTS[@]}"; do
          file="${d}_${p}_${pct}_${s}"
          run_main "$d" "$TARGET" "$m" "$p" "$file"
        done
      done
    done
  done
done

# ---- unbalanced (no seed) ----
TARGET="unbalanced"
p="unbalanced"
for d in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    for pct in "${PCTS[@]}"; do
      file="${d}_${p}_${pct}"
      run_main "$d" "$TARGET" "$m" "$p" "$file"
    done
  done
done
```

Save it as, for example:

```bash
run_all_models.sh
```

Then run:

```bash
bash run_all_models.sh
```

---

## 8. Results Directory Structure

The `results/` folder stores all training outputs generated after running experiments.

In general, the folder is organized as:

```text
results/<dataset>/<target>/<model>/<perturbation>/<run_name>/
```

This means the saved outputs are organized step by step by:
1. dataset
2. target
3. model
4. perturbation type
5. specific run or file name

### What is saved inside `results/`
The `results/` directory contains experiment outputs generated during training and evaluation, such as:
- generated CSV result files
- log files
- model checkpoints

### General structure

```text
results/
├── mimic_iii/
│   ├── in_hospital_mortality/
│   │   ├── gru/
│   │   │   ├── sparsified-patientwise/
│   │   │   ├── sparsified-tsid-varid/
│   │   │   └── subsampled/
│   │   ├── grud/
│   │   ├── sand/
│   │   ├── strats/
│   │   └── tcn/
│   └── unbalanced/
│       ├── gru/
│       ├── grud/
│       ├── sand/
│       ├── strats/
│       └── tcn/
└── physionet_2012/
```

### Example of how results are saved

For example, if you run:
- dataset: `mimic_iii`
- target: `in_hospital_mortality`
- model: `gru`
- perturbation: `subsampled`

then the outputs are saved under a path like:

```text
results/mimic_iii/in_hospital_mortality/gru/subsampled/
```

If the script saves run-specific folders using the file name, a full run may look like:

```text
results/mimic_iii/in_hospital_mortality/gru/subsampled/mimic_iii_subsampled_20_0/
```

Inside that run folder, you would typically find:
- CSV result files
- training and evaluation logs
- checkpoints

Another example for unbalanced experiments:

```text
results/mimic_iii/unbalanced/strats/unbalanced/
```

or, if run names are used:

```text
results/mimic_iii/unbalanced/strats/unbalanced/mimic_iii_unbalanced_80/
```

---

## 9. Plot-Per-Metric Directory Structure

The `plots_per_metric/` folder stores visualization outputs generated from the experiment results.

In general, the plots are organized as:

```text
plots_per_metric/<dataset>/<target>/<perturbation>/
```

This makes it easier to compare how model performance changes under each perturbation setting.

### General structure

```text
plots_per_metric/
├── mimic_iii/
│   ├── in_hospital_mortality/
│   │   ├── subsampled/
│   │   ├── sparsified-patientwise/
│   │   └── sparsified-tsid-varid/
│   └── unbalanced/
│       └── unbalanced/
└── physionet_2012/
```

### Why `unbalanced/unbalanced` appears

For the unbalanced setting, the plot folder can appear as:

```text
plots_per_metric/<dataset>/unbalanced/unbalanced/
```

Here:
- the first `unbalanced` refers to the **target or task folder**
- the second `unbalanced` refers to the **perturbation folder**

So even though the name repeats, it follows the same folder logic consistently.

### Example of how plots are saved

For example, for MIMIC-III unbalanced experiments, the plots may be stored in:

```text
plots_per_metric/mimic_iii/unbalanced/unbalanced/
```

Inside that folder, metric-wise plots are saved as image files such as:

```text
plots_per_metric/mimic_iii/unbalanced/unbalanced/
├── test_accuracy_at_0.5.png
├── test_auprc.png
├── test_auroc.png
├── test_balanced_accuracy_at_0.5.png
├── test_f1_at_0.5.png
├── test_f2_at_0.5.png
├── test_minrp.png
├── test_precision_at_0.5.png
└── test_recall_at_0.5.png
```

These files represent one plot per evaluation metric.

---

## 10. About Generated Files

The repository uses three main storage locations:

- `data/processed/`  
  stores the base dataset `.pkl` files and all generated perturbed `.pkl` files

- `results/`  
  stores experiment outputs such as CSV files, logs, checkpoints, and other run-specific saved outputs

- `plots_per_metric/`  
  stores metric-wise plots generated from the experiment results

### Storage summary
- base and perturbed `.pkl` files -> `data/processed/`
- CSV files, logs, and checkpoints -> `results/...`
- plots -> `plots_per_metric/...`

---

## 11. Typical Workflow

A normal end-to-end workflow is:

1. Put the base dataset files `mimic_iii.pkl` and `physionet_2012.pkl` into `data/processed/`
2. Generate perturbed `.pkl` files using the preprocessing scripts
3. Keep all generated `.pkl` files in `data/processed/`
4. Run training for all models
5. Store training outputs, CSV files, logs, and checkpoints in `results/`
6. Use `src/plot_metrics.py` to generate plots
7. Store plots in `plots_per_metric/`

---

## 12. Notes

- Run all commands from the repository root
- Perturbations with seed use `seed = 0` and `seed = 2`
- Unbalanced preprocessing does not use a seed
- Percentages are evaluated at `10, 20, ..., 90`
- Generated file names should be consistent with the `--file` argument used during training
- If your local project uses a different target name, update the training script accordingly
