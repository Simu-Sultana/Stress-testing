# Stress-testing Deep Learning Models on Sparse Clinical Time-Series Data

This project investigates the robustness of temporal deep learning models on sparse and irregular clinical time-series data.

We systematically evaluate how model performance degrades under realistic challenges such as:
- reduced cohort size,
- increased sparsity,
- and class imbalance.

The framework supports multiple datasets and architectures, enabling controlled stress-testing of model behavior.

---

## Overview

This project benchmarks temporal deep learning models under controlled perturbation settings to analyze how robust they are when the training data become smaller, sparser, or more imbalanced.

The codebase is organized as a modular Python package (`src/`) with dedicated submodules for training, models, perturbation, and evaluation.

### Supported datasets
- MIMIC-III
- PhysioNet 2012

### Supported models
- GRU
- GRU-D
- TCN
- SaND
- STraTS

### Supported perturbations
- Subsampled
- Sparsified (tsid-varid)
- Unbalanced

---

## Important

All scripts must be run using module syntax:

```bash
python -m src.<module>
```

Running scripts using `python src/...` may break imports after restructuring.

---

## Quick Start

> Run all commands from the repository root.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the base dataset files

Place the processed base files in the following location:

```text
data/processed/mimic_iii.pkl
data/processed/physionet_2012.pkl
```

### 3. Generate perturbed data

Example:

```bash
python -m src.perturbation.preprocess_mimic_iii_subsampled \
  --data_dir data/processed \
  --out_dir data/processed \
  --pct 20 \
  --seed 0
```

### 4. Train a model

Example:

```bash
python -m src.training.main \
  --dataset mimic_iii \
  --target ihm \
  --model_type gru \
  --file mimic_iii_subsampled_20_0
```

### 5. Generate plots

```bash
python -m src.pipeline.plot_metrics
```

---

## Repository Structure

```text
.
├── src/
│   ├── training/        # training logic (main, dataset, evaluator)
│   ├── models/          # model architectures
│   ├── perturbation/    # preprocessing / perturbation scripts
│   ├── pipeline/        # plotting and analysis
│   ├── utils/           # helper utilities
│   └── debug/           # debugging tools
│
├── data/
│   └── processed/       # base and generated .pkl datasets
│
├── results/             # run-wise experiment outputs
├── All_results/         # aggregated results
├── plots_per_metric/    # metric-wise plots
├── final_plots/         # final selected plots for reporting
├── docs/                # detailed documentation
├── README.md
├── Reproduce_Guide.md
├── Insights.md
└── requirements.txt
```

---

## Detailed Documentation

For more detailed instructions, see the [`docs/`](docs/) folder:

- [GPU / HPC usage](docs/gpu_hpc.md)
- [Hyperparameter settings](docs/hyperparameters.md)
- [PKL generation pipeline](docs/pkl_pipeline.md)
- [Model training and execution](docs/training.md)
- [Repository structure](docs/structure.md)

For full end-to-end reproduction of all experiments, see [`Reproduce_Guide.md`](Reproduce_Guide.md).

---

## Dataset Access

This project uses the following datasets:

- **MIMIC-III** — Requires credentialed access via [PhysioNet MIMIC-III](https://physionet.org/content/mimiciii/1.4/).
- **PhysioNet 2012 Challenge dataset** — Freely available at [PhysioNet 2012](https://physionet.org/content/challenge-2012/1.0.0/).

After obtaining and processing the datasets, place the base `.pkl` files in:

```text
data/processed/
```

---

## Environment Setup

### Python version

```text
Python 3.10.9
```

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Optional: manual installation if PyTorch CUDA wheels fail on your system

```bash
pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 pytz==2023.3.post1 tqdm==4.66.1 matplotlib==3.7.5 scikit-learn==1.2.2 transformers==4.35.2
pip install torch torchvision torchaudio
```

---

## Preprocessing

Perturbed `.pkl` files are generated from the base dataset files.

### Example: MIMIC-III subsampling

```bash
python -m src.perturbation.preprocess_mimic_iii_subsampled \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 2 \
  --pct 20
```

Generated files follow the project naming convention, for example:

```text
mimic_iii_subsampled_20_0.pkl
physionet_2012_subsampled_20_0.pkl
```

For more preprocessing details, see: [PKL generation pipeline](docs/pkl_pipeline.md)

---

## Training

Train a model using `src.training.main`.

### Example

```bash
python -m src.training.main \
  --dataset mimic_iii \
  --target ihm \
  --model_type gru \
  --file mimic_iii_subsampled_20_0
```

### Important arguments

| Argument | Description | Example |
|---|---|---|
| `--dataset` | Dataset name | `mimic_iii`, `physionet_2012` |
| `--target` | Prediction target | `ihm`, `los` |
| `--model_type` | Model type | `gru`, `grud`, `tcn`, `sand`, `strats` |
| `--file` | Dataset file tag | `mimic_iii_subsampled_20_0` |
| `--output_dir` | Optional custom output directory | `results/my_run` |

For more training details, see: [Model training and execution](docs/training.md) and [Hyperparameter settings](docs/hyperparameters.md)

---

## Results

Each experiment stores outputs in a structured directory format:

```text
results/<dataset>/<target>/<model>/<perturbation>/<run_name>/
```

### Example

```text
results/mimic_iii/ihm/gru/subsampled/mimic_iii_subsampled_20_0/
```

### Typical contents

| File | Description |
|---|---|
| `log.txt` | Training progress per epoch |
| `best.pt` | Best model checkpoint |
| `last.pt` | Final model checkpoint after all epochs |
| `<run_name>.csv` | Evaluation metrics for that run |

---

## Evaluation Metrics

The project reports the following evaluation metrics:

| Metric | Used for | Key finding from experiments |
|---|---|---|
| AUROC | Subsampling, Sparsification | Performance stays stable above moderate cohort sizes and drops sharply below 10–20% of the data |
| AUPRC | Class imbalance (Unbalanced) | Primary metric for the unbalanced perturbation; drops sharply below 20% positive class |
| AUPRC Gain | Class imbalance (Unbalanced) | Measures improvement over a random baseline and declines strongly at very low positive rates |
| F1-score | Class imbalance (Unbalanced) | Mirrors AUPRC behavior under imbalance |

Higher values indicate better predictive performance across all metrics.

> **Interpreting degradation across perturbations**
> - **Subsampling**: Performance drops sharply below 10–20% of the cohort.
> - **Sparsification**: Performance declines more gradually than subsampling.
> - **Class imbalance**: AUPRC collapses below 20% positive class.
> - **GRU-D** may fail under extreme sparsity because it requires sufficient observations to compute delta values.

---

## Plots

Plotting scripts generate summary figures from aggregated experiment results.

### Plot folders

| Folder | Description |
|---|---|
| `plots_per_metric/` | Metric-wise comparison plots across perturbation levels |
| `final_plots/` | Final selected plots used for reporting |

### Example

```bash
python -m src.pipeline.plot_metrics
```

---

## Workflow

### 1. Add the base `.pkl` files

```text
data/processed/mimic_iii.pkl
data/processed/physionet_2012.pkl
```

### 2. Generate perturbed datasets

```bash
python -m src.perturbation.preprocess_mimic_iii_subsampled \
  --data_dir data/processed \
  --out_dir data/processed \
  --pct 20 \
  --seed 0
```

### 3. Train models

```bash
python -m src.training.main \
  --dataset mimic_iii \
  --target ihm \
  --model_type gru \
  --file mimic_iii_subsampled_20_0
```

### 4. Save results

Outputs are stored in:

```text
results/<dataset>/<target>/<model>/<perturbation>/<run_name>/
```

### 5. Generate plots

```bash
python -m src.pipeline.plot_metrics
```

### 6. Analyze results

Use the generated CSV files and plots to compare model robustness across perturbation settings. See [`Insights.md`](Insights.md) for a summary of key findings.

---

## Contribution

This project provides a systematic stress-testing framework for temporal deep learning models by:

- designing controlled perturbations for clinical time-series datasets,
- evaluating robustness across multiple temporal architectures,
- standardizing experiments across datasets and perturbation settings,
- and analyzing performance degradation under realistic data limitations.

---

## Notes

- Run all commands from the repository root.
- Seeds commonly used in experiments: `0`, `2`.
- Percentages are typically varied across a range such as `10, 20, 30, 40, 50, 60, 70, 80, 90`.
- The unbalanced perturbation does not use seeds.
- File naming must match the `--file` argument exactly.
- Some datasets require credentialed access before preprocessing and use.
- GRU-D may fail on very sparse data — this is expected behavior.

---

## Additional Guides

See the [`docs/`](docs/) folder for complete instructions:

- [GPU / HPC usage](docs/gpu_hpc.md)
- [Hyperparameter settings](docs/hyperparameters.md)
- [PKL generation pipeline](docs/pkl_pipeline.md)
- [Model training and execution](docs/training.md)
- [Repository structure](docs/structure.md)