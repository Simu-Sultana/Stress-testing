# Stress-testing Deep Learning Models on Sparse Clinical Time-Series Data

This project investigates the robustness of temporal deep learning models on sparse and irregular clinical time-series data.

We systematically evaluate how model performance degrades under realistic challenges such as:
- reduced cohort size,
- increased sparsity,
- and class imbalance.

The framework supports multiple datasets and architectures, enabling controlled stress-testing of model behavior.

---

## Overview

The project evaluates temporal deep learning models under controlled perturbations applied to clinical time-series datasets. The goal is to understand how robust different architectures are when the data become smaller, sparser, or more imbalanced.

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

## Quick Start

Run all commands from the repository root.

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Place base dataset files

Make sure the following files are available:

data/processed/mimic_iii.pkl
data/processed/physionet_2012.pkl
3. Generate perturbed data

Example:

python src/preprocess_mimic_iii_subsampled.py --pct 20 --seed 0
4. Train a model

Example:

python src/main.py --dataset mimic_iii --model_type gru --file mimic_iii_subsampled_20_0
5. Generate plots
python src/plot_metrics.py
Repository Structure
.
├── src/                   # core implementation: models, training, preprocessing
├── data/processed/        # base and generated .pkl files
├── results/               # experiment outputs
├── All_results/           # aggregated outputs such as CSV files and logs
├── plots_per_metric/      # metric-wise plots
├── final_plots/           # final selected plots for reporting
├── docs/                  # detailed project documentation
├── README.md
├── Reproduce_Guide.md
├── Insights.md
└── requirements.txt
Detailed Documentation

The following guides provide detailed instructions for running and understanding the project:

GPU / HPC usage
Hyperparameter settings
PKL generation pipeline
Model training and execution
Repository structure
Dataset Access

This project uses the following datasets:

MIMIC-III
Requires credentialed access.
PhysioNet 2012 Challenge dataset

Dataset source:

PhysioNet

Place the processed dataset files in:

data/processed/
Environment Setup
Python version
Python 3.10.9
Create a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
Install PyTorch with CUDA 11.7
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
Install the remaining dependencies
pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 pytz==2023.3.post1 tqdm==4.66.1 matplotlib==3.7.5 scikit-learn==1.2.2 transformers==4.35.2

You can also install all required packages using:

pip install -r requirements.txt
Preprocessing

Perturbed .pkl files are generated from the base processed datasets.

Example command:

python src/preprocess_mimic_iii_subsampled.py \
    --data_dir data

### Install the remaining dependencies Bash pip install numpy\==1.26.4 pandas\==2.0.3 scipy\==1.10.1 pytz\==2023.3.post1 tqdm\==4.66.1 matplotlib\==3.7.5 scikit-learn\==1.2.2 transformers\==4.35.2 You can also install all required packages using: Bash pip install \-r requirements.txt * * * Preprocessing ------------- Perturbed `.pkl` files are generated from the base processed datasets. Example command:

Again, you made the same mistake. See, this is not in the markdown file. How many times I have to tell you or I have to use a slang for you to do the work perfectly? Please listen to me.
Thought for a couple of seconds
# Stress-testing Deep Learning Models on Sparse Clinical Time-Series Data

This project investigates the robustness of temporal deep learning models on sparse and irregular clinical time-series data.

We systematically evaluate how model performance degrades under realistic challenges such as:
- reduced cohort size,
- increased sparsity,
- and class imbalance.

The framework supports multiple datasets and architectures, enabling controlled stress-testing of model behavior.

---

## Overview

The project benchmarks temporal deep learning models under controlled perturbation settings to analyze how robust they are when the training data become smaller, sparser, or more imbalanced.

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

## Quick Start

Run all commands from the repository root.

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Place the base dataset files

Put the processed base files here:

data/processed/mimic_iii.pkl
data/processed/physionet_2012.pkl
3. Generate perturbed data

Example:

python src/preprocess_mimic_iii_subsampled.py --pct 20 --seed 0
4. Train a model

Example:

python src/main.py --dataset mimic_iii --model_type gru --file mimic_iii_subsampled_20_0
5. Generate plots
python src/plot_metrics.py
Repository Structure
.
├── src/                    # core implementation: models, training, preprocessing
├── data/processed/         # base and generated .pkl datasets
├── results/                # run-wise experiment outputs
├── All_results/            # aggregated results
├── plots_per_metric/       # metric-wise plots
├── final_plots/            # final selected plots for reporting
├── docs/                   # detailed documentation
├── README.md
├── Reproduce_Guide.md
├── Insights.md
└── requirements.txt
Detailed Documentation

For more detailed instructions, see:

GPU / HPC usage
Hyperparameter settings
PKL generation pipeline
Model training and execution
Repository structure
Dataset Access

This project uses the following datasets:

MIMIC-III
Requires credentialed access.
PhysioNet 2012 Challenge dataset

Dataset source:

PhysioNet

Place the processed base .pkl files in:

data/processed/
Environment Setup
Python version
Python 3.10.9
Create a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
Install PyTorch with CUDA 11.7
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
Install the remaining dependencies
pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 pytz==2023.3.post1 tqdm==4.66.1 matplotlib==3.7.5 scikit-learn==1.2.2 transformers==4.35.2

You can also install all required packages using:

pip install -r requirements.txt
You motherfucker, can you listen to me? For example, start with the place, the number second, place the base data file. Why did you write it in the text file, motherfucker? I told you to write it in the markdown file.
Thought for 4s
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

## Quick Start

Run all commands from the repository root.

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Place the base dataset files

Place the processed base files in the following location:

data/processed/mimic_iii.pkl
data/processed/physionet_2012.pkl
3. Generate perturbed data

Example:

python src/preprocess_mimic_iii_subsampled.py --pct 20 --seed 0
4. Train a model

Example:

python src/main.py --dataset mimic_iii --model_type gru --file mimic_iii_subsampled_20_0
5. Generate plots
python src/plot_metrics.py
Repository Structure
.
├── src/                    # core implementation: models, training, preprocessing
├── data/
│   └── processed/          # base and generated .pkl datasets
├── results/                # run-wise experiment outputs
├── All_results/            # aggregated results
├── plots_per_metric/       # metric-wise plots
├── final_plots/            # final selected plots for reporting
├── docs/                   # detailed documentation
├── README.md
├── Reproduce_Guide.md
├── Insights.md
└── requirements.txt
Detailed Documentation

For more detailed instructions, see:

GPU / HPC usage
Hyperparameter settings
PKL generation pipeline
Model training and execution
Repository structure
Dataset Access

This project uses the following datasets:

MIMIC-III
Requires credentialed access.
PhysioNet 2012 Challenge dataset

Dataset source:

PhysioNet

After obtaining and processing the datasets, place the base .pkl files in:

data/processed/
Environment Setup
Python version
Python 3.10.9
Create a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
Install PyTorch with CUDA 11.7
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
Install the remaining dependencies
pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 pytz==2023.3.post1 tqdm==4.66.1 matplotlib==3.7.5 scikit-learn==1.2.2 transformers==4.35.2
Alternative

You can also install all required packages using:

pip install -r requirements.txt
Preprocessing

Perturbed .pkl files are generated from the base dataset files.

Example: MIMIC-III subsampling
python src/preprocess_mimic_iii_subsampled.py \
    --data_dir data/processed \
    --out_dir data/processed \
    --seed 2 \
    --pct 20

Generated files should follow the project naming convention, for example:

mimic_iii_subsampled_20_0.pkl
physionet_2012_subsampled_20_0.pkl

For more preprocessing details, see:

PKL generation pipeline
Training

Train a model using src/main.py.

Example
python src/main.py --dataset mimic_iii --model_type gru --file mimic_iii_subsampled_20_0
Important arguments
--dataset : dataset name, for example mimic_iii or physionet_2012
--model_type : model type, for example gru, grud, tcn, sand, strats
--file : dataset file tag, for example mimic_iii_subsampled_20_0
--output_dir : optional custom output directory

For more training details, see:

Model training and execution
Hyperparameter settings
Results

Each experiment stores outputs in a structured directory format:

results/<dataset>/<target>/<model>/<perturbation>/<run_name>/
Example
results/mimic_iii/IHM/GRU/subsampled/mimic_iii_subsampled_20_0/
Typical contents
log.txt : training progress
best.pt : best model checkpoint
last.pt : final model checkpoint
<run_name>.csv : evaluation metrics
Evaluation Metrics

The project reports several evaluation metrics, including:

AUROC : overall ranking performance
AUPRC : especially important under class imbalance
Accuracy : threshold-based classification performance
F1-score : balance between precision and recall
Recall : sensitivity to positive cases

Higher values indicate better predictive performance.

Plots

Plotting scripts generate summary figures from aggregated experiment results.

Plot folders
plots_per_metric/ : metric-wise comparison plots
final_plots/ : final selected plots used for reporting
Example
python src/plot_metrics.py
Workflow
1. Add the base .pkl files
data/processed/mimic_iii.pkl
data/processed/physionet_2012.pkl
2. Generate perturbed datasets
python src/preprocess_mimic_iii_subsampled.py --pct 20 --seed 0
3. Train models
python src/main.py --dataset mimic_iii --model_type gru --file mimic_iii_subsampled_20_0
4. Save results

Outputs are stored in:

results/<dataset>/<target>/<model>/<perturbation>/<run_name>/
5. Generate plots
python src/plot_metrics.py
6. Analyze results

Use the generated CSV files and plots to compare model robustness across perturbation settings.

Contribution

This project provides a systematic stress-testing framework for temporal deep learning models by:

designing controlled perturbations for clinical time-series datasets,
evaluating robustness across multiple temporal architectures,
standardizing experiments across datasets and perturbation settings,
and analyzing performance degradation under realistic data limitations.
Notes
Run all commands from the repository root.
Seeds commonly used in experiments: 0, 2
Percentages are typically varied across multiple perturbation levels.
The unbalanced perturbation does not use seeds.
File naming must match the --file argument exactly.
Some datasets require credentialed access before preprocessing and use.
Additional Guides

See the docs/ folder for complete instructions:

GPU / HPC usage
Hyperparameter settings
PKL generation pipeline
Model training and execution
Repository structure