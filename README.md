# Stress-testing Deep Learning Models on Sparse Laboratory Data

This project investigates the robustness of temporal deep learning models on sparse clinical time-series data.

We systematically evaluate how model performance degrades under realistic challenges such as:
- reduced cohort size,
- increased sparsity,
- and class imbalance.

The framework supports multiple datasets and architectures, enabling controlled stress-testing of model behavior.

---

## 🚀 Quick Start

1. Install dependencies:
   pip install -r requirements.txt

2. Place dataset files:
   data/processed/mimic_iii.pkl  
   data/processed/physionet_2012.pkl

3. Generate perturbed data:
   python src/preprocess_mimic_iii_subsampled.py --pct 20 --seed 0

4. Train a model:
   python src/main.py --dataset mimic_iii --model_type gru --file mimic_iii_subsampled_20_0

5. Generate plots:
   python src/plot_metrics.py

---

## 📊 Supported Components

### Datasets
- MIMIC-III
- PhysioNet 2012

### Models
- GRU
- GRU-D
- TCN
- SaND
- STraTS

### Perturbations
- Subsampled
- Sparsified (tsid-varid)
- Unbalanced

---

## 📁 Repository Structure

- `src/` → core implementation (models, training, preprocessing)
- `All_results/` → aggregated outputs (CSV, logs)
- `final_plots/` → final selected plots for reporting
- `README.md`
- `Reproduce_Guide.md`
- `requirements.txt`
- `Insights.md`

---

## 📂 Dataset Access

This project uses:

- MIMIC-III (requires credentialed access)
- PhysioNet 2012 Challenge dataset

Download from:
https://physionet.org/

---

## ⚙️ Environment Setup

Python version:
- Python 3.10.9

Create environment:

    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip

Install PyTorch:

    pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117

Install remaining packages:

    pip install numpy==1.26.4 pandas==2.0.3 scipy==1.10.1 pytz==2023.3.post1 tqdm==4.66.1 matplotlib==3.7.5 scikit-learn==1.2.2 transformers==4.35.2

---

## 🔄 Preprocessing

Example:

    python src/preprocess_mimic_iii_subsampled.py \
        --data_dir data/processed \
        --out_dir data/processed \
        --seed 2 \
        --pct 20

---

## 🏋️ Training

Run:

    python src/main.py

Important arguments:
- `--dataset`
- `--model_type`
- `--file`
- `--output_dir`

---

## 📊 Results

Structure:

    results/<dataset>/<target>/<model>/<perturbation>/<run_name>/

Contains:
- CSV metrics
- logs
- checkpoints

---

## 📈 Plots

- `plots_per_metric/` → metric-wise plots
- `final_plots/` → final selected plots for reporting

---

## 📌 Evaluation Metrics

- AUROC → ranking performance
- AUPRC → better for imbalance
- Accuracy, F1, Recall → threshold-based

Higher values = better performance

---

## 🔁 Workflow

1. Add base `.pkl` files
2. Generate perturbations
3. Train models
4. Save results
5. Generate plots
6. Analyze final plots

---

## ⚠️ Notes

- Run from repo root
- Seeds: 0, 2
- Percentages: 10–90
- Unbalanced has no seed
- File naming must match `--file`