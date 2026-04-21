# Reproducing All Experiments

This guide provides a complete, step-by-step pipeline to reproduce all experiments, including preprocessing, training, and evaluation.

---

## 1. Environment Setup

### Create environment

```bash
conda create -n strats python=3.10.9 -y
conda activate strats
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import torch, pandas, numpy; print('Environment OK')"
```

---

## 2. Dataset Setup

Place the base dataset files inside:

```text
data/processed/
├── mimic_iii.pkl
├── physionet_2012.pkl
```

### Important

- These are the base datasets
- All generated files will also be stored here

---

## 3. Generate Perturbed Data

### Settings

- Percentages: 10–90
- Seeds: 0–9
- Perturbations:
  - subsampled
  - sparsified-tsid-varid
  - unbalanced

---

### Subsampled + Sparsified

```bash
PCTS=(10 20 30 40 50 60 70 80 90)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
PERTS=(subsampled sparsified-tsid-varid)

for p in "${PERTS[@]}"; do
  for s in "${SEEDS[@]}"; do
    for pct in "${PCTS[@]}"; do
      python -m src.perturbation.preprocess_mimic_iii_${p} \
        --data_dir data/processed \
        --out_dir data/processed \
        --seed $s \
        --pct $pct

      python -m src.perturbation.preprocess_physionet_2012_${p} \
        --data_dir data/processed \
        --out_dir data/processed \
        --seed $s \
        --pct $pct
    done
  done
done
```

---

### Unbalanced (no seed)

```bash
PCTS=(10 20 30 40 50 60 70 80 90)

for pct in "${PCTS[@]}"; do
  python -m src.perturbation.preprocess_mimic_iii_unbalanced \
    --data_dir data/processed \
    --out_dir data/processed \
    --pct $pct

  python -m src.perturbation.preprocess_physionet_2012_unbalanced \
    --data_dir data/processed \
    --out_dir data/processed \
    --pct $pct
done
```

---

## 4. Training

Run all experiments:

```bash
bash src/pipeline/run_perturbations_training.sh
```

### This will

- train all models
- run all perturbations
- save outputs in `results/`

---

## 5. Results Structure

```text
results/<dataset>/<target>/<model>/<perturbation>/<run_name>/
```

Each run contains:

- CSV metrics
- logs
- checkpoint (`best.pt`)

---

## 6. Plot Generation

```bash
python -m src.pipeline.plot_metrics
```

Outputs will be stored in:

```text
plots_per_metric/
```

---

## 7. HPC Execution (Optional)

```bash
conda activate strats
srun --exclusive -N1 -n1 --gres=gpu:1 bash -lc "bash src/pipeline/run_perturbations_training.sh"
```

---

## 8. Reproducibility Notes

- Seeds: 0–9
- Percentages: 10–90
- Unbalanced has no seed
- Ensure correct file naming
- Ensure all `.pkl` files exist before training

---

## 9. Common Issues

- Missing files → run preprocessing
- Wrong naming → check file format
- GRU-D fails → data too sparse (expected behavior)

---

## 10. Workflow Summary

1. Setup environment
2. Add datasets
3. Generate perturbed data
4. Train models
5. Generate plots

---

## 11. Analysis

Use the generated CSV files and plots to compare model robustness across perturbation settings.

See [`Insights.md`](Insights.md) for a summary of key findings.