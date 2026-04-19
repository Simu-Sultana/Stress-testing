# Insights and Results

## Project Overview

**Project Title:** Stress-testing Deep Learning Models on Sparse Laboratory Data

This project evaluates how temporal deep learning models behave under **non-ICU-like data conditions**, focusing on robustness against three controlled perturbations:

- reduced cohort size (subsampling)
- increased measurement sparsity (sparsification)
- class imbalance (unbalanced labels)

The motivation comes from Rahimi et al. (2025), who showed that models trained and evaluated on ICU data perform significantly worse on non-ICU cohorts with sparse, irregular, and imbalanced laboratory data. This project systematically stress-tests ICU-trained models by artificially introducing those non-ICU characteristics.

---

## Datasets

| Dataset | ICU stays | Temporal variables | Task | Positive class |
|---|---|---|---|---|
| MIMIC-III | 52,871 | 129 | 24-hour mortality | 9.7% |
| PhysioNet 2012 | 11,988 | 37 | 48-hour mortality | 14.2% |

Both datasets are modified to simulate non-ICU characteristics such as sparse observations, irregular sampling, and imbalanced labels.

---

## Models Evaluated

| Model | Type | Key characteristic |
|---|---|---|
| GRU | RNN | Discretized regular time series |
| GRU-D | RNN | Handles irregular time series via exponential decay |
| TCN | CNN | Dilated causal convolutions |
| SAnD | Transformer | Attention-based on discretized input |
| STraTS | Transformer | Designed natively for sparse irregular time series |

---

## Perturbation Types

### 1. Subsampled
Reduces the number of patients by randomly selecting a subset of admission IDs from the training and validation splits. The test set remains unchanged.

### 2. Sparsified (tsid-varid)
Randomly removes measurements for each admission-variable pair. Demographics are unchanged. Retention percentage `p` controls how much data remains.

### 3. Unbalanced
Introduces class imbalance by converting the continuous Length-of-Stay (LOS) value into a binary target via thresholding. The positive-class rate `p` is controlled directly.

---

## Key Results

### 1. Subsampling (Cohort Size Reduction)

- Performance (AUROC) remains relatively stable above 20% of the data.
- Sharp decline begins below **10–20% of the cohort**:
  - PhysioNet-2012: drop below ~2,000 admissions
  - MIMIC-III: drop below ~5,000 admissions
- At very low percentages (1–5%), GRU-D and STraTS degrade the most.
- At full data: MIMIC-III achieves ~0.875–0.900 AUROC; PhysioNet achieves ~0.825–0.850 AUROC.

> **Insight:** There is a clear sample size threshold below which all models become unreliable. Models do not degrade gracefully — performance drops sharply once cohort size falls below a critical level.

---

### 2. Sparsification

- Performance decreases **gradually and steadily** with increasing sparsification.
- Overall impact is **smaller compared to subsampling and imbalance**.
- Outlier seeds cause notable drops at high sparsity levels, indicating instability.
- GRU-D breaks down entirely under heavy sparsification — it requires at least two measurements per variable to compute delta time values.

> **Insight:** Sparsification is the least damaging perturbation for most models. However, GRU-D — despite being specifically designed for irregular time series — is paradoxically the most vulnerable to extreme sparsity.

---

### 3. Class Imbalance

- AUPRC (the primary metric for imbalanced settings) drops sharply below **20% positive class**.
- AUPRC collapses to near-zero below 10% positive class.
- AUPRC Gain over a random baseline peaks around 30–40% positive class, then declines — models lose their meaningful advantage over random at very low positive rates.
- Performance drops are consistent across all models.

> **Insight:** Class imbalance has the strongest negative effect on model performance. No model is more robust to imbalance than others — all degrade at the same rate.

---

## Comparison with Non-ICU Cohort (NF)

To validate the stress-testing approach, PhysioNet-2012 was perturbed to match the data characteristics of the real-world Neutropenic Fever (NF) non-ICU cohort from Rahimi et al. (2025):

| Characteristic | NF cohort | PhysioNet original | PhysioNet multiperturbed |
|---|---|---|---|
| Training admissions | 3,389 | 7,672 | 3,389 |
| Avg measurements per adm-variable | 6.2 | 14.5 | 6.5 |
| Positive class | 4.8% | 14.2% | 4.8% |

When PhysioNet is perturbed to match NF characteristics, model performance drops to match NF levels. This confirms that the low performance observed in real non-ICU settings is driven by **data characteristics**, not by task difficulty.

---

## Model Behaviour Summary

- All models show **similar trends** across all perturbation types.
- No single model is consistently more robust across all settings.
- Performance differences between models are smaller than the effect of the perturbations themselves.
- Model choice matters less than data quality and structure.

---

## Special Observation: GRU-D

GRU-D was designed specifically to handle irregular time series via exponential decay of missing values. However, it is the most vulnerable model under sparsification — when fewer than two measurements exist per variable, delta time cannot be computed and the model fails entirely.

This is a fundamental architectural limitation that makes GRU-D unsuitable for very sparse non-ICU data despite its design intent.

---

## Discussion

The results suggest that the performance gap between ICU and non-ICU settings reported in Rahimi et al. (2025) is largely explained by three compounding data characteristics: smaller cohort size, higher sparsity, and greater class imbalance. Each of these individually degrades model performance, and in real non-ICU settings they tend to co-occur.

The finding that sparsification has the smallest individual impact is somewhat counterintuitive — one might expect models like STraTS, designed for sparse irregular input, to maintain performance better than RNN-based models. However, the results show that all models are similarly affected, suggesting that the bottleneck is not the model architecture but the information content available in the data.

Limitations of this study include the use of fixed hyperparameters (taken from Tipirneni et al., 2022) due to computational cost, a limited number of seeds for the subsampling and sparsification experiments, and the fact that perturbations were applied independently rather than in combination.

---

## Conclusions

- Model performance declines as cohort size decreases, with a critical threshold around 2,000–5,000 admissions depending on the dataset.
- Class imbalance has the strongest negative effect on performance — AUPRC collapses below 20% positive class.
- Sparsification has a smaller but steady impact, with GRU-D being the exception.
- Results are consistent across all five model architectures — data conditions dominate model choice.
- The stress-testing framework successfully reproduces the performance degradation observed in real non-ICU cohorts, validating the approach.

---

## Key Takeaways for Practitioners

- Data quality and structure matter more than model architecture in sparse clinical settings.
- Models trained on ICU data should not be deployed in non-ICU settings without re-evaluation.
- Collecting sufficient labeled data and addressing class imbalance (via oversampling, undersampling, or weighted loss) are higher priority than model selection.
- GRU-D should be avoided in very sparse settings despite its design intent.

---

## References

[1] S. Tipirneni and C. K. Reddy, "STraTS: Self-supervised transformer for time-series representation learning in healthcare," in *Proc. AAAI Conf. Artif. Intell.*, 2022.

[2] Rahimi et al., "Non-temporal tree-based models outperform temporal deep learning models in the prediction of chemotherapy-induced side effects from longitudinal laboratory data," 2025.
