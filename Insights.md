# Insights and Results

## Project Overview

**Project Title:**  
Stress-testing Deep Learning Models on Sparse Laboratory Data  

This project evaluates how different deep learning models behave under **non-ICU-like data conditions**, focusing on robustness against:

- small cohort size  
- sparsity in measurements  
- class imbalance  

---

## Datasets

- PhysioNet 2012 (ICU dataset)  
- MIMIC-III (ICU dataset)  

These datasets are modified to simulate **non-ICU characteristics** such as:
- sparse observations  
- irregular sampling  
- imbalanced labels  

---

## Models Evaluated

- GRU  
- GRU-D  
- TCN  
- SaND  
- STraTS  

---

## Perturbation Types

The following perturbations were applied:

### 1. Subsampled
- Reduces number of patients  
- Simulates smaller cohort sizes  

### 2. Sparsified
- Randomly removes measurements  
- Uses tsid-varid strategy  

### 3. Imbalance
- Introduces class imbalance  
- Based on length-of-stay (LOS) thresholding  

---

## Key Results

### 1. Subsampled (Cohort Size Reduction)

- Model performance **decreases significantly** as dataset size reduces  
- Performance drops:
  - below ~2000 samples (PhysioNet)  
  - below ~5000 samples (MIMIC)  
- At very low percentages (e.g., 5–10%), performance degrades sharply  

---

### 2. Sparsification

- Performance **decreases gradually** with increasing sparsity  
- Overall impact is **smaller compared to other perturbations**  
- Some variability across seeds (outliers observed)  

**Insight:**  
Sparsification has **less impact** on model performance compared to subsampling and imbalance.

---

### 3. Imbalance

- Performance decreases when positive class < 30%  
- Significant drop when positive class < 20%  

**Insight:**  
Class imbalance has a **strong negative impact** on model performance.

---

## Model Behavior

- All models show **similar trends across perturbations**  
- No model is consistently robust across all settings  

**Important Observations:**
- Model performance is driven more by **data characteristics** than model choice  
- Differences between models are **smaller than the effect of perturbations**

---

## Special Observation: GRU-D

- GRU-D struggles under high sparsity  
- Requires at least two observations to compute time gaps (delta)  

**Insight:**  
GRU-D is **not robust to extreme sparsity**

---

## Overall Conclusions

- Model performance declines as:
  - dataset size decreases  
  - class imbalance increases  

- Sparsification has:
  - **smaller overall effect**, but can cause instability  

- Imbalance has:
  - **strongest negative effect**  

- Across all perturbations:
  - **models behave similarly**  
  - performance differences are mainly driven by data conditions  

---

## Key Takeaways

- Data quality and structure matter more than model complexity  
- Models trained on ICU data do **not generalize well** to non-ICU conditions  
- Handling imbalance and small datasets is critical for real-world deployment  

---