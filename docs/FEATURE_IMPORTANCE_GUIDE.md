# Feature Importance Analysis Guide

## Overview

This guide explains how to perform feature importance analysis on your trained GCN model to understand which microbial features (bacteria/taxa) are most important for predicting each disease.

## Why Feature Importance Matters for Microbiome Analysis

- **Biological Insight**: Identify specific bacteria associated with diseases (e.g., which microbes are CRC biomarkers)
- **Model Interpretability**: Understand what your model learned (not just black-box predictions)
- **Hypothesis Generation**: Guide future wet-lab experiments
- **Biomarker Discovery**: Find candidate diagnostic markers for diseases

---

## Available Methods

### 1. **Integrated Gradients** (Recommended) ⭐

**How it works**: Computes the path integral of gradients from a baseline (zeros) to the actual input. More robust than simple gradients.

**Advantages**:

- Theoretically grounded (satisfies sensitivity and implementation invariance axioms)
- Less noisy than raw gradients
- Works well for deep networks

**When to use**: Default choice for most analyses

**Example**:

```bash
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method integrated_gradients --ig-steps 50 --output processed_data/feature_importance_ig.csv
```

---

### 2. **Saliency Maps**

**How it works**: Computes gradient of output w.r.t. input features (∂output/∂input)

**Advantages**:

- Fast (single backward pass)
- Simple to understand

**Disadvantages**:

- Can be noisy
- May saturate for deep networks

**When to use**: Quick exploratory analysis or when computational resources are limited

**Example**:

```bash
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method saliency --output processed_data/feature_importance_saliency.csv
```

---

### 3. **Feature Ablation**

**How it works**: Removes each feature one at a time, measures performance drop

**Advantages**:

- Very interpretable (direct cause-effect)
- Model-agnostic

**Disadvantages**:

- Computationally expensive (requires forward pass for each feature)
- Can be slow for thousands of features

**When to use**:

- When you need highly interpretable results
- For validating other methods
- When you have limited features (<500)

**Example**:

```bash
# Use subset of samples for speed
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method ablation --ablation-samples 500 --output processed_data/feature_importance_ablation.csv
```

---

## Step-by-Step Workflow

### Step 1: Train Your Model (if not done already)

```bash
python model/gcn_model.py --epochs 50 --hidden_dim 64 --use-posweight --tune-thresholds --val-split 0.2 --save-model processed_data/gcn_model.pt
```

### Step 2: Run Feature Importance Analysis

**For Integrated Gradients (recommended)**:

```bash
python -m model.feature_importance ^
    --model processed_data/gcn_model.pt ^
    --data processed_data/gcn_data.pt ^
    --method integrated_gradients ^
    --ig-steps 50 ^
    --output processed_data/feature_importance.csv ^
    --top-k 50
```

**For faster analysis (fewer IG steps)**:

```bash
python -m model.feature_importance ^
    --model processed_data/gcn_model.pt ^
    --data processed_data/gcn_data.pt ^
    --method integrated_gradients ^
    --ig-steps 20 ^
    --output processed_data/feature_importance_fast.csv
```

**For GPU acceleration**:

```bash
python -m model.feature_importance ^
    --model processed_data/gcn_model.pt ^
    --data processed_data/gcn_data.pt ^
    --method integrated_gradients ^
    --device cuda ^
    --output processed_data/feature_importance.csv
```

### Step 3: Examine Results

The script generates:

1. **CSV file** (`feature_importance.csv`):

   - Columns: `feature`, `global_importance`, `CRC_importance`, `T2D_importance`, `IBD_importance`, `Cirrhosis_importance`, `OBT_importance`
   - Sorted by global importance (descending)

2. **Visualizations** in `processed_data/importance_plots/`:
   - `global_importance_barplot.png` - Top features overall
   - `importance_heatmap.png` - Feature importance across all diseases
   - `CRC_top_features.png` - Top 20 features for CRC
   - `T2D_top_features.png` - Top 20 features for T2D
   - `IBD_top_features.png` - Top 20 features for IBD
   - `Cirrhosis_top_features.png` - Top 20 features for Cirrhosis
   - `OBT_top_features.png` - Top 20 features for OBT

---

## Interpreting Results

### Global Importance Score

- Higher score = more important for overall disease prediction
- Use for finding general dysbiosis markers

### Disease-Specific Importance

- Shows which features are important for each specific disease
- Example: High `CRC_importance` but low `T2D_importance` suggests CRC-specific biomarker

### Example Interpretation

If you see:

```
feature                           global_importance  CRC_importance  T2D_importance
Fusobacterium_nucleatum          0.245              0.421           0.089
Bacteroides_fragilis             0.198              0.312           0.156
```

**Interpretation**:

- _Fusobacterium nucleatum_ is highly important for CRC (0.421) but not T2D (0.089)
- This aligns with known biology (F. nucleatum is associated with colorectal cancer)
- _Bacteroides fragilis_ is moderately important for both diseases

---

## Advanced Usage

### Compare Multiple Methods

```bash
# Run all three methods
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method integrated_gradients --output processed_data/importance_ig.csv

python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method saliency --output processed_data/importance_saliency.csv

python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method ablation --ablation-samples 500 --output processed_data/importance_ablation.csv
```

Then compare results to identify robust important features (those ranked high by all methods).

### Custom Analysis in Python

```python
import torch
import numpy as np
from model.feature_importance import IntegratedGradients
from model.gcn_model import ResidualGCN

# Load model and data
model = ResidualGCN(in_dim=100, hidden_dim=64, num_labels=5)
model.load_state_dict(torch.load('processed_data/gcn_model.pt'))
data = torch.load('processed_data/gcn_data.pt')

# Compute importance for a specific disease (e.g., CRC = label 0)
ig = IntegratedGradients(model, device='cpu')
crc_importance = ig.compute(
    data.x,
    data.edge_index,
    data.edge_weight,
    target_label_idx=0,  # CRC
    steps=50
)

# Get top 10 features for CRC
feature_importance = np.abs(crc_importance).mean(axis=0)
top_10_idx = np.argsort(feature_importance)[-10:][::-1]
print("Top 10 features for CRC:", top_10_idx)
```

---

## Expected Runtime

| Method                          | Features | Samples | Approximate Time |
| ------------------------------- | -------- | ------- | ---------------- |
| Integrated Gradients (50 steps) | 1000     | 500     | ~2-5 min (GPU)   |
| Saliency Maps                   | 1000     | 500     | ~10-30 sec (GPU) |
| Feature Ablation (all samples)  | 1000     | 500     | ~30-60 min (GPU) |
| Feature Ablation (200 samples)  | 1000     | 500     | ~10-15 min (GPU) |

_Times are approximate and depend on hardware_

---

## Troubleshooting

### Error: "CUDA out of memory"

**Solution**: Use CPU or reduce batch processing

```bash
python -m model.feature_importance --device cpu ...
```

### Error: "Could not load feature names"

**Solution**: Make sure `processed_data/processed_gcn_dataset.npz` exists. If not, feature names will default to `feature_0`, `feature_1`, etc.

### Error: Model architecture mismatch

**Solution**: Ensure you pass the same `--hidden_dim`, `--batchnorm`, `--layernorm` flags used during training:

```bash
python -m model.feature_importance --model ... --hidden_dim 128 --batchnorm
```

---

## Biological Validation

After identifying important features:

1. **Literature Search**: Check if identified bacteria are known disease markers
2. **Cross-Validation**: Ensure features are consistent across different train/test splits
3. **Domain Knowledge**: Consult with microbiologists to validate findings
4. **Experimental Validation**: Design wet-lab experiments to validate computational findings

---

## Citation & References

If you use this for research, consider citing:

**Integrated Gradients**:

- Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks"

**GCN Interpretability**:

- Ying et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks"

**Original GDmicro**:

- Liao, H., Shang, J., & Sun, Y. (2023). "GDmicro: Classifying host disease status with GCN and deep adaptation network based on the human gut microbiome data"

---

## Next Steps

1. **Run basic analysis**: Start with Integrated Gradients
2. **Examine top features**: Look at `processed_data/importance_plots/`
3. **Validate findings**: Compare with known disease-microbiome associations
4. **Iterate**: Try different methods, compare results
5. **Document**: Keep notes on biologically meaningful features found

For questions or issues, check the main README.md or consult the course materials.
