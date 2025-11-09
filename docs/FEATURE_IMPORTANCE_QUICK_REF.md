# Feature Importance - Quick Reference

## TL;DR - Run This First

```bash
# After training your model, run this:
python -m model.feature_importance ^
    --model processed_data/gcn_model.pt ^
    --data processed_data/gcn_data.pt ^
    --method integrated_gradients ^
    --output processed_data/feature_importance.csv

# Check results in:
# - processed_data/feature_importance.csv (all scores)
# - processed_data/importance_plots/ (visualizations)
```

---

## What You'll Get

### 📊 CSV Output

| feature                 | global_importance | CRC_importance | T2D_importance | IBD_importance | Cirrhosis_importance | OBT_importance |
| ----------------------- | ----------------- | -------------- | -------------- | -------------- | -------------------- | -------------- |
| Bacteroides_fragilis    | 0.245             | 0.421          | 0.089          | 0.234          | 0.156                | 0.178          |
| Fusobacterium_nucleatum | 0.198             | 0.512          | 0.045          | 0.123          | 0.089                | 0.067          |
| ...                     | ...               | ...            | ...            | ...            | ...                  | ...            |

### 📈 Visualizations

- `global_importance_barplot.png` - Top features overall
- `importance_heatmap.png` - Heatmap across diseases
- `{disease}_top_features.png` - Top 20 per disease (5 plots)

---

## Methods Comparison

| Method                   | Speed       | Accuracy    | Use Case                |
| ------------------------ | ----------- | ----------- | ----------------------- |
| **integrated_gradients** | ⚡⚡ Medium | ⭐⭐⭐ High | **Recommended default** |
| **saliency**             | ⚡⚡⚡ Fast | ⭐⭐ Medium | Quick exploration       |
| **ablation**             | ⚡ Slow     | ⭐⭐⭐ High | Validation/few features |

---

## Common Commands

### Basic Analysis

```bash
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method integrated_gradients
```

### Fast Analysis (Fewer Steps)

```bash
python -m model.feature_importance --method integrated_gradients --ig-steps 20
```

### GPU Accelerated

```bash
python -m model.feature_importance --method integrated_gradients --device cuda
```

### Feature Ablation (Sample 500 nodes for speed)

```bash
python -m model.feature_importance --method ablation --ablation-samples 500
```

### Show Top 100 Features

```bash
python -m model.feature_importance --method integrated_gradients --top-k 100
```

---

## Interpreting Scores

### Global Importance

- **High (>0.2)**: Critical feature for overall disease prediction
- **Medium (0.1-0.2)**: Moderately important
- **Low (<0.1)**: Minor contribution

### Disease-Specific Patterns

- **High for one disease, low for others** → Disease-specific biomarker
- **High across all diseases** → General dysbiosis marker
- **Zero across all** → Uninformative feature (consider removing)

---

## Troubleshooting

| Problem                     | Solution                                                   |
| --------------------------- | ---------------------------------------------------------- |
| CUDA out of memory          | Add `--device cpu`                                         |
| Model architecture mismatch | Match `--hidden_dim`, `--batchnorm` flags to training      |
| Takes too long              | Use `--ig-steps 20` or `--method saliency`                 |
| Missing feature names       | Feature names auto-generated as feature_0, feature_1, etc. |

---

## Programmatic Usage

```python
from model.feature_importance import IntegratedGradients
from model.gcn_model import ResidualGCN
import torch

# Load
model = ResidualGCN(in_dim=100, hidden_dim=64, num_labels=5)
model.load_state_dict(torch.load('processed_data/gcn_model.pt'))
data = torch.load('processed_data/gcn_data.pt')

# Analyze
ig = IntegratedGradients(model, device='cpu')
importance_per_label, global_imp = ig.compute_global_importance(
    data.x, data.edge_index, data.edge_weight, num_labels=5, steps=50
)

# Get top features for CRC (label 0)
import numpy as np
crc_top_10 = np.argsort(importance_per_label[0])[-10:][::-1]
print("Top 10 CRC features:", crc_top_10)
```

---

## Next Steps

1. ✅ Run basic analysis (`integrated_gradients`)
2. 📊 Check visualizations in `processed_data/importance_plots/`
3. 📖 Read top features in CSV file
4. 🔬 Validate findings with literature
5. 🧪 Consider wet-lab validation for top biomarkers

**See [FEATURE_IMPORTANCE_GUIDE.md](FEATURE_IMPORTANCE_GUIDE.md) for detailed documentation.**
