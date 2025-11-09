# Complete Pipeline - Run From Scratch

This document lists all commands needed to run the entire GCN multi-label microbiome classification pipeline from start to finish.

---

## Prerequisites

- Conda installed
- Git repository cloned
- Input CSV files in `Input_files/` directory

---

## Step 0: Environment Setup

### Create Conda Environment

```cmd
conda env create -f clean_environment.yml --name gcn_multi
```

### Activate Environment

```cmd
conda activate gcn_multi
```

### Verify Installation (Optional)

```cmd
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__)"
```

---

## Step 1: Data Preparation

### Process Raw Data into Multi-Label Dataset

```cmd
python -m data_pipeline.dataset --map_csv data_map.csv --out processed_data/processed_gcn_dataset.npz --csv_out processed_data/combined_dataset_with_labels.csv
```

**Expected Output:**

- `processed_data/processed_gcn_dataset.npz` - Normalized features and labels
- `processed_data/combined_dataset_with_labels.csv` - Human-readable combined dataset

**Check Success:**

```cmd
dir processed_data/processed_gcn_dataset.npz
```

---

## Step 2: Graph Construction

### Build k-NN Graph (Default: k=8, Euclidean)

```cmd
python graph/graph_builder.py --verbose
```

### OR: Build with Custom Parameters (Recommended)

```cmd
python graph/graph_builder.py --k 10 --metric cosine --verbose
```

### OR: Build with PCA Dimensionality Reduction

```cmd
python graph/graph_builder.py --k 10 --metric cosine --latent pca --latent-dim 32 --verbose
```

**Expected Output:**

- `processed_data/gcn_data.pt` - PyTorch Geometric Data object with graph structure

**Check Success:**

```cmd
dir processed_data/gcn_data.pt
```

---

## Step 3: Model Training

### Basic Training (Quick Test)

```cmd
python model/gcn_model.py --epochs 50 --hidden_dim 64
```

### Recommended Training (With All Features)

```cmd
python model/gcn_model.py --epochs 100 --hidden_dim 64 --use-posweight --tune-thresholds --val-split 0.2 --early-stop 15 --save-model processed_data/gcn_model.pt --output processed_data/prediction_output.csv
```

### Advanced Training (With BatchNorm and Larger Hidden Dimension)

```cmd
python model/gcn_model.py --epochs 100 --hidden_dim 128 --batchnorm --use-posweight --tune-thresholds --val-split 0.2 --dropout 0.3 --early-stop 15 --save-model processed_data/gcn_model.pt
```

**Expected Output:**

- `processed_data/gcn_model.pt` - Trained model weights
- `processed_data/prediction_output.csv` - Predictions on all samples
- Terminal output showing metrics (Micro F1, Macro F1, per-disease accuracy)

---

## Step 4: Feature Importance Analysis (NEW!)

### Method 1: Integrated Gradients (Recommended)

```cmd
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method integrated_gradients --output processed_data/feature_importance.csv --top-k 50
```

In case of you are using advanced training in step 3

```cmd
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method integrated_gradients --output processed_data/feature_importance.csv --top-k 50 --hidden_dim 128 --batchnorm
```

### Method 2: Fast Analysis (Fewer Integration Steps)

```cmd
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method integrated_gradients --ig-steps 20 --output processed_data/feature_importance_fast.csv
```

### Method 3: Saliency Maps (Fastest)

```cmd
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method saliency --output processed_data/feature_importance_saliency.csv
```

### Method 4: Feature Ablation (Most Interpretable, Slowest)

```cmd
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method ablation --ablation-samples 500 --output processed_data/feature_importance_ablation.csv
```

### With GPU Acceleration (If Available)

```cmd
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method integrated_gradients --device cuda --output processed_data/feature_importance.csv
```

**Expected Output:**

- `processed_data/feature_importance.csv` - Importance scores per disease
- `processed_data/importance_plots/` - Visualization plots
  - `global_importance_barplot.png`
  - `importance_heatmap.png`
  - `CRC_top_features.png`
  - `T2D_top_features.png`
  - `IBD_top_features.png`
  - `Cirrhosis_top_features.png`
  - `OBT_top_features.png`

**Check Results:**

```cmd
type processed_data/feature_importance.csv | more
dir processed_data/importance_plots
```

---

## Step 5: Advanced - Domain Adaptation (Optional)

### Train with Domain Adaptation (MMD)

```cmd
python -m model.train_with_domain_adaptation --data processed_data/gcn_data.pt --da-method mmd --da-weight 0.1 --latent-dim 128 --hidden-dim 64 --epochs 100 --lr 1e-3 --dropout 0.3 --val-split 0.2 --early-stop 15 --use-posweight --save-model processed_data/gcn_model_da.pt
```

### Rebuild Graph with Domain-Adapted Encoder

```cmd
python -m graph.graph_builder_with_da --input processed_data/processed_gcn_dataset.npz --encoder-model processed_data/gcn_model_da.pt --output processed_data/gcn_data_da_refined.pt --k 8
```

### Re-train on Refined Graph

```cmd
python -m model.train_with_domain_adaptation --data processed_data/gcn_data_da_refined.pt --da-method mmd --da-weight 0.1 --latent-dim 128 --hidden-dim 64 --epochs 100 --save-model processed_data/final_gcn_da.pt
```

---

## Complete Pipeline Summary (Copy-Paste All)

```cmd
REM ===== STEP 0: Environment Setup =====
conda activate gcn_multi

REM ===== STEP 1: Data Preparation =====
python -m data_pipeline.dataset --map_csv data_map.csv --out processed_data/processed_gcn_dataset.npz --csv_out processed_data/combined_dataset_with_labels.csv

REM ===== STEP 2: Graph Construction =====
python graph/graph_builder.py --k 10 --metric cosine --verbose

REM ===== STEP 3: Model Training =====
python model/gcn_model.py --epochs 100 --hidden_dim 64 --use-posweight --tune-thresholds --val-split 0.2 --early-stop 15 --save-model processed_data/gcn_model.pt --output processed_data/prediction_output.csv

REM ===== STEP 4: Feature Importance Analysis =====
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method integrated_gradients --output processed_data/feature_importance.csv --top-k 50

REM ===== Done! Check Results =====
dir processed_data
dir processed_data/importance_plots
```

---

## Quick Start (Minimal Commands)

For a quick test run with default settings:

```cmd
conda activate gcn_multi
python -m data_pipeline.dataset --map_csv data_map.csv --out processed_data/processed_gcn_dataset.npz
python graph/graph_builder.py --verbose
python model/gcn_model.py --epochs 50
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method saliency
```

---

## Expected Runtime (Approximate)

| Step                              | Time (CPU)     | Time (GPU)    |
| --------------------------------- | -------------- | ------------- |
| Data Preparation                  | ~1-2 min       | ~1-2 min      |
| Graph Construction                | ~2-5 min       | ~2-5 min      |
| Model Training (100 epochs)       | ~10-20 min     | ~3-5 min      |
| Feature Importance (IG, 50 steps) | ~5-10 min      | ~2-3 min      |
| Feature Importance (Saliency)     | ~30 sec        | ~10 sec       |
| **Total (Recommended Pipeline)**  | **~20-40 min** | **~8-15 min** |

---

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:** Activate conda environment

```cmd
conda activate gcn_multi
```

### Issue: "CUDA out of memory"

**Solution:** Use CPU or reduce batch size

```cmd
python model/gcn_model.py --device cpu ...
```

### Issue: "File not found: processed_data/..."

**Solution:** Create directory

```cmd
mkdir processed_data
```

### Issue: Model architecture mismatch in feature importance

**Error:** `RuntimeError: Error(s) in loading state_dict... size mismatch for conv1.bias`

**Solution 1:** Inspect your model first to see what parameters were used during training

```bash
python -m model.inspect_model --model processed_data/gcn_model.pt
```

This will show you the exact architecture and print the correct command to use.

**Solution 2:** Match training parameters manually

```cmd
REM If you trained with --hidden_dim 128 --batchnorm
python -m model.feature_importance --model processed_data/gcn_model.pt --data processed_data/gcn_data.pt --method integrated_gradients --hidden_dim 128 --batchnorm
```

Common mismatches:

- `--hidden_dim` (default: 64, check if you used 128 or 256)
- `--batchnorm` flag (check if you added this during training)
- `--layernorm` flag (check if you added this during training)

### Issue: "ImportError: cannot import name 'ResidualGCN'"

**Solution:** Make sure you're running from project root directory

```cmd
cd d:\Documents\College-Year-3\2110581-BIOINFORMATIC\gcn_multi_label
```

---

## Validation Checklist

After running the pipeline, verify:

- [ ] `processed_data/processed_gcn_dataset.npz` exists
- [ ] `processed_data/gcn_data.pt` exists
- [ ] `processed_data/gcn_model.pt` exists
- [ ] `processed_data/prediction_output.csv` exists
- [ ] `processed_data/feature_importance.csv` exists
- [ ] `processed_data/importance_plots/` contains 7 PNG files
- [ ] Terminal shows Micro F1 score > 0.5 (model is learning)
- [ ] CSV files open correctly in Excel/text editor

---

## Next Steps After Running

1. **Review Predictions**: Open `processed_data/prediction_output.csv`
2. **Check Metrics**: Look at terminal output for F1 scores
3. **Analyze Important Features**: Open `processed_data/feature_importance.csv`
4. **View Visualizations**: Open plots in `processed_data/importance_plots/`
5. **Compare with Literature**: Validate top bacterial features against published research
6. **Iterate**: Adjust hyperparameters and re-run if needed

---

## Parameter Tuning Guide

### For Better Performance:

- Increase `--epochs` (100 → 200)
- Increase `--hidden_dim` (64 → 128 or 256)
- Add `--batchnorm` flag
- Tune `--dropout` (try 0.3, 0.4, 0.5)
- Adjust `--k` in graph building (try 5, 8, 10, 15)
- Try different `--metric` (euclidean, cosine, correlation)

### For Faster Execution:

- Decrease `--epochs` (100 → 50)
- Decrease `--ig-steps` (50 → 20)
- Use `--method saliency` instead of integrated_gradients
- Use `--ablation-samples 200` for feature ablation

### For Better Interpretability:

- Use `--tune-thresholds` flag
- Increase `--ig-steps` (50 → 100)
- Run all three feature importance methods and compare
- Use `--top-k 100` to see more features

---

## File Structure After Completion

```
gcn_multi_label/
├── processed_data/
│   ├── processed_gcn_dataset.npz      # Processed features & labels
│   ├── combined_dataset_with_labels.csv  # Human-readable dataset
│   ├── gcn_data.pt                    # Graph structure
│   ├── gcn_model.pt                   # Trained model
│   ├── prediction_output.csv          # Model predictions
│   ├── feature_importance.csv         # Feature importance scores
│   └── importance_plots/              # Visualization plots (7 PNGs)
├── Input_files/                       # Original data (unchanged)
├── data_pipeline/                     # Code (unchanged)
├── graph/                             # Code (unchanged)
└── model/                             # Code (unchanged)
```

---

## Citation

If you use this pipeline in your research, consider citing:

**Original GDmicro Paper:**

- Liao, H., Shang, J., & Sun, Y. (2023). "GDmicro: Classifying host disease status with GCN and deep adaptation network based on the human gut microbiome data"

**Feature Importance Methods:**

- Integrated Gradients: Sundararajan et al. (2017)
- Graph Neural Networks: Kipf & Welling (2017)

---

**Good luck with your bioinformatics analysis! 🧬🔬**
