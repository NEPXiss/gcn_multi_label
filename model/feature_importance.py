# model/feature_importance.py
"""
Feature importance analysis for GCN multi-label microbiome classification

Supports multiple methods:
1. Integrated Gradients (recommended)
2. Saliency Maps (gradient-based)
3. Feature Ablation (perturbation-based)
4. Attention weights (if model has attention)

Usage:
    python -m model.feature_importance --model processed_data/gcn_model.pt \
        --data processed_data/gcn_data.pt --method integrated_gradients \
        --output processed_data/feature_importance.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .gcn_model import ResidualGCN


# ============= Integrated Gradients =============
class IntegratedGradients:
    """
    Integrated Gradients: Computes path integral of gradients from baseline to input
    
    More robust than simple gradients, satisfies axioms like:
    - Sensitivity: If feature changes output, it gets non-zero attribution
    - Implementation Invariance: Functionally equivalent networks give same attributions
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def compute(self, x, edge_index, edge_weight, target_label_idx, baseline=None, steps=50):
        """
        Compute integrated gradients for all features w.r.t. specific label
        
        Args:
            x: [N, F] node features
            edge_index: [2, E] edge indices
            edge_weight: [E] edge weights
            target_label_idx: int, which label to compute importance for (0-4 for your 5 diseases)
            baseline: [N, F] baseline features (default: zeros)
            steps: number of integration steps
            
        Returns:
            attributions: [N, F] feature importance per node
        """
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        baseline = baseline.to(self.device).requires_grad_(False)
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device) if edge_weight is not None else None
        
        # Generate interpolated inputs along straight-line path
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        attributions = []
        for alpha in alphas:
            # Interpolate
            x_interp = baseline + alpha * (x - baseline)
            x_interp.requires_grad_(True)
            
            # Forward pass
            if edge_weight is not None:
                logits = self.model(x_interp, edge_index, edge_weight=edge_weight)
            else:
                logits = self.model(x_interp, edge_index)
            
            # Get output for target label
            output = logits[:, target_label_idx].sum()
            
            # Backward
            self.model.zero_grad()
            output.backward()
            
            # Store gradients
            attributions.append(x_interp.grad.detach().cpu())
        
        # Average gradients and multiply by (input - baseline)
        avg_gradients = torch.stack(attributions).mean(dim=0)
        integrated_grads = (x.cpu() - baseline.cpu()) * avg_gradients
        
        return integrated_grads.numpy()
    
    def compute_global_importance(self, x, edge_index, edge_weight, num_labels, baseline=None, steps=50):
        """
        Compute feature importance averaged across all samples and all labels
        
        Returns:
            importance_per_label: dict {label_idx: [F] importance scores}
            global_importance: [F] importance averaged across all labels
        """
        N, F = x.shape
        importance_per_label = {}
        
        print(f"Computing Integrated Gradients for {num_labels} labels...")
        for label_idx in range(num_labels):
            print(f"  Processing label {label_idx}...")
            attr = self.compute(x, edge_index, edge_weight, label_idx, baseline, steps)
            # Average across all nodes, take absolute value
            importance_per_label[label_idx] = np.abs(attr).mean(axis=0)
        
        # Global importance: average across all labels
        global_importance = np.mean([importance_per_label[i] for i in range(num_labels)], axis=0)
        
        return importance_per_label, global_importance


# ============= Saliency Maps =============
class SaliencyMaps:
    """
    Simple gradient-based attribution
    Computes gradient of output w.r.t. input features
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def compute(self, x, edge_index, edge_weight, target_label_idx):
        """
        Compute saliency map for target label
        
        Returns:
            saliency: [N, F] gradient magnitudes
        """
        x = x.to(self.device).requires_grad_(True)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device) if edge_weight is not None else None
        
        if edge_weight is not None:
            logits = self.model(x, edge_index, edge_weight=edge_weight)
        else:
            logits = self.model(x, edge_index)
        
        output = logits[:, target_label_idx].sum()
        
        self.model.zero_grad()
        output.backward()
        
        saliency = torch.abs(x.grad).detach().cpu().numpy()
        return saliency
    
    def compute_global_importance(self, x, edge_index, edge_weight, num_labels):
        """
        Compute saliency-based importance across all labels
        """
        importance_per_label = {}
        
        print(f"Computing Saliency Maps for {num_labels} labels...")
        for label_idx in range(num_labels):
            print(f"  Processing label {label_idx}...")
            saliency = self.compute(x, edge_index, edge_weight, label_idx)
            importance_per_label[label_idx] = saliency.mean(axis=0)
        
        global_importance = np.mean([importance_per_label[i] for i in range(num_labels)], axis=0)
        
        return importance_per_label, global_importance


# ============= Feature Ablation =============
class FeatureAblation:
    """
    Measure feature importance by removing features and observing performance drop
    
    More computationally expensive but very interpretable
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def compute_global_importance(self, data, num_labels, sample_size=None, ablation_value=0.0):
        """
        Compute importance by feature ablation
        
        Args:
            data: PyG Data object
            num_labels: number of labels
            sample_size: if not None, only use random subset of samples for speed
            ablation_value: value to replace feature with (0 or mean)
            
        Returns:
            importance_per_label: dict {label_idx: [F] importance scores}
            global_importance: [F] importance scores
        """
        data = data.to(self.device)
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        y = data.y
        
        N, F = x.shape
        
        # Sample subset if requested
        if sample_size is not None and sample_size < N:
            sample_idx = torch.randperm(N)[:sample_size]
            x = x[sample_idx]
            y = y[sample_idx]
        
        # Baseline performance (no ablation)
        with torch.no_grad():
            if edge_weight is not None:
                logits_baseline = self.model(x, edge_index, edge_weight=edge_weight)
            else:
                logits_baseline = self.model(x, edge_index)
            probs_baseline = torch.sigmoid(logits_baseline)
        
        # Per-label baseline F1 scores
        baseline_f1 = {}
        for label_idx in range(num_labels):
            preds = (probs_baseline[:, label_idx] >= 0.5).cpu().numpy()
            from sklearn.metrics import f1_score
            if sample_size is not None and sample_size < N:
                y_true = y[sample_idx, label_idx].cpu().numpy()
            else:
                y_true = y[:, label_idx].cpu().numpy()
            baseline_f1[label_idx] = f1_score(y_true, preds, zero_division=0)
        
        # Ablate each feature
        importance_per_label = {i: np.zeros(F) for i in range(num_labels)}
        
        print(f"Computing Feature Ablation for {F} features...")
        for feat_idx in tqdm(range(F)):
            # Ablate feature
            x_ablated = x.clone()
            x_ablated[:, feat_idx] = ablation_value
            
            with torch.no_grad():
                if edge_weight is not None:
                    logits_ablated = self.model(x_ablated, edge_index, edge_weight=edge_weight)
                else:
                    logits_ablated = self.model(x_ablated, edge_index)
                probs_ablated = torch.sigmoid(logits_ablated)
            
            # Compute performance drop for each label
            for label_idx in range(num_labels):
                preds = (probs_ablated[:, label_idx] >= 0.5).cpu().numpy()
                if sample_size is not None and sample_size < N:
                    y_true = y[sample_idx, label_idx].cpu().numpy()
                else:
                    y_true = y[:, label_idx].cpu().numpy()
                f1_ablated = f1_score(y_true, preds, zero_division=0)
                
                # Importance = performance drop
                importance_per_label[label_idx][feat_idx] = max(0, baseline_f1[label_idx] - f1_ablated)
        
        global_importance = np.mean([importance_per_label[i] for i in range(num_labels)], axis=0)
        
        return importance_per_label, global_importance


# ============= Visualization & Export =============
def save_importance_results(
    importance_per_label,
    global_importance,
    feature_names,
    disease_names,
    output_path,
    top_k=50
):
    """
    Save feature importance to CSV and create visualizations
    
    Args:
        importance_per_label: dict {label_idx: [F] scores}
        global_importance: [F] scores
        feature_names: list of feature names
        disease_names: list of disease names
        output_path: path to save CSV
        top_k: number of top features to visualize
    """
    num_labels = len(disease_names)
    F = len(feature_names)
    
    # Create DataFrame
    df_data = {'feature': feature_names, 'global_importance': global_importance}
    for i in range(num_labels):
        df_data[f'{disease_names[i]}_importance'] = importance_per_label[i]
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('global_importance', ascending=False)
    
    # Save to CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved importance scores to {out_path}")
    
    # Print top features
    print(f"\n{'='*60}")
    print(f"Top {min(top_k, len(df))} Most Important Features (Global)")
    print(f"{'='*60}")
    print(df.head(top_k)[['feature', 'global_importance']].to_string(index=False))
    
    # Visualizations
    vis_dir = out_path.parent / 'importance_plots'
    vis_dir.mkdir(exist_ok=True)
    
    # 1. Top-k global importance bar plot
    plt.figure(figsize=(12, 8))
    top_features = df.head(top_k)
    plt.barh(range(len(top_features)), top_features['global_importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_k} Most Important Features (Global)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(vis_dir / 'global_importance_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {vis_dir / 'global_importance_barplot.png'}")
    
    # 2. Heatmap of top features across diseases
    top_k_vis = min(30, len(df))
    heatmap_data = df.head(top_k_vis)[[f'{d}_importance' for d in disease_names]].values.T
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, 
                xticklabels=df.head(top_k_vis)['feature'].values,
                yticklabels=disease_names,
                cmap='YlOrRd', 
                cbar_kws={'label': 'Importance Score'})
    plt.xlabel('Features')
    plt.ylabel('Diseases')
    plt.title(f'Feature Importance Heatmap (Top {top_k_vis} Features)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(vis_dir / 'importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {vis_dir / 'importance_heatmap.png'}")
    
    # 3. Per-disease top features
    for i, disease in enumerate(disease_names):
        df_disease = df.sort_values(f'{disease}_importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(df_disease)), df_disease[f'{disease}_importance'].values)
        plt.yticks(range(len(df_disease)), df_disease['feature'].values)
        plt.xlabel('Importance Score')
        plt.title(f'Top 20 Features for {disease}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(vis_dir / f'{disease}_top_features.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_labels} disease-specific plots to {vis_dir}")
    
    return df


# ============= Main CLI =============
def main():
    parser = argparse.ArgumentParser(description='Feature importance analysis for GCN microbiome classifier')
    parser.add_argument('--model', required=True, help='Path to trained model (.pt file with state_dict)')
    parser.add_argument('--data', required=True, help='Path to PyG Data object (.pt file)')
    parser.add_argument('--method', default='integrated_gradients',
                        choices=['integrated_gradients', 'saliency', 'ablation'],
                        help='Feature importance method')
    parser.add_argument('--output', default='processed_data/feature_importance.csv',
                        help='Output CSV path for importance scores')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim of model (must match training)')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--layernorm', action='store_true')
    parser.add_argument('--ig-steps', type=int, default=50, help='Steps for integrated gradients')
    parser.add_argument('--ablation-samples', type=int, default=None,
                        help='Sample size for ablation (None = use all)')
    parser.add_argument('--top-k', type=int, default=50, help='Number of top features to display')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = torch.load(args.data)
    
    # Get feature names from original dataset if available
    try:
        data_npz = np.load('processed_data/processed_gcn_dataset.npz', allow_pickle=True)
        feature_names = data_npz['feature_cols'].tolist()
    except Exception:
        print("Warning: Could not load feature names from processed_gcn_dataset.npz")
        feature_names = [f'feature_{i}' for i in range(data.num_node_features)]
    
    disease_names = ['CRC', 'T2D', 'IBD', 'Cirrhosis', 'OBT']
    
    # Load model
    print(f"Loading model from {args.model}...")
    in_dim = data.num_node_features
    num_labels = data.y.shape[1]
    
    model = ResidualGCN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_labels=num_labels,
        dropout=args.dropout,
        use_bn=args.batchnorm,
        use_layernorm=args.layernorm
    )
    
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    # Compute importance
    if args.method == 'integrated_gradients':
        print(f"\nUsing Integrated Gradients (steps={args.ig_steps})...")
        ig = IntegratedGradients(model, device=args.device)
        importance_per_label, global_importance = ig.compute_global_importance(
            data.x, data.edge_index, 
            data.edge_weight if hasattr(data, 'edge_weight') else None,
            num_labels, steps=args.ig_steps
        )
    
    elif args.method == 'saliency':
        print("\nUsing Saliency Maps...")
        sm = SaliencyMaps(model, device=args.device)
        importance_per_label, global_importance = sm.compute_global_importance(
            data.x, data.edge_index,
            data.edge_weight if hasattr(data, 'edge_weight') else None,
            num_labels
        )
    
    elif args.method == 'ablation':
        print("\nUsing Feature Ablation...")
        if args.ablation_samples:
            print(f"Using {args.ablation_samples} samples for efficiency")
        fa = FeatureAblation(model, device=args.device)
        importance_per_label, global_importance = fa.compute_global_importance(
            data, num_labels, sample_size=args.ablation_samples
        )
    
    # Save and visualize results
    print("\nGenerating visualizations...")
    save_importance_results(
        importance_per_label,
        global_importance,
        feature_names,
        disease_names,
        args.output,
        top_k=args.top_k
    )
    
    print(f"\n{'='*60}")
    print("Feature importance analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
