"""
Example: Feature Importance Analysis for GCN Microbiome Classifier

This script demonstrates how to run feature importance analysis
and interpret the results programmatically.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Import your custom modules
from model.gcn_model import ResidualGCN
from model.feature_importance import IntegratedGradients, SaliencyMaps, save_importance_results


def load_model_and_data():
    """Load trained model and data"""
    print("Loading model and data...")
    
    # Load data
    data = torch.load('processed_data/gcn_data.pt')
    
    # Load feature names
    try:
        data_npz = np.load('processed_data/processed_gcn_dataset.npz', allow_pickle=True)
        feature_names = data_npz['feature_cols'].tolist()
    except Exception:
        print("Warning: Using generic feature names")
        feature_names = [f'feature_{i}' for i in range(data.num_node_features)]
    
    # Initialize model (must match training parameters)
    model = ResidualGCN(
        in_dim=data.num_node_features,
        hidden_dim=64,  # Match your training config
        num_labels=5,
        dropout=0.5,
        use_bn=False,  # Set to True if you used --batchnorm during training
        use_layernorm=False
    )
    
    # Load trained weights
    model.load_state_dict(torch.load('processed_data/gcn_model.pt'))
    model.eval()
    
    print(f"✓ Model loaded: {data.num_nodes} samples, {data.num_node_features} features")
    
    return model, data, feature_names


def run_integrated_gradients_example():
    """Example 1: Integrated Gradients (recommended)"""
    print("\n" + "="*60)
    print("Example 1: Integrated Gradients Analysis")
    print("="*60)
    
    model, data, feature_names = load_model_and_data()
    disease_names = ['CRC', 'T2D', 'IBD', 'Cirrhosis', 'OBT']
    
    # Initialize IG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ig = IntegratedGradients(model, device=device)
    
    # Compute global importance
    print("\nComputing Integrated Gradients (this may take a few minutes)...")
    importance_per_label, global_importance = ig.compute_global_importance(
        data.x, 
        data.edge_index, 
        data.edge_weight if hasattr(data, 'edge_weight') else None,
        num_labels=5,
        steps=30  # Reduced for faster demo (use 50-100 for production)
    )
    
    # Save results
    print("\nSaving results...")
    save_importance_results(
        importance_per_label,
        global_importance,
        feature_names,
        disease_names,
        'processed_data/feature_importance_example_ig.csv',
        top_k=30
    )
    
    print("\n✓ Analysis complete! Check processed_data/importance_plots/")


def run_saliency_example():
    """Example 2: Saliency Maps (fast)"""
    print("\n" + "="*60)
    print("Example 2: Saliency Maps Analysis (Fast)")
    print("="*60)
    
    model, data, feature_names = load_model_and_data()
    disease_names = ['CRC', 'T2D', 'IBD', 'Cirrhosis', 'OBT']
    
    # Initialize Saliency
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sm = SaliencyMaps(model, device=device)
    
    # Compute importance
    print("\nComputing Saliency Maps...")
    importance_per_label, global_importance = sm.compute_global_importance(
        data.x,
        data.edge_index,
        data.edge_weight if hasattr(data, 'edge_weight') else None,
        num_labels=5
    )
    
    # Save results
    save_importance_results(
        importance_per_label,
        global_importance,
        feature_names,
        disease_names,
        'processed_data/feature_importance_example_saliency.csv',
        top_k=30
    )
    
    print("\n✓ Saliency analysis complete!")


def analyze_disease_specific_biomarkers():
    """Example 3: Find disease-specific biomarkers"""
    print("\n" + "="*60)
    print("Example 3: Disease-Specific Biomarker Discovery")
    print("="*60)
    
    model, data, feature_names = load_model_and_data()
    disease_names = ['CRC', 'T2D', 'IBD', 'Cirrhosis', 'OBT']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ig = IntegratedGradients(model, device=device)
    
    # Compute importance for each disease
    print("\nComputing disease-specific importance...")
    importance_per_label, _ = ig.compute_global_importance(
        data.x,
        data.edge_index,
        data.edge_weight if hasattr(data, 'edge_weight') else None,
        num_labels=5,
        steps=30
    )
    
    # Find disease-specific markers (high importance for one disease, low for others)
    print("\n" + "-"*60)
    print("Disease-Specific Biomarker Candidates")
    print("-"*60)
    
    for disease_idx, disease_name in enumerate(disease_names):
        # Get importance scores
        disease_imp = importance_per_label[disease_idx]
        other_diseases_imp = np.mean([importance_per_label[i] for i in range(5) if i != disease_idx], axis=0)
        
        # Specificity score: high for target disease, low for others
        specificity = disease_imp - other_diseases_imp
        
        # Get top specific features
        top_specific_idx = np.argsort(specificity)[-10:][::-1]
        
        print(f"\n{disease_name} - Top 10 Specific Biomarkers:")
        for rank, idx in enumerate(top_specific_idx, 1):
            if idx < len(feature_names):
                print(f"  {rank}. {feature_names[idx]:<40} (specificity: {specificity[idx]:.4f})")


def compare_methods():
    """Example 4: Compare different feature importance methods"""
    print("\n" + "="*60)
    print("Example 4: Comparing Feature Importance Methods")
    print("="*60)
    
    model, data, feature_names = load_model_and_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Compute with both methods
    print("\n1. Computing Integrated Gradients...")
    ig = IntegratedGradients(model, device=device)
    _, ig_importance = ig.compute_global_importance(
        data.x, data.edge_index, 
        data.edge_weight if hasattr(data, 'edge_weight') else None,
        num_labels=5, steps=20
    )
    
    print("\n2. Computing Saliency Maps...")
    sm = SaliencyMaps(model, device=device)
    _, sm_importance = sm.compute_global_importance(
        data.x, data.edge_index,
        data.edge_weight if hasattr(data, 'edge_weight') else None,
        num_labels=5
    )
    
    # Compare top features
    top_k = 20
    ig_top_idx = np.argsort(ig_importance)[-top_k:][::-1]
    sm_top_idx = np.argsort(sm_importance)[-top_k:][::-1]
    
    print(f"\n{'='*60}")
    print(f"Top {top_k} Features Comparison")
    print(f"{'='*60}")
    
    overlap = set(ig_top_idx) & set(sm_top_idx)
    print(f"\nOverlap: {len(overlap)}/{top_k} features agree between methods")
    
    print("\nFeatures identified by BOTH methods (robust):")
    for idx in sorted(overlap, key=lambda x: ig_importance[x], reverse=True)[:10]:
        if idx < len(feature_names):
            print(f"  • {feature_names[idx]:<40} (IG: {ig_importance[idx]:.4f}, Saliency: {sm_importance[idx]:.4f})")


if __name__ == '__main__':
    print("GCN Microbiome Feature Importance - Examples")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        'processed_data/gcn_model.pt',
        'processed_data/gcn_data.pt'
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print("\n⚠ Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease train your model first:")
        print("  python model/gcn_model.py --epochs 50 --save-model processed_data/gcn_model.pt")
        exit(1)
    
    # Run examples
    try:
        # Example 1: Basic IG analysis (recommended starting point)
        run_integrated_gradients_example()
        
        # Example 2: Fast saliency analysis
        # run_saliency_example()
        
        # Example 3: Disease-specific biomarkers
        # analyze_disease_specific_biomarkers()
        
        # Example 4: Compare methods
        # compare_methods()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Check generated plots in processed_data/importance_plots/")
        print("  2. Review CSV files for detailed scores")
        print("  3. Uncomment other examples above to explore more")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your model architecture matches training parameters")
        print("Adjust hidden_dim, use_bn, use_layernorm in load_model_and_data() if needed")
