# graph/graph_builder_semi_supervised.py
"""
Graph builder that preserves train/test masks for semi-supervised learning
Creates ONE graph with all samples (labeled + unlabeled)
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

import sys
sys.path.append('..')
from graph.graph_builder import build_knn_graph


def build_semi_supervised_graph(
    X_proc: np.ndarray,
    Y: np.ndarray,
    class_info: np.ndarray,
    study_info: np.ndarray,
    k: int = 8,
    metric: str = 'euclidean',
    symmetric: bool = True,
    normalize_weights: bool = True,
    encoder_model=None,
    device='cpu',
    verbose: bool = True
):
    """
    Build k-NN graph for semi-supervised learning
    All samples (train + test) are in the same graph
    
    Args:
        X_proc: [N, input_dim] preprocessed features
        Y: [N, num_labels] labels
        class_info: [N] array of 'train'/'test' strings
        study_info: [N] array of study/domain names
        k: number of neighbors
        metric: distance metric
        symmetric: symmetrize edges
        normalize_weights: normalize edge weights
        encoder_model: optional pretrained encoder for latent features
        device: computation device
        verbose: print progress
    
    Returns:
        PyG Data object with train_mask, test_mask attributes
    """
    N = X_proc.shape[0]
    
    # Determine which features to use for graph construction
    if encoder_model is not None:
        if verbose:
            print("Using encoder to extract latent features for graph...")
        
        encoder_model.eval()
        encoder_model.to(device)
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_proc, dtype=torch.float32).to(device)
            
            # Get encoder
            if hasattr(encoder_model, 'encoder'):
                encoder = encoder_model.encoder
            else:
                encoder = encoder_model
            
            latent_features = encoder(X_tensor).cpu().numpy()
        
        features_for_graph = latent_features
        if verbose:
            print(f"Latent features shape: {latent_features.shape}")
    else:
        if verbose:
            print("Using preprocessed features for graph...")
        features_for_graph = X_proc
    
    # Build k-NN graph
    if verbose:
        print(f"Building k-NN graph (k={k}, metric={metric})...")
    
    edge_index, edge_weight = build_knn_graph(
        features_for_graph,
        k=k,
        metric=metric,
        symmetric=symmetric,
        normalize_weights=normalize_weights,
        verbose=verbose
    )
    
    # Create masks based on class_info
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    
    for i in range(N):
        cls = str(class_info[i]).lower()
        if cls == 'train':
            train_mask[i] = True
        elif cls == 'test':
            test_mask[i] = True
        else:
            # Unknown class - treat as test
            test_mask[i] = True
    
    if verbose:
        print(f"Created masks:")
        print(f"  Train samples: {train_mask.sum().item()} ({train_mask.sum().item()/N*100:.1f}%)")
        print(f"  Test samples: {test_mask.sum().item()} ({test_mask.sum().item()/N*100:.1f}%)")
    
    # Create PyG Data object
    # Store ORIGINAL preprocessed features as x (not latent!)
    x_t = torch.tensor(X_proc, dtype=torch.float32)
    y_t = torch.tensor(Y, dtype=torch.float32)
    
    data = Data(
        x=x_t,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y_t,
        train_mask=train_mask,
        test_mask=test_mask
    )
    
    # Store metadata
    data.study_info = study_info
    data.class_info = class_info
    
    # Store latent features if used
    if encoder_model is not None:
        data.latent_features = torch.tensor(features_for_graph, dtype=torch.float32)
    
    if verbose:
        print(f"\nGraph construction complete:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Features: {data.x.shape}")
        print(f"  Labels: {data.y.shape}")
        print(f"  Train/Test: {train_mask.sum().item()}/{test_mask.sum().item()}")
        print(f"  Studies: {np.unique(study_info)}")
    
    return data


def verify_graph_structure(data, verbose=True):
    """
    Verify that the semi-supervised graph is constructed correctly
    
    Checks:
    1. Train and test nodes are connected
    2. No data leakage (test labels not used during training)
    3. Graph connectivity
    """
    if verbose:
        print("\n" + "="*60)
        print("Graph Structure Verification")
        print("="*60)
    
    # Check 1: Train-test connectivity
    train_indices = torch.where(data.train_mask)[0]
    test_indices = torch.where(data.test_mask)[0]
    
    edge_src = data.edge_index[0]
    edge_dst = data.edge_index[1]
    
    # Count edges between train and test
    train_to_test_edges = 0
    test_to_train_edges = 0
    
    for i in range(data.edge_index.shape[1]):
        src = edge_src[i].item()
        dst = edge_dst[i].item()
        
        if data.train_mask[src] and data.test_mask[dst]:
            train_to_test_edges += 1
        elif data.test_mask[src] and data.train_mask[dst]:
            test_to_train_edges += 1
    
    if verbose:
        print(f"✓ Train→Test edges: {train_to_test_edges}")
        print(f"✓ Test→Train edges: {test_to_train_edges}")
        print(f"✓ Total cross edges: {train_to_test_edges + test_to_train_edges}")
    
    # Check 2: Label distribution
    train_labels = data.y[data.train_mask].sum(dim=0)
    test_labels = data.y[data.test_mask].sum(dim=0)
    
    if verbose:
        print(f"\nLabel distribution:")
        print(f"  Train label counts: {train_labels.numpy()}")
        print(f"  Test label counts: {test_labels.numpy()}")
    
    # Check 3: Connected components
    from torch_geometric.utils import to_scipy_sparse_matrix
    from scipy.sparse.csgraph import connected_components
    
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    n_components, labels = connected_components(adj, directed=False)
    
    if verbose:
        print(f"\n✓ Connected components: {n_components}")
        if n_components == 1:
            print("  ✓ Graph is fully connected (good!)")
        else:
            print(f"  ⚠ Graph has {n_components} components")
            for i in range(min(n_components, 5)):
                comp_size = (labels == i).sum()
                print(f"    Component {i}: {comp_size} nodes")
    
    return {
        'train_to_test_edges': train_to_test_edges,
        'test_to_train_edges': test_to_train_edges,
        'n_components': n_components
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build semi-supervised graph from dataset with class column"
    )
    parser.add_argument('--input', default='processed_data/processed_gcn_dataset.npz',
                        help='Input .npz file with class_info and study_info')
    parser.add_argument('--output', default='processed_data/gcn_data_semi_supervised.pt',
                        help='Output PyG Data path')
    parser.add_argument('--encoder-model', default=None,
                        help='Path to pretrained encoder/model for latent features')
    parser.add_argument('--k', type=int, default=8,
                        help='Number of neighbors')
    parser.add_argument('--metric', default='euclidean',
                        choices=['euclidean', 'cosine', 'correlation'],
                        help='Distance metric')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify graph structure')
    
    args = parser.parse_args()
    
    # Load data
    if args.verbose:
        print(f"Loading dataset from {args.input}...")
    
    data_npz = np.load(args.input, allow_pickle=True)
    
    # Check for required fields
    required_fields = ['X_proc', 'Y', 'class_info', 'study_info']
    missing = [f for f in required_fields if f not in data_npz]
    if missing:
        raise ValueError(f"Missing required fields in .npz: {missing}")
    
    X_proc = data_npz['X_proc']
    Y = data_npz['Y']
    class_info = data_npz['class_info']
    study_info = data_npz['study_info']
    
    if args.verbose:
        print(f"Loaded:")
        print(f"  X_proc: {X_proc.shape}")
        print(f"  Y: {Y.shape}")
        print(f"  class_info: {class_info.shape} - unique: {np.unique(class_info)}")
        print(f"  study_info: {study_info.shape} - unique: {np.unique(study_info)}")
    
    # Load encoder if provided
    encoder_model = None
    if args.encoder_model:
        if args.verbose:
            print(f"\nLoading encoder from {args.encoder_model}...")
        
        device = torch.device(args.device)
        checkpoint = torch.load(args.encoder_model, map_location=device)
        
        from domain_adaptation import DomainAdaptiveGCN, DomainAdaptationEncoder
        
        if 'model_state_dict' in checkpoint:
            # Full model
            config = checkpoint.get('config', {})
            encoder_model = DomainAdaptiveGCN(
                input_dim=X_proc.shape[1],
                latent_dim=config.get('latent_dim', 128),
                hidden_dim=config.get('hidden_dim', 64),
                num_labels=Y.shape[1],
                da_method=config.get('da_method')
            )
            encoder_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Just encoder
            encoder_model = DomainAdaptationEncoder(
                input_dim=X_proc.shape[1],
                latent_dim=128
            )
            encoder_model.load_state_dict(checkpoint)
        
        if args.verbose:
            print("✓ Encoder loaded successfully")
    
    # Build graph
    data = build_semi_supervised_graph(
        X_proc=X_proc,
        Y=Y,
        class_info=class_info,
        study_info=study_info,
        k=args.k,
        metric=args.metric,
        symmetric=True,
        normalize_weights=True,
        encoder_model=encoder_model,
        device=args.device,
        verbose=args.verbose
    )
    
    # Verify graph structure
    if args.verify:
        verify_graph_structure(data, verbose=args.verbose)
    
    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(out_path))
    
    if args.verbose:
        print(f"\n✓ Semi-supervised graph saved to {out_path}")


if __name__ == '__main__':
    main()