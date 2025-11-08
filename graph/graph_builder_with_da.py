# graph/graph_builder_with_da.py
"""
Graph builder that uses learned latent features from Domain Adaptation Encoder
This ensures the k-NN graph is built on domain-invariant representations
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

# Import your existing graph building functions
import sys
sys.path.append('..')
from graph.graph_builder import build_knn_graph


def build_graph_with_encoder(
    X_raw,
    Y,
    encoder_model,
    k=8,
    metric='euclidean',
    symmetric=True,
    normalize_weights=True,
    device='cpu',
    verbose=True
):
    """
    Build k-NN graph using latent features from pretrained encoder
    
    Args:
        X_raw: [N, input_dim] raw features
        Y: [N, num_labels] labels
        encoder_model: trained DomainAdaptationEncoder or full DomainAdaptiveGCN
        k: number of neighbors
        metric: distance metric
        symmetric: whether to symmetrize edges
        normalize_weights: whether to normalize edge weights
        device: computation device
        verbose: print progress
    Returns:
        PyG Data object with graph built on latent features
    """
    if verbose:
        print("Extracting latent features using trained encoder...")
    
    encoder_model.eval()
    encoder_model.to(device)
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
        
        # Get encoder
        if hasattr(encoder_model, 'encoder'):
            # Full DomainAdaptiveGCN
            encoder = encoder_model.encoder
        else:
            # Just the encoder
            encoder = encoder_model
        
        # Extract latent features
        latent_features = encoder(X_tensor).cpu().numpy()
    
    if verbose:
        print(f"Latent features shape: {latent_features.shape}")
        print(f"Building k-NN graph with k={k}, metric={metric}...")
    
    # Build graph on latent features
    edge_index, edge_weight = build_knn_graph(
        latent_features,
        k=k,
        metric=metric,
        symmetric=symmetric,
        normalize_weights=normalize_weights,
        verbose=verbose
    )
    
    # Create PyG Data object
    # IMPORTANT: Store ORIGINAL features as x, not latent
    # The model will encode them again during training
    x_t = torch.tensor(X_raw, dtype=torch.float32)
    y_t = torch.tensor(Y, dtype=torch.float32)
    
    data = Data(
        x=x_t,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y_t
    )
    
    # Optionally store latent features for inspection
    data.latent_features = torch.tensor(latent_features, dtype=torch.float32)
    
    if verbose:
        print(f"Graph constructed: {data.num_nodes} nodes, {data.num_edges} edges")
    
    return data


def pretrain_encoder_for_graph(
    X_raw,
    encoder_config,
    epochs=50,
    lr=1e-3,
    device='cpu',
    verbose=True
):
    """
    Pretrain encoder using reconstruction loss or contrastive learning
    This is useful if you don't have domain labels yet
    
    Args:
        X_raw: [N, input_dim]
        encoder_config: dict with encoder parameters
        epochs: pretraining epochs
        lr: learning rate
        device: computation device
        verbose: print progress
    Returns:
        pretrained encoder
    """
    from model.domain_adaptation import DomainAdaptationEncoder
    
    encoder = DomainAdaptationEncoder(
        input_dim=encoder_config['input_dim'],
        latent_dim=encoder_config['latent_dim'],
        hidden_dims=encoder_config.get('hidden_dims', [256, 128]),
        dropout=encoder_config.get('dropout', 0.3)
    )
    
    # Simple autoencoder for pretraining
    decoder = torch.nn.Sequential(
        torch.nn.Linear(encoder_config['latent_dim'], 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, encoder_config['input_dim'])
    )
    
    encoder.to(device)
    decoder.to(device)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr
    )
    criterion = torch.nn.MSELoss()
    
    X_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
    
    if verbose:
        print("Pretraining encoder with reconstruction loss...")
    
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        
        optimizer.zero_grad()
        
        # Encode-decode
        latent = encoder(X_tensor)
        reconstructed = decoder(latent)
        
        # Reconstruction loss
        loss = criterion(reconstructed, X_tensor)
        
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    if verbose:
        print("Pretraining complete!")
    
    return encoder


# ============= Main Pipeline =============
def main():
    parser = argparse.ArgumentParser(
        description="Build graph using latent features from Domain Adaptation Encoder"
    )
    parser.add_argument('--input', default='processed_data/processed_gcn_dataset.npz',
                        help='Input .npz from dataset.py')
    parser.add_argument('--output', default='processed_data/gcn_data_da.pt',
                        help='Output PyG Data path')
    parser.add_argument('--encoder-model', default=None,
                        help='Path to pretrained encoder/model (.pt file). If not provided, will pretrain.')
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent dimension for encoder')
    parser.add_argument('--k', type=int, default=8,
                        help='Number of neighbors')
    parser.add_argument('--metric', default='euclidean',
                        choices=['euclidean', 'cosine', 'correlation'])
    parser.add_argument('--pretrain-epochs', type=int, default=50,
                        help='Epochs for encoder pretraining (if no pretrained model)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Load data
    if args.verbose:
        print(f"Loading dataset from {args.input}...")
    
    data_npz = np.load(args.input, allow_pickle=True)
    X = data_npz['X_proc'] if 'X_proc' in data_npz else data_npz['X']
    Y = data_npz['Y']
    
    if args.verbose:
        print(f"Loaded X: {X.shape}, Y: {Y.shape}")
    
    device = torch.device(args.device)
    
    # Get or create encoder
    if args.encoder_model:
        if args.verbose:
            print(f"Loading pretrained encoder from {args.encoder_model}...")
        
        checkpoint = torch.load(args.encoder_model, map_location=device)
        
        # Check if it's a full model or just encoder
        from model.domain_adaptation import DomainAdaptiveGCN, DomainAdaptationEncoder
        
        if 'model_state_dict' in checkpoint:
            # Full trained model
            config = checkpoint.get('config', {})
            model = DomainAdaptiveGCN(
                input_dim=X.shape[1],
                latent_dim=config.get('latent_dim', args.latent_dim),
                hidden_dim=config.get('hidden_dim', 64),
                num_labels=Y.shape[1],
                da_method=config.get('da_method', 'mmd')
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            encoder = model
        else:
            # Just encoder state dict
            encoder = DomainAdaptationEncoder(
                input_dim=X.shape[1],
                latent_dim=args.latent_dim
            )
            encoder.load_state_dict(checkpoint)
    
    else:
        # Pretrain new encoder
        if args.verbose:
            print("No pretrained encoder provided. Pretraining new encoder...")
        
        encoder = pretrain_encoder_for_graph(
            X_raw=X,
            encoder_config={
                'input_dim': X.shape[1],
                'latent_dim': args.latent_dim,
                'hidden_dims': [256, 128],
                'dropout': 0.3
            },
            epochs=args.pretrain_epochs,
            lr=1e-3,
            device=device,
            verbose=args.verbose
        )
    
    # Build graph with latent features
    data = build_graph_with_encoder(
        X_raw=X,
        Y=Y,
        encoder_model=encoder,
        k=args.k,
        metric=args.metric,
        symmetric=True,
        normalize_weights=True,
        device=device,
        verbose=args.verbose
    )
    
    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(out_path))
    
    if args.verbose:
        print(f"\nSaved graph to {out_path}")
        print(f"Nodes: {data.num_nodes}")
        print(f"Edges: {data.num_edges}")
        print(f"Features: {data.x.shape}")
        print(f"Labels: {data.y.shape}")
        print(f"Latent features: {data.latent_features.shape}")


if __name__ == '__main__':
    main()