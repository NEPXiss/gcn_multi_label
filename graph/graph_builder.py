"""
graph/graph_builder.py
Refactor of build_gcn.py for Phase 0
- Loads processed_gcn_dataset.npz
- Builds k-NN graph (symmetric)
- Supports Euclidean, Cosine, Correlation similarity
- Saves PyTorch Geometric Data object
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def compute_distance_matrix(X, metric='euclidean'):
    """
    Compute distance matrix for k-NN graph.
    metric: 'euclidean', 'cosine', 'correlation'
    """
    if metric == 'euclidean':
        return X
    elif metric == 'cosine':
        # sklearn NearestNeighbors handles 'cosine'
        return X
    elif metric == 'correlation':
        # correlation distance = 1 - correlation coefficient
        corr = np.corrcoef(X)
        dist = 1 - corr
        # clip small numerical errors
        dist = np.clip(dist, 0, 2)
        return dist
    else:
        raise ValueError(f"Unknown metric {metric}")


def build_knn_graph(X, k=8, metric='euclidean', verbose=True):
    """
    Build symmetric k-NN graph from feature matrix X
    Returns: edge_index [2, num_edges] torch tensor
    """
    if verbose:
        print(f"Building k-NN graph: k={k}, metric={metric}...")

    if metric in ['euclidean', 'cosine']:
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
    elif metric == 'correlation':
        # correlation distance handled manually
        dist_matrix = compute_distance_matrix(X, metric='correlation')
        nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed')
        nbrs.fit(dist_matrix)
        distances, indices = nbrs.kneighbors(dist_matrix)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Build symmetric edge_index
    edge_list = []
    num_nodes = X.shape[0]
    for i in range(num_nodes):
        for j in indices[i]:
            edge_list.append([i, j])
            edge_list.append([j, i])  # ensure symmetry

    edge_index = np.unique(edge_list, axis=0).T  # shape [2, num_edges]
    if verbose:
        print(f"Graph built: {num_nodes} nodes, {edge_index.shape[1]} edges")
    return torch.tensor(edge_index, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Build k-NN graph from processed dataset for PyG")
    parser.add_argument('--input', default='processed_data/processed_gcn_dataset.npz',
                        help='Input .npz file from dataset.py')
    parser.add_argument('--output', default='gcn_data.pt',
                        help='Output PyG Data object path')
    parser.add_argument('--k', type=int, default=8, help='Number of neighbors')
    parser.add_argument('--metric', default='euclidean',
                        choices=['euclidean', 'cosine', 'correlation'], help='Distance metric for k-NN')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()

    # Load dataset
    if args.verbose:
        print(f"Loading dataset from {args.input}...")
    data_npz = np.load(args.input, allow_pickle=True)
    X = data_npz['X_proc']
    Y = data_npz['Y']

    # Build graph
    edge_index = build_knn_graph(X, k=args.k, metric=args.metric, verbose=args.verbose)

    # Convert to PyG Data object
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(Y, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)

    # Save
    torch.save(data, args.output)
    if args.verbose:
        print(f"Saved PyG Data object to {args.output}")
        print("Graph data ready:")
        print("  Number of nodes:", data.num_nodes)
        print("  Number of edges:", data.num_edges)
        print("  Feature shape:", data.x.shape)
        print("  Label shape:", data.y.shape)


if __name__ == '__main__':
    main()
