import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def _pearson_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation between rows of X.
    Returns matrix in [-1, 1], with diagonal = 1.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(Xc, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    Xn = Xc / denom
    corr = Xn @ Xn.T
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def _build_knn_indices(features: np.ndarray, k: int, metric: str, verbose: bool):
    """
    Return neighbors indices and similarity values (if available).
    For 'cosine' we use sklearn NearestNeighbors metric='cosine' and convert to similarity.
    For 'euclidean' we use NearestNeighbors and return indices (similarities not computed).
    For 'correlation' we compute full Pearson corr matrix and pick top-k per row.
    """
    N = features.shape[0]
    if N == 0:
        raise ValueError("Empty feature matrix")

    if metric == "euclidean":
        nn = NearestNeighbors(n_neighbors=min(k, N - 1), metric="euclidean", algorithm="auto")
        nn.fit(features)
        distances, indices = nn.kneighbors(features)
        similarities = None

    elif metric == "cosine":
        nn = NearestNeighbors(n_neighbors=min(k, N - 1), metric="cosine", algorithm="auto")
        nn.fit(features)
        distances, indices = nn.kneighbors(features)
        similarities = 1.0 - distances  # shape (N, k)
        similarities = np.clip(similarities, -1.0, 1.0)

    elif metric == "correlation":
        corr = _pearson_similarity_matrix(features)  # (N, N)
        np.fill_diagonal(corr, -np.inf)
        kk = min(k, N - 1)
        indices = np.argpartition(-corr, kk, axis=1)[:, :kk]
        row_idx = np.arange(N)[:, None]
        similarities = corr[row_idx, indices]
        similarities = np.nan_to_num(similarities, nan=0.0, neginf=0.0)
        similarities = np.clip(similarities, -1.0, 1.0)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return indices, similarities


def build_knn_graph(
    X: np.ndarray,
    k: int = 8,
    metric: str = "euclidean",
    symmetric: bool = True,
    normalize_weights: bool = True,
    convert_sim_to_weight: str = "shift_clip",
    verbose: bool = True,
):
    """
    Build weighted (optionally symmetric) k-NN graph from X.

    convert_sim_to_weight:
      - 'shift_clip': transform sim in [-1,1] -> weight in [0,1] via (sim + 1)/2
      - 'clip': clip negative sims to 0 (max(0, sim))
      - 'exp': use exp(sim), then normalize

    Returns:
      edge_index: torch.LongTensor shape [2, E]
      edge_weight: torch.FloatTensor shape [E]
    """
    if verbose:
        print(f"[build_knn_graph] k={k}, metric={metric}, symmetric={symmetric}, normalize_weights={normalize_weights}")

    N = X.shape[0]
    if N == 0:
        raise ValueError("Empty feature matrix")

    indices, similarities = _build_knn_indices(X, k=k, metric=metric, verbose=verbose)

    src_list = []
    dst_list = []
    w_list = []

    for i in range(N):
        nbrs = indices[i]
        if similarities is None:
            for nb in nbrs:
                src_list.append(i)
                dst_list.append(int(nb))
                w_list.append(1.0)
        else:
            sims = similarities[i]
            for nb, sim in zip(nbrs, sims):
                src_list.append(i)
                dst_list.append(int(nb))
                if convert_sim_to_weight == "shift_clip":
                    weight = float((sim + 1.0) / 2.0)
                elif convert_sim_to_weight == "clip":
                    weight = float(max(0.0, sim))
                elif convert_sim_to_weight == "exp":
                    weight = float(np.exp(sim))
                else:
                    weight = float((sim + 1.0) / 2.0)
                w_list.append(weight)

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float32)

    # Symmetrize
    if symmetric:
        src2 = np.concatenate([src, dst], axis=0)
        dst2 = np.concatenate([dst, src], axis=0)
        w2 = np.concatenate([w, w], axis=0)
        edge_dict = {}
        for s, d, ww in zip(src2.tolist(), dst2.tolist(), w2.tolist()):
            key = (s, d)
            if key in edge_dict:
                edge_dict[key].append(ww)
            else:
                edge_dict[key] = [ww]
        src_final = []
        dst_final = []
        w_final = []
        for (s, d), wvals in edge_dict.items():
            src_final.append(s)
            dst_final.append(d)
            w_final.append(float(np.mean(wvals)))
        src = np.array(src_final, dtype=np.int64)
        dst = np.array(dst_final, dtype=np.int64)
        w = np.array(w_final, dtype=np.float32)

    # Normalize outgoing weights per source node (L1)
    if normalize_weights:
        sums = {}
        for s_idx, weight in zip(src.tolist(), w.tolist()):
            sums[s_idx] = sums.get(s_idx, 0.0) + weight
        norm_w = []
        for s_idx, weight in zip(src.tolist(), w.tolist()):
            denom = sums.get(s_idx, 1.0)
            if denom == 0:
                norm_w.append(0.0)
            else:
                norm_w.append(weight / denom)
        w = np.array(norm_w, dtype=np.float32)

    edge_index = np.vstack([src, dst])
    edge_index_t = torch.from_numpy(edge_index).long()
    edge_weight_t = torch.from_numpy(w).float()

    if verbose:
        num_nodes = N
        num_edges = edge_index_t.shape[1]
        print(f"[build_knn_graph] nodes={num_nodes}, edges={num_edges}")

    return edge_index_t, edge_weight_t


def main():
    parser = argparse.ArgumentParser(description="Build k-NN graph from processed dataset for PyG")
    parser.add_argument('--input', default='processed_data/processed_gcn_dataset.npz',
                        help='Input .npz file from dataset.py')
    parser.add_argument('--output', default='processed_data/gcn_data.pt',
                        help='Output PyG Data object path (default: processed_data/gcn_data.pt)')
    parser.add_argument('--k', type=int, default=8, help='Number of neighbors')
    parser.add_argument('--metric', default='euclidean',
                        choices=['euclidean', 'cosine', 'correlation'], help='Distance metric for k-NN')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    parser.add_argument('--no-symmetric', dest='symmetric', action='store_false',
                        help='Do not symmetrize edges (by default symmetrize)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Do not normalize edge weights per source (default: normalize)')
    parser.add_argument('--latent', default=None, choices=[None, 'pca'],
                        help='Optional latent transform to apply before k-NN (e.g., pca)')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension when using --latent pca')
    parser.set_defaults(verbose=False, symmetric=True, normalize=True)

    args = parser.parse_args()

    if args.verbose:
        print(f"Loading dataset from {args.input}...")

    data_npz = np.load(args.input, allow_pickle=True)
    if 'X_proc' in data_npz:
        X = data_npz['X_proc']
    elif 'X' in data_npz:
        X = data_npz['X']
    else:
        raise KeyError("Input .npz must contain 'X_proc' or 'X' array")

    if 'Y' in data_npz:
        Y = data_npz['Y']
    else:
        raise KeyError("Input .npz must contain 'Y' array")

    if args.verbose:
        print(f"Loaded features X shape: {X.shape}, labels Y shape: {Y.shape}")

    features_used = X
    if args.latent == 'pca':
        if args.verbose:
            print(f"Applying PCA (n_components={args.latent_dim}) to features before k-NN...")
        pca = PCA(n_components=min(args.latent_dim, X.shape[1]))
        features_used = pca.fit_transform(X)

    edge_index, edge_weight = build_knn_graph(
        features_used,
        k=args.k,
        metric=args.metric,
        symmetric=args.symmetric,
        normalize_weights=args.normalize,
        verbose=args.verbose
    )

    x_t = torch.tensor(X, dtype=torch.float)  # store original processed features as x
    y_t = torch.tensor(Y, dtype=torch.float)
    data = Data(x=x_t, edge_index=edge_index, edge_weight=edge_weight, y=y_t)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(out_path))
    if args.verbose:
        print(f"Saved PyG Data object to {out_path}")
        print("Graph data ready:")
        print("  Number of nodes:", data.num_nodes)
        print("  Number of edges:", data.num_edges)
        print("  Feature shape:", data.x.shape)
        print("  Label shape:", data.y.shape)


if __name__ == '__main__':
    main()
