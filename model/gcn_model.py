# model/gcn_model.py
"""
Phase 1 GCN multi-label model and training utilities.

Features:
- 3-layer GCN with residual connections
- Optionally BatchNorm / LayerNorm, dropout
- Uses BCEWithLogitsLoss (supports pos_weight tensor)
- Accepts PyG Data with data.edge_weight (used in conv forward)
- Validation and per-disease threshold tuning (F1-based)
- CLI for training / evaluation / prediction CSV output
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_fscore_support

# ---------------- Model ----------------
class ResidualGCN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_labels,
        dropout=0.5,
        use_bn=True,
        use_layernorm=False,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_labels)

        self.use_bn = use_bn
        self.use_layernorm = use_layernorm
        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
        if use_layernorm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.ln3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # conv1
        h1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        if self.use_bn:
            h1 = self.bn1(h1)
        if self.use_layernorm:
            h1 = self.ln1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # conv2 (residual from h1)
        h2 = self.conv2(h1, edge_index, edge_weight=edge_weight)
        if self.use_bn:
            h2 = self.bn2(h2)
        if self.use_layernorm:
            h2 = self.ln2(h2)
        h2 = F.relu(h2 + h1)  # residual connection
        h2 = self.dropout(h2)

        # conv3 (residual from h2)
        h3 = self.conv3(h2, edge_index, edge_weight=edge_weight)
        if self.use_bn:
            h3 = self.bn3(h3)
        if self.use_layernorm:
            h3 = self.ln3(h3)
        h3 = F.relu(h3 + h2)  # residual
        h3 = self.dropout(h3)

        logits = self.head(h3)  # return raw logits for BCEWithLogitsLoss
        return logits


# ---------------- Utilities ----------------
def compute_pos_weight(y: torch.Tensor, eps=1e-8):
    """
    Compute pos_weight tensor for BCEWithLogitsLoss:
      pos_weight = (num_negative / (num_positive + eps)) per label
    y: tensor shape [N, D] of 0/1 floats or ints
    returns: tensor shape [D]
    """
    y_np = y.detach().cpu().numpy()
    pos = y_np.sum(axis=0)
    neg = y_np.shape[0] - pos
    pos_weight = (neg / (pos + eps)).astype(np.float32)
    pos_weight = torch.from_numpy(pos_weight)
    return pos_weight


def metrics_from_preds(y_true, y_probs, thresholds=None):
    """
    Compute common metrics:
      - per-label ROC-AUC, PR-AUC
      - micro/macro F1 at given thresholds
    y_true: np.array [N, D]
    y_probs: np.array [N, D] (sigmoid probabilities)
    thresholds: None or array-like length D (if None uses 0.5)
    returns dict of metrics and per-label dicts
    """
    N, D = y_true.shape
    if thresholds is None:
        thresholds = np.array([0.5] * D)
    preds = (y_probs >= thresholds[None, :]).astype(int)

    # global metrics
    micro_f1 = f1_score(y_true, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    # per-label ROC-AUC and PR-AUC
    roc_list = []
    pr_list = []
    for j in range(D):
        try:
            roc = roc_auc_score(y_true[:, j], y_probs[:, j])
        except Exception:
            roc = float('nan')
        try:
            pr = average_precision_score(y_true[:, j], y_probs[:, j])
        except Exception:
            pr = float('nan')
        roc_list.append(roc)
        pr_list.append(pr)

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'roc_auc_per_label': np.array(roc_list),
        'pr_auc_per_label': np.array(pr_list),
        'preds': preds
    }


def tune_thresholds(y_true: np.ndarray, y_probs: np.ndarray, search_space=None):
    """
    Per-label threshold tuning to maximize F1 on provided validation set.
    y_true: [N, D], y_probs: [N, D]
    Returns thresholds (D,) and per-label best F1
    """
    N, D = y_true.shape
    if search_space is None:
        search_space = np.linspace(0.05, 0.95, 91)  # step 0.01

    best_thresh = np.zeros(D, dtype=np.float32)
    best_f1 = np.zeros(D, dtype=np.float32)

    for j in range(D):
        best = -1.0
        best_t = 0.5
        yj = y_true[:, j]
        pj = y_probs[:, j]
        # if no positives, keep default threshold 0.5 and f1 = 0
        if yj.sum() == 0:
            best_thresh[j] = 0.5
            best_f1[j] = 0.0
            continue
        for t in search_space:
            pred_j = (pj >= t).astype(int)
            f1 = f1_score(yj, pred_j, zero_division=0)
            if f1 > best:
                best = f1
                best_t = t
        best_thresh[j] = best_t
        best_f1[j] = best
    return best_thresh, best_f1


# ---------------- Training / Eval ----------------
def train_epoch(model, data, optimizer, criterion, device):
    model.train()
    model.to(device)
    data = data.to(device)
    optimizer.zero_grad()
    if hasattr(data, 'edge_weight'):
        logits = model(data.x, data.edge_index, edge_weight=data.edge_weight)
    else:
        logits = model(data.x, data.edge_index, edge_weight=None)
    loss = criterion(logits, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(model, data, device, thresholds=None):
    model.eval()
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        if hasattr(data, 'edge_weight'):
            logits = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            logits = model(data.x, data.edge_index, edge_weight=None)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = data.y.cpu().numpy()

    metrics = metrics_from_preds(y_true, probs, thresholds)
    return y_true, probs, metrics


def train(
    model,
    data,
    device='cpu',
    epochs=50,
    lr=1e-3,
    weight_decay=1e-5,
    pos_weight=None,
    val_split=0.2,
    seed=42,
    early_stop=10,
    verbose=True,
):
    """
    Training wrapper that does a random node-level train/val split (if val_split>0)
    and returns trained model + best thresholds tuned on validation.
    Note: For small experiments you may want to provide an external train/val split.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = data.num_nodes
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * (1 - val_split))
    train_idx = idx[:split]
    val_idx = idx[split:] if val_split > 0 else np.array([], dtype=int)

    # create train / val Data objects (shallow copy but with subsetted x,y and same edges)
    # Here we keep full graph but will mask losses by indexing outputs; simpler: compute logits and mask
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    best_thresholds = None

    for epoch in range(1, epochs + 1):
        model.train()
        model.to(device)
        optimizer.zero_grad()
        if hasattr(data, 'edge_weight'):
            logits = model(data.x.to(device), data.edge_index.to(device), edge_weight=data.edge_weight.to(device))
        else:
            logits = model(data.x.to(device), data.edge_index.to(device), edge_weight=None)
        # compute loss only on training nodes
        train_logits = logits[train_idx]
        train_labels = data.y[train_idx].to(device)
        loss = criterion(train_logits, train_labels)
        loss.backward()
        optimizer.step()

        if val_split > 0:
            # validation: compute logits for all nodes then evaluate on val_idx
            model.eval()
            with torch.no_grad():
                if hasattr(data, 'edge_weight'):
                    all_logits = model(data.x.to(device), data.edge_index.to(device), edge_weight=data.edge_weight.to(device))
                else:
                    all_logits = model(data.x.to(device), data.edge_index.to(device), edge_weight=None)
                all_probs = torch.sigmoid(all_logits).cpu().numpy()
                y_true = data.y.cpu().numpy()
                val_probs = all_probs[val_idx]
                val_true = y_true[val_idx]
                # tune thresholds per-label on validation (small search)
                thresholds, f1s = tune_thresholds(val_true, val_probs)
                # compute micro F1 at these thresholds on validation set
                preds_val = (val_probs >= thresholds[None, :]).astype(int)
                micro_f1 = f1_score(val_true, preds_val, average='micro', zero_division=0)
        else:
            micro_f1 = None
            thresholds = np.array([0.5] * data.y.shape[1])

        if verbose and (epoch % 5 == 0 or epoch == 1):
            info = f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}"
            if micro_f1 is not None:
                info += f", Val micro-F1: {micro_f1:.4f}"
            print(info)

        # early stopping on validation micro_f1
        if val_split > 0:
            if micro_f1 > best_val_f1:
                best_val_f1 = micro_f1
                best_state = model.state_dict()
                best_thresholds = thresholds
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if early_stop and epochs_no_improve >= early_stop:
                if verbose:
                    print(f"No improvement for {epochs_no_improve} epochs. Early stopping.")
                break
    # load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    # return model and thresholds tuned on validation (or default 0.5)
    if best_thresholds is None:
        best_thresholds = np.array([0.5] * data.y.shape[1])
    return model, best_thresholds


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Residual GCN for multi-label classification")
    parser.add_argument('--data', default='processed_data/gcn_data.pt', help='Input PyG Data object (.pt)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batchnorm', action='store_true', help='Use BatchNorm (default False)')
    parser.add_argument('--layernorm', action='store_true', help='Use LayerNorm (default False)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='processed_data/prediction_output.csv', help='Prediction CSV output')
    parser.add_argument('--save-model', default='processed_data/gcn_model.pt', help='Save trained model path')
    parser.add_argument('--val-split', type=float, default=0.2, help='Random val split fraction (0-1)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early-stop', type=int, default=15)
    parser.add_argument('--tune-thresholds', action='store_true', help='Tune per-disease thresholds on validation')
    parser.add_argument('--use-posweight', action='store_true', help='Compute and use pos_weight from labels')
    args = parser.parse_args()

    # Load data
    data = torch.load(args.data)
    device = torch.device(args.device)

    in_dim = data.num_node_features
    num_labels = data.y.shape[1]

    model = ResidualGCN(in_dim, args.hidden_dim, num_labels, dropout=args.dropout,
                        use_bn=args.batchnorm, use_layernorm=args.layernorm)
    # compute pos_weight if requested
    pos_weight = None
    if args.use_posweight:
        pos_weight = compute_pos_weight(data.y)
        print("Using pos_weight:", pos_weight.numpy())

    # Train
    model, best_thresholds = train(
        model,
        data,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        val_split=args.val_split,
        seed=args.seed,
        early_stop=args.early_stop,
        verbose=True,
    )

    # Save model
    out_model = Path(args.save_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_model))
    print(f"Saved trained model to {out_model}")

    # Final evaluation on full data
    y_true, y_probs, metrics = evaluate_model(model, data, device, thresholds=None)
    # If requested, tune thresholds on val-split region (we already tuned during training and returned best_thresholds)
    if args.tune_thresholds:
        # If val_split>0, we already tuned thresholds during training and best_thresholds returned
        thresholds = best_thresholds
    else:
        thresholds = np.array([0.5] * num_labels)

    preds = (y_probs >= thresholds[None, :]).astype(int)

    # Compute final metrics
    micro_f1 = f1_score(y_true, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    print("Final metrics (using thresholds):")
    print(f"  Micro F1: {micro_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")

    # Save predictions CSV
    df = pd.DataFrame({'sample_id': np.arange(data.num_nodes)})
    for j in range(num_labels):
        df[f'true_label_{j}'] = y_true[:, j]
        df[f'prob_label_{j}'] = np.round(y_probs[:, j], 4)
        df[f'pred_label_{j}'] = preds[:, j]
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")


if __name__ == '__main__':
    main()
