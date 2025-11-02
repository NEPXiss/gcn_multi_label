# model/semi_supervised_training.py
"""
Semi-supervised learning for GCN with Domain Adaptation
Supports training with both labeled and unlabeled nodes in the same graph
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import argparse
from pathlib import Path


# ============= Data Preparation =============
class SemiSupervisedData:
    """
    Wrapper for semi-supervised graph data with train/test masks
    """
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        test_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        study_info: Optional[np.ndarray] = None
    ):
        """
        Args:
            x: [N, input_dim] node features (all samples)
            y: [N, num_labels] labels (including unlabeled as -1 or 0)
            edge_index: [2, E] graph edges
            edge_weight: [E] edge weights (optional)
            train_mask: [N] boolean mask for labeled training samples
            test_mask: [N] boolean mask for unlabeled test samples
            val_mask: [N] boolean mask for validation samples (optional)
            study_info: [N] domain/study labels for each sample
        """
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.study_info = study_info
        
        # Create masks if not provided
        if train_mask is None:
            # Default: 80% train, 20% test
            N = x.shape[0]
            perm = torch.randperm(N)
            split = int(0.8 * N)
            train_mask = torch.zeros(N, dtype=torch.bool)
            train_mask[perm[:split]] = True
        
        if test_mask is None:
            test_mask = ~train_mask
        
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.val_mask = val_mask
        
        # Compute statistics
        self.num_nodes = x.shape[0]
        self.num_edges = edge_index.shape[1]
        self.num_features = x.shape[1]
        self.num_labels = y.shape[1] if len(y.shape) > 1 else 1
        self.num_train = train_mask.sum().item()
        self.num_test = test_mask.sum().item()
        self.num_val = val_mask.sum().item() if val_mask is not None else 0
    
    def to(self, device):
        """Move all tensors to device"""
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.to(device)
        self.train_mask = self.train_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        if self.val_mask is not None:
            self.val_mask = self.val_mask.to(device)
        return self
    
    def __repr__(self):
        return (f"SemiSupervisedData(nodes={self.num_nodes}, "
                f"edges={self.num_edges}, "
                f"train={self.num_train}, "
                f"val={self.num_val}, "
                f"test={self.num_test})")


def create_semi_supervised_split(
    data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    stratify: bool = True,
    seed: int = 42
):
    """
    Split data into train/val/test with masks for semi-supervised learning
    
    Args:
        data: PyG Data object with x, y, edge_index
        train_ratio: fraction for training (labeled)
        val_ratio: fraction for validation (labeled, for hyperparameter tuning)
        test_ratio: fraction for testing (unlabeled during training)
        stratify: whether to stratify by labels
        seed: random seed
    
    Returns:
        SemiSupervisedData object
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = data.x.shape[0]
    indices = np.arange(N)
    
    if stratify and len(data.y.shape) > 1:
        # Multi-label stratification: use first label
        labels = data.y[:, 0].cpu().numpy()
    else:
        labels = None
    
    # Shuffle
    np.random.shuffle(indices)
    
    # Compute splits
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Create semi-supervised data object
    ss_data = SemiSupervisedData(
        x=data.x,
        y=data.y,
        edge_index=data.edge_index,
        edge_weight=data.edge_weight if hasattr(data, 'edge_weight') else None,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        study_info=data.study_info if hasattr(data, 'study_info') else None
    )
    
    print(f"Created semi-supervised split:")
    print(f"  Train: {ss_data.num_train} ({train_ratio*100:.1f}%)")
    print(f"  Val: {ss_data.num_val} ({val_ratio*100:.1f}%)")
    print(f"  Test: {ss_data.num_test} ({test_ratio*100:.1f}%)")
    
    return ss_data


# ============= Training Functions =============
def train_epoch_semi_supervised(
    model,
    data: SemiSupervisedData,
    optimizer,
    criterion,
    device='cpu',
    da_method: Optional[str] = None,
    da_weight: float = 0.1,
    mmd_loss_fn=None,
    coral_loss_fn=None
):
    """
    Training epoch for semi-supervised learning
    
    Key: Loss computed ONLY on labeled (train_mask) nodes,
         but forward pass includes ALL nodes for message passing
    
    Args:
        model: DomainAdaptiveGCN or similar
        data: SemiSupervisedData object
        optimizer: torch optimizer
        criterion: loss function (BCEWithLogitsLoss for multi-label)
        device: computation device
        da_method: 'mmd', 'coral', 'dann', or None
        da_weight: weight for domain adaptation loss
        mmd_loss_fn: MMDLoss instance
        coral_loss_fn: CoralLoss instance
    
    Returns:
        dict with losses
    """
    model.train()
    model.to(device)
    data = data.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass on ALL nodes (labeled + unlabeled)
    if hasattr(model, 'forward') and 'return_latent' in model.forward.__code__.co_varnames:
        logits, latent = model(data.x, data.edge_index, 
                               edge_weight=data.edge_weight, 
                               return_latent=True)
    else:
        logits = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        latent = None
    
    # Classification loss ONLY on labeled training nodes
    train_logits = logits[data.train_mask]
    train_labels = data.y[data.train_mask]
    
    cls_loss = criterion(train_logits, train_labels)
    
    # Domain adaptation loss (if applicable)
    da_loss = torch.tensor(0.0).to(device)
    
    if da_method and da_weight > 0 and latent is not None:
        if data.study_info is not None:
            # Get domain information
            train_domains = data.study_info[data.train_mask.cpu().numpy()]
            unique_domains = np.unique(train_domains)
            
            if len(unique_domains) > 1:
                # Use largest domain as source, others as target
                domain_counts = {d: np.sum(train_domains == d) for d in unique_domains}
                source_domain = max(domain_counts, key=domain_counts.get)
                
                # Get indices within training set
                train_indices = torch.where(data.train_mask)[0]
                source_mask_in_train = train_domains == source_domain
                target_mask_in_train = ~source_mask_in_train
                
                source_indices = train_indices[torch.from_numpy(source_mask_in_train)]
                target_indices = train_indices[torch.from_numpy(target_mask_in_train)]
                
                if len(source_indices) > 0 and len(target_indices) > 0:
                    source_latent = latent[source_indices]
                    target_latent = latent[target_indices]
                    
                    # Compute DA loss
                    if da_method == 'mmd':
                        if mmd_loss_fn is None:
                            from domain_adaptation import MMDLoss
                            mmd_loss_fn = MMDLoss().to(device)
                        da_loss = mmd_loss_fn(source_latent, target_latent)
                    
                    elif da_method == 'coral':
                        if coral_loss_fn is None:
                            from domain_adaptation import CoralLoss
                            coral_loss_fn = CoralLoss().to(device)
                        da_loss = coral_loss_fn(source_latent, target_latent)
                    
                    elif da_method == 'dann':
                        # DANN implementation
                        batch_size_source = source_latent.size(0)
                        batch_size_target = target_latent.size(0)
                        
                        domain_label_source = torch.zeros(batch_size_source, 1).to(device)
                        domain_label_target = torch.ones(batch_size_target, 1).to(device)
                        
                        domain_pred_source = model.get_domain_logits(source_latent)
                        domain_pred_target = model.get_domain_logits(target_latent)
                        
                        da_loss = F.binary_cross_entropy_with_logits(
                            torch.cat([domain_pred_source, domain_pred_target], dim=0),
                            torch.cat([domain_label_source, domain_label_target], dim=0)
                        )
    
    # Total loss
    total_loss = cls_loss + da_weight * da_loss
    
    total_loss.backward()
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'cls_loss': cls_loss.item(),
        'da_loss': da_loss.item()
    }


@torch.no_grad()
def evaluate_semi_supervised(
    model,
    data: SemiSupervisedData,
    mask: torch.Tensor,
    device='cpu',
    thresholds: Optional[np.ndarray] = None
):
    """
    Evaluate model on specified nodes (train/val/test)
    
    Args:
        model: trained model
        data: SemiSupervisedData object
        mask: boolean mask for nodes to evaluate
        device: computation device
        thresholds: per-label thresholds (or None for 0.5)
    
    Returns:
        dict with metrics
    """
    model.eval()
    model.to(device)
    data = data.to(device)
    
    # Forward pass on ALL nodes
    logits = model(data.x, data.edge_index, edge_weight=data.edge_weight)
    
    # Extract predictions for specified mask
    eval_logits = logits[mask]
    eval_labels = data.y[mask]
    
    # Convert to probabilities
    probs = torch.sigmoid(eval_logits).cpu().numpy()
    y_true = eval_labels.cpu().numpy()
    
    # Apply thresholds
    if thresholds is None:
        thresholds = np.array([0.5] * y_true.shape[1])
    
    y_pred = (probs >= thresholds[None, :]).astype(int)
    
    # Compute metrics
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-label metrics
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Try to compute AUC (might fail if only one class present)
    try:
        micro_auc = roc_auc_score(y_true, probs, average='micro')
        macro_auc = roc_auc_score(y_true, probs, average='macro')
    except:
        micro_auc = float('nan')
        macro_auc = float('nan')
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'per_label_f1': per_label_f1,
        'micro_auc': micro_auc,
        'macro_auc': macro_auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': probs
    }


def train_semi_supervised(
    model,
    data: SemiSupervisedData,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    da_method: Optional[str] = None,
    da_weight: float = 0.1,
    early_stop: int = 15,
    device='cpu',
    pos_weight: Optional[torch.Tensor] = None,
    verbose: bool = True
):
    """
    Complete semi-supervised training loop
    
    Args:
        model: DomainAdaptiveGCN
        data: SemiSupervisedData object
        epochs: number of training epochs
        lr: learning rate
        weight_decay: L2 regularization
        da_method: domain adaptation method
        da_weight: weight for DA loss
        early_stop: early stopping patience
        device: computation device
        pos_weight: class weights for BCEWithLogitsLoss
        verbose: print progress
    
    Returns:
        trained model, best thresholds, training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Initialize DA loss functions
    mmd_loss_fn = None
    coral_loss_fn = None
    if da_method == 'mmd':
        from .domain_adaptation import MMDLoss
        mmd_loss_fn = MMDLoss().to(device)
    elif da_method == 'coral':
        from .domain_adaptation import CoralLoss
        coral_loss_fn = CoralLoss().to(device)
    
    best_val_f1 = -1.0
    best_state = None
    best_thresholds = None
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_f1': [],
        'val_auc': []
    }
    
    for epoch in range(1, epochs + 1):
        # Train
        loss_dict = train_epoch_semi_supervised(
            model=model,
            data=data,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            da_method=da_method,
            da_weight=da_weight,
            mmd_loss_fn=mmd_loss_fn,
            coral_loss_fn=coral_loss_fn
        )
        
        history['train_loss'].append(loss_dict['total_loss'])
        
        # Evaluate on validation set (if available)
        if data.val_mask is not None and data.val_mask.sum() > 0:
            val_results = evaluate_semi_supervised(
                model, data, data.val_mask, device=device
            )
            val_f1 = val_results['micro_f1']
            val_auc = val_results['micro_auc']
            
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)
        else:
            val_f1 = None
            val_auc = None
        
        # Print progress
        if verbose and (epoch % 5 == 0 or epoch == 1):
            info = f"Epoch {epoch:3d}/{epochs} | Loss: {loss_dict['total_loss']:.4f} "
            info += f"(Cls: {loss_dict['cls_loss']:.4f}, DA: {loss_dict['da_loss']:.4f})"
            if val_f1 is not None:
                info += f" | Val F1: {val_f1:.4f}"
            if val_auc is not None and not np.isnan(val_auc):
                info += f", AUC: {val_auc:.4f}"
            print(info)
        
        # Early stopping based on validation F1
        if val_f1 is not None:
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = model.state_dict()
                epochs_no_improve = 0
                
                # Tune thresholds on validation set
                from .gcn_model import tune_thresholds
                best_thresholds, _ = tune_thresholds(
                    val_results['y_true'],
                    val_results['y_probs']
                )
            else:
                epochs_no_improve += 1
            
            if early_stop and epochs_no_improve >= early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Default thresholds if no validation
    if best_thresholds is None:
        best_thresholds = np.array([0.5] * data.num_labels)
    
    return model, best_thresholds, history


# ============= Transductive Learning =============
def predict_unlabeled_nodes(
    model,
    data: SemiSupervisedData,
    device='cpu',
    thresholds: Optional[np.ndarray] = None
):
    """
    Predict labels for unlabeled (test_mask) nodes
    This is transductive learning: test nodes were in the graph during training!
    
    Args:
        model: trained model
        data: SemiSupervisedData object
        device: computation device
        thresholds: per-label thresholds
    
    Returns:
        predictions for test nodes
    """
    return evaluate_semi_supervised(model, data, data.test_mask, device, thresholds)


# ============= Inductive Learning =============
def predict_new_samples_inductive(
    model,
    encoder,
    train_data: SemiSupervisedData,
    new_samples: torch.Tensor,
    k: int = 5,
    device='cpu',
    thresholds: Optional[np.ndarray] = None
):
    """
    Predict labels for NEW samples not seen during training (inductive)
    
    Process:
    1. Encode new samples to latent space
    2. Find k nearest neighbors in training set (using latent features)
    3. Create edges from new samples to neighbors
    4. Run GCN forward pass
    
    Args:
        model: trained GCN model
        encoder: trained encoder (extract from model)
        train_data: original training graph
        new_samples: [M, input_dim] new samples to predict
        k: number of neighbors to connect
        device: computation device
        thresholds: per-label thresholds
    
    Returns:
        predictions for new samples
    """
    from sklearn.neighbors import NearestNeighbors
    
    model.eval()
    encoder.eval()
    
    new_samples = new_samples.to(device)
    
    # Encode new samples and training samples
    with torch.no_grad():
        new_latent = encoder(new_samples).cpu().numpy()
        train_latent = encoder(train_data.x.to(device)).cpu().numpy()
    
    # Find k-NN in latent space
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(train_latent)
    distances, indices = nn.kneighbors(new_latent)
    
    # Build augmented graph: [train_nodes + new_nodes]
    M = new_samples.shape[0]
    N = train_data.num_nodes
    
    # Augmented node features
    x_augmented = torch.cat([train_data.x, new_samples], dim=0)
    
    # Augmented edges: original edges + new edges to neighbors
    new_edges = []
    for i in range(M):
        new_node_idx = N + i
        for neighbor_idx in indices[i]:
            # Bidirectional edges
            new_edges.append([new_node_idx, neighbor_idx])
            new_edges.append([neighbor_idx, new_node_idx])
    
    new_edges = torch.tensor(new_edges, dtype=torch.long).t()
    edge_index_augmented = torch.cat([train_data.edge_index, new_edges], dim=1)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x_augmented.to(device), edge_index_augmented.to(device))
        new_logits = logits[N:]  # Extract predictions for new nodes
        probs = torch.sigmoid(new_logits).cpu().numpy()
    
    # Apply thresholds
    if thresholds is None:
        thresholds = np.array([0.5] * probs.shape[1])
    
    y_pred = (probs >= thresholds[None, :]).astype(int)
    
    return {
        'y_pred': y_pred,
        'y_probs': probs
    }


# ============= CLI =============
def main():
    parser = argparse.ArgumentParser(description="Semi-supervised GCN Training")
    parser.add_argument('--data', default='processed_data/gcn_data.pt')
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--da-method', choices=['mmd', 'coral', 'dann', 'none'], default='mmd')
    parser.add_argument('--da-weight', type=float, default=0.1)
    parser.add_argument('--early-stop', type=int, default=15)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-model', default='processed_data/gcn_semi_supervised.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-posweight', action='store_true')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = torch.load(args.data)
    
    # Create semi-supervised split
    ss_data = create_semi_supervised_split(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Create model
    from .domain_adaptation import DomainAdaptiveGCN
    
    model = DomainAdaptiveGCN(
        input_dim=ss_data.num_features,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_labels=ss_data.num_labels,
        dropout=args.dropout,
        da_method=args.da_method if args.da_method != 'none' else None
    )
    
    # Compute pos_weight if needed
    pos_weight = None
    if args.use_posweight:
        from gcn_model import compute_pos_weight
        train_labels = ss_data.y[ss_data.train_mask]
        pos_weight = compute_pos_weight(train_labels)
        print(f"Using pos_weight: {pos_weight.numpy()}")
    
    # Train
    print("\n" + "="*60)
    print("Starting Semi-supervised Training")
    print("="*60)
    
    model, best_thresholds, history = train_semi_supervised(
        model=model,
        data=ss_data,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-5,
        da_method=args.da_method if args.da_method != 'none' else None,
        da_weight=args.da_weight,
        early_stop=args.early_stop,
        device=args.device,
        pos_weight=pos_weight,
        verbose=True
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    # Evaluate on train set
    train_results = evaluate_semi_supervised(
        model, ss_data, ss_data.train_mask, args.device, best_thresholds
    )
    print(f"Train - Micro F1: {train_results['micro_f1']:.4f}, "
          f"Macro F1: {train_results['macro_f1']:.4f}")
    
    # Evaluate on validation set
    if ss_data.val_mask is not None:
        val_results = evaluate_semi_supervised(
            model, ss_data, ss_data.val_mask, args.device, best_thresholds
        )
        print(f"Val   - Micro F1: {val_results['micro_f1']:.4f}, "
              f"Macro F1: {val_results['macro_f1']:.4f}")
    
    # Evaluate on test set (transductive)
    test_results = evaluate_semi_supervised(
        model, ss_data, ss_data.test_mask, args.device, best_thresholds
    )
    print(f"Test  - Micro F1: {test_results['micro_f1']:.4f}, "
          f"Macro F1: {test_results['macro_f1']:.4f}")
    
    # Per-label results
    print("\nPer-label F1 scores (Test set):")
    for i, f1 in enumerate(test_results['per_label_f1']):
        print(f"  Label {i}: {f1:.4f}")
    
    # Save model
    out_model = Path(args.save_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'thresholds': best_thresholds,
        'config': vars(args),
        'history': history,
        'train_mask': ss_data.train_mask,
        'val_mask': ss_data.val_mask,
        'test_mask': ss_data.test_mask
    }, str(out_model))
    
    print(f"\nModel saved to {out_model}")

        # =======================================================
    # TODO --- Save predictions like old version ---
    # =======================================================
    import numpy as np
    import pandas as pd

    # ใช้ผลจาก test set (หรือจะบันทึก train/val ด้วยก็ได้)
    y_true = test_results['y_true']
    y_probs = test_results['y_probs']
    preds = test_results['y_pred']

    # ปรับชื่อโรคให้ตรงกับ label
    disease_names = ['CRC', 'T2D', 'IBD', 'Cirrhosis', 'OBT'][:y_true.shape[1]]

    print("\n--- Per-disease accuracy (%) ---")
    for j, name in enumerate(disease_names):
        correct = (y_true[:, j] == preds[:, j]).sum()
        acc = correct / y_true.shape[0] * 100
        print(f"{name}: {acc:.2f}%")

    # สร้าง DataFrame
    df = pd.DataFrame({'sample_id': np.arange(len(y_true))})
    for j, name in enumerate(disease_names):
        df[f'true_{name}'] = y_true[:, j]
        df[f'prob_{name}'] = np.round(y_probs[:, j], 4)
        df[f'pred_{name}'] = preds[:, j]

    # บันทึกเป็นไฟล์ CSV
    out_csv = Path(args.save_model).with_suffix('.csv')
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved predictions to {out_csv}")



if __name__ == '__main__':
    main()