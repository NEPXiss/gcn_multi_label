# model/train_with_domain_adaptation.py
"""
Training script with Domain Adaptation support
Supports MMD, CORAL, and DANN methods
Compatible with multi-label classification
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score

from .domain_adaptation import (
    DomainAdaptiveGCN,
    MMDLoss,
    CoralLoss,
    compute_domain_adaptation_loss
)


# ============= Domain-Aware Data Preparation =============
def prepare_domain_splits(data, domain_col='study', val_split=0.2, seed=42):
    """
    Split data into train/val while tracking domain information
    
    Args:
        data: PyG Data object with data.study_info (numpy array of study names)
        domain_col: column name for domain/study information
        val_split: fraction for validation
        seed: random seed
    Returns:
        splits: dict with 'train_idx', 'val_idx', 'train_domains', 'val_domains'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = data.num_nodes
    idx = np.arange(N)
    np.random.shuffle(idx)
    
    split_point = int(N * (1 - val_split))
    train_idx = idx[:split_point]
    val_idx = idx[split_point:]
    
    # If domain information available
    if hasattr(data, 'study_info'):
        study_info = data.study_info
        train_domains = study_info[train_idx]
        val_domains = study_info[val_idx]
    else:
        # If no domain info, treat all as same domain
        train_domains = np.array(['train'] * len(train_idx))
        val_domains = np.array(['val'] * len(val_idx))
    
    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'train_domains': train_domains,
        'val_domains': val_domains
    }


def get_domain_batches(indices, domains, batch_size=64):
    """
    Create batches with domain information for DA training
    
    Args:
        indices: node indices
        domains: domain labels for each index
        batch_size: batch size
    Returns:
        list of (batch_indices, batch_domains)
    """
    # Group by domain
    unique_domains = np.unique(domains)
    domain_to_indices = {d: indices[domains == d] for d in unique_domains}
    
    batches = []
    # Simple random sampling from all domains
    remaining = list(indices)
    np.random.shuffle(remaining)
    
    for i in range(0, len(remaining), batch_size):
        batch_idx = remaining[i:i+batch_size]
        batch_domains = domains[[np.where(indices == idx)[0][0] for idx in batch_idx]]
        batches.append((batch_idx, batch_domains))
    
    return batches


# ============= Training Function with DA =============
def train_epoch_with_da(
    model,
    data,
    optimizer,
    train_idx,
    train_domains,
    criterion,
    da_method='mmd',
    da_weight=0.1,
    device='cpu'
):
    """
    Training epoch with domain adaptation
    
    Args:
        model: DomainAdaptiveGCN
        data: PyG Data object
        optimizer: torch optimizer
        train_idx: training node indices
        train_domains: domain labels for training nodes
        criterion: classification loss (BCEWithLogitsLoss)
        da_method: 'mmd', 'coral', 'dann', or None
        da_weight: weight for domain adaptation loss
        device: computation device
    Returns:
        total_loss, cls_loss, da_loss
    """
    model.train()
    model.to(device)
    data = data.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass with latent features
    if hasattr(data, 'edge_weight'):
        logits, latent = model(data.x, data.edge_index, 
                               edge_weight=data.edge_weight, 
                               return_latent=True)
    else:
        logits, latent = model(data.x, data.edge_index, 
                               edge_weight=None, 
                               return_latent=True)
    
    # Classification loss (only on training nodes)
    train_logits = logits[train_idx]
    train_labels = data.y[train_idx]
    cls_loss = criterion(train_logits, train_labels)
    
    # Domain adaptation loss
    da_loss = torch.tensor(0.0).to(device)
    
    if da_method and da_weight > 0:
        # Get unique domains in training set
        unique_domains = np.unique(train_domains)
        
        if len(unique_domains) > 1:
            # Split into source and target domains
            # Strategy: use largest domain as source, others as target
            domain_counts = {d: np.sum(train_domains == d) for d in unique_domains}
            source_domain = max(domain_counts, key=domain_counts.get)
            
            source_mask = train_domains == source_domain
            target_mask = ~source_mask
            
            source_indices = train_idx[source_mask]
            target_indices = train_idx[target_mask]
            
            if len(source_indices) > 0 and len(target_indices) > 0:
                source_latent = latent[source_indices]
                target_latent = latent[target_indices]
                
                # Compute DA loss
                if da_method == 'mmd':
                    mmd_loss_fn = MMDLoss().to(device)
                    da_loss = mmd_loss_fn(source_latent, target_latent)
                
                elif da_method == 'coral':
                    coral_loss_fn = CoralLoss().to(device)
                    da_loss = coral_loss_fn(source_latent, target_latent)
                
                elif da_method == 'dann':
                    # DANN: domain classifier loss with gradient reversal
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
    
    return total_loss.item(), cls_loss.item(), da_loss.item()


# ============= Evaluation Function =============
def evaluate_with_da(model, data, eval_idx, device='cpu', thresholds=None):
    """
    Evaluate model on given indices
    """
    model.eval()
    model.to(device)
    data = data.to(device)
    
    with torch.no_grad():
        if hasattr(data, 'edge_weight'):
            logits = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            logits = model(data.x, data.edge_index, edge_weight=None)
        
        probs = torch.sigmoid(logits[eval_idx]).cpu().numpy()
        y_true = data.y[eval_idx].cpu().numpy()
    
    # Apply thresholds
    if thresholds is None:
        thresholds = np.array([0.5] * y_true.shape[1])
    
    preds = (probs >= thresholds[None, :]).astype(int)
    
    micro_f1 = f1_score(y_true, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'y_true': y_true,
        'y_probs': probs,
        'y_pred': preds
    }


# ============= Full Training Loop =============
def train_with_domain_adaptation(
    model,
    data,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    da_method='mmd',
    da_weight=0.1,
    val_split=0.2,
    early_stop=15,
    device='cpu',
    verbose=True,
    pos_weight=None,
    seed=42
):
    """
    Complete training loop with domain adaptation
    
    Args:
        model: DomainAdaptiveGCN
        data: PyG Data with optional 'study_info' attribute
        epochs: number of training epochs
        lr: learning rate
        weight_decay: L2 regularization
        da_method: 'mmd', 'coral', 'dann', or None
        da_weight: weight for DA loss (lambda)
        val_split: validation split ratio
        early_stop: early stopping patience
        device: 'cpu' or 'cuda'
        verbose: print progress
        pos_weight: class weights for imbalanced data
        seed: random seed
    Returns:
        trained model, best thresholds
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prepare splits
    splits = prepare_domain_splits(data, val_split=val_split, seed=seed)
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    train_domains = splits['train_domains']
    val_domains = splits['val_domains']
    
    if verbose:
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")
        print(f"Training domains: {np.unique(train_domains)}")
        print(f"DA method: {da_method}, weight: {da_weight}")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    best_thresholds = None
    
    for epoch in range(1, epochs + 1):
        # Train
        total_loss, cls_loss, da_loss = train_epoch_with_da(
            model=model,
            data=data,
            optimizer=optimizer,
            train_idx=train_idx,
            train_domains=train_domains,
            criterion=criterion,
            da_method=da_method,
            da_weight=da_weight,
            device=device
        )
        
        # Validate
        if val_split > 0:
            val_results = evaluate_with_da(model, data, val_idx, device=device)
            val_f1 = val_results['micro_f1']
        else:
            val_f1 = None
        
        # Print progress
        if verbose and (epoch % 5 == 0 or epoch == 1):
            info = f"Epoch {epoch}/{epochs} | Total: {total_loss:.4f} | Cls: {cls_loss:.4f} | DA: {da_loss:.4f}"
            if val_f1 is not None:
                info += f" | Val F1: {val_f1:.4f}"
            print(info)
        
        # Early stopping
        if val_split > 0:
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
    
    if best_thresholds is None:
        best_thresholds = np.array([0.5] * data.y.shape[1])
    
    return model, best_thresholds


# ============= CLI =============
def main():
    parser = argparse.ArgumentParser(description="Train GCN with Domain Adaptation")
    parser.add_argument('--data', default='processed_data/gcn_data.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--da-method', choices=['mmd', 'coral', 'dann', 'none'], default='mmd')
    parser.add_argument('--da-weight', type=float, default=0.1, 
                        help='Weight for domain adaptation loss (lambda)')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--early-stop', type=int, default=15)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-model', default='processed_data/gcn_model_da.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-posweight', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Load data
    data = torch.load(args.data)
    device = torch.device(args.device)
    
    input_dim = data.num_node_features
    num_labels = data.y.shape[1]
    
    # Create model
    model = DomainAdaptiveGCN(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_labels=num_labels,
        dropout=args.dropout,
        da_method=args.da_method if args.da_method != 'none' else None
    )
    
    # Compute pos_weight if needed
    pos_weight = None
    if args.use_posweight:
        from .gcn_model import compute_pos_weight
        pos_weight = compute_pos_weight(data.y)
        if args.verbose:
            print("Using pos_weight:", pos_weight.numpy())
    
    # Train
    model, best_thresholds = train_with_domain_adaptation(
        model=model,
        data=data,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        da_method=args.da_method if args.da_method != 'none' else None,
        da_weight=args.da_weight,
        val_split=args.val_split,
        early_stop=args.early_stop,
        device=device,
        verbose=args.verbose,
        pos_weight=pos_weight,
        seed=args.seed
    )
    
    # Save model
    out_model = Path(args.save_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'thresholds': best_thresholds,
        'config': vars(args)
    }, str(out_model))
    
    print(f"\nSaved trained model to {out_model}")
    
    # Final evaluation
    results = evaluate_with_da(model, data, np.arange(data.num_nodes), device, best_thresholds)
    print(f"\nFinal Micro F1: {results['micro_f1']:.4f}")
    print(f"Final Macro F1: {results['macro_f1']:.4f}")

    # TODO --- Save predictions like old version ---
    import pandas as pd
    y_true = results['y_true']
    y_probs = results['y_probs']
    preds = results['y_pred']

    disease_names = ['CRC', 'T2D', 'IBD', 'Cirrhosis', 'OBT']

    print("\n--- Per-disease accuracy (%) ---")
    for j, name in enumerate(disease_names):
        correct = (y_true[:, j] == preds[:, j]).sum()
        acc = correct / y_true.shape[0] * 100
        print(f"{name}: {acc:.2f}%")

    # Save to CSV
    df = pd.DataFrame({'sample_id': np.arange(len(y_true))})
    for j, name in enumerate(disease_names):
        df[f'true_{name}'] = y_true[:, j]
        df[f'prob_{name}'] = np.round(y_probs[:, j], 4)
        df[f'pred_{name}'] = preds[:, j]

    out_csv = Path(args.save_model).with_suffix('.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

if __name__ == '__main__':
    main()