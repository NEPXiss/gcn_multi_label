"""
model/gcn_model.py
Phase 0 refactor for GCN multi-label classification
- Defines GCNMultiLabel model
- Training & evaluation functions
- CLI for train/eval + prediction CSV
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, hamming_loss

# ---------------- Model ----------------
class GCNMultiLabel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_labels):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_labels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        logits = self.out(x)
        return logits  # raw logits for BCEWithLogitsLoss

# ---------------- Training ----------------
def train(model, data, optimizer, criterion, epochs=50, device='cpu'):
    model.to(device)
    data = data.to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model

# ---------------- Evaluation ----------------
def evaluate(model, data, threshold=0.3, disease_names=None, device='cpu'):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int()
        y_true = data.y.cpu().numpy()
        y_pred = preds.cpu().numpy()

    # Metrics
    metrics = {
        'micro_f1': f1_score(y_true, y_pred, average='micro'),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'hamming_loss': hamming_loss(y_true, y_pred)
    }

    # Per-disease accuracy
    per_disease_acc = {}
    if disease_names is not None:
        for i, name in enumerate(disease_names):
            correct = (y_true[:, i] == y_pred[:, i]).sum()
            acc = correct / y_true.shape[0] * 100
            per_disease_acc[name] = acc

    return y_true, y_pred, probs.cpu().numpy(), metrics, per_disease_acc

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GCN multi-label")
    parser.add_argument('--data', default='processed_data/gcn_data.pt', help='Input PyG Data object (.pt)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--threshold', type=float, default=0.3, help='Sigmoid threshold for label prediction')
    parser.add_argument('--output', default='processed_data/prediction_output.csv', help='Prediction CSV output')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load data
    data = torch.load(args.data)
    device = torch.device(args.device)

    # Model setup
    in_dim = data.num_node_features
    num_labels = data.y.shape[1]
    model = GCNMultiLabel(in_dim, args.hidden_dim, num_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Train
    model = train(model, data, optimizer, criterion, epochs=args.epochs, device=device)

    # Evaluate
    disease_names = ['CRC', 'T2D', 'IBD', 'Cirrhosis', 'OBT']
    y_true, y_pred, probs, metrics, per_disease_acc = evaluate(
        model, data, threshold=args.threshold, disease_names=disease_names, device=device
    )

    # Print metrics
    print("\n--- Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\n--- Per-disease accuracy (%) ---")
    for name, acc in per_disease_acc.items():
        print(f"{name}: {acc:.2f}%")

    # Save predictions
    df = pd.DataFrame({'sample_id': np.arange(data.num_nodes)})
    for i, name in enumerate(disease_names):
        df[f'true_{name}'] = y_true[:, i]
        df[f'prob_{name}'] = np.round(probs[:, i], 4)
        df[f'pred_{name}'] = y_pred[:, i]
        df[f'match_{name}'] = df[f'true_{name}'] == df[f'pred_{name}']
    df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")

if __name__ == '__main__':
    main()
