import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import f1_score, hamming_loss

# ---------------- Load Data ----------------
data = torch.load('gcn_data.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# ---------------- Define GCN ----------------
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

in_dim = data.num_node_features
hidden_dim = 64
num_labels = data.y.shape[1]

model = GCNMultiLabel(in_dim, hidden_dim, num_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

# ---------------- Training Loop ----------------
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits, data.y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ---------------- Evaluation ----------------
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits)  # probabilities
    threshold = 0.3
    preds = (probs >= threshold).int()
    y_true = data.y.cpu().numpy()
    y_pred = preds.cpu().numpy()

# Micro/macro/Hamming metrics
print("\n--- Evaluation ---")
print("Micro F1:", f1_score(y_true, y_pred, average='micro'))
print("Macro F1:", f1_score(y_true, y_pred, average='macro'))
print("Hamming loss:", hamming_loss(y_true, y_pred))

# ---------------- Per-disease accuracy ----------------
disease_names = ['CRC', 'T2D', 'IBD', 'Cirrhosis', 'OBT']
print("\n--- Per-disease accuracy (%) ---")
for i, name in enumerate(disease_names):
    correct = (y_true[:, i] == y_pred[:, i]).sum()
    acc = correct / y_true.shape[0] * 100
    print(f"{name}: {acc:.2f}%")

# ---------------- Export predictions ----------------
df = pd.DataFrame({
    'sample_id': np.arange(data.num_nodes)
})

# True labels
for i, name in enumerate(disease_names):
    df[f'true_{name}'] = y_true[:, i]

# Predicted probabilities
for i, name in enumerate(disease_names):
    df[f'prob_{name}'] = np.round(probs[:, i].cpu().numpy(), 4)

# Predicted labels
for i, name in enumerate(disease_names):
    df[f'pred_{name}'] = y_pred[:, i]

# Predicted vs True match (True/False)
for i, name in enumerate(disease_names):
    df[f'match_{name}'] = df[f'true_{name}'] == df[f'pred_{name}']

df.to_csv('prediction_output.csv', index=False)
print("\nPredictions saved to prediction_output.csv")
