import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

# Load processed dataset
data_npz = np.load('processed_gcn_dataset.npz', allow_pickle=True)
X = data_npz['X_proc']
Y = data_npz['Y']

# Construct k-NN graph
k = 8  # number of neighbors
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
distances, indices = nbrs.kneighbors(X)

# Build symmetric edge_index
# indices[i] contains k neighbors of node i
edge_index_list = []
num_nodes = X.shape[0]

for i in range(num_nodes):
    for j in indices[i]:
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])  # ensure symmetry

edge_index = np.unique(edge_index_list, axis=0).T  # shape [2, num_edges]

# Convert to torch tensors
x = torch.tensor(X, dtype=torch.float)
y = torch.tensor(Y, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long)

# 3) Create PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)
torch.save(data, 'gcn_data.pt')

print("Saved PyG Data object to gcn_data.pt")
print("Graph data ready:")
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Feature shape:", data.x.shape)
print("Label shape:", data.y.shape)
