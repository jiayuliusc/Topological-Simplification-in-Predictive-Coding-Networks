import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

def load_data(filepath):
    data = np.load(filepath)
    X = data['points'].astype(np.float32)
    y = data['labels'].astype(np.int64)
    return X, y
    
# 2. ANN Architecture: 30x4 + 18x4 ReLU
class TopologyNet(nn.Module):
    def __init__(self):
        super(TopologyNet, self).__init__()
        # Architecture: Input(2) -> 30x4 -> 18x4 -> Output(2)
        self.layers = nn.ModuleList([
            nn.Linear(2, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30, 18), nn.ReLU(),
            nn.Linear(18, 18), nn.ReLU(),
            nn.Linear(18, 18), nn.ReLU(),
            nn.Linear(18, 18), nn.ReLU(),
            nn.Linear(18, 2)
        ])
    
    def forward_with_activations(self, x):
        activations = [x.detach().numpy()]
        out = x
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, nn.ReLU) or layer == self.layers[-1]:
                activations.append(out.detach().numpy())
        return out, activations

# 3. Training Loop
X, y = load_data('full_dataset.npz')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = TopologyNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Training to achieve 99% accuracy...")
for epoch in range(1000):
    optimizer.zero_grad()
    outputs, _ = model.forward_with_activations(torch.tensor(X_train))
    loss = criterion(outputs, torch.tensor(y_train))
    loss.backward()
    optimizer.step()
    
    acc = (outputs.argmax(1) == torch.tensor(y_train)).float().mean().item()
    if acc >= 0.99 and epoch > 50:
        print(f"Goal reached at epoch {epoch}: Accuracy {acc:.4f}")
        break

# 4. Topological Analysis (Beta_0 at each layer)
# Using Class A (the 9 disks) to track simplification
class_a_test = torch.tensor(X_test[y_test == 0])
_, activations = model.forward_with_activations(class_a_test)

beta_0_values = []
k, eta = 15, 2.5 # Parameters from prompt

for i, act in enumerate(activations):
    # Create KNN graph (distance=1 if neighbors)
    # The graph distance metric implies path length
    knn_graph = kneighbors_graph(act, n_neighbors=k, mode='connectivity', include_self=False)
    
    # In the paper's metric, points are connected if path distance <= eta (2.5)
    # Since path distances are integers, this means points are connected if they are
    # 1-hop or 2-hop neighbors in the KNN graph.
    # Note: For beta_0, connectivity is preserved if the base KNN graph is connected.
    # We find the number of connected components in the adjacency matrix.
    n_components, labels = connected_components(knn_graph, directed=False)
    beta_0_values.append(n_components)

# 5. Save Results to File
# Define your figure and axis labels, along with the grid
plt.figure(figsize=(10, 6))  # Same as the figsize in the graph_betti_numbers function
plt.plot(range(len(beta_0_values)), beta_0_values, marker='s', linestyle='-', color='blue', linewidth=2.0)  # Style from the graph_betti_mean_band function

# Title and labels
plt.title(r"30x4 + 18x4", fontsize=15, pad=10)  # Adjusted font size and padding
plt.xlabel("Layer", fontsize=14, labelpad=6)  # Matching the fontsize and label padding
plt.ylabel(r"Betti Number $\beta_0$", fontsize=14, labelpad=6)  # Same fontsize and padding as in the function

# Customize the ticks and grid
plt.xticks(range(len(beta_0_values)), fontsize=11)  # Ensuring proper tick size
plt.yticks(fontsize=11)  # Ensuring proper tick size
plt.grid(True, alpha=0.3)  # Matching the grid transparency

# Save the plot as PNG
save_path = "topology_results.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save with tight layout and high resolution
print(f"Graph saved successfully to {save_path}")

# Show the plot
plt.show()