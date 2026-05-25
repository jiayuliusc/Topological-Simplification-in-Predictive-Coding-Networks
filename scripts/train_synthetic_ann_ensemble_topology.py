import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

# 1. Dataset Loading
def load_data(filepath):
    data = np.load(filepath)
    X = data['points'].astype(np.float32)
    y = data['labels'].astype(np.int64)
    return X, y

# 2. ANN Architecture (30x4 + 18x4 ReLU) [cite: 540, 557]
class TopologyNet(nn.Module):
    def __init__(self):
        super(TopologyNet, self).__init__()
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
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def get_activations(self, x):
        """Returns activations at input and after each ReLU/Output layer."""
        activations = [x.detach().numpy()]
        out = x
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, nn.ReLU) or layer == self.layers[-1]:
                activations.append(out.detach().numpy())
        return activations

# 3. Training Function with Paper's LR Scheduler [cite: 544, 545, 546]
def train_model(X_train, y_train, seed):
    torch.manual_seed(seed)
    model = TopologyNet()
    criterion = nn.CrossEntropyLoss()
    
    # Paper settings for bottleneck architectures [cite: 545, 546]
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: 0.5 ** (t / 4000))
    
    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)
    
    for epoch in range(2000): # Increased max epochs to ensure 99% accuracy
        optimizer.zero_grad()
        outputs = model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        acc = (outputs.argmax(1) == y_t).float().mean().item()
        if acc >= 0.99 and epoch > 50:
            return model, acc, epoch
            
    return model, acc, 2000

# 4. Pipeline Execution
X, y = load_data('full_dataset.npz')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class_a_test = torch.tensor(X_test[y_test == 0])

os.makedirs("models", exist_ok=True)
all_betti_runs = []
num_runs = 30
k, eta = 14, 2.5 # Parameters specified for D-I dataset [cite: 593]

for run in range(num_runs):
    print(f"Training Model {run+1}/{num_runs}...")
    model, final_acc, final_epoch = train_model(X_train, y_train, seed=run)
    torch.save(model.state_dict(), f"models/model_seed_{run}.pth")
    
    # Analysis
    model.eval()
    activations = model.get_activations(class_a_test)
    run_b0 = []
    for act in activations:
        # Intrinsic graph metric (delta_k) implementation [cite: 568, 570]
        knn_graph = kneighbors_graph(act, n_neighbors=k, mode='connectivity', include_self=False)
        n_components, _ = connected_components(knn_graph, directed=False)
        run_b0.append(n_components)
    all_betti_runs.append(run_b0)

# 5. Statistics Calculation
all_betti_runs = np.array(all_betti_runs)
mean_per_layer = np.mean(all_betti_runs, axis=0)
std_per_layer = np.std(all_betti_runs, axis=0)
layers = range(len(mean_per_layer))

# 6. Plotting with Paper Style [cite: 707, 708]
plt.figure(figsize=(10, 6))
color = 'blue'

# Thin Blue Lines for Individual Runs
for i in range(num_runs):
    plt.plot(
        layers, 
        all_betti_runs[i], 
        color=color, 
        alpha=0.15, # Faint alpha for individual runs [cite: 707]
        linewidth=0.8
    )

# Thick Blue Line for Mean
plt.plot(
    layers, 
    mean_per_layer, 
    color=color, 
    linewidth=3.0, # Bold mean curve [cite: 707]
    marker='s',
    label='Mean β0'
)

# Shaded Band for Standard Deviation 
lower = mean_per_layer - (0.5 * std_per_layer) # Paper uses half-std region 
upper = mean_per_layer + (0.5 * std_per_layer)
plt.fill_between(layers, lower, upper, color=color, alpha=0.2, linewidth=0, label='±0.5 Std Dev')

# Final Formatting
plt.title(r"Topological Change Robustness (30x4 + 18x4 ReLU)", fontsize=15, pad=10)
plt.xlabel("Layer", fontsize=14, labelpad=6)
plt.ylabel(r"Betti Number $\beta_0$", fontsize=14, labelpad=6)
plt.xticks(layers, fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig("topology_robustness_plot.png", dpi=300, bbox_inches="tight")
print("Pipeline complete. Results saved in 'models/' and 'topology_robustness_plot.png'.")