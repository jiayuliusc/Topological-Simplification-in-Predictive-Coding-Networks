import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from ripser import ripser
from tqdm import tqdm
import os
import warnings
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from topological_dl.config import CONFIG, dataset_results_dir
from topological_dl.trainer import Trainer

# Suppress warnings
warnings.filterwarnings("ignore")

def get_raw_mnist_by_class():
    """
    Loads MNIST, normalizes to [0,1], flattens to 784 dim,
    and separates data by label (0-9).
    Returns a dictionary: {0: array_of_zeros, 1: array_of_ones, ...}
    """
    print("Loading and segregating MNIST by class...")
    
    # Standard transform: [0,1] scaling
    # transform = transforms.Compose([transforms.ToTensor()])
    
    # dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataset = torchvision.datasets.MNIST(root=str(CONFIG.data_dir), train=True, download=True)
    loader = DataLoader(dataset, batch_size=60000, shuffle=False)
    
    # Load all at once
    images, labels = next(iter(loader))
    
    # Flatten: (N, 28, 28) -> (N, 784)
    X_all = images.view(images.size(0), -1).numpy()
    y_all = labels.numpy()
    
    data_by_class = {}
    for i in range(10):
        data_by_class[i] = X_all[y_all == i]
        print(f"  Digit {i}: {data_by_class[i].shape[0]} samples")
        
    return data_by_class

def run_class_based_monte_carlo(n_trials=30, subsample_size=1500, output_dir="mnist_class_trials2"):
    # 1. Load Data
    # data_by_class = get_raw_mnist_by_class()
    import dill
    data_path = dataset_results_dir("MNIST") / "data" / "label_0_1500.dill"
    with data_path.open('rb') as f:
        data_by_class = dill.load(f)
    
    # 2. Setup Parameters
    k_range = range(3, 16)     
    eta_range = range(1, 14)   
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n--- Starting Monte Carlo ({n_trials} Trials) ---")
    print(f"Subsample per class: {subsample_size}")
    print(f"Scanning k: {list(k_range)}")
    print(f"Scanning eta: {list(eta_range)}")
    
    # 3. Main Trial Loop
    for trial in tqdm(range(n_trials), desc="Trials"):
        
        # Prepare content for this trial's CSV
        trial_filename = f"{output_dir}/trial_{trial}_all_classes.csv"
        
        with open(trial_filename, 'w') as f:
            f.write(f"Trial ID: {trial}\n")
            f.write(f"Subsample Size: {subsample_size}\n")
            f.write("="*40 + "\n\n")
            
            # 4. Iterate through Classes (0-9) inside this trial
            ## CHANGE BACK TO RANGE(10)
            for digit in range(1):
                # X_pool = data_by_class[digit]

                # rows = []
                # for x, y in data_by_class:
                #     x = x.unsqueeze(0)  # add batch dimension
                #     x_flat = x.reshape(x.shape[0], -1)
                #     x_jax = jnp.asarray(x_flat)
                
                #     x_np = x_flat.detach().cpu().numpy()  # reuse x_flat
                #     assert x_np.shape[1] == self.input_dim, f'Expected {self.input_dim} features, got {x_np.shape[1]}'
                #     rows.append(x_np)
                # X_sub = np.vstack(rows)

                base_dataset = data_by_class.dataset.base

                # Save and remove the problematic transform
                original_transform = base_dataset.transform
                base_dataset.transform = None  # Disable all transforms
                
                try:
                    # Now safely extract data
                    X_pool = []
                    for i in range(len(data_by_class)):
                        img, label = data_by_class[i]  # This will now work without transforms
                        
                        # Convert to numpy
                        if isinstance(img, torch.Tensor):
                            img = img.numpy()
                        else:
                            img = np.array(img)
                        
                        # Normalize to [0, 1] if needed (raw MNIST is 0-255)
                        if img.max() > 1.0:
                            img = img.astype(np.float32) / 255.0
                        
                        X_pool.append(img.flatten())
                    
                    X_pool = np.array(X_pool)
                    
                finally:
                    # Restore original transform (good practice)
                    base_dataset.transform = original_transform
                
                X_sub = X_pool
                
                # Subsample specific to this class
                # indices = np.random.choice(X_pool.shape[0], subsample_size, replace=False)
                # X_sub = X_pool[indices]
                
                # Containers for Pivot Tables
                records = []
                
                # Sweep parameters
                for k in k_range:
                    # Build Graph
                    nbrs = NearestNeighbors(n_neighbors=k).fit(X_sub)
                    G = nbrs.kneighbors_graph(X_sub, mode='connectivity')   # 0/1 adjacency
                    D = shortest_path(G, directed=False, unweighted=True)   # full NxN distances
                    
                    # Compute Persistence
                    result = ripser(D, distance_matrix=True, maxdim=1, thresh=20)
                    dgm0 = result['dgms'][0]
                    dgm1 = result['dgms'][1]
                    
                    for eta in eta_range:
                        b0 = np.sum(dgm0[:, 1] > eta)
                        b1 = np.sum((dgm1[:, 0] <= eta) & (dgm1[:, 1] > eta))
                        
                        records.append({'k': k, 'eta': eta, 'B0': b0, 'B1': b1})
                
                # Create Pivot Tables for this Digit
                df = pd.DataFrame(records)
                matrix_b0 = df.pivot(index='k', columns='eta', values='B0')
                matrix_b1 = df.pivot(index='k', columns='eta', values='B1')
                
                # Write to the single CSV
                f.write(f"=== DIGIT {digit} ANALYSIS ===\n")
                
                f.write(f"--- Digit {digit}: B0 (Components) ---\n")
                matrix_b0.to_csv(f)
                f.write("\n") # Spacer
                
                f.write(f"--- Digit {digit}: B1 (Holes) ---\n")
                matrix_b1.to_csv(f)
                f.write("\n" + "#"*40 + "\n\n") # Large separator between classes

    print("\nSimulation Complete.")
    print(f"Data saved to folder: {output_dir}")

if __name__ == "__main__":
    print('new version')
    run_class_based_monte_carlo(n_trials=1)
