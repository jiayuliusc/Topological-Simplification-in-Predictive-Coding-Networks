import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import dill
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from ripser import ripser
from tqdm import tqdm

from topological_dl.config import CONFIG, dataset_results_dir
from topological_dl.trainer import Trainer 

# --- Configuration ---
ROOT_DIR = dataset_results_dir("MNIST")
DATA_DIR = ROOT_DIR / "data_by_class"
STUDY_NAME = '256x8_ReLU' # This matches your folder name

RESULTS_DIR = ROOT_DIR / 'topology_csvs'
NUM_MODELS = 1
NUM_CLASSES = 10
EPSILONS = np.arange(1.5, 11.5, 1.0) # {1.5, 2.5, ... 10.5}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def betti_at_epsilon(diagram, epsilon):
    """
    Counts intervals (birth, death) such that birth <= epsilon < death.
    """
    if diagram is None or len(diagram) == 0:
        return 0
    # diagram is (N, 2) array of [birth, death]
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    is_alive = (births <= epsilon) & (deaths > epsilon)
    return np.sum(is_alive)

def calculate_betti_sums(diagrams, epsilons):
    """
    diagrams: List of [H0, H1, H2] arrays for a specific layer.
    Returns: Dictionary {epsilon_value: sum_betti}
    """
    sums = {}
    for eps in epsilons:
        b0 = betti_at_epsilon(diagrams[0], eps)
        b1 = betti_at_epsilon(diagrams[1], eps)
        # Check if H2 exists (ripser sometimes returns fewer dims if empty)
        b2 = betti_at_epsilon(diagrams[2], eps) if len(diagrams) > 2 else 0
        
        sums[f'Sum_Eps_{eps:.1f}'] = b0 + b1 + b2
    return sums

def run_analysis(trainer_instance):
    
    for class_idx in range(NUM_CLASSES):
        print(f"\n--- Processing Class {class_idx} ---")
        
        # 1. Load Data (label_X_1500.dill)
        filename = f'label_{class_idx}_1500.dill'
        data_path = DATA_DIR / filename
        
        try:
            with open(data_path, 'rb') as f:
                # Assuming dill loads a list/array of data points
                X_raw = dill.load(f)
                
            # Create dummy dataset for Trainer.get_layers (expects (x, y) tuples)
            # We wrap X in jax array just to be safe for the model calls
            dummy_dataset = [(jnp.array(x), class_idx) for x in X_raw]
            
        except FileNotFoundError:
            print(f"Skipping Class {class_idx}: File {data_path} not found.")
            continue

        # List to collect all rows for this class's CSV
        csv_rows = []

        for model_idx in tqdm(range(NUM_MODELS), desc=f"Class {class_idx} Models"):
            
            # 2. Get Layers
            try:
                # get_layers uses the internal path: {root}/{study_name}/trained_models/model_{id}
                layers = trainer_instance.get_layers(
                    dataset=dummy_dataset, 
                    model_id=model_idx, 
                    input_layer=True, 
                    return_labels=False
                )
            except Exception as e:
                print(f"Error loading model {model_idx}: {e}")
                continue

            # 3. Compute PH for each layer
            for layer_idx, layer_activation in enumerate(layers):
                data_points = np.array(layer_activation)
                
                # Check for NaNs or Infs which crash Ripser
                if not np.isfinite(data_points).all():
                    print(f"Warning: Non-finite values in Model {model_idx} Layer {layer_idx}")
                    continue

                try:
                    # Run Ripser (maxdim=2 for H0, H1, H2)
                    # Reducing thresh slightly can speed it up if diagrams are huge
                    dgms = ripser(data_points, maxdim=2)['dgms']
                    
                    # 4. Calculate Sums
                    eps_sums = calculate_betti_sums(dgms, EPSILONS)
                    
                    # 5. Build CSV Row
                    row = {
                        'Model_ID': model_idx,
                        'Layer_ID': layer_idx,
                        'Num_Points': len(data_points)
                    }
                    # Merge the epsilon sums into the row dictionary
                    row.update(eps_sums)
                    
                    csv_rows.append(row)
                    
                except Exception as e:
                    print(f"Ripser failed Model {model_idx} Layer {layer_idx}: {e}")

        # 6. Save CSV for this Class
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            # Reorder columns to put Model/Layer first
            cols = ['Model_ID', 'Layer_ID'] + [c for c in df.columns if c not in ['Model_ID', 'Layer_ID']]
            df = df[cols]
            
            save_path = RESULTS_DIR / f'results_class_{class_idx}.csv'
            df.to_csv(save_path, index=False)
            print(f"Saved {save_path}")

if __name__ == "__main__":
    # Initialize Trainer with the correct study name
    # The Trainer class automatically looks for 'trained_models' inside the study folder
    trainer = Trainer(
        dataset='MNIST', 
        hidden_dims=[256]*8, 
        act_fn=jax.nn.relu, 
        study_name=STUDY_NAME,
        root=ROOT_DIR 
    )
    
    run_analysis(trainer)
