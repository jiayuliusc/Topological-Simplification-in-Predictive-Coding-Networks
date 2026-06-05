"""Trainer class and analysis utilities."""

from __future__ import annotations

import sys
from typing import Callable, Dict, Tuple
import ast
from pathlib import Path

try:
    from .config import CONFIG, dataset_results_dir
except ImportError:
    from config import CONFIG, dataset_results_dir

for dependency_path in (
    CONFIG.pcx_dir,
    CONFIG.ripser_plusplus_dir,
):
    if dependency_path is not None and dependency_path.exists():
        sys.path.append(str(dependency_path))

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import equinox as eqx

try:
    import pcx as px
    import pcx.predictive_coding as pxc
    import pcx.nn as pxnn
    import pcx.functional as pxf
    import pcx.utils as pxu
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import pcx. Install pcx, set TDL_PCX_DIR, "
        "or edit config/local_config.json to point to a local checkout. "
        "See docs/replication.md for configuration details."
    ) from exc

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, ConcatDataset
from torchvision import datasets, transforms
import optax

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, pairwise_distances

from scipy import sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path, minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dill
import pickle
from ripser import ripser
from tqdm import tqdm
import json
import re
from collections import defaultdict
import itertools

import optuna
from optuna_dashboard import run_server

import threading
import time
import os
import socket
import gc
import subprocess


try:
    from .data_loading import load_dataset
    from .pcn_model import Model, forward
    from .pcn_training import eval, get_opts, train
except ImportError:
    from data_loading import load_dataset
    from pcn_model import Model, forward
    from pcn_training import eval, get_opts, train


class Trainer:
    root = str(CONFIG.results_dir)
    results_root = str(CONFIG.results_dir)
    train_dataset_d1 = None
    test_dataset_d1 = None
    full_dataset_d1 = None
    all_green = None
    all_red = None
    svm = None

    _keys_path = CONFIG.results_dir / "keys.npz"
    if _keys_path.exists():
        with np.load(_keys_path) as f:
            model_keys = f["model_keys"]
            epoch_keys_per_model = f["epoch_keys_per_model"]
    else:
        _base_key = jax.random.PRNGKey(0)
        model_keys = jax.random.split(_base_key, 256)
        epoch_keys_per_model = jax.random.split(jax.random.PRNGKey(1), 256 * 1000).reshape(256, 1000, 2)

    @classmethod
    def _load_d1_assets(cls):
        if cls.train_dataset_d1 is not None:
            return

        d1_root = dataset_results_dir("D1")
        split_path = d1_root / "train_test_split.pkl"
        full_dataset_path = d1_root / "full_dataset.npz"
        svm_path = d1_root / "svm.dill"

        missing = [str(p) for p in (split_path, full_dataset_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "D1 dataset assets are missing. Set TDL_RESULTS_DIR or "
                "config/local_config.json so these files exist: " + ", ".join(missing)
            )

        with split_path.open("rb") as f:
            train_dataset, test_dataset = dill.load(f)

        all_data = np.load(full_dataset_path)
        all_points = all_data["points"]
        all_labels = all_data["labels"]

        cls.train_dataset_d1 = train_dataset
        cls.test_dataset_d1 = test_dataset
        cls.full_dataset_d1 = (all_points, all_labels)

        green_indices = np.where(all_labels == 0)
        cls.all_green = (all_points[green_indices], all_labels[green_indices])

        red_indices = np.where(all_labels == 1)
        cls.all_red = (all_points[red_indices], all_labels[red_indices])

        if svm_path.exists():
            with svm_path.open("rb") as f:
                cls.svm = dill.load(f)
    
    def __init__(
        self,
        dataset: str,  # 'D1', 'D2', 'D3', or 'MNIST'
        hidden_dims: list[int],
        act_fn,
        study_name: str,
        root: str = None,
        residual=False,
        num_models=50,
        num_epochs=200,
        batch_size=32
    ):
        if dataset not in ['D1', 'D2', 'D3', 'MNIST', 'FASHIONMNIST']:
            raise Exception('dataset argument must be D1, D2, D3, MNIST, or FASHIONMNIST')
        
        self.dataset = dataset
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn
        self.study_name = study_name
        self.residual = residual
        self.num_models = num_models
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.root = str(Path(root).expanduser() if root is not None else dataset_results_dir(dataset))

        if dataset == 'D1':
            Trainer._load_d1_assets()
            self.train_dataset = Trainer.train_dataset_d1
            self.test_dataset = Trainer.test_dataset_d1
            self.input_dim = 2
            self.output_dim = 2
            self.true_b = [9, None]
        else:
            # Standard datasets like MNIST/FashionMNIST/CIFAR10
            self.train_dataset, self.test_dataset, self.input_dim, self.output_dim = load_dataset(dataset)
            self.true_b = [None, None, None]  # Not meaningful for real image datasets
            
        self.full_dataset = ConcatDataset([self.train_dataset, self.test_dataset])

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(f'{self.root}/{self.study_name}', exist_ok=True)
        os.makedirs(f'{self.root}/{self.study_name}/trained_models', exist_ok=True)
            
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def get_epoch_dataloader(self, model_id: int, epoch: int, train_dataset=None) -> DataLoader:
        """Get a DataLoader with epoch-specific seeding"""
        if train_dataset is None:
            train_dataset = self.train_dataset
        
        epoch_key = Trainer.epoch_keys_per_model[model_id][epoch]
        epoch_seed = int(jax.random.randint(epoch_key, (), 0, 2**31 - 1))
    
        # Set seeds for reproducibility
        torch.manual_seed(epoch_seed)
        np.random.seed(epoch_seed)
    
        # Create generator for this epoch
        generator = torch.Generator()
        generator.manual_seed(epoch_seed)
    
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            generator=generator
        )
    
        return train_dataloader

    def train_model(
        self,
        model_id,
        T,
        start_lr_w,
        start_lr_h,
        trans_mult,
        decay_rate,
        l2_w,
        l2_x,
        l2_h,
        early_stopping
    ):
        model_key = Trainer.model_keys[model_id]
        model = Model(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            act_fn=self.act_fn,
            model_key=model_key,
            residual=self.residual,
            l2_w=l2_w,
            l2_x=l2_x,
            l2_h=l2_h
        )
    
        # Dummy forward pass to initialize Vodes
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            forward(jax.numpy.zeros((self.batch_size, model.input_dim.get())), None, model=model)
    
        optim_w, optim_h = get_opts(
            model=model,
            init_w=start_lr_w,
            init_h=start_lr_h,
            transition_steps=(len(self.train_dataset) // self.batch_size) * trans_mult,
            decay_rate=decay_rate,
            T=T
        )
    
        best_test_acc = 0
        for epoch in range(self.num_epochs):
            train_loader = self.get_epoch_dataloader(model_id, epoch)
            train(train_loader, T=T, model=model, optim_w=optim_w, optim_h=optim_h)

            # track norms
            # hs = [model.vodes[i].get("h") for i in range(len(model.vodes)-1)]
            # norms = [
            #     jnp.mean(jnp.linalg.norm(h_i, axis=-1)) if h_i.ndim > 1 else jnp.linalg.norm(h_i)
            #     for h_i in hs
            # ]
            # print(norms)

            train_acc, _ = eval(train_loader, model=model)
            test_acc, _ = eval(self.test_loader, model=model)
            print(f'Epoch {epoch}: {train_acc}, {test_acc}')
            if test_acc > best_test_acc:
                pxu.save_params(model, f'{self.root}/{self.study_name}/trained_models/model_{model_id}')
            
            best_test_acc = max(best_test_acc, test_acc)
            
            if test_acc == 1 or early_stopping(epoch, test_acc):
                break
    
        with open(f'{self.root}/{self.study_name}/accuracies.txt', 'a') as f:
            f.write(f'{model_id}: {best_test_acc}\n')

        # clean up memory usage
        del model, train_loader
        gc.collect()
        jax.clear_caches()

    def train_all_models(
        self,
        T,
        start_lr_w,
        start_lr_h,
        trans_mult,
        decay_rate,
        l2_w=0.0,
        l2_x=0.0,
        l2_h=0.0,
        model_ids=None,
        early_stopping=lambda epoch, test_acc: epoch > 100 and test_acc < 0.8
    ):
        if model_ids is None:
            model_ids = range(self.num_models)

        for i in tqdm(model_ids):
            self.train_model(
                model_id=i,
                T=T,
                start_lr_w=start_lr_w,
                start_lr_h=start_lr_h,
                trans_mult=trans_mult,
                decay_rate=decay_rate,
                l2_w=l2_w,
                l2_x=l2_x,
                l2_h=l2_h,
                early_stopping=early_stopping
            )

    @staticmethod
    def start_optuna_dashboard():
        def find_available_port(start_port=8080):
            for port in range(start_port, start_port + 100):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(('localhost', port)) != 0:
                        return port
            raise Exception("No available ports found")
        
        def start_dashboard(storage_url, port=8080):
            """Start Optuna dashboard in a separate thread"""
            try:
                print(f"Starting Optuna dashboard on http://localhost:{port}")
                run_server(storage_url, host="0.0.0.0", port=port)
            except Exception as e:
                print(f"Error starting dashboard: {e}")
                
        
        home_dir = os.path.expanduser("~")
        storage_url = f"sqlite:///{home_dir}/optimization.db"
        port = find_available_port(8080)
        
        # Start dashboard
        dashboard_thread = threading.Thread(
            target=start_dashboard, 
            args=(storage_url, port),
            daemon=True
        )
        dashboard_thread.start()
        time.sleep(3)  # Give it time to start
        
        ssh_host = os.getenv("TDL_SSH_HOST", "<username>@<cluster-login-host>")
        print("Run the following command in your local terminal, not on the compute node:")
        print(f"ssh -L {port}:{socket.gethostname()}:{port} {ssh_host}")
        print(f"Then open http://localhost:{port} in a web browser.")

    def run_optuna(
        self,
        num_models,
        T_func,
        start_lr_w_func,
        start_lr_h_func,
        trans_mult_func,
        decay_rate_func,
        l2_w_func=lambda trial: 0.0,
        l2_x_func=lambda trial: 0.0,
        l2_h_func=lambda trial: 0.0,
        early_stopping_within_key=lambda epoch, test_acc: epoch > 100 and test_acc < 0.8,
        early_stopping_whole_trial=lambda trial_num, test_acc: trial_num == 0 and test_acc == 0.61328125,
        prune_after_num_trials=8,  # let this many trials run fully first
        prune_after_keys_tried=2,  # start pruning after this many keys tried
        add_trials: list[dict] = None
    ):
        filetype = 'pkl' if self.dataset == 'D1' else 'dill'
        with open(f'{self.root}/train_test_split_25_percent.{filetype}', 'rb') as f:
            train_sub, test_sub = dill.load(f)

        test_loader = DataLoader(test_sub, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        def objective(trial):
            T = T_func(trial)
            start_lr_w = start_lr_w_func(trial)
            start_lr_h = start_lr_h_func(trial)
            trans_mult = trans_mult_func(trial)
            decay_rate = decay_rate_func(trial)
            l2_w = l2_w_func(trial)
            l2_x = l2_x_func(trial)
            l2_h = l2_h_func(trial)
        
            results = []
            for i in range(num_models):
                model_key = Trainer.model_keys[i]
                model = Model(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    hidden_dims=self.hidden_dims,
                    act_fn=self.act_fn,
                    model_key=model_key,
                    residual=self.residual,
                    l2_w=l2_w,
                    l2_x=l2_x,
                    l2_h=l2_h
                )
            
                # Dummy forward pass to initialize Vodes
                with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
                    forward(jax.numpy.zeros((self.batch_size, model.input_dim.get())), None, model=model)
            
                optim_w, optim_h = get_opts(
                    model=model,
                    init_w=start_lr_w,
                    init_h=start_lr_h,
                    transition_steps=(len(train_sub) // self.batch_size) * trans_mult,
                    decay_rate=decay_rate,
                    T=T
                )
            
                best_test_acc = 0
                for epoch in range(self.num_epochs):
                    train_loader = self.get_epoch_dataloader(i, epoch, train_sub)
                    train(train_loader, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
            
                    test_acc, _ = eval(test_loader, model=model)
                    best_test_acc = max(best_test_acc, test_acc)
            
                    if test_acc == 1 or (early_stopping_within_key is not None and early_stopping_within_key(epoch, test_acc)):
                        break

                results.append(best_test_acc)

                trial.report(np.mean(results), i+1)

                del model, optim_w, optim_h, train_loader
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if early_stopping_whole_trial is not None and early_stopping_whole_trial(i, best_test_acc):
                    return np.mean(results)

            mean_acc = np.mean(results)

            gc.collect()
            jax.clear_caches()
            
            return mean_acc

        
        home_dir = os.path.expanduser("~")
        storage_url = f"sqlite:///{home_dir}/optimization.db"
        
        study = optuna.create_study(
            study_name=f'{self.dataset}_{self.study_name}',
            storage=storage_url,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=prune_after_num_trials,  # let this many trials run fully first
                n_warmup_steps=prune_after_keys_tried,  # start pruning after this many keys tried
                interval_steps=1
            )
        )

        if add_trials is not None:
            for trial in add_trials:
                study.enqueue_trial(trial)
        
        study.optimize(objective)

    def get_layers(
        self,
        dataset,
        input_layer: bool,
        return_labels: bool,
        model=None,
        model_id=None
    ):
        model_created = False
        
        if model is None:
            if model_id is None:
                raise Exception('Either model or model_id must be provided')

            model_created = True
                
            model = Model(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dims=self.hidden_dims,
                act_fn=self.act_fn,
                model_key=Trainer.model_keys[model_id],
                residual=self.residual
            )
            pxu.load_params(model, f'{self.root}/{self.study_name}/trained_models/model_{model_id}')
        
        all_h = [[] for _ in range(len(model.vodes))]
        if input_layer:
            rows = []
        if return_labels:
            all_labels = []
        
        for x, y in dataset:
            x = x.unsqueeze(0)  # add batch dimension
            x_flat = x.reshape(x.shape[0], -1)
            x_jax = jnp.asarray(x_flat)
        
            # Forward pass to get representations
            with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
                _ = forward(x_jax, None, model=model)
        
            # Collect hidden representations
            for i in range(len(model.vodes)):
                all_h[i].append(model.vodes[i].get('h'))
        
            # Collect labels
            if return_labels:
                y_np = y.detach().cpu().numpy()
                all_labels.append(np.atleast_1d(y_np))
        
            # Collect input rows
            if input_layer:
                x_np = x_flat.detach().cpu().numpy()  # reuse x_flat
                assert x_np.shape[1] == self.input_dim, f'Expected {self.input_dim} features, got {x_np.shape[1]}'
                rows.append(x_np)
        
        # Concatenate everything
        all_h_concat = [jnp.concatenate(h, axis=0) for h in all_h]
        if return_labels:
            all_labels = np.concatenate(all_labels, axis=0)
        if input_layer:
            first_layer = np.vstack(rows)
            all_layers = [first_layer, *all_h_concat]
        else:
            all_layers = all_h_concat

        if model_created:
            del model
        
        if return_labels:
            return all_layers, all_labels
        return all_layers

    class _LabelFilterDataset(Dataset):
        """Wrap any (x, y) dataset; keep only items with y == label."""
        def __init__(self, base_ds: Dataset, label: int):
            self.base = base_ds
            self.label = int(label)
            keep = []
            # Build index list once (robust to datasets without .targets)
            for i in range(len(base_ds)):
                _, y = base_ds[i]
                # y can be scalar tensor, numpy, or int
                yv = int(y.detach().cpu().item() if torch.is_tensor(y) else y)
                if yv == self.label:
                    keep.append(i)
            self._idx = np.asarray(keep, dtype=np.int64)
    
        def __len__(self):
            return self._idx.size
    
        def __getitem__(self, i):
            return self.base[self._idx[i]]


    def run_ripser(
        self,
        dataset=None,
        dir_name='ripser_only_0_k14',
        k=14,  # set k = None to not use k-NN
        normalize=False,
        maxdim=0,
        thresh=5,
        model_ids=None,
        num_models=30,
        min_accuracy=None
    ):
        if dataset is None:
            with open(f'{self.root}/only_0_25_percent.dill', 'rb') as f:
                dataset = dill.load(f)

        if self.dataset == 'FASHIONMNIST':
            dir_name = 'ripser/' + dir_name
        
        def get_optimal_k():
            rows = []
            for xb, *_ in dl:                     # xb: [B, 2] (or [B, *, 2] → we’ll flatten)
                # move to CPU, detach from autograd, convert to NumPy
                x_np = xb.detach().cpu().numpy()  # torch → numpy (safe) :contentReference[oaicite:0]{index=0}
                x_np = x_np.reshape(x_np.shape[0], -1)
                input_dim = 2 if self.dataset == 'D1' else 3
                assert x_np.shape[1] == input_dim, f"Expected {input_dim} features, got {x_np.shape[1]}"
                rows.append(x_np)
            first_layer = np.vstack(rows)
        
            # Find optimal k. Should be k = 5
            k = 1
            while k < 100:
                # # Create k-NN graph
                # neighbors = NearestNeighbors(n_neighbors=k).fit(first_layer)
                # graph = neighbors.kneighbors_graph(first_layer, mode='connectivity')
        
                # # Check if the graph is connected
                # graph_undirected = graph.maximum(graph.T)
                # graph_undirected.setdiag(0)
                # graph_undirected.eliminate_zeros()
                # n_components, _ = connected_components(csgraph=graph_undirected, directed=False)

                ##############################
                # Create k-NN graph
                neighbors = NearestNeighbors(n_neighbors=k).fit(first_layer)
                graph = neighbors.kneighbors_graph(first_layer, mode='connectivity')
        
                # Create distance matrix
                distance_matrix = shortest_path(graph, directed=False, unweighted=True)

                dgms = ripser(distance_matrix, distance_matrix=True, maxdim=0, thresh=2)["dgms"]
                H0 = dgms[0]
                n_components = np.sum((H0[:,0] <= 1) & (H0[:,1] > 1))

                print(f'n_components at k = {k}: {n_components}')
                ##############################
                
                if n_components == self.true_b[0]:
                    break  # Stop at the lowest k that has b0 = true_b0
                k += 1
        
            if k == 100:
                raise Exception(f"No k found with n_components = {self.true_b[0]}")

            return k

        def get_distance_matrices(layers, k=5, find_k=False, dl_for_k=None):
            if find_k or k is None:
                assert dl_for_k is not None, "No dl_for_knn argument given"
                k = get_optimal_k(dl_for_knn)
        
            distance_matrices = []
            for X in layers:
                # Create k-NN graph
                neighbors = NearestNeighbors(n_neighbors=k).fit(X)
                graph = neighbors.kneighbors_graph(X, mode='connectivity')
        
                # Create distance matrix
                distance_matrix = shortest_path(graph, directed=False, unweighted=True)
                distance_matrices.append(distance_matrix)
        
            return distance_matrices

        def get_top_ids(self):
            lines = []
            with open(f'{self.root}/{self.study_name}/accuracies.txt', 'r') as f:
                for line in f:
                    parts = line.split(':')
                    model_num = int(parts[0])
                    accuracy = float(parts[1])
                    if min_accuracy is None or accuracy >= min_accuracy:
                        lines.append([model_num, accuracy])
            sorted_lines = sorted(lines, key=lambda x: x[1], reverse=True)
            sorted_ids = [x[0] for x in sorted_lines]
            return sorted_ids[:num_models]

        
        if model_ids is None:
            model_ids = get_top_ids(self)

        print(f'Model_ids: {model_ids}')

        ripser_root = f'{self.root}/{self.study_name}/{dir_name}'
        os.makedirs(ripser_root, exist_ok=True)

        use_knn = (k is not None)
        if use_knn:
            print('Using k-NN')
        else:
            print('Not using k-NN')

        for i, model_id in enumerate(tqdm(model_ids, desc='Running ripser')):
            all_layers = self.get_layers(
                dataset=dataset,
                model_id=model_id,
                input_layer=False,
                return_labels=False
            )

            diagrams = []
            if use_knn:
                distance_matrices = get_distance_matrices(all_layers, k=k)
                for dist_mat in distance_matrices:
                    dgm = ripser(dist_mat, distance_matrix=True, maxdim=maxdim, thresh=thresh)['dgms']
                    diagrams.append(dgm)
            else:
                for layer in all_layers:
                    if normalize:
                        D = pairwise_distances(layer, metric='euclidean')
                        dmax = float(np.max(D))
                        D_norm = D / dmax
                        dgm = ripser(D_norm, distance_matrix=True, maxdim=maxdim, thresh=thresh)['dgms']
                    else:
                        dgm = ripser(layer, maxdim=maxdim, thresh=thresh)['dgms']
                        
                    diagrams.append(dgm)
                
            with open(f'{ripser_root}/model_{model_id}.dill', 'wb') as f:
                dill.dump(diagrams, f)

            # clean up memory usage
            del all_layers, diagrams, dgm
            if use_knn:
                del distance_matrices, dist_mat
            gc.collect()

        # run on input layer since we don't know true topology
        if self.dataset == 'MNIST':
            rows = []
            for x, y in dataset:
                x = x.unsqueeze(0)  # add batch dimension
                x_flat = x.reshape(x.shape[0], -1)
                x_np = x_flat.detach().cpu().numpy()
                rows.append(x_np)
            input_layer = np.vstack(rows)

            if normalize:
                D = pairwise_distances(input_layer, metric='euclidean')
                dmax = float(np.max(D))
                D_norm = D / dmax
                input_dgm = ripser(D_norm, distance_matrix=True, maxdim=maxdim, thresh=thresh)['dgms']
            else:
                input_dgm = ripser(input_layer, maxdim=maxdim, thresh=thresh)['dgms']

            with open(f'{ripser_root}/input_layer.dill', 'wb') as f:
                dill.dump(input_dgm, f)

    @staticmethod
    def _betti_at_eta_one_dim(diagram_dim, eta):
        """
        diagram_dim: array-like of shape (n_intervals, 2) with (birth, death)
        """        
        if diagram_dim is None:
            return 0
        arr = np.asarray(diagram_dim)
        if arr.size == 0:
            return 0
        births = arr[:, 0]
        deaths = arr[:, 1]
        # Count intervals "alive" at eta: birth <= eta < death
        return int(np.count_nonzero((births <= eta) & (eta < deaths)))

    @staticmethod
    def _betti_at_eta(diagram, eta, dim=0):
        """
        diagram: list-like where diagram[d] is an array of (birth, death) for homology dim d
        """
        return Trainer._betti_at_eta_one_dim(diagram[dim], eta)

    def graph_betti_numbers(
        self,
        title=None,
        dir_name='ripser_only_0_k14',
        k=14,  # set k = None when not using k-NN
        etas=[2.5],
        maxdim=0,
        use_running_min=False,
        sum_betti_numbers=False,
        figsize=(10, 6),
        save=False,
        filename=None
    ):
        input_layer_dgm = None
        
        def graph_betti_mean_band(
            diagrams_list,
            eta,
            dims=[0],
            color="C0",
            plot_individual=True,
            alpha_individual=0.12,
            linewidth_individual=1.0,
            linewidth_mean=2.0,
            marker="s",
            layer_labels=None,
            title=None
        ):
            # assert len(diagrams_list) > 0, "diagrams_list must be non-empty"
        
            # K = len(diagrams_list)  # number of models
            # L = len(diagrams_list[0])  # number of layers
            # # Ensure consistent layer counts
            # for ds in diagrams_list:
            #     if len(ds) != L:
            #         raise ValueError("All networks must have the same number of layers.")
        
            # # Compute Betti matrix: shape (K, L)
            # betti_mat = np.zeros((K, L), dtype=float)
            # for i, diagrams in enumerate(diagrams_list):
            #     for ell, diagram in enumerate(diagrams):
            #         betti_mat[i, ell] = Trainer._betti_at_eta(diagram, eta=eta, dim=dim)

            betti_mat = self.get_betti_mat(dir_name=dir_name, eta=eta, dims=dims, use_running_min=use_running_min)

            mean_per_layer = betti_mat.mean(axis=0)
            std_per_layer  = betti_mat.std(axis=0, ddof=1)

            K = len(betti_mat)  # number of models
            L = len(betti_mat[0])  # number of layers

            # # Add input layer stats
            # if input_layer_dgm is not None:
            #     self.true_b[dim] = Trainer._betti_at_eta(input_layer_dgm, eta=eta, dim=dim)
            # betti_mat = np.hstack([np.full((K,1), self.true_b[dim]), betti_mat])
            # mean_per_layer = np.insert(mean_per_layer, 0, self.true_b[dim])
            # std_per_layer = np.insert(std_per_layer, 0, 0.0)
        
            x = np.arange(len(mean_per_layer))
        
            if layer_labels is None:
                layer_labels = ['Input'] + [str(i+1) for i in range(L-2)] + ['Output']
        
            plt.figure(figsize=figsize)
        
            # Plot individual network lines (faint)
            if plot_individual:
                for i in range(K):
                    plt.plot(
                        x,
                        betti_mat[i],
                        color=color,
                        alpha=alpha_individual,
                        linewidth=linewidth_individual,
                    )
        
            # Mean line
            plt.plot(
                x, mean_per_layer,
                color=color,
                linewidth=linewidth_mean,
                marker=marker
            )
        
            # Std band
            lower = mean_per_layer - std_per_layer
            upper = mean_per_layer + std_per_layer
            plt.fill_between(x, lower, upper, color=color, alpha=0.2, linewidth=0)
        
            # Axes, labels, grid
            plt.xticks(x, layer_labels, rotation=0)
            plt.xlabel("Layer", fontsize=14, labelpad=6)

            betti_in_title = rf'\beta_{dims[0]}'
            for d in dims[1:]:
                betti_in_title += rf'+ \beta_{d}'
            betti_in_title = rf'${betti_in_title}$'
            
            plt.ylabel(betti_in_title, fontsize=14, labelpad=6)
            plt.grid(True, alpha=0.3)
        
            if title is None:
                title = rf"{betti_in_title} for {self.study_name}, {rf'$k={k}$ ' if k is not None else ''}at $\eta={eta}${' (Euclidean)' if k is None else ''}"
            plt.title(title, fontsize=15, pad=10)

            ymax = int(np.max([np.max(betti_mat), np.max(mean_per_layer + std_per_layer)]))
            if self.dataset == 'D1':
                plt.yticks(np.arange(0, ymax + 1, 1))

            plt.tick_params(axis='both', which='major', labelsize=11)
        
            plt.tight_layout()

            if save:
                out_name = filename if filename is not None else self.study_name
                dim_tag = ''.join(str(d) for d in dims)
                out_path = Path(self.root) / "figures" / "individual" / f"{out_name}_B{dim_tag}_{f'k{k}' if k is not None else 'no_knn'}_eta{eta}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
            
            plt.show()
        
            # Return the stats in case you want to reuse them
            return mean_per_layer, std_per_layer, betti_mat

        ripser_root = f'{self.root}/{self.study_name}/{'ripser/' if self.dataset == 'FASHIONMNIST' else ''}{dir_name}'
        all_diagrams = []
        for file in os.listdir(ripser_root):
            with open(f'{ripser_root}/{file}', 'rb') as f:
                diagram = dill.load(f)
                
                if 'model' in file:
                    all_diagrams.append(diagram)
                elif file == 'input_layer.dill':
                    input_layer_dgm = diagram
                    
        for eta in etas:
            if not sum_betti_numbers:
                for d in range(maxdim + 1):
                    graph_betti_mean_band(
                        all_diagrams,
                        title=title,
                        eta=eta,
                        dims=[d],
                        color="blue"
                    )
            else:
                graph_betti_mean_band(
                    all_diagrams,
                    title=title,
                    eta=eta,
                    dims=range(maxdim + 1),
                    color="blue"
                )

    def get_betti_mat(self, dir_name='ripser_only_0_k14', eta=2.5, dims=[0], use_running_min=False):
        """
        Returns a matrix of size (num_models, num_layers)
        betti_mat[0] is a list of size num_layers representing each layer of the first model
        For mat[0] to be a list of all models for the first layer, use betti_mat.T

        If dims has multiple elements, the betti numbers of each dim will be SUMMED in the matrix
        """
        input_layer_dgm = None
        
        ripser_root = f'{self.root}/{self.study_name}/{'ripser/' if self.dataset == 'FASHIONMNIST' else ''}{dir_name}'
        all_diagrams = []
        for file in os.listdir(ripser_root):
            with open(f'{ripser_root}/{file}', 'rb') as f:
                if 'model' in file:
                    diagram = dill.load(f)
                    all_diagrams.append(diagram)
                elif 'input_layer' in file:
                    input_layer_dgm = dill.load(f)

        if input_layer_dgm is None and self.dataset == 'FASHIONMNIST':
            print('Input layer from folder')
            input_path = dataset_results_dir("FASHIONMNIST") / "data" / "input_layer_ripser" / f"{dir_name}_input.dill"
            with input_path.open('rb') as f:
                input_layer_dgm = dill.load(f)
                    
        # for i, file in enumerate(os.listdir(ripser_root)):
        #     print(f'{i}: {file}')
        #     with open(f'{ripser_root}/{file}', 'rb') as f:
        #         diagram = dill.load(f)
        #         all_diagrams.append(diagram)
        
        K = len(all_diagrams)
        L = len(all_diagrams[0])
        
        betti_mat = np.zeros((K, L), dtype=float)
        for j, diagrams in enumerate(all_diagrams):
            for ell, diagram in enumerate(diagrams):
                entry = 0
                for dim in dims:
                    entry += Trainer._betti_at_eta(diagram, eta=eta, dim=dim)
                betti_mat[j, ell] = entry

        if input_layer_dgm is not None:
            for dim in dims:
                self.true_b[dim] = Trainer._betti_at_eta(input_layer_dgm, eta=eta, dim=dim)

        # Add input layer stats
        input_b = sum(self.true_b[dim] for dim in dims)
        betti_mat = np.hstack([np.full((K,1), input_b), betti_mat])

        if use_running_min:
            for i in range(len(betti_mat)):
                betti_mat[i] = np.minimum.accumulate(betti_mat[i])  # monotone non-increasing

        return betti_mat

    @staticmethod
    def compare_graphs(
        title: str,
        studies: list[str] = None,  # actual names of studies to include
        include: list[str] = None,  # regexes
        exclude: list[str] = None,  # regexes
        dataset='D1',
        dir_names: list[str] = None,
        eta=2.5,
        k=14,
        dims=[0],
        use_running_min=False,
        legend: list[str] = None,
        colors=['blue', 'red', 'green', 'purple', 'orange'],
        save=False,
        save_data=False,
        filename=None
    ):
        if (save or save_data) and filename is None:
            raise Exception('Provide a filename to save to')

        if include is not None:
            selected = Trainer.filter_studies(include, exclude)
        else:
            selected = studies

        if dir_names is None:
            dir_names = ['ripser_only_0_k14'] * len(selected)

        if len(selected) > len(colors):
            # not enough colors
            extra_needed = len(selected) - len(colors)
            cmap = plt.get_cmap('tab20')
            extra_colors = [cmap(i / extra_needed) for i in range(extra_needed)]
            colors = colors + extra_colors
        
        plt.figure(figsize=(10, 6))

        K = None
        L = None
        ymax = 0

        data = {}

        for i, study in enumerate(selected):
            # All we need here is the study_name
            # Everything else is arbitrary
            trainer = Trainer(
                dataset=dataset,
                hidden_dims=[30]*8,
                act_fn=jax.nn.relu,
                study_name=study,
                residual=False
            )
            dir_name = dir_names[i]

            # ripser_root = f'{trainer.root}/{trainer.study_name}/{dir_name}'
            # all_diagrams = []
            # for file in os.listdir(ripser_root):
            #     with open(f'{ripser_root}/{file}', 'rb') as f:
            #         diagram = dill.load(f)
            #         all_diagrams.append(diagram)
            
            # K = len(all_diagrams)
            
            # LL = len(all_diagrams[0])
            # if i != 0 and LL != L:
            #     raise Exception(f'Number of layers changed from {L} to {LL}')
            # L = LL
                
            # # Ensure consistent layer counts
            # for ds in all_diagrams:
            #     if len(ds) != L:
            #         raise ValueError("All networks must have the same number of layers.")
    
            # betti_mat = np.zeros((K, L), dtype=float)
            # for j, diagrams in enumerate(all_diagrams):
            #     for ell, diagram in enumerate(diagrams):
            #         betti_mat[j, ell] = Trainer._betti_at_eta(diagram, eta=eta, dim=dim)

            betti_mat = trainer.get_betti_mat(dir_name=dir_name, eta=eta, dims=dims, use_running_min=use_running_min)
    
            mean_per_layer = betti_mat.mean(axis=0)
            std_per_layer  = betti_mat.std(axis=0, ddof=1)
    
            # # Add input layer stats
            # betti_mat = np.hstack([np.full((K,1), trainer.true_b0), betti_mat])
            # mean_per_layer = np.insert(mean_per_layer, 0, trainer.true_b0)
            # std_per_layer = np.insert(std_per_layer, 0, 0.0)
        
            x = np.arange(len(mean_per_layer))

            # Mean line
            label = legend[i] if legend is not None else trainer.study_name
            plt.plot(
                x, mean_per_layer,
                linewidth=2.0,
                marker='s',
                label=label,
                color=colors[i]
            )

            ymax_here = int(np.max([np.max(betti_mat), np.max(mean_per_layer)]))
            ymax = max(ymax, ymax_here)

            if save_data:
                data[label] = betti_mat
                
                # for layer_idx in range(len(mean_per_layer)):
                #     csv_rows.append({
                #         'model': label,
                #         'layer': layer_idx,
                #         f'b{dim}_mean': float(mean_per_layer[layer_idx]),
                #         f'b{dim}_std': float(std_per_layer[layer_idx])
                #     })

        layer_labels = ['Input'] + [str(i+1) for i in range(8)] + ['Output']
        plt.xticks(x, layer_labels, rotation=0)
        if dataset == 'D1':
            plt.yticks(np.arange(0, ymax + 1, 1))
        plt.grid(True, alpha=0.3)
        
        plt.xlabel('Layer', fontsize=14, labelpad=6)

        betti_in_title = rf'\beta_{dims[0]}'
        for d in dims[1:]:
            betti_in_title += rf'+ \beta_{d}'
        betti_in_title = rf'${betti_in_title}$'
        plt.ylabel(betti_in_title, fontsize=14, labelpad=6)
        
        plt.title(title, fontsize=16)

        plt.legend(fontsize=13)
        plt.tight_layout()

        if save:
            plt.savefig(f'{trainer.root}/figures/comparisons/{filename}_B{''.join(str(d) for d in dims)}_eta{eta}.png', dpi=300, bbox_inches="tight")
        
        plt.show()

        if save_data:
            # csv_path = f'{trainers[0].root}/betti_data/{filename}_B{dim}_k{k}_eta{eta}.csv'
            # with open(csv_path, 'w', newline='') as csvfile:
            #     fieldnames = ['model', 'layer', f'b{dim}_mean', f'b{dim}_std']
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #     writer.writeheader()
            #     writer.writerows(csv_rows)

            dim_tag = ''.join(str(d) for d in dims)
            out_path = dataset_results_dir(dataset) / "betti_data" / f"{filename}_B{dim_tag}_k{k}_eta{eta}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **data)








    @staticmethod
    def all_8_layer_models():
        with (dataset_results_dir("D1") / "all_8_layer_models.txt").open('r') as f:
            txt = f.read()
        return ast.literal_eval(txt)

    @staticmethod
    def filter_studies(regexes=[], exclude=[], all_studies=None):
        if all_studies is None:
            all_studies = Trainer.all_8_layer_models()

        _include = [re.compile(p) for p in regexes]
        _exclude = [re.compile(p) for p in exclude]
        
        def matches(name):
            # must match at least one include-regex
            inc_ok = not _include or any(r.search(name) for r in _include)
            # must match *no* exclude-regex
            exc_ok = not _exclude or not any(r.search(name) for r in _exclude)
            return inc_ok and exc_ok

        selected = [s for s in all_studies if matches(s)]
        if not selected:
            raise ValueError("No matching studies found.")

        return selected

    @staticmethod
    def get_bootstrap_data(
        data_filename='D1/betti_data/all_8_layer_models_all_stats_mean_alpha0.05',
        studies: list[str] = [],  # e.g., [r"30x8_relu", r"tanh"]
        exclude: list[str] = []
    ):
        filepath = CONFIG.results_dir / f"{data_filename}.json"
        with filepath.open('r') as f:
            data = json.load(f)
            
        if studies is None and exclude is None:
            return data

        _include = [re.compile(p) for p in studies]
        _exclude = [re.compile(p) for p in exclude]
        
        def matches(name):
            # must match at least one include-regex
            inc_ok = not _include or any(r.search(name) for r in _include)
            # must match *no* exclude-regex
            exc_ok = not _exclude or not any(r.search(name) for r in _exclude)
            return inc_ok and exc_ok
    
        filtered = {
            A: {B: vals for B, vals in inner.items() if matches(B)}
            for A, inner in data.items() if matches(A)
        }

        return filtered

    @staticmethod
    def all_bootstrap_stats(
        filename: str,
        studies: list[str] = None,
        dataset='D1',
        dir_name='ripser_only_0_k14',
        B=10000,
        seed=42,
        use_mean=True,
        p=0.95,
        alpha=0.05,
        save=True
    ):
        if studies is None:
            studies = Trainer.all_8_layer_models()

        print(studies)
        print('Num studies:', len(studies))
        
        study_to_betti_by_layer = {}
        for s in studies:
            # All we need here is the study_name
            # Everything else is arbitrary
            trainer = Trainer(
                dataset=dataset,
                hidden_dims=[30]*8,
                act_fn=jax.nn.relu,
                study_name=s,
                residual=False
            )
            betti_mat = trainer.get_betti_mat(dir_name=dir_name).T
            if dataset == 'MNIST':
                betti_mat[0][:] = 1.0  # dummy input layer data
            study_to_betti_by_layer[s] = betti_mat
    
        def _bootstrap_layer_stats(X, Y):
            rng = np.random.default_rng(seed)
            
            X = np.asarray(X)
            Y = np.asarray(Y)
            nX, nY = len(X), len(Y)
            
            diffs = np.empty(B)
            for b in range(B):
                Xb = rng.choice(X, size=nX, replace=True)
                Yb = rng.choice(Y, size=nY, replace=True)
                statX = np.mean(Xb) if use_mean else np.quantile(Xb, p)
                statY = np.mean(Yb) if use_mean else np.quantile(Yb, p)
                diffs[b] = statX - statY
            diffs.sort()
            
            p_gt0 = np.mean(diffs > 0.0)  # P(A>B)
            p_lt0 = np.mean(diffs < 0.0)  # P(A<B)
            p_eq0 = 1.0 - p_gt0 - p_lt0  # P(A=B)
            lower = np.quantile(diffs, alpha)
            upper = np.quantile(diffs, 1 - alpha)
            mean_d = np.mean(diffs)
            se_d = np.std(diffs, ddof=1)
            
            return dict(
                p_gt0=float(p_gt0),
                p_lt0=float(p_lt0),
                p_eq0=float(p_eq0),
                lower=float(lower),
                upper=float(upper),
                mean=float(mean_d),
                se=float(se_d),
            )

        
        out = defaultdict(dict)
        n_layers = min(len(study_to_betti_by_layer[s]) for s in studies)
    
        for A, B_name in tqdm(itertools.combinations(studies, 2),
                              total=len(studies)*(len(studies)-1)//2):
            layers_stats_AB = []
            for layer in range(n_layers):
                X = study_to_betti_by_layer[A][layer]
                Y = study_to_betti_by_layer[B_name][layer]
                stats = _bootstrap_layer_stats(X, Y)
                layers_stats_AB.append(stats)
    
            out[A][B_name] = layers_stats_AB
    
            # Generate complementary stats for (B,A)
            layers_stats_BA = []
            for s in layers_stats_AB:
                layers_stats_BA.append({
                    "p_gt0": s["p_lt0"],
                    "p_lt0": s["p_gt0"],
                    "p_eq0": s["p_eq0"],
                    "lower": -s["upper"],
                    "upper": -s["lower"],
                    "mean": -s["mean"],
                    "se": s["se"]
                })
            out[B_name][A] = layers_stats_BA
    
        # Self-comparisons
        for s in studies:
            out[s][s] = [
                {
                    "p_gt0": 0.0,
                    "p_lt0": 0.0,
                    "p_eq0": 1.0,
                    "lower": 0.0,
                    "upper": 0.0,
                    "mean": 0.0,
                    "se": 0.0,
                }
                for _ in range(n_layers)
            ]

        if save:
            p_or_mean = 'mean' if use_mean else f'p{p}'
            filepath = dataset_results_dir(dataset) / "betti_data" / f"{filename}_all_stats_{p_or_mean}_alpha{alpha}.json"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open("w") as f:
                json.dump(out, f, indent=2)

        return out

    @staticmethod
    def _com_of_drops_one_seed(
        betti_curve: np.ndarray,
        use_running_min: bool = True,
        include_output: bool = True,
        no_drop_value=None,
    ) -> float:
        """
        Compute a single scalar COM for one model run's Betti curve.

        betti_curve: shape (n_layers_total,) where index 0 is the input layer value.
                    (This is exactly one row of get_betti_mat output.)
        use_running_min:
            - True: count only new record lows (robust to rebounds).
            - False: count only positive stepwise drops on the raw curve.
        include_output:
            - True: include the last transition into the final layer.
            - False: ignore the last transition.
        no_drop_value:
            Value returned if there are no simplifying drops (denominator = 0).
            If None, defaults to n_layers_total (a "worst" sentinel).
        """
        beta = np.asarray(betti_curve, dtype=float)
        n_layers_total = beta.shape[0]

        max_transition = (n_layers_total - 1) if include_output else (n_layers_total - 2)
        if max_transition < 1:
            return float(no_drop_value if no_drop_value is not None else n_layers_total)

        beta_use = beta[: max_transition + 1]  # layers 0..max_transition

        if use_running_min:
            beta_use = np.minimum.accumulate(beta_use)  # monotone non-increasing

        drops = beta_use[:-1] - beta_use[1:]
        if not use_running_min:
            drops = np.maximum(0.0, drops)

        total_drop = float(np.sum(drops))
        if total_drop <= 0.0:
            return float(no_drop_value if no_drop_value is not None else n_layers_total)

        ell = np.arange(1, max_transition + 1, dtype=float)
        return float(np.sum(ell * drops) / total_drop)

    @staticmethod
    def get_com(
        study: str,
        dataset: str,  # e.g., MNIST
        dir_name: str = 'ripser_class_0_norm',
        eta: float = 0.2,
        dims: list[int] = [0],
        use_running_min: bool = True,
        include_output: bool = True
    ):
        trainer = Trainer(
            dataset=dataset,
            hidden_dims=[30] * 8,
            act_fn=jax.nn.relu,
            study_name=study,
            residual=False,
        )

        betti_mat = trainer.get_betti_mat(dir_name=dir_name, eta=eta, dims=dims)  # (K, n_layers_total)
        K, n_layers_total = betti_mat.shape

        no_drop_value = float(n_layers_total)  # sentinel "worse than last transition index"
        com = np.empty(K, dtype=float)
        for i in range(K):
            com[i] = Trainer._com_of_drops_one_seed(
                betti_mat[i],
                use_running_min=use_running_min,
                include_output=include_output,
                no_drop_value=no_drop_value,
            )

        return com

    @staticmethod
    def visualize_com(
        studies: list[str],
        dataset: str,  # e.g., MNIST
        dir_name: str = 'ripser_class_0_norm',
        eta: float = 0.2,
        dims: list[int] = [0],
        use_running_min: bool = True,
        include_output: bool = True,
        title='',
        x_labels: list[str] = None,
        x_axis_label: str = 'Architecture',
        save=False,
        filename=None,
        figsize=(10, 6)
    ):
        """
        For each study_name in `studies`, loads Betti curves via get_betti_mat(),
        computes per-model COM scalars, then bootstraps pairwise differences in
        the *mean* COM across seeds.

        Returns out[A][B] = [stats_dict] (a list of length 1 for compatibility with
        code that expects a "per-layer list" like all_bootstrap_stats).

        Interpretation:
          - Smaller COM => earlier simplification.
          - For pair (A,B), p_lt0 ≈ P( mean(COM_A) - mean(COM_B) < 0 )
                           ≈ P( A earlier than B )
        """
        study_to_com = {}
        for s in studies:
            com = Trainer.get_com(
                study=s,
                dataset=dataset,
                dir_name=dir_name,
                eta=eta,
                dims=dims,
                use_running_min=use_running_min,
                include_output=include_output
            )
            study_to_com[s] = com

        rows = []
        for A in studies:
            mean_com = np.mean(study_to_com[A])
            print(f'mean({A}) = {mean_com}')
            for v in study_to_com[A]:
                rows.append({"arch": A, "COM": float(v)})
    
        df = pd.DataFrame(rows)
    
        # plt.figure()
        # sns.violinplot(data=df, x="arch", y="COM", inner="box")
        # plt.ylabel("Center of Mass of Betti Drops (lower = earlier)")
        # plt.xticks(rotation=30, ha="right")
        # plt.tight_layout()
        # plt.show()

        from matplotlib.ticker import MaxNLocator

        # --- Plot: match trainer styling (figsize, grid alpha, clean axes) ---
        fig, ax = plt.subplots(figsize=figsize)
    
        # Preserve the given order of `studies` (instead of seaborn sorting)
        sns.violinplot(
            data=df,
            x="arch",
            y="COM",
            order=studies,
            inner="box",
            # cut=0,              # don't extend beyond data range
            linewidth=0.75,
            ax=ax,
        )
    
        ax.set_xlabel(x_axis_label, fontsize=14, labelpad=10)
        ax.set_ylabel("COM", fontsize=14, labelpad=6)
        
        betti_in_title = rf'\beta_{dims[0]}'
        for d in dims[1:]:
            betti_in_title += rf'+ \beta_{d}'
        betti_in_title = rf'${betti_in_title}$'

        if title is not None:
            ax.set_title(rf"COM of {betti_in_title} drops, $\eta$={eta}", fontsize=16)
    
        # Light grid like your other plots
        # ax.grid(True, axis="y", alpha=0.3)
    
        # Tick formatting similar vibe (not too busy)
        # ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

        if x_labels is not None:
            ax.set_xticks(range(len(studies)))
            ax.set_xticklabels(x_labels, fontsize=11)
        
        # Rotate only if needed (keeps the “flat” look when short labels)
        rotate = 30 if len(studies) > 6 else 0
        ax.tick_params(axis="x", rotation=rotate, size=11)
        if rotate:
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
    
        # Optional: small padding so violins don't touch the frame
        # y = df["COM"].to_numpy()
        # if y.size:
        #     lo, hi = float(np.min(y)), float(np.max(y))
        #     pad = 0.05 * (hi - lo) if hi > lo else 0.5
        #     ax.set_ylim(lo - pad, hi + pad)
    
        plt.tight_layout()

        if save:
            out_path = dataset_results_dir(dataset) / "figures" / "COM" / f"{filename}_B{''.join(str(d) for d in dims)}_eta{eta}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
        
        plt.show()

    @staticmethod
    def _hit_time_one_seed(
        betti_curve: np.ndarray,
        thresh: float,
        use_running_min: bool = True
    ) -> float:
        beta = np.asarray(betti_curve, dtype=float)
        n_layers_total = beta.shape[0]

        max_transition = n_layers_total - 1
        beta_use = beta[: max_transition + 1]  # layers 0..max_transition

        if use_running_min:
            beta_use = np.minimum.accumulate(beta_use)  # monotone non-increasing

        hit = np.where(beta <= thresh * beta[0])[0]
        return float(hit[0])

    @staticmethod
    def get_hit_time(
        study: str,
        dataset: str,  # e.g., MNIST
        dir_name: str = 'ripser_class_0_norm',
        eta: float = 0.2,
        dims: list[int] = [0],
        thresh=0.9,
        use_running_min: bool = True,
    ):
        trainer = Trainer(
            dataset=dataset,
            hidden_dims=[30] * 8,
            act_fn=jax.nn.relu,
            study_name=study,
            residual=False,
        )

        betti_mat = trainer.get_betti_mat(dir_name=dir_name, eta=eta, dims=dims)  # (K, n_layers_total)
        K, n_layers_total = betti_mat.shape

        hitting_time = np.empty(K, dtype=float)
        for i in range(K):
            hitting_time[i] = Trainer._hit_time_one_seed(
                betti_mat[i],
                thresh=thresh,
                use_running_min=use_running_min
            )

        return hitting_time

    @staticmethod
    def visualize_hit_time(
        studies: list[str],
        dataset: str,  # e.g., MNIST
        dir_name: str = 'ripser_class_0_norm',
        eta: float = 0.2,
        dims: list[int] = [0],
        thresh=[0.9],
        use_running_min: bool = True,
        title='',
        x_labels: list[str] = None,
        x_axis_label: str = 'Architecture',
        save=False,
        filename=None,
        figsize=(10, 6)
    ):
        study_to_hit_time = {}
        for s in studies:
            summ = 0
            for t in thresh:
                hit_time = Trainer.get_hit_time(
                    study=s,
                    dataset=dataset,
                    dir_name=dir_name,
                    eta=eta,
                    dims=dims,
                    thresh=t,
                    use_running_min=use_running_min
                )
                summ += hit_time
            study_to_hit_time[s] = summ

        rows = []
        for A in studies:
            median_hit_time = np.median(study_to_hit_time[A])
            print(f'median({A}) = {median_hit_time}')
            for v in study_to_hit_time[A]:
                rows.append({"arch": A, "hit time": float(v)})
    
        df = pd.DataFrame(rows)
    
        # plt.figure()
        # sns.violinplot(data=df, x="arch", y="COM", inner="box")
        # plt.ylabel("Center of Mass of Betti Drops (lower = earlier)")
        # plt.xticks(rotation=30, ha="right")
        # plt.tight_layout()
        # plt.show()

        from matplotlib.ticker import MaxNLocator

        # --- Plot: match trainer styling (figsize, grid alpha, clean axes) ---
        fig, ax = plt.subplots(figsize=figsize)
    
        # Preserve the given order of `studies` (instead of seaborn sorting)
        sns.violinplot(
            data=df,
            x="arch",
            y="hit time",
            order=studies,
            inner="box",
            # cut=0,              # don't extend beyond data range
            linewidth=1.2,
            ax=ax,
        )
    
        ax.set_xlabel(x_axis_label, fontsize=14, labelpad=10)
        ax.set_ylabel("Hit Time", fontsize=14, labelpad=6)
        
        betti_in_title = rf'\beta_{dims[0]}'
        for d in dims[1:]:
            betti_in_title += rf'+ \beta_{d}'
        betti_in_title = rf'${betti_in_title}$'

        if title is not None:
            ax.set_title(rf"Hit Time of {betti_in_title} at thresh {thresh}, $\eta$={eta}", fontsize=16)
    
        # Light grid like your other plots
        # ax.grid(True, axis="y", alpha=0.3)
    
        # Tick formatting similar vibe (not too busy)
        # ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

        if x_labels is not None:
            ax.set_xticks(range(len(studies)))
            ax.set_xticklabels(x_labels, fontsize=11)
        
        # Rotate only if needed (keeps the “flat” look when short labels)
        rotate = 30 if len(studies) > 6 else 0
        ax.tick_params(axis="x", rotation=rotate, size=11)
        if rotate:
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
    
        # Optional: small padding so violins don't touch the frame
        # y = df["COM"].to_numpy()
        # if y.size:
        #     lo, hi = float(np.min(y)), float(np.max(y))
        #     pad = 0.05 * (hi - lo) if hi > lo else 0.5
        #     ax.set_ylim(lo - pad, hi + pad)
    
        plt.tight_layout()

        if save:
            out_path = dataset_results_dir(dataset) / "figures" / "hit_time" / f"{filename}_B{''.join(str(d) for d in dims)}_eta{eta}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
        
        plt.show()

    @staticmethod
    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        """
        Cliff's delta: P(x > y) - P(x < y).
        Negative => x tends to be smaller than y.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            return np.nan
    
        # O(n*m) is fine for ~30x30. If you ever go huge, we can optimize.
        gt = 0
        lt = 0
        for xi in x:
            gt += np.sum(xi > y)
            lt += np.sum(xi < y)
        return (gt - lt) / (x.size * y.size)

    @staticmethod
    def bootstrap_ci_arb_stat(
        x: np.ndarray,
        y: np.ndarray,
        stat_fn,
        *,
        n_boot: int = 10000,
        ci: float = 0.95,
        seed: int = 0,
    ) -> dict:
        """
        Bootstrap CI by resampling within x and within y.
        Percentile interval.
    
        Returns dict with:
          - stat: observed statistic
          - ci_low, ci_high
          - boot: bootstrap samples (array)
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            return dict(stat=np.nan, ci_low=np.nan, ci_high=np.nan, boot=np.array([]))
    
        rng = np.random.default_rng(seed)
        n, m = x.size, y.size
    
        boot = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            xb = x[rng.integers(0, n, size=n)]
            yb = y[rng.integers(0, m, size=m)]
            boot[b] = stat_fn(xb, yb)
    
        alpha = 1.0 - ci
        lo = np.quantile(boot, alpha / 2.0)
        hi = np.quantile(boot, 1.0 - alpha / 2.0)
    
        return dict(
            stat=float(stat_fn(x, y)),
            ci_low=float(lo),
            ci_high=float(hi),
            boot=boot,
        )

    @staticmethod
    def bootstrap_cliffs_delta(
        studies: list[str],
        dataset: str,
        dir_name: str = 'ripser_class_0_norm',
        eta: float = 0.2,
        dims: list[int] = [0],
        use_running_min: bool = True,
        n_boot=10000,
        ci=0.95
    ):
        for study1, study2 in itertools.combinations(studies, 2):
            com1 = Trainer.get_com(
                study=study1,
                dataset=dataset,
                dir_name=dir_name,
                eta=eta,
                dims=dims,
                use_running_min=use_running_min
            )
            com2 = Trainer.get_com(
                study=study2,
                dataset=dataset,
                dir_name=dir_name,
                eta=eta,
                dims=dims,
                use_running_min=use_running_min
            )

            boot_dict = Trainer.bootstrap_ci_arb_stat(
                x=com1,
                y=com2,
                stat_fn=Trainer.cliffs_delta,
                n_boot=n_boot,
                ci=ci
            )

            print(rf"Cliff's $\delta$ ({study1}, {study2}) = [{boot_dict['ci_low']}, {boot_dict['ci_high']}]")
                

    # @staticmethod
    # def plot_betti_heatmap(
    #     data, layer, metric="p_gt0", title=None, filename=None, cmap="coolwarm", annot=True, save=False
    # ):
    #     """
    #     data: output of all_bootstrap_stats
    #     metric: "p_gt0" (recommended), or "lower", "upper", "mean", "se"
    #     Produces a square heatmap for a chosen layer.
    #     """
    #     studies = sorted(data.keys())
    
    #     # Build matrix for selected metric @ layer
    #     M = pd.DataFrame(index=studies, columns=studies, dtype=float)
    #     for A in studies:
    #         for B in studies:
    #             val = data[A][B][layer][metric]
    #             M.loc[A, B] = val
    
    #     plt.figure(figsize=(1.2*len(studies), 1.0*len(studies)))
    #     if metric == "p_gt0":
    #         ax = sns.heatmap(M, vmin=0, vmax=1, cmap=cmap, annot=annot, fmt=".2f",
    #                          cbar_kws={"label": r"$P\,(A > B)$"})
    #     else:
    #         ax = sns.heatmap(M, cmap=cmap, annot=annot, fmt=".2f",
    #                          cbar_kws={"label": metric})
    #     ax.set_xlabel("Architecture B")
    #     ax.set_ylabel("Architecture A")
    #     if not title and metric == 'p_gt0':
    #         title = r'$P\,(A > B)$' + f' @ Layer {layer}'
    #     ax.set_title(title or f"{metric} @ Layer {layer}")
    #     plt.tight_layout()

    #     if save:
    #         if not filename:
    #             raise Exception('Must provide filename if saving')
    #         filepath = dataset_results_dir("D1") / "figures" / "heatmaps" / f"{filename}_heatmap_{metric}_layer{layer}.png"
    #         plt.savefig(filepath, dpi=300, bbox_inches="tight")

    #     plt.show()

    @staticmethod
    def get_tsc(
        data,  # should be the output of get_bootstrap_data
        to_calculate: list[str] = None,  # regex
        metric="p_gt0",
        mean=True,
        only_hidden=True,
        title=None,
        legend: list[str] = None,
        legend_title: str = None,
        plot=False,
        figsize=(10, 6),
        save=False,
        filename=None
    ):
        """
        Plots the topology simplification curve.
        The topology simplification score of model i at layer l is:
        TSS_i^l = 1/(N-1) sum_{j != i} P(B0(model i, layer l) < B0(model j, layer l))
        
        This measures how consistently each architecture simplifies topology
        relative to the other architectures.

        TSS = 1 --> simplifies more than others
        TSS = 0.5 --> average simplification
        TSS = 0 --> simplifies less than others
        """
        all_studies = list(data.keys())
        selected = all_studies
        
        if to_calculate is not None:
            regexes = [re.compile(p) for p in to_calculate]
            def matches(name):
                return any(r.search(name) for r in regexes)
    
            selected = [s for s in all_studies if matches(s)]
            if not selected:
                raise ValueError("No matching studies found in data for to_calculate.")
        
        n_layers = len(next(iter(next(iter(data.values())).values())))

        # Compute TSS_i^(ell) for each architecture and layer
        if only_hidden:
            tss = {A: np.zeros(n_layers - 2) for A in selected}
        else:
            tss = {A: np.zeros(n_layers) for A in selected}

        # for A in selected:
        #     for ell in range(n_layers):
        #         vals = []
        #         for B in all_studies:
        #             if A == B:
        #                 continue
        #             node = data[A][B][ell]
        #             # "tie-aware TSS" -- search ChatGPT project for explanation
        #             p_win = node["p_lt0"] + 0.5 * node["p_eq0"]
        #             vals.append(p_win)
        #             # vals.append(1.0 - data[A][B][ell]["p_gt0"])  # P(A < B)
        #         tss[A][ell] = np.mean(vals)

        layers = np.arange(n_layers) if not only_hidden else np.arange(1, n_layers - 1)
        for A in selected:
            for ell in layers:
                vals = []
                for B in all_studies:
                    if A == B:
                        continue
                    node = data[A][B][ell]
                    # "tie-aware TSS" -- search ChatGPT project for explanation
                    p_win = node["p_lt0"] + 0.5 * node["p_eq0"]
                    vals.append(p_win)
                    # vals.append(1.0 - data[A][B][ell]["p_gt0"])  # P(A < B)
                if only_hidden:
                    tss[A][ell-1] = np.mean(vals)
                else:
                    tss[A][ell] = np.mean(vals)

        if plot:
            plt.figure(figsize=figsize)
    
            # Plot one line per architecture
            for i, A in enumerate(selected):
                label = legend[i] if legend is not None else A
                plt.plot(
                    layers,
                    tss[A],
                    linewidth=2.0,
                    marker="s",
                    label=label,
                )
    
            layer_labels = [str(i + 1) for i in range(n_layers - 2)]
            if not only_hidden:
                layer_labels = ['Input'] + layer_labels + ['Output']
            plt.xticks(layers, layer_labels, rotation=0)
            plt.yticks(np.linspace(0, 1, 6))
            plt.ylim(-0.05, 1.05)
    
            plt.grid(True, alpha=0.3)
            if only_hidden:
                plt.xlabel("Hidden Layer", fontsize=14, labelpad=6)
            else:
                plt.xlabel("Layer", fontsize=14, labelpad=6)
            plt.ylabel("TSS", fontsize=14, labelpad=6)
            plt.title(title, fontsize=15, pad=10)
            plt.tick_params(axis='both', which='major', labelsize=11)
            plt.legend(
                title=legend_title,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                frameon=True,
                fontsize=11,
                title_fontsize=13,
                handlelength=1.5,
                labelspacing=0.3,
                borderpad=0.3,
                handletextpad=0.4,
                alignment="left"
            )
            plt.tight_layout()

        if save:
            if not plot:
                raise Exception('Marked save but not plot')
            if filename is None:
                raise Exception("Must provide filename if save=True")
            mean_or_p95 = 'mean' if mean else 'p95'
            filepath = dataset_results_dir("D1") / "figures" / "tsc" / mean_or_p95 / f"{filename}_{mean_or_p95}_TSC.png"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")

        if plot:
            plt.show()

        return tss

    @staticmethod
    def _energy_given_xy(x, y, *, model: "Model"):
        model.eval()
        x = jnp.atleast_2d(x)  # (B,D)
        y = jnp.atleast_2d(y)  # (B,C)
    
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            _ = forward(x, y, model=model)
            e = model.energy()  # (B,)

            # regularization as per Orchard & Sun
            l2_h = model.l2_h.get()
            if l2_h > 0:
                h_pen = 0.0
                # exclude output layer
                for vode in model.vodes[:-1]:
                    h = vode.get("h")
                    h_pen = h_pen + jnp.sum(h * h)
                x_pen = jnp.sum(x * x)
                # x_pen = jnp.linalg.norm(x)
                e = e + 0.5 * l2_h * (h_pen + x_pen)
    
        return jnp.sum(e)  # scalar

    @staticmethod
    def _rand_init(key, shape, kind="normal", scale=1.0, box=None):
        """
        Draw a random initial x_0.
        kind: "normal" or "uniform"
        scale: std (normal) or half-width (uniform)
        box:  tuple (lo, hi) → clip to [lo, hi] after sampling
        """
        if kind == "normal":
            x0 = scale * jax.random.normal(key, shape)
        elif kind == "uniform":
            x0 = jax.random.uniform(key, shape, minval=-scale, maxval=scale)
        else:
            raise ValueError("kind must be 'normal' or 'uniform'")
        if box is not None:
            lo, hi = box
            x0 = jnp.clip(x0, lo, hi)
        return x0

    # -------- a single Langevin / noisy gradient descent trajectory ----------
    @staticmethod
    def _invert_once(
        rng_key: "jax.Array",
        y_target: "jax.Array",
        *,
        model: "Model",
        steps: int = 200,
        lr: float = 0.05,
        noise_sigma: float = 0.0,
        noise_type: str = "normal",
        temp: float = 1.0,
        init_kind: str = "normal",
        init_scale: float = 1.0,
        early_stop_energy: float | None = None,
        track_input_norm=True
    ):
        """
        Reparameterized inversion: optimize z in R^D with x = tanh(z) in [-1,1]^D.
        """
        y_target = jnp.atleast_1d(y_target)
        input_dim = model.input_dim.get()

        # initialize z (unconstrained latent variable)
        k_init, k_noise = jax.random.split(rng_key)
        z = Trainer._rand_init(k_init, (1, input_dim), kind=init_kind, scale=init_scale)

        # energy as a function of z, via x = tanh(z)
        def E_of_z(z_flat):
            x = jnp.tanh(z_flat)
            return Trainer._energy_given_xy(x[None, :], y_target[None, :], model=model)

        grad_E = jax.grad(lambda z_flat: E_of_z(z_flat.squeeze(0)))

        def step(carry, i):
            z_cur, key = carry
            g = jax.grad(Trainer._energy_given_xy)(
                jnp.tanh(z_cur), y_target[None, :], model=model
            ) * (1.0 - jnp.tanh(z_cur) ** 2)  # chain rule: dE/dz = dE/dx * (1 - tanh^2 z)

            # gradient step in z-space
            z_next = z_cur - lr * g

            # Langevin-like noise in z-space
            def add_noise(kn, zval):
                if noise_sigma <= 0:
                    return zval
                if noise_type == "normal":
                    z = jax.random.normal(kn, zval.shape)
                elif noise_type == "uniform":
                    z = jax.random.uniform(kn, zval.shape, minval=-1.0, maxval=1.0)
                else:
                    raise ValueError("noise_type must be 'normal' or 'uniform'")
                return zval + noise_sigma * jnp.sqrt(2.0 * lr * temp) * z

            key, kn = jax.random.split(key)
            z_next = add_noise(kn, z_next)

            x_next = jnp.tanh(z_next)
            e = Trainer._energy_given_xy(x_next, y_target[None, :], model=model)

            # Track input norm (L2 norm of the input layer)
            if track_input_norm:
                # Compute L2 norm of the input: ||x||_2 = sqrt(sum(x^2))
                # input_norm = jnp.sum(x_next.squeeze(0) * x_next.squeeze(0))
                input_norm = jnp.linalg.norm(x_next.squeeze(0))
                return (z_next, key), (e, input_norm)
            else:
                return (z_next, key), e

        # run gradient descent / Langevin steps
        if track_input_norm:
            (z_final, _), (e_trace, input_norm_trace) = jax.lax.scan(step, (z, k_noise), jnp.arange(steps))
        else:
            (z_final, _), e_trace = jax.lax.scan(step, (z, k_noise), jnp.arange(steps))
        x_final = jnp.tanh(z_final)
        
        # (z_final, _), e_trace = jax.lax.scan(step, (z, k_noise), jnp.arange(steps))
        # x_final = jnp.tanh(z_final)

        # early-stop handling (optional)
        if early_stop_energy is not None:
            idx = jnp.argmax(e_trace <= early_stop_energy)
            use_idx = jnp.where(jnp.any(e_trace <= early_stop_energy), idx, steps - 1)

            def step_noiseless(carry, i):
                z_cur = carry
                g = jax.grad(Trainer._energy_given_xy)(
                    jnp.tanh(z_cur), y_target[None, :], model=model
                ) * (1.0 - jnp.tanh(z_cur) ** 2)
                z_next = z_cur - lr * g
                return z_next, z_next

            z_star, _ = jax.lax.scan(step_noiseless, z, jnp.arange(use_idx + 1))
            x_star = jnp.tanh(z_star)
            return x_star.squeeze(0), e_trace

        return x_final.squeeze(0), e_trace, input_norm_trace

    # -------- batched sampling of many reconstructions for a given y --------
    def invert_output(
        self,
        target_label: int | None = None,
        target_logits: "np.ndarray | jax.Array" = None,
        *,
        num_samples: int = 2048,
        model_ids: list[int] | None = None,
        num_top_models=10,
        steps: int = 100,
        lr: float = 0.005,
        noise_sigma: float = 0,      # set >0 to make it *non-deterministic*, e.g., 0.07
        noise_type: str = "normal",
        temp: float = 1.0,
        init_kind: str = "normal",
        init_scale: float = 0.5,
        batch_size: int = 32,
        early_stop_energy: float | None = None,
        l2_w=0.0,
        l2_x=0.0,
        l2_h=0.0,
        save=False,
        save_dir=None
    ):
        """
        Produce many reconstructions x* such that the PCN believes y(x*) ≈ target.
        - If target_logits is None, we clamp to one-hot for target_label.
        - Non-determinism comes from: random x0 and per-step noise (Langevin).
        Returns a dict: {model_id: {"x": np.ndarray [N,2], "E": np.ndarray [N], "trace": optional}}
        """
        assert (target_label is not None) ^ (target_logits is not None), \
            "Provide exactly one of target_label or target_logits."
        output_dim = self.output_dim
        if target_logits is None:
            y = jnp.asarray(jax.nn.one_hot(jnp.asarray(target_label), output_dim))
        else:
            y = jnp.asarray(target_logits).astype(jnp.float32)
            if y.ndim == 1 and y.shape[0] == output_dim:
                pass
            else:
                raise ValueError(f"target_logits must have shape ({output_dim},)")

        if save_dir is None:
            save_dir = f'{self.root}/{self.study_name}'
        else:
            save_dir = str(CONFIG.root_dir / save_dir)
        save_filepath = f'{save_dir}/reconstructions_s{steps}'

        def get_top_ids(self):
            lines = []
            with open(f"{self.root}/{self.study_name}/accuracies.txt", "r") as f:
                for line in f:
                    model_num, acc = line.strip().split(":")
                    lines.append([int(model_num), float(acc)])
            sorted_lines = sorted(lines, key=lambda x: x[1], reverse=True)
            return [x[0] for x in sorted_lines]
    
        if model_ids is None:
            model_ids = get_top_ids(self)[:min(self.num_models, num_top_models)]
            print(f'Using models: {model_ids}')

        results = {}

        for mid in tqdm(model_ids):
            # Load model weights
            model = Model(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dims=self.hidden_dims,
                act_fn=self.act_fn,
                model_key=Trainer.model_keys[mid],
                residual=self.residual,
                l2_w=l2_w,
                l2_x=l2_x,
                l2_h=l2_h
            )
            pxu.load_params(model, f'{self.root}/{self.study_name}/trained_models/model_{mid}')

            # jit a single-trajectory kernel for speed
            kernel = jax.jit(lambda k: Trainer._invert_once(
                k, y, model=model,
                steps=steps, lr=lr,
                noise_sigma=noise_sigma, noise_type=noise_type, temp=temp,
                init_kind=init_kind, init_scale=init_scale,
                early_stop_energy=early_stop_energy
            ))

            # sample many x*
            xs = []
            Es = []

            # batched PRNG
            main_key = jax.random.PRNGKey(int(Trainer.model_keys[mid][0]))
            keys = jax.random.split(main_key, num_samples)

            # loop in manageable chunks to avoid host <-> device thrash
            for start in range(0, num_samples, batch_size):
                k_chunk = keys[start:start+batch_size]
                # vmap over the kernel to parallelize several samples at once
                x_chunk, e_traces, input_norm_trace = jax.vmap(kernel)(k_chunk)
                xs.append(np.asarray(x_chunk))
                # final energies are last elements of traces
                Es.append(np.asarray(e_traces[:, -1]))

            X = np.concatenate(xs, axis=0)  # shape (N, 2)
            E = np.concatenate(Es, axis=0)  # shape (N,)

            if save:
                os.makedirs(save_filepath, exist_ok=True)
                with open(f'{save_filepath}/model_{mid}_label_{int(target_label)}.dill', 'wb') as f:
                    dill.dump(X, f)

            results[mid] = X

            # cleanup (helps on long runs)
            # del model
            # gc.collect()
            # jax.clear_caches()

        return results, input_norm_trace[0], e_traces[0]


    def invert_output_new(
        self,
        target_train_index: int = 0,
        target_example: "np.ndarray | jax.Array | torch.Tensor" = None,
        *,
        num_samples: int = 1,
        model_ids: list[int] | None = None,
        num_top_models=10,
        steps: int = 100,
        lr: float = 0.005,
        steps_values: list[int] | None = None,
        lr_values: list[float] | None = None,
        l2_w_values: list[float] | None = None,
        l2_x_values: list[float] | None = None,
        l2_h_values: list[float] | None = None,
        noise_sigma: float = 0,      # set >0 to make it *non-deterministic*, e.g., 0.07
        noise_type: str = "normal",
        temp: float = 1.0,
        init_kind: str = "normal",
        init_scale: float = 0.5,
        batch_size: int = 32,
        early_stop_energy: float | None = None,
        l2_w=0.0,
        l2_x=0.0,
        l2_h=0.0,
        save=False,
        save_dir=None,
        save_grid_png: bool = True,
        return_target_info: bool = False
    ):
        """
        Produce reconstructions x* by clamping the output layer to the output
        state produced by a real training example, rather than to a one-hot class label.

        To sweep inversion hyperparameters, pass ranges such as:
            steps_values=[50, 100, 200]
            lr_values=[1e-3, 5e-3, 1e-2]
            l2_w_values=[0.0]
            l2_x_values=[0.0, 1e-4, 1e-3]
            l2_h_values=[0.0, 1e-4, 1e-3]

        One reconstruction is generated for every combination by default because
        num_samples defaults to 1. If num_samples > 1, each combination produces
        num_samples reconstructions.

        Returns:
            If no *_values ranges are supplied, preserves the old-style return:
                (results, one_input_norm_trace, one_energy_trace)
                where results[model_id] = np.ndarray [num_samples, input_dim]

            If any *_values range is supplied, returns:
                (grid_results, one_input_norm_trace, one_energy_trace)
                where grid_results[model_id][combo_key] = {
                    "params": dict,
                    "x": np.ndarray [num_samples, input_dim],
                    "E": np.ndarray [num_samples],
                }

            If return_target_info=True, target_info is appended to the return tuple.
        """
        sweep_mode = any(v is not None for v in [
            steps_values, lr_values, l2_w_values, l2_x_values, l2_h_values
        ])

        steps_values = [steps] if steps_values is None else list(steps_values)
        lr_values = [lr] if lr_values is None else list(lr_values)
        l2_w_values = [l2_w] if l2_w_values is None else list(l2_w_values)
        l2_x_values = [l2_x] if l2_x_values is None else list(l2_x_values)
        l2_h_values = [l2_h] if l2_h_values is None else list(l2_h_values)

        param_grid = list(itertools.product(
            steps_values, l2_w_values, l2_x_values, l2_h_values, lr_values
        ))

        if save_dir is None:
            save_dir = f'{self.root}/{self.study_name}'
        else:
            save_dir = str(CONFIG.root_dir / save_dir)

        sweep_tag = f'grid_{len(param_grid)}combos' if sweep_mode else f's{steps_values[0]}'
        save_filepath = f'{save_dir}/reconstructions_{sweep_tag}'

        def get_top_ids(self):
            lines = []
            with open(f"{self.root}/{self.study_name}/accuracies.txt", "r") as f:
                for line in f:
                    model_num, acc = line.strip().split(":")
                    lines.append([int(model_num), float(acc)])
            sorted_lines = sorted(lines, key=lambda x: x[1], reverse=True)
            return [x[0] for x in sorted_lines]

        def _as_single_input(x):
            """Convert a dataset example into shape (1, input_dim) as a JAX array."""
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            else:
                x = np.asarray(x)

            x = np.asarray(x, dtype=np.float32).reshape(1, -1)
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"Reference example has {x.shape[1]} features after flattening, "
                    f"but this trainer expects input_dim={self.input_dim}."
                )
            return jnp.asarray(x)

        def _label_to_int(y_ref):
            if y_ref is None:
                return None
            if torch.is_tensor(y_ref):
                return int(y_ref.detach().cpu().item())
            return int(np.asarray(y_ref).item())

        def get_reference_example():
            if target_example is not None:
                return _as_single_input(target_example), None

            x_ref, y_ref = self.train_dataset[int(target_train_index)]
            return _as_single_input(x_ref), _label_to_int(y_ref)

        def _combo_key(params):
            combo_steps, combo_l2_w, combo_l2_x, combo_l2_h, combo_lr = params
            return (
                f"steps={combo_steps}__"
                f"l2w={combo_l2_w:g}__"
                f"l2x={combo_l2_x:g}__"
                f"l2h={combo_l2_h:g}__"
                f"lr={combo_lr:g}"
            )

        def _save_mnist_grid(mid, model_results, y_ref_label):
            if not save or not save_grid_png or self.input_dim != 28 * 28:
                return

            os.makedirs(save_filepath, exist_ok=True)
            n = len(model_results)
            ncols = min(6, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.8 * nrows))
            axes = np.atleast_1d(axes).reshape(-1)

            for ax in axes:
                ax.axis("off")

            for ax, (key, record) in zip(axes, model_results.items()):
                img = record["x"][0].reshape(28, 28)
                # Your MNIST loader normalizes to [-1, 1], so map back to [0, 1].
                img = (img + 1.0) / 2.0
                ax.imshow(np.clip(img, 0.0, 1.0), cmap="gray")
                p = record["params"]
                ax.set_title(
                    f"s={p['steps']}, lr={p['lr']:g}\n"
                    f"w={p['l2_w']:g}, x={p['l2_x']:g}, h={p['l2_h']:g}\n"
                    f"E={record['E'][0]:.3g}",
                    fontsize=8
                )

            target_part = f'trainidx_{int(target_train_index)}' if target_example is None else 'custom_example'
            if y_ref_label is not None:
                target_part += f'_label_{y_ref_label}'

            fig.tight_layout()
            fig.savefig(f'{save_filepath}/model_{mid}_{target_part}_grid.png', dpi=200, bbox_inches="tight")
            plt.close(fig)

        def _invert_once_eager(
            rng_key,
            y_target,
            *,
            model,
            steps,
            lr,
            noise_sigma,
            noise_type,
            temp,
            init_kind,
            init_scale,
            early_stop_energy
        ):
            """
            Eager/Python-loop version of _invert_once.

            This avoids wrapping PCX/Vode state mutation inside jax.jit, jax.vmap, or
            jax.lax.scan. Those transformations can leak tracers because pxu.step and
            VodeParam updates are stateful.
            """
            y_target = jnp.asarray(jnp.atleast_1d(y_target), dtype=jnp.float32)
            input_dim = model.input_dim.get()

            k_init, k_noise = jax.random.split(rng_key)
            z = Trainer._rand_init(
                k_init,
                (1, input_dim),
                kind=init_kind,
                scale=init_scale
            )

            e_trace = []
            input_norm_trace = []

            def energy_from_x(x_current):
                return Trainer._energy_given_xy(
                    x_current,
                    y_target[None, :],
                    model=model
                )

            for _ in range(int(steps)):
                x_current = jnp.tanh(z)

                # dE/dz = dE/dx * dx/dz, with x = tanh(z).
                grad_x = jax.grad(energy_from_x)(x_current)
                grad_z = grad_x * (1.0 - x_current ** 2)
                z = z - float(lr) * grad_z

                if noise_sigma > 0:
                    k_noise, kn = jax.random.split(k_noise)
                    if noise_type == "normal":
                        noise = jax.random.normal(kn, z.shape)
                    elif noise_type == "uniform":
                        noise = jax.random.uniform(kn, z.shape, minval=-1.0, maxval=1.0)
                    else:
                        raise ValueError("noise_type must be 'normal' or 'uniform'")
                    z = z + noise_sigma * jnp.sqrt(2.0 * float(lr) * temp) * noise

                x_next = jnp.tanh(z)
                e = energy_from_x(x_next)
                e_trace.append(e)
                input_norm_trace.append(jnp.linalg.norm(x_next.squeeze(0)))

                if early_stop_energy is not None and float(e) <= early_stop_energy:
                    break

            return (
                jnp.asarray(x_next.squeeze(0)),
                jnp.asarray(e_trace),
                jnp.asarray(input_norm_trace),
            )

        if model_ids is None:
            model_ids = get_top_ids(self)[:min(self.num_models, num_top_models)]
            print(f'Using models: {model_ids}')

        results = {}
        target_info = {}
        one_input_norm_trace = None
        one_energy_trace = None

        for mid in tqdm(model_ids):
            # Compute the clamp target once using the first l2 setting. The l2 values do
            # not affect this ordinary forward pass, but the model object needs the same
            # static fields as the inversion model.
            target_model = Model(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dims=self.hidden_dims,
                act_fn=self.act_fn,
                model_key=Trainer.model_keys[mid],
                residual=self.residual,
                l2_w=l2_w_values[0],
                l2_x=l2_x_values[0],
                l2_h=l2_h_values[0]
            )
            pxu.load_params(target_model, f'{self.root}/{self.study_name}/trained_models/model_{mid}')

            x_ref_jax, y_ref_label = get_reference_example()
            target_model.eval()
            with pxu.step(target_model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
                _ = forward(x_ref_jax, None, model=target_model)
                y = jnp.asarray(target_model.vodes[-1].get("h")).squeeze(0).astype(jnp.float32)

            if y.ndim != 1 or y.shape[0] != self.output_dim:
                raise ValueError(
                    f"Expected output-layer clamp target to have shape ({self.output_dim},), "
                    f"but got {tuple(y.shape)}."
                )

            target_info[mid] = {
                "target_train_index": None if target_example is not None else int(target_train_index),
                "target_label": y_ref_label,
                "target_output_state": np.asarray(y),
            }

            if sweep_mode:
                results[mid] = {}

            for combo_idx, combo in enumerate(tqdm(param_grid, desc=f'Model {mid} sweep', leave=False)):
                combo_steps, combo_l2_w, combo_l2_x, combo_l2_h, combo_lr = combo

                # Reload the same trained weights into a model with this combo's
                # regularization settings.
                model = Model(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    hidden_dims=self.hidden_dims,
                    act_fn=self.act_fn,
                    model_key=Trainer.model_keys[mid],
                    residual=self.residual,
                    l2_w=combo_l2_w,
                    l2_x=combo_l2_x,
                    l2_h=combo_l2_h
                )
                pxu.load_params(model, f'{self.root}/{self.study_name}/trained_models/model_{mid}')

                xs = []
                Es = []
                e_traces_for_combo = []
                input_norm_traces_for_combo = []

                # Use a deterministic but different key stream for each combo.
                main_key = jax.random.PRNGKey(int(Trainer.model_keys[mid][0]) + 1009 * combo_idx)
                keys = jax.random.split(main_key, num_samples)

                # Important: do NOT use jax.jit/jax.vmap here. PCX/Vode state is
                # mutated inside the energy call, and vectorizing/jitting this path
                # can produce UnexpectedTracerError leaks.
                for k in keys:
                    x_one, e_trace_one, input_norm_trace_one = _invert_once_eager(
                        k,
                        y,
                        model=model,
                        steps=int(combo_steps),
                        lr=float(combo_lr),
                        noise_sigma=noise_sigma,
                        noise_type=noise_type,
                        temp=temp,
                        init_kind=init_kind,
                        init_scale=init_scale,
                        early_stop_energy=early_stop_energy,
                    )
                    xs.append(np.asarray(x_one)[None, :])
                    Es.append(np.asarray(e_trace_one[-1])[None])
                    e_traces_for_combo.append(e_trace_one)
                    input_norm_traces_for_combo.append(input_norm_trace_one)

                X = np.concatenate(xs, axis=0)
                E = np.concatenate(Es, axis=0)

                if one_input_norm_trace is None:
                    one_input_norm_trace = input_norm_traces_for_combo[0]
                    one_energy_trace = e_traces_for_combo[0]

                record = {
                    "params": {
                        "steps": int(combo_steps),
                        "l2_w": float(combo_l2_w),
                        "l2_x": float(combo_l2_x),
                        "l2_h": float(combo_l2_h),
                        "lr": float(combo_lr),
                    },
                    "x": X,
                    "E": E,
                }

                if sweep_mode:
                    results[mid][_combo_key(combo)] = record
                else:
                    results[mid] = X

                if save:
                    os.makedirs(save_filepath, exist_ok=True)
                    target_part = f'trainidx_{int(target_train_index)}' if target_example is None else 'custom_example'
                    if y_ref_label is not None:
                        target_part += f'_label_{y_ref_label}'

                    if sweep_mode:
                        filename = f'model_{mid}_{target_part}_{_combo_key(combo)}.dill'
                        with open(f'{save_filepath}/{filename}', 'wb') as f:
                            dill.dump(record, f)
                    else:
                        with open(f'{save_filepath}/model_{mid}_{target_part}.dill', 'wb') as f:
                            dill.dump(X, f)

            if sweep_mode:
                _save_mnist_grid(mid, results[mid], y_ref_label)

            # cleanup (helps on long runs)
            # del model
            # gc.collect()
            # jax.clear_caches()

        if return_target_info:
            return results, one_input_norm_trace, one_energy_trace, target_info
        return results, one_input_norm_trace, one_energy_trace


    @staticmethod
    def mrd(recons, dataset, graph=False):
        """
        Compute minimum reconstruction distance (MRD) for a given target class.
        A lower MRD means the reconstructions better fit the target data.
        
        dataset should only contain data from one class, no labels.
        """
        if graph:
            plt.scatter(dataset[:,0], dataset[:,1], s=4, color='green', label='True Dataset')
            plt.scatter(recons[:,0], recons[:,1], s=4, color='red', label='Reconstructions')
            plt.axis('equal')
            plt.legend()
            plt.show()
            
        D = cdist(recons, dataset)
        mrd = np.mean(np.min(D, axis=1))
        return float(mrd)

    def get_mrd_list(self, dir_name='reconstructions_s100', dataset=None, include_ids=False):
        if dataset is None:
            dataset = Trainer.all_green[0]
            
        recon_root = f'{self.root}/{self.study_name}/{dir_name}'
        mrds = []
        for file in os.listdir(recon_root):
            if 'dill' in file:
                with open(f'{recon_root}/{file}', 'rb') as f:
                    recons = dill.load(f)
                    mrd = Trainer.mrd(recons, dataset)
                    if include_ids:
                        mrds.append((file, mrd))
                    else:
                        mrds.append(mrd)
                
        return mrds

    @staticmethod
    def filter_with_regex(
        data,
        to_calculate: list[str] = None,  # regex
        check_recons_exist=False
    ):
        all_studies = list(data.keys())
        to_select = all_studies
        
        if to_calculate is not None:
            regexes = [re.compile(p) for p in to_calculate]
            def matches(name):
                return any(r.search(name) for r in regexes)
    
            to_select = [s for s in all_studies if matches(s)]
            if not to_select:
                raise ValueError("No matching studies found in data for to_calculate.")

        if not check_recons_exist:
            return to_select
        
        selected = []
        for study in to_select:
            if os.path.isdir(dataset_results_dir("D1") / study / "reconstructions_s100"):
                selected.append(study)
        return selected

    @staticmethod
    def get_model_sizes(dataset='D1'):
        import json
        with (dataset_results_dir(dataset) / "model_sizes.json").open('rb') as f:
            model_sizes_dict = json.load(f)
        return model_sizes_dict

    @staticmethod
    def plot_mrd_tss(
        data,
        to_calculate: list[str] = None,  # regex
        selected=None,
        dataset='D1',
        x_metric='tss',
        y_metric='mrd',
        corr='spearman',
        dims=[0],  # only used if either metric == 'com'
        eta=2.5,  # only used if either metric == 'com'
        dir_name='ripser_only_0_k14',  # only used if either metric == 'com'
        use_running_min=True,  # only used if either metric == 'com'
        include_output=True,  # only used if either metric == 'com'
        metric="p_gt0",
        mean=True,
        title='MRD vs. TSS',
        figsize=(9, 6),
        s=100,
        lowess_extra_gap=0,
        legend_left=True,
        save=False,
        filename=None,
        add_d=False,
        print_all=False,
        print_outliers=False
    ):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])

        if selected is None:
            selected = Trainer.filter_with_regex(data, to_calculate, check_recons_exist=True)
        print(selected)
        print(f'Num models: {len(selected)}')

        if x_metric == 'tss' or y_metric == 'tss':
            all_tss = Trainer.get_tsc(
                data,
                to_calculate,
                metric,
                mean
            )

        if x_metric == 'hidden' or y_metric == 'hidden':
            model_sizes = Trainer.get_model_sizes(dataset)

        x_vals = []
        y_vals = []

        ########
        from scipy.spatial.distance import mahalanobis

        def mahalanobis_outlier_scores(xs, ys):
            points = np.column_stack((xs, ys))

            mean = points.mean(axis=0)
            cov = np.cov(points, rowvar=False)
            inv_cov = np.linalg.inv(cov)
        
            scores = np.array([mahalanobis(p, mean, inv_cov) for p in points])
            ranking = np.argsort(scores)[::-1]  # highest first
        
            return scores, ranking
        ########

        if print_all:
            # newline
            print()

        for study in selected:
            # All we need here is the study_name
            # Everything else is arbitrary
            trainer = Trainer(
                dataset=dataset,
                hidden_dims=[30]*8,
                act_fn=jax.nn.relu,
                study_name=study,
                residual=False
            )

            if x_metric == 'mrd' or y_metric == 'mrd':
                mrd = trainer.get_mrd_list()
                if x_metric == 'mrd':
                    x_val = np.mean(mrd)
                else:
                    y_val = np.mean(mrd)

            if x_metric == 'tss' or y_metric == 'tss':
                tss = all_tss[study]
                if x_metric == 'tss':
                    x_val = np.mean(tss)
                else:
                    y_val = np.mean(tss)

            if x_metric == 'com' or y_metric == 'com':
                com = Trainer.get_com(
                    study=study,
                    dataset=dataset,
                    dir_name=dir_name,
                    eta=eta,
                    dims=dims,
                    use_running_min=use_running_min,
                    include_output=include_output
                )
                if x_metric == 'com':
                    x_val = np.mean(com)
                else:
                    y_val = np.mean(com)

            if x_metric == 'hidden':
                x_val = model_sizes[study]
            elif y_metric == 'hidden':
                y_val = model_sizes[study]

            if print_all:
                print(f'{study}: {x_val:.3f}, {y_val:.3f}')

            x_vals.append(x_val)
            y_vals.append(y_val)

        if print_outliers:
            print()
            scores, rank = mahalanobis_outlier_scores(x_vals, y_vals)
            for i in rank:
                print(f'{selected[i]}: {scores[i]}')

        pearson_corr, p_pearson = pearsonr(x_vals, y_vals)
        print(f'\nPearson correlation coefficient: {pearson_corr}')
        print(f'P-value: {p_pearson}\n')

        spearman_corr, p_spearman = spearmanr(x_vals, y_vals)
        print(f'Spearman correlation: {spearman_corr}')
        print(f'P-value: {p_spearman}\n')

        from scipy.stats import kendalltau
        tau, p_tau = kendalltau(x_vals, y_vals)
        print(f"Kendall's Tau: {tau}")
        print(f'P-value: {p_tau}')

        # ---- Plot ----
        plt.figure(figsize=figsize)
        sns.set_style("whitegrid")
        sns.set_context("talk")

        # ---- Determine activation type (color) ----
        def get_activation_type(name):
            name = name.lower()
            if "leaky" in name:
                return "Leaky ReLU"
            elif "tanh" in name:
                return "Tanh"
            else:
                return "ReLU"

        act_types = [get_activation_type(s) for s in selected]
        palette = {
            "ReLU": "blue",
            "Tanh": "red",
            "Leaky ReLU": "green"
        }

        # ---- Determine architecture type (marker) ----
        if add_d:
            def extract_d_value(study_name):
                match1 = re.search(r'(\d+)x8', study_name)
                if match1:
                    return int(match1.group(1))
                    
                match2 = re.search(r'_(\d+)x4', study_name)
                if match2:
                    return int(match2.group(1))
                    
                return None
    
            arch_vals = [extract_d_value(s) for s in selected]
            unique_ds = sorted(set(v for v in arch_vals if v is not None), reverse=True)
    
            marker_cycle = ['s', 'o', '^', 'D', 'P', 'v', '*']
            d_to_marker = {d: marker_cycle[i % len(marker_cycle)] for i, d in enumerate(unique_ds)}

        # ---- Scatter points ----
        for i, study in enumerate(selected):
            act_type = act_types[i]
            if add_d:
                d_val = arch_vals[i]
                marker = d_to_marker.get(d_val, 's')
            plt.scatter(
                x_vals[i],
                y_vals[i],
                label=act_type,  # used for color legend
                s=s,
                color=palette[act_type],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
                marker=marker if add_d else 's'
            )

        # ---- Nonparametric local regression trend ----
        sns.regplot(
            x=x_vals, y=y_vals,
            scatter=False, color="black",
            lowess=True,
            line_kws={"lw": 1.8, "ls": "--", "alpha": 0.6}
        )

        # ---- Axis labels, title, grid ----
        label_dict = {
            'tss': 'TSS',
            'mrd': 'MRD',
            'com': 'COM',
            'hidden': 'Sum of Hidden Layer Sizes'
        }
        
        plt.xlabel(label_dict[x_metric], fontsize=14, labelpad=6)
        plt.ylabel(label_dict[y_metric], fontsize=14, labelpad=6)
        if title is not None:
            plt.title(
                title,
                fontsize=15,
                pad=10
            )
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.grid(True, alpha=0.25)

        # ---- Activation legend ----
        leg1 = None
        if len(set(act_types)) > 1:
            from matplotlib.lines import Line2D
            order = [a for a in ["ReLU", "Tanh", "Leaky ReLU"] if a in act_types]
            act_handles = [
                Line2D([], [], color=palette[a], marker='s', linestyle='None',
                       markersize=6, markeredgecolor='black') for a in order
            ]
            act_labels = order
    
            leg1 = plt.legend(
                act_handles,
                act_labels,
                title="Activation Function",
                loc = 'upper left' if legend_left else 'upper right',
                bbox_to_anchor = (0.02, 0.98) if legend_left else (0.98, 0.98),
                frameon=True,
                fontsize=10,
                title_fontsize=12,
                handlelength=1.5,
                labelspacing=0.3,
                borderpad=0.3,
                handletextpad=0.4,
                alignment="left"
            )
            plt.gca().add_artist(leg1)

        # ---- Architecture legend (marker shapes) ----
        if add_d:
            arch_handles = [
                Line2D([], [], color='gray', marker=d_to_marker[d], linestyle='None',
                       markersize=6, markeredgecolor='black')
                for d in unique_ds
            ]
            arch_labels = [rf"$d={d}$" for d in unique_ds]
    
            leg2 = plt.legend(
                arch_handles,
                arch_labels,
                title="Architecture",
                loc="upper left",
                bbox_to_anchor=(0.26, 0.98),   # positions second legend beside first
                frameon=True,
                fontsize=10,
                title_fontsize=12,
                handlelength=1.0,
                labelspacing=0.3,
                borderpad=0.3,
                handletextpad=0.4,
                alignment="left"
            )
            plt.gca().add_artist(leg2)

        # ---- Trend line legend entry ----
        from matplotlib.lines import Line2D

        trend_handle = Line2D(
            [0], [0],
            color='black',
            lw=1.8,
            ls='--',
            alpha=0.6
        )

        lowess_offset = 0.03
        if leg1 is not None:
            fig = plt.gcf()
            ax  = plt.gca()
            
            # Force a draw
            fig.canvas.draw()
            
            # Get legend 1 height in axes coordinates
            bbox1 = leg1.get_window_extent().transformed(ax.transAxes.inverted())
            h1 = bbox1.height
            gap = 0.05

            lowess_offset = h1 + gap

        trend_leg = plt.legend(
            [trend_handle],
            ["LOWESS trend line"],
            loc = 'upper left' if legend_left else 'upper right',
            bbox_to_anchor = (0.03 if legend_left else 0.97, 1 - lowess_offset - lowess_extra_gap),
            # bbox_to_anchor=(0.03, 0.775),  # slightly below the two legends
            frameon=True,
            fontsize=9,
            title_fontsize=11,
            handlelength=2.0,
            borderpad=0.3,
            handletextpad=0.4,
            alignment="left"
        )
        plt.gca().add_artist(trend_leg)

        # ---- Display correlation ----
        if corr == 'spearman':
            corr_text = rf"$\rho_s = {spearman_corr:.2f}$" + f"\n(p={p_spearman:.3g})"
        elif corr == 'pearson':
            corr_text = rf"$\rho = {pearson_corr:.2f}$" + f"\n(p={p_pearson:.3g})"
            
        plt.text(
            0.97 if legend_left else 0.03,
            0.03,
            corr_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment = 'right' if legend_left else 'left',
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.6,
                edgecolor="gray"
            )
        )

        plt.tight_layout()

        if save:
            if filename is None:
                raise Exception("Must provide filename if save=True")
            out_path = dataset_results_dir(dataset) / "figures" / f"{filename}_{y_metric.upper()}_vs_{x_metric.upper()}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")

        plt.show()



    def class_mean_activations(
        self,
        dataset,
        class_label: int,
        model=None,
        model_id=None,
    ) -> list[jax.Array]:
        """
        Compute the mean hidden-layer activations over all correctly classified
        examples of a given class.
    
        Parameters
        ----------
        dataset : torch Dataset
            The full dataset (e.g. train_dataset). Filtering to `class_label` is
            done internally via _LabelFilterDataset.
        class_label : int
            The class whose examples are used (e.g. 3 for MNIST digit 3).
        model : Model, optional
            A pre-loaded Model instance.  If None, `model_id` must be provided.
        model_id : int, optional
            ID of a saved model to load.  Used only when `model` is None.
    
        Returns
        -------
        list of jax.Array
            One array of shape ``(vode_dim,)`` per Vode, in the same order as
            ``model.vodes``.  Pass these directly to ``vode.h.set(...)`` to
            initialise hidden states before generation.
        """
        # --- 1. Filter dataset to the target class ---
        class_dataset = Trainer._LabelFilterDataset(dataset, class_label)
    
        # --- 2. Run forward passes and collect activations + labels ---
        # get_layers iterates the dataset, runs a feedforward initialisation for
        # each sample, and returns the settled h values for every Vode.
        # We ask for labels so we can identify which samples were correct.
        all_layers = self.get_layers(
            dataset=class_dataset,
            input_layer=True,
            return_labels=False,
            model=model,
            model_id=model_id,
        )
    
        # all_layers : list of (N, vode_dim) arrays, one per Vode (plus raw pixels at index 0)
    
        # --- 3. Determine which samples were correctly classified ---
        # The output Vode activations (all_layers[-1]) are the raw logits.
        logits        = all_layers[-1]                        # (N, output_dim)
        predicted     = jnp.argmax(logits, axis=-1)           # (N,)
        correct_mask  = np.array(predicted) == class_label    # (N,) bool
    
        n_correct = correct_mask.sum()
        n_total   = len(correct_mask)
        print(
            f"Class {class_label}: {n_correct}/{n_total} correctly classified. "
            f"Computing mean activations over correct examples only."
        )
    
        if n_correct == 0:
            raise RuntimeError(
                f"No correctly classified examples found for class {class_label}. "
                "Ensure the model is trained before calling this function."
            )
    
        # --- 4. Take the mean over correctly classified examples per Vode ---
        mean_activations = [
            jnp.mean(layer[correct_mask], axis=0)
            for layer in all_layers
        ]
    
        return mean_activations

    def recons_from_means(
        self,
        class_label: int,
        dataset=None,
        mean_activations=None,
        model: Model = None,
        model_id=None,
        num_samples: int = 10,
        T: int = 100,
        lr_h: float = 0.01,
        lr_x: float = 0.01,
        noise_sigma: float = 0.1,
    ) -> np.ndarray:
        """
        Generate image reconstructions from a trained PCN by running top-down
        inference with hidden layers seeded from per-class mean activations and
        the image layer seeded from the per-class mean image plus Gaussian noise.
    
        Parameters
        ----------
        model : Model
            A trained PCN with weights already loaded.
        mean_activations : list of array-like
            Output of compute_class_mean_activations.  Index 0 is the mean input
            image (784-dim for MNIST); indices 1..-1 are the mean hidden vode
            activations in bottom-up order; the last index is the output vode.
        class_label : int
            The class to generate (used to build the one-hot target clamped at
            the output vode).
        num_samples : int
            How many independent reconstructions to generate.
        T : int
            Number of inference (state-update) steps per reconstruction.
        lr_h : float
            Learning rate for the vode-state optimizer.
        lr_x : float
            Learning rate for the image-layer optimizer (gradient descent on x).
        noise_sigma : float
            Standard deviation of the Gaussian noise added to the mean image to
            seed the image layer.  Larger values give more diverse reconstructions.
    
        Returns
        -------
        np.ndarray of shape (num_samples, input_dim)
            The generated images, one per row, in the same pixel-value range as
            the training data.
        """
        model_created = False
        
        if model is None:
            if model_id is None:
                raise Exception('Either model or model_id must be provided')

            model_created = True
                
            model = Model(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dims=self.hidden_dims,
                act_fn=self.act_fn,
                model_key=Trainer.model_keys[model_id],
                residual=self.residual
            )
            pxu.load_params(model, f'{self.root}/{self.study_name}/trained_models/model_{model_id}')

        if dataset is None:
            dataset = self.full_dataset

        if mean_activations is None:
            # mean_activations = self.class_mean_activations(dataset=dataset, class_label=class_label, model=model)
            mean_path = dataset_results_dir("MNIST") / "class_mean_activations" / f"class_{class_label}.pkl"
            with mean_path.open("rb") as f:
                mean_activations = pickle.load(f)

        print('Mean input image:')
        input_mean = mean_activations[0]
        input_mean_side = int(np.sqrt(len(input_mean)))  # 28 for MNIST
        X = input_mean.reshape(input_mean_side, input_mean_side)
        plt.imshow(X, cmap='gray')
        plt.axis('off')
        plt.show()
        
        model.eval()
    
        one_hot_y = jnp.asarray(
            jax.nn.one_hot(jnp.asarray(class_label), model.output_dim.get())
        )                                           # (output_dim,)
    
        mean_image   = jnp.asarray(mean_activations[0])          # (input_dim,)
        mean_h       = [jnp.asarray(a) for a in mean_activations[1:-1]]  # hidden vodes only
    
        reconstructions = []
    
        # We create a single optim_h that is re-initialised for every sample.
        optim_h = pxu.Optim(lambda: optax.sgd(lr_h, momentum=0.9, nesterov=True))

        def energy_single_sample(x_in, *, model: "Model"):
            """
            Scalar energy for a single reconstruction sample.
            Kept local to avoid nesting `value_and_grad` over the transformed
            global `energy(...)` helper, which can leak tracers under newer JAX.
            """
            x_in = jnp.atleast_2d(x_in)  # (1, input_dim)
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                _ = forward(x_in, None, model=model)
                e = model.energy()  # shape (1,)

                l2_h = model.l2_h.get()
                if l2_h > 0:
                    h_pen = 0.0
                    for vode in model.vodes[:-1]:
                        h = vode.get("h")
                        h_pen = h_pen + jnp.sum(h * h)
                    x_pen = jnp.sum(x_in * x_in)
                    e = e + 0.5 * l2_h * (h_pen + x_pen)

            return jnp.sum(e)
    
        for i in tqdm(range(num_samples), desc='Generating reconstructions'):
            # ── 1. Seed the image layer: mean image + Gaussian noise ──────────
            noise = noise_sigma * np.random.randn(*mean_image.shape)
            x = jnp.asarray(mean_image + noise, dtype=jnp.float32)[None, :]  # (1, input_dim)
    
            # ── 2. Seed hidden vode states with per-class means ───────────────
            #    We run a STATUS.INIT forward pass first so that every vode is
            #    allocated, then immediately overwrite the hidden states with the
            #    mean activations.  The output vode is clamped to the one-hot.
            with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
                forward(x, one_hot_y[None, :], model=model)
    
            # Overwrite hidden vode h values with class means (skip output vode)
            for vode, mean_h_i in zip(model.vodes[:-1], mean_h):
                vode.h.set(mean_h_i[None, :])   # add batch dim to match vmap shape
    
            # ── 3. Inference loop: update vode states AND image simultaneously ─
            optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    
            x_current = x  # will be updated each step via gradient on energy
    
            for _ in range(T):
                # Gradient w.r.t. vode states (standard PCN inference)
                e, g_h = pxf.value_and_grad(
                    pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                    has_aux=False,
                )(energy_single_sample)(x_current, model=model)
    
                # Update vode hidden states
                optim_h.step(model, g_h["model"])
    
                # Gradient w.r.t. image x (treat x as the free variable)
                grad_x = jax.grad(
                    lambda x_: energy_single_sample(x_, model=model)
                )(x_current)
    
                x_current = x_current - lr_x * grad_x
    
            optim_h.clear()
    
            # ── 4. Collect the final image ─────────────────────────────────────
            reconstructions.append(np.array(x_current.squeeze(0)))

        if model_created:
            del model

        recons = np.stack(reconstructions, axis=0)   # (num_samples, input_dim)
 
        # ── 5. Plot a grid of all reconstructions ─────────────────────────────
        img_side = int(np.sqrt(recons.shape[1]))  # 28 for MNIST
        ncols = int(np.ceil(np.sqrt(num_samples)))
        nrows = int(np.ceil(num_samples / ncols))
     
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
        axes = np.array(axes).reshape(-1)  # flatten for easy indexing
     
        for idx, ax in enumerate(axes):
            if idx < num_samples:
                img = recons[idx].reshape(img_side, img_side)
                ax.imshow(img, cmap='gray')
                ax.axis('off')
            else:
                ax.set_visible(False)  # hide unused grid cells
     
        fig.suptitle(f'Reconstructions for class {class_label}', fontsize=13)
        plt.tight_layout()
        plt.show()
     
        return recons
