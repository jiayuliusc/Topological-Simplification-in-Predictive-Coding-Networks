"""Predictive-coding backend imports and shared third-party dependencies."""

import sys
from typing import Callable, Dict, Tuple
import ast
from pathlib import Path

try:
    from .config import CONFIG, dataset_results_dir
except ImportError:
    from config import CONFIG, dataset_results_dir

USE_PCX2 = CONFIG.use_pcx2

for dependency_path in (
    CONFIG.pcx2_dir if USE_PCX2 else CONFIG.pcx_dir,
    CONFIG.ripser_plusplus_dir,
):
    if dependency_path is not None and dependency_path.exists():
        sys.path.append(str(dependency_path))

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import equinox as eqx

try:
    if USE_PCX2:
        import pcx2 as px
        import pcx2.predictive_coding as pxc
        import pcx2.nn as pxnn
        import pcx2.functional as pxf
        import pcx2.utils as pxu
    else:
        import pcx as px
        import pcx.predictive_coding as pxc
        import pcx.nn as pxnn
        import pcx.functional as pxf
        import pcx.utils as pxu
except ModuleNotFoundError as exc:
    package_name = "pcx2" if USE_PCX2 else "pcx"
    path_var = "TDL_PCX2_DIR" if USE_PCX2 else "TDL_PCX_DIR"
    raise ModuleNotFoundError(
        f"Could not import {package_name}. Install {package_name}, set {path_var}, "
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

