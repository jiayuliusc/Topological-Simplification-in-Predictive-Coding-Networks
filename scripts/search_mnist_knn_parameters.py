#!/usr/bin/env python3
"""Search MNIST kNN graph parameters for class-wise persistent homology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import dill
import numpy as np
import pandas as pd
import torch
import torchvision
from ripser import ripser
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from topological_dl.config import CONFIG, dataset_results_dir


def parse_int_range(value: str) -> range:
    """Parse inclusive integer ranges like '3:15' or a single integer."""
    if ":" in value:
        start, end = [int(part) for part in value.split(":", maxsplit=1)]
        return range(start, end + 1)
    item = int(value)
    return range(item, item + 1)


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_raw_mnist_by_class(data_root: Path) -> dict[int, np.ndarray]:
    """Load train MNIST as flattened [0, 1] arrays grouped by digit."""
    dataset = torchvision.datasets.MNIST(root=str(data_root), train=True, download=True)
    loader = DataLoader(dataset, batch_size=60_000, shuffle=False)
    images, labels = next(iter(loader))
    x_all = images.view(images.size(0), -1).numpy().astype(np.float32) / 255.0
    y_all = labels.numpy()
    return {digit: x_all[y_all == digit] for digit in range(10)}


def load_class_dill(path: Path) -> np.ndarray:
    """Load a class-specific dill file and coerce it to a 2D numpy array."""
    with path.open("rb") as f:
        obj = dill.load(f)

    if isinstance(obj, np.ndarray):
        arr = obj
    else:
        rows = []
        for item in obj:
            if isinstance(item, tuple):
                item = item[0]
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            rows.append(np.asarray(item).reshape(-1))
        arr = np.vstack(rows)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D class array, got shape {arr.shape}")
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def load_data_by_class(data_root: Path, class_data_dir: Path | None) -> dict[int, np.ndarray]:
    if class_data_dir is None:
        return load_raw_mnist_by_class(data_root)

    data_by_class = {}
    for path in sorted(class_data_dir.glob("label_*_1500.dill")):
        digit = int(path.stem.split("_")[1])
        data_by_class[digit] = load_class_dill(path)
    if not data_by_class:
        raise FileNotFoundError(f"No label_*_1500.dill files found in {class_data_dir}")
    return data_by_class


def run_knn_search(
    *,
    data_by_class: dict[int, np.ndarray],
    output_dir: Path,
    n_trials: int,
    subsample_size: int,
    classes: list[int],
    k_range: range,
    eta_range: range,
    maxdim: int,
    thresh: float,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    print(f"Writing trial reports to {output_dir}")

    for trial in tqdm(range(n_trials), desc="Trials"):
        trial_path = output_dir / f"trial_{trial}_all_classes.csv"
        with trial_path.open("w") as f:
            f.write(f"Trial ID: {trial}\n")
            f.write(f"Subsample Size: {subsample_size}\n")
            f.write("=" * 40 + "\n\n")

            for digit in classes:
                if digit not in data_by_class:
                    print(f"Skipping digit {digit}: no data available")
                    continue

                x_pool = np.asarray(data_by_class[digit], dtype=np.float32)
                if subsample_size and x_pool.shape[0] > subsample_size:
                    indices = rng.choice(x_pool.shape[0], subsample_size, replace=False)
                    x_sub = x_pool[indices]
                else:
                    x_sub = x_pool

                records = []
                for k in k_range:
                    n_neighbors = min(k, x_sub.shape[0] - 1)
                    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(x_sub)
                    graph = neighbors.kneighbors_graph(x_sub, mode="connectivity")
                    distances = shortest_path(graph.maximum(graph.T), directed=False, unweighted=True)
                    distances[~np.isfinite(distances)] = thresh + 1

                    dgms = ripser(distances, distance_matrix=True, maxdim=maxdim, thresh=thresh)["dgms"]
                    dgm0 = dgms[0]
                    dgm1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

                    for eta in eta_range:
                        b0 = int(np.sum(dgm0[:, 1] > eta))
                        b1 = int(np.sum((dgm1[:, 0] <= eta) & (eta < dgm1[:, 1])))
                        records.append({"k": k, "eta": eta, "B0": b0, "B1": b1})

                df = pd.DataFrame(records)
                matrix_b0 = df.pivot(index="k", columns="eta", values="B0")
                matrix_b1 = df.pivot(index="k", columns="eta", values="B1")

                f.write(f"=== DIGIT {digit} ANALYSIS ===\n")
                f.write(f"--- Digit {digit}: B0 (Components) ---\n")
                matrix_b0.to_csv(f)
                f.write("\n")
                f.write(f"--- Digit {digit}: B1 (Holes) ---\n")
                matrix_b1.to_csv(f)
                f.write("\n" + "#" * 40 + "\n\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search MNIST kNN graph parameters for persistent homology.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=CONFIG.data_dir)
    parser.add_argument(
        "--class-data-dir",
        type=Path,
        default=None,
        help="Optional directory containing label_*_1500.dill files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dataset_results_dir("MNIST") / "knn_parameter_trials",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--subsample-size", type=int, default=1500)
    parser.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--k-range", default="3:15")
    parser.add_argument("--eta-range", default="1:13")
    parser.add_argument("--maxdim", type=int, default=1)
    parser.add_argument("--thresh", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_by_class = load_data_by_class(args.data_root, args.class_data_dir)
    run_knn_search(
        data_by_class=data_by_class,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        subsample_size=args.subsample_size,
        classes=parse_int_csv(args.classes),
        k_range=parse_int_range(args.k_range),
        eta_range=parse_int_range(args.eta_range),
        maxdim=args.maxdim,
        thresh=args.thresh,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
