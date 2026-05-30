#!/usr/bin/env python3
"""Train a fixed FFN ensemble and summarize beta_0 across layers."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph


def load_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(filepath)
    x = data["points"].astype(np.float32)
    y = data["labels"].astype(np.int64)
    return x, y


class TopologyNet(nn.Module):
    """Fixed 30x4 + 18x4 ReLU FFN used as a simple synthetic baseline."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(2, 30),
                nn.ReLU(),
                nn.Linear(30, 30),
                nn.ReLU(),
                nn.Linear(30, 30),
                nn.ReLU(),
                nn.Linear(30, 30),
                nn.ReLU(),
                nn.Linear(30, 18),
                nn.ReLU(),
                nn.Linear(18, 18),
                nn.ReLU(),
                nn.Linear(18, 18),
                nn.ReLU(),
                nn.Linear(18, 18),
                nn.ReLU(),
                nn.Linear(18, 2),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def get_activations(self, x: torch.Tensor) -> list[np.ndarray]:
        activations = [x.detach().cpu().numpy()]
        out = x
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, nn.ReLU) or layer == self.layers[-1]:
                activations.append(out.detach().cpu().numpy())
        return activations


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    max_epochs: int,
    target_accuracy: float,
    learning_rate: float,
) -> tuple[TopologyNet, float, int]:
    torch.manual_seed(seed)
    model = TopologyNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: 0.5 ** (t / 4000))
    x_tensor = torch.tensor(x_train)
    y_tensor = torch.tensor(y_train)
    accuracy = 0.0

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        accuracy = (outputs.argmax(1) == y_tensor).float().mean().item()
        if accuracy >= target_accuracy and epoch > 50:
            return model, accuracy, epoch

    return model, accuracy, max_epochs


def beta0_by_layer(activations: list[np.ndarray], *, k: int) -> list[int]:
    beta_0_values = []
    for activation in activations:
        knn_graph = kneighbors_graph(
            activation,
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
        )
        n_components, _ = connected_components(knn_graph, directed=False)
        beta_0_values.append(int(n_components))
    return beta_0_values


def plot_ensemble_beta0(all_betti_runs: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_per_layer = np.mean(all_betti_runs, axis=0)
    std_per_layer = np.std(all_betti_runs, axis=0)
    layers = np.arange(len(mean_per_layer))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    color = "#0072B2"
    for run in all_betti_runs:
        ax.plot(layers, run, color=color, alpha=0.15, linewidth=0.8)

    ax.plot(layers, mean_per_layer, color=color, linewidth=2.4, marker="s", label="Mean beta_0")
    ax.fill_between(
        layers,
        mean_per_layer - 0.5 * std_per_layer,
        mean_per_layer + 0.5 * std_per_layer,
        color=color,
        alpha=0.2,
        linewidth=0,
        label="+/- 0.5 SD",
    )
    ax.set_title("Topology robustness for 30x4 + 18x4 ReLU FFNs")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Betti number $\beta_0$")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved ensemble topology plot to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fixed FFN ensemble and summarize beta_0 across layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to full_dataset.npz.")
    parser.add_argument("--output-dir", type=Path, default=Path("ann_ensemble_results"))
    parser.add_argument("--num-runs", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--k", type=int, default=14)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_data(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed_offset,
    )
    class_zero_test = torch.tensor(x_test[y_test == 0])
    all_betti_runs = []
    accuracy_rows = []

    for run in range(args.num_runs):
        seed = args.seed_offset + run
        print(f"Training model {run + 1}/{args.num_runs} with seed {seed}...")
        model, final_accuracy, final_epoch = train_model(
            x_train,
            y_train,
            seed=seed,
            max_epochs=args.max_epochs,
            target_accuracy=args.target_accuracy,
            learning_rate=args.learning_rate,
        )
        torch.save(model.state_dict(), model_dir / f"model_seed_{seed}.pth")
        all_betti_runs.append(beta0_by_layer(model.get_activations(class_zero_test), k=args.k))
        accuracy_rows.append((seed, final_accuracy, final_epoch))

    all_betti_array = np.asarray(all_betti_runs, dtype=float)
    np.savetxt(args.output_dir / "beta0_by_seed.csv", all_betti_array, delimiter=",")
    np.savetxt(
        args.output_dir / "accuracies.csv",
        np.asarray(accuracy_rows, dtype=float),
        delimiter=",",
        header="seed,final_accuracy,final_epoch",
        comments="",
    )
    plot_ensemble_beta0(all_betti_array, args.output_dir / "topology_robustness_plot.png")


if __name__ == "__main__":
    main()
