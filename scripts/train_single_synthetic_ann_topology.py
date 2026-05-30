#!/usr/bin/env python3
"""Train one fixed FFN baseline and plot beta_0 across layers."""

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

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, list[np.ndarray]]:
        activations = [x.detach().cpu().numpy()]
        out = x
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, nn.ReLU) or layer == self.layers[-1]:
                activations.append(out.detach().cpu().numpy())
        return out, activations


def train_single_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_epochs: int,
    target_accuracy: float,
    learning_rate: float,
    seed: int,
) -> TopologyNet:
    torch.manual_seed(seed)
    model = TopologyNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    x_tensor = torch.tensor(x_train)
    y_tensor = torch.tensor(y_train)

    print(f"Training until train accuracy reaches {target_accuracy:.3f}...")
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs, _ = model.forward_with_activations(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == y_tensor).float().mean().item()
        if accuracy >= target_accuracy and epoch > 50:
            print(f"Goal reached at epoch {epoch}: accuracy={accuracy:.4f}")
            break
    else:
        print(f"Max epochs reached: final train accuracy={accuracy:.4f}")

    return model


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


def plot_beta0(beta_0_values: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = range(len(beta_0_values))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(layers, beta_0_values, marker="s", linestyle="-", color="#0072B2", linewidth=2.0)
    ax.set_title("30x4 + 18x4 ReLU FFN", fontsize=12, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Betti number $\beta_0$")
    ax.set_xticks(list(layers))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved topology plot to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one fixed FFN baseline and plot beta_0 across layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to full_dataset.npz.")
    parser.add_argument("--output", type=Path, default=Path("topology_results.png"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    x, y = load_data(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )
    model = train_single_model(
        x_train,
        y_train,
        max_epochs=args.max_epochs,
        target_accuracy=args.target_accuracy,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    class_zero_test = torch.tensor(x_test[y_test == 0])
    _, activations = model.forward_with_activations(class_zero_test)
    plot_beta0(beta0_by_layer(activations, k=args.k), args.output)


if __name__ == "__main__":
    main()
