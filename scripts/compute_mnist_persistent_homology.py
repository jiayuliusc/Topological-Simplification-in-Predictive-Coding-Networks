#!/usr/bin/env python3
"""Compute MNIST layer-wise Betti summaries from trained PCN models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import dill
import numpy as np
import pandas as pd
from ripser import ripser
from tqdm import tqdm

from topological_dl.config import dataset_results_dir


def parse_float_csv(value: str) -> np.ndarray:
    return np.asarray([float(part.strip()) for part in value.split(",") if part.strip()])


def betti_at_epsilon(diagram: np.ndarray, epsilon: float) -> int:
    """Count persistence intervals alive at epsilon."""
    if diagram is None or len(diagram) == 0:
        return 0
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    return int(np.sum((births <= epsilon) & (epsilon < deaths)))


def calculate_betti_sums(diagrams: list[np.ndarray], epsilons: np.ndarray) -> dict[str, int]:
    """Return summed Betti numbers across available homology dimensions."""
    sums = {}
    for epsilon in epsilons:
        total = sum(betti_at_epsilon(diagram, epsilon) for diagram in diagrams)
        sums[f"sum_eps_{epsilon:.1f}"] = int(total)
    return sums


def load_class_dataset(data_dir: Path, class_idx: int) -> list[tuple[jnp.ndarray, int]]:
    import jax.numpy as jnp

    data_path = data_dir / f"label_{class_idx}_1500.dill"
    with data_path.open("rb") as f:
        x_raw = dill.load(f)
    return [(jnp.asarray(x), class_idx) for x in x_raw]


def run_analysis(
    *,
    trainer_instance,
    data_dir: Path,
    output_dir: Path,
    num_models: int,
    classes: list[int],
    epsilons: np.ndarray,
    maxdim: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_idx in classes:
        print(f"\n--- Processing class {class_idx} ---")
        try:
            class_dataset = load_class_dataset(data_dir, class_idx)
        except FileNotFoundError:
            print(f"Skipping class {class_idx}: missing {data_dir / f'label_{class_idx}_1500.dill'}")
            continue

        rows = []
        for model_idx in tqdm(range(num_models), desc=f"Class {class_idx} models"):
            try:
                layers = trainer_instance.get_layers(
                    dataset=class_dataset,
                    model_id=model_idx,
                    input_layer=True,
                    return_labels=False,
                )
            except Exception as exc:
                print(f"Skipping model {model_idx}: {exc}")
                continue

            for layer_idx, layer_activation in enumerate(layers):
                data_points = np.asarray(layer_activation)
                if not np.isfinite(data_points).all():
                    print(f"Skipping model {model_idx}, layer {layer_idx}: non-finite activations")
                    continue
                try:
                    dgms = ripser(data_points, maxdim=maxdim)["dgms"]
                except Exception as exc:
                    print(f"Ripser failed for model {model_idx}, layer {layer_idx}: {exc}")
                    continue

                row = {
                    "model_id": model_idx,
                    "layer_id": layer_idx,
                    "num_points": len(data_points),
                }
                row.update(calculate_betti_sums(dgms, epsilons))
                rows.append(row)

        if rows:
            output_path = output_dir / f"results_class_{class_idx}.csv"
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"Saved {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute MNIST layer-wise Betti summaries from trained PCN models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_root = dataset_results_dir("MNIST")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--data-dir", type=Path, default=default_root / "data_by_class")
    parser.add_argument("--output-dir", type=Path, default=default_root / "topology_csvs")
    parser.add_argument("--study-name", default="256x8_ReLU")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--num-models", type=int, default=1)
    parser.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--epsilons", default="1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5")
    parser.add_argument("--maxdim", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import jax
    from topological_dl.trainer import Trainer

    trainer = Trainer(
        dataset="MNIST",
        hidden_dims=[args.hidden_dim] * args.num_hidden_layers,
        act_fn=jax.nn.relu,
        study_name=args.study_name,
        root=args.root,
    )
    classes = [int(part.strip()) for part in args.classes.split(",") if part.strip()]
    run_analysis(
        trainer_instance=trainer,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_models=args.num_models,
        classes=classes,
        epsilons=parse_float_csv(args.epsilons),
        maxdim=args.maxdim,
    )


if __name__ == "__main__":
    main()
