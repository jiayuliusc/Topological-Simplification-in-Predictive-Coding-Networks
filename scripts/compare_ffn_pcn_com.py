#!/usr/bin/env python3
"""Compare topology simplification COM between FFNs and PCNs.

This script is designed for CARC runs where FFN activations already live at:

    activations/<arch>/<activation>/seed_<0..29>/*.pkl

Each seed folder should contain `input_layer.pkl` plus one `.pkl` file per
captured layer. The script computes k-NN graph geodesic distances, runs Ripser,
computes per-seed COM, bootstraps 30 seeds with replacement, and compares mean
COM for matched FFN/PCN architecture-activation groups.

PCN input can be supplied in either of two ways:

1. Generic layout, same as FFN:
       --pcn-activations-root /path/to/pcn/activations
       or
       --pcn-ripser-root /path/to/pcn/ripser_cache

2. Existing Trainer.py layout:
       --pcn-trainer-root /path/to/results/D1
       --pcn-study-template "{arch}_{activation}"
       --pcn-dir-name ripser_only_0_k14

For the Trainer layout, the script expects files like:

    <pcn-trainer-root>/<study>/<pcn-dir-name>/model_<seed>.dill

where each `model_<seed>.dill` is a list of per-layer Ripser diagrams.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dill
import numpy as np
import pandas as pd
from ripser import ripser
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


DEFAULT_ARCHITECTURES = (
    "18x8",
    "24x8",
    "30x8",
    "30x4_12x4",
    "30x4_18x4",
    "30x4_24x4",
)
DEFAULT_ACTIVATIONS = ("relu", "tanh", "leaky_relu")


@dataclass(frozen=True)
class Group:
    network: str
    arch: str
    activation: str


def parse_csv_arg(value: str | None, default: Iterable[str]) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def seed_number(path: Path) -> int:
    match = re.search(r"seed[_-]?(\d+)", path.stem if path.is_file() else path.name)
    if not match:
        raise ValueError(f"Could not parse seed number from {path}")
    return int(match.group(1))


def layer_sort_key(path: Path) -> tuple[int, str]:
    if path.name == "input_layer.pkl":
        return (-1, path.name)
    numbers = [int(x) for x in re.findall(r"\d+", path.stem)]
    return (numbers[-1] if numbers else 10_000, path.name)


def load_pickle(path: Path):
    with path.open("rb") as f:
        try:
            return dill.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        dill.dump(obj, f)


def finite_point_cloud(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud, got shape {arr.shape}")
    ok = np.isfinite(arr).all(axis=1)
    arr = arr[ok]
    if arr.shape[0] < 3:
        raise ValueError(f"Need at least 3 finite points, got {arr.shape[0]}")
    return arr


def knn_geodesic_distance_matrix(x: np.ndarray, k: int, max_distance: float) -> np.ndarray:
    """Build an unweighted k-NN graph and return graph shortest-path distances."""
    arr = finite_point_cloud(x)
    n_neighbors = min(k, arr.shape[0] - 1)
    if n_neighbors < 1:
        raise ValueError("Need at least two points for k-NN graph")

    graph = NearestNeighbors(n_neighbors=n_neighbors).fit(arr).kneighbors_graph(
        arr,
        mode="connectivity",
    )
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()

    distances = shortest_path(graph, directed=False, unweighted=True)
    distances = np.asarray(distances, dtype=np.float32)
    distances[~np.isfinite(distances)] = max_distance
    return distances


def compute_diagrams(
    point_cloud: np.ndarray,
    *,
    k: int,
    maxdim: int,
    thresh: float,
) -> list[np.ndarray]:
    max_distance = float(thresh) + 1.0
    distances = knn_geodesic_distance_matrix(point_cloud, k=k, max_distance=max_distance)
    return ripser(
        distances,
        distance_matrix=True,
        maxdim=maxdim,
        thresh=float(thresh),
    )["dgms"]


def activation_layer_files(seed_dir: Path, curve_scope: str) -> list[Path]:
    files = sorted(seed_dir.glob("*.pkl"), key=layer_sort_key)
    if curve_scope == "hidden":
        files = [p for p in files if p.name != "input_layer.pkl"]
    if not files:
        raise FileNotFoundError(f"No activation .pkl files found in {seed_dir}")
    return files


def compute_ripser_for_activation_seed(
    seed_dir: Path,
    output_dir: Path,
    *,
    k: int,
    maxdim: int,
    thresh: float,
    curve_scope: str,
    force: bool,
) -> Path:
    model_path = output_dir / "model.pkl"
    metadata_path = output_dir / "metadata.json"
    if model_path.exists() and not force:
        return model_path

    layer_files = activation_layer_files(seed_dir, curve_scope=curve_scope)
    diagrams = []
    for layer_path in layer_files:
        point_cloud = load_pickle(layer_path)
        diagrams.append(compute_diagrams(point_cloud, k=k, maxdim=maxdim, thresh=thresh))

    save_pickle(diagrams, model_path)
    metadata = {
        "source_activation_dir": str(seed_dir),
        "curve_scope": curve_scope,
        "k": k,
        "maxdim": maxdim,
        "thresh": thresh,
        "layer_files": [p.name for p in layer_files],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model_path


def ensure_generic_ripser(
    *,
    network: str,
    activations_root: Path | None,
    existing_ripser_root: Path | None,
    output_root: Path,
    groups: list[Group],
    seeds: list[int],
    k: int,
    maxdim: int,
    thresh: float,
    curve_scope: str,
    force: bool,
) -> Path | None:
    if existing_ripser_root is not None:
        return existing_ripser_root
    if activations_root is None:
        return None

    ripser_root = output_root / "ripser_cache" / f"{network}_k{k}_thresh{thresh:g}_{curve_scope}"
    for group in tqdm(groups, desc=f"Ripser {network} activations"):
        for seed in seeds:
            seed_dir = activations_root / group.arch / group.activation / f"seed_{seed}"
            if not seed_dir.exists():
                print(f"[skip] missing activation seed dir: {seed_dir}")
                continue
            out_dir = ripser_root / group.arch / group.activation / f"seed_{seed}"
            compute_ripser_for_activation_seed(
                seed_dir,
                out_dir,
                k=k,
                maxdim=maxdim,
                thresh=thresh,
                curve_scope=curve_scope,
                force=force,
            )
    return ripser_root


def betti_at_eta(diagrams: list[np.ndarray], dim: int, eta: float) -> int:
    if dim >= len(diagrams):
        return 0
    arr = np.asarray(diagrams[dim], dtype=float)
    if arr.size == 0:
        return 0
    births = arr[:, 0]
    deaths = arr[:, 1]
    return int(np.count_nonzero((births <= eta) & (eta < deaths)))


def betti_curve_from_diagrams(
    layer_diagrams: list[list[np.ndarray]],
    *,
    eta: float,
    dims: list[int],
) -> np.ndarray:
    return np.asarray(
        [sum(betti_at_eta(diagrams, dim, eta) for dim in dims) for diagrams in layer_diagrams],
        dtype=float,
    )


def com_from_betti_curve(
    betti_curve: np.ndarray,
    *,
    use_running_min: bool = True,
    no_drop_value: float = math.nan,
) -> float:
    beta = np.asarray(betti_curve, dtype=float)
    if beta.size < 2:
        return float(no_drop_value)

    beta_use = np.minimum.accumulate(beta) if use_running_min else beta.copy()
    drops = beta_use[:-1] - beta_use[1:]
    if not use_running_min:
        drops = np.maximum(drops, 0.0)

    total_drop = float(np.sum(drops))
    if total_drop <= 0:
        return float(no_drop_value)

    transitions = np.arange(1, len(beta), dtype=float)
    return float(np.dot(transitions, drops) / total_drop)


def load_generic_layer_diagrams(ripser_root: Path, group: Group, seed: int) -> list[list[np.ndarray]]:
    model_path = ripser_root / group.arch / group.activation / f"seed_{seed}" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    diagrams = load_pickle(model_path)
    if not isinstance(diagrams, list):
        raise TypeError(f"Expected list of layer diagrams in {model_path}")
    return diagrams


def load_trainer_layer_diagrams(
    pcn_trainer_root: Path,
    group: Group,
    seed: int,
    *,
    study_template: str,
    dir_name: str,
    curve_scope: str,
) -> list[list[np.ndarray]]:
    study = study_template.format(arch=group.arch, activation=group.activation, act=group.activation)
    model_path = pcn_trainer_root / study / dir_name / f"model_{seed}.dill"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    diagrams = load_pickle(model_path)
    if curve_scope == "with-input":
        input_path = pcn_trainer_root / study / dir_name / "input_layer.dill"
        if input_path.exists():
            diagrams = [load_pickle(input_path), *diagrams]
    return diagrams


def collect_com_rows(
    *,
    network: str,
    groups: list[Group],
    seeds: list[int],
    eta: float,
    dims: list[int],
    use_running_min: bool,
    generic_ripser_root: Path | None = None,
    pcn_trainer_root: Path | None = None,
    pcn_study_template: str = "{arch}_{activation}",
    pcn_dir_name: str = "ripser_only_0_k14",
    curve_scope: str = "with-input",
) -> pd.DataFrame:
    rows = []
    for group in groups:
        for seed in seeds:
            try:
                if pcn_trainer_root is not None:
                    layer_diagrams = load_trainer_layer_diagrams(
                        pcn_trainer_root,
                        group,
                        seed,
                        study_template=pcn_study_template,
                        dir_name=pcn_dir_name,
                        curve_scope=curve_scope,
                    )
                else:
                    if generic_ripser_root is None:
                        raise ValueError(f"No Ripser root provided for {network}")
                    layer_diagrams = load_generic_layer_diagrams(generic_ripser_root, group, seed)

                curve = betti_curve_from_diagrams(layer_diagrams, eta=eta, dims=dims)
                com = com_from_betti_curve(curve, use_running_min=use_running_min)
                rows.append(
                    {
                        "network": network,
                        "arch": group.arch,
                        "activation": group.activation,
                        "seed": seed,
                        "com": com,
                        "valid_com": bool(np.isfinite(com)),
                        "betti_curve": json.dumps(curve.tolist()),
                    }
                )
            except FileNotFoundError as exc:
                print(f"[skip] {network} {group.arch}/{group.activation}/seed_{seed}: missing {exc}")
            except Exception as exc:
                print(f"[error] {network} {group.arch}/{group.activation}/seed_{seed}: {exc}")

    return pd.DataFrame(rows)


def bootstrap_compare(
    com_df: pd.DataFrame,
    *,
    seeds_per_bootstrap: int,
    n_bootstrap: int,
    rng_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows = []
    valid = com_df[com_df["valid_com"]].copy()

    for (arch, activation), group_df in valid.groupby(["arch", "activation"]):
        ffn = group_df[group_df["network"] == "ffn"]["com"].to_numpy(dtype=float)
        pcn = group_df[group_df["network"] == "pcn"]["com"].to_numpy(dtype=float)
        if ffn.size == 0 or pcn.size == 0:
            print(f"[skip] bootstrap {arch}/{activation}: need both FFN and PCN COM values")
            continue

        ffn_means = np.empty(n_bootstrap, dtype=float)
        pcn_means = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            ffn_means[b] = np.mean(rng.choice(ffn, size=seeds_per_bootstrap, replace=True))
            pcn_means[b] = np.mean(rng.choice(pcn, size=seeds_per_bootstrap, replace=True))

        diff = pcn_means - ffn_means
        rows.append(
            {
                "arch": arch,
                "activation": activation,
                "n_ffn_valid": int(ffn.size),
                "n_pcn_valid": int(pcn.size),
                "ffn_mean_com_observed": float(np.mean(ffn)),
                "pcn_mean_com_observed": float(np.mean(pcn)),
                "pcn_minus_ffn_observed": float(np.mean(pcn) - np.mean(ffn)),
                "bootstrap_ffn_mean": float(np.mean(ffn_means)),
                "bootstrap_pcn_mean": float(np.mean(pcn_means)),
                "bootstrap_diff_mean": float(np.mean(diff)),
                "bootstrap_diff_ci_low": float(np.quantile(diff, 0.025)),
                "bootstrap_diff_ci_high": float(np.quantile(diff, 0.975)),
                "p_pcn_gt_ffn": float(np.mean(diff > 0)),
                "p_pcn_lt_ffn": float(np.mean(diff < 0)),
                "winner_by_observed_mean": "pcn" if np.mean(pcn) > np.mean(ffn) else "ffn",
            }
        )
    return pd.DataFrame(rows)


def make_groups(architectures: list[str], activations: list[str], network: str) -> list[Group]:
    return [Group(network=network, arch=arch, activation=act) for arch in architectures for act in activations]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ffn-activations-root", type=Path, required=True)
    parser.add_argument("--ffn-ripser-root", type=Path, default=None)
    parser.add_argument("--pcn-activations-root", type=Path, default=None)
    parser.add_argument("--pcn-ripser-root", type=Path, default=None)
    parser.add_argument("--pcn-trainer-root", type=Path, default=None)
    parser.add_argument("--pcn-study-template", default="{arch}_{activation}")
    parser.add_argument("--pcn-dir-name", default="ripser_only_0_k14")
    parser.add_argument("--output-root", type=Path, default=Path("com_comparison_results"))
    parser.add_argument("--architectures", default=",".join(DEFAULT_ARCHITECTURES))
    parser.add_argument("--activations", default=",".join(DEFAULT_ACTIVATIONS))
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--k", type=int, default=14)
    parser.add_argument("--eta", type=float, default=2.5)
    parser.add_argument("--maxdim", type=int, default=1)
    parser.add_argument("--dims", default="0,1")
    parser.add_argument("--curve-scope", choices=("with-input", "hidden"), default="with-input")
    parser.add_argument("--no-running-min", action="store_true")
    parser.add_argument("--bootstrap-sample-size", type=int, default=30)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--force-ripser", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    architectures = parse_csv_arg(args.architectures, DEFAULT_ARCHITECTURES)
    activations = parse_csv_arg(args.activations, DEFAULT_ACTIVATIONS)
    dims = [int(x) for x in parse_csv_arg(args.dims, ("0", "1"))]
    seeds = list(range(args.num_seeds))
    thresh = args.eta
    maxdim = max(args.maxdim, max(dims))

    args.output_root.mkdir(parents=True, exist_ok=True)
    ffn_groups = make_groups(architectures, activations, "ffn")
    pcn_groups = make_groups(architectures, activations, "pcn")

    ffn_ripser_root = ensure_generic_ripser(
        network="ffn",
        activations_root=args.ffn_activations_root,
        existing_ripser_root=args.ffn_ripser_root,
        output_root=args.output_root,
        groups=ffn_groups,
        seeds=seeds,
        k=args.k,
        maxdim=maxdim,
        thresh=thresh,
        curve_scope=args.curve_scope,
        force=args.force_ripser,
    )

    pcn_ripser_root = None
    if args.pcn_trainer_root is None:
        pcn_ripser_root = ensure_generic_ripser(
            network="pcn",
            activations_root=args.pcn_activations_root,
            existing_ripser_root=args.pcn_ripser_root,
            output_root=args.output_root,
            groups=pcn_groups,
            seeds=seeds,
            k=args.k,
            maxdim=maxdim,
            thresh=thresh,
            curve_scope=args.curve_scope,
            force=args.force_ripser,
        )

    if args.pcn_trainer_root is None and pcn_ripser_root is None:
        raise SystemExit(
            "Provide one PCN source: --pcn-activations-root, --pcn-ripser-root, "
            "or --pcn-trainer-root. If PCN results are not ready yet, run this "
            "script later with the PCN path filled in."
        )

    ffn_com = collect_com_rows(
        network="ffn",
        groups=ffn_groups,
        seeds=seeds,
        eta=args.eta,
        dims=dims,
        use_running_min=not args.no_running_min,
        generic_ripser_root=ffn_ripser_root,
        curve_scope=args.curve_scope,
    )
    pcn_com = collect_com_rows(
        network="pcn",
        groups=pcn_groups,
        seeds=seeds,
        eta=args.eta,
        dims=dims,
        use_running_min=not args.no_running_min,
        generic_ripser_root=pcn_ripser_root,
        pcn_trainer_root=args.pcn_trainer_root,
        pcn_study_template=args.pcn_study_template,
        pcn_dir_name=args.pcn_dir_name,
        curve_scope=args.curve_scope,
    )

    com_df = pd.concat([ffn_com, pcn_com], ignore_index=True)
    com_path = args.output_root / "com_by_seed.csv"
    com_df.to_csv(com_path, index=False)

    comparison_df = bootstrap_compare(
        com_df,
        seeds_per_bootstrap=args.bootstrap_sample_size,
        n_bootstrap=args.n_bootstrap,
        rng_seed=args.bootstrap_seed,
    )
    comparison_path = args.output_root / "bootstrap_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    config_path = args.output_root / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str))

    print(f"\nSaved per-seed COM: {com_path}")
    print(f"Saved bootstrap comparison: {comparison_path}")
    if not comparison_df.empty:
        print("\nComparison summary:")
        print(
            comparison_df[
                [
                    "arch",
                    "activation",
                    "ffn_mean_com_observed",
                    "pcn_mean_com_observed",
                    "pcn_minus_ffn_observed",
                    "p_pcn_gt_ffn",
                    "winner_by_observed_mean",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
