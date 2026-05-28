#!/usr/bin/env python3
"""Seed-level COM comparison between feedforward networks and PCNs.

Procedure implemented here:

1. FFN: load activation matrices from
   <ffn-activations-root>/<arch>/<activation>/seed_0 ... seed_29.
2. FFN: convert each seed's activation matrices to persistence diagrams with
   the same k-NN graph + Ripser pipeline used for the synthetic-data topology
   experiments.
3. PCN: load model_*.dill files from
   <pcn-results-root>/<pcn-study>/<pcn-dir-name>. If they already contain
   persistence diagrams, compute COM directly.
4. Compute one COM value per seed/model with the same
   compute_com_from_diagrams(...) function.
5. Bootstrap at the seed level: resample 30 FFN COMs and 30 PCN COMs with
   replacement, compare mean_pcn - mean_ffn, and save summaries.

The default architecture/activation grid is the overlap requested by the
project:

    architectures: 18x8, 24x8, 30x4_12x4, 30x4_18x4, 30x4_24x4, 30x8
    activations:   leaky_relu, relu, tanh

On CARC, a typical command is:

    python scripts/compare_ffn_pcn_com.py \
      --ffn-activations-root /project2/alvinjin_1630/John/ANN/activations \
      --pcn-results-root /project2/alvinjin_1630/results/D1 \
      --output-root /project2/alvinjin_1630/John/ANN/com_comparison_results
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

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
    "30x4_12x4",
    "30x4_18x4",
    "30x4_24x4",
    "30x8",
)
DEFAULT_ACTIVATIONS = ("leaky_relu", "relu", "tanh")


def parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def load_pickle_or_dill(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return dill.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f)


def save_dill(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        dill.dump(obj, f)


def layer_sort_key(path: Path) -> tuple[int, str]:
    if path.name == "input_layer.pkl":
        return (-1, path.name)
    nums = [int(x) for x in re.findall(r"\d+", path.stem)]
    return (nums[-1] if nums else 10_000, path.name)


def model_id_from_path(path: Path) -> int:
    match = re.search(r"model_(\d+)\.dill$", path.name)
    if not match:
        raise ValueError(f"Cannot parse model id from {path}")
    return int(match.group(1))


def normalize_activation_for_pcn(activation: str) -> str:
    """Map FFN activation folder names to PCN study-name fragments."""
    return "leaky" if activation == "leaky_relu" else activation


def pcn_study_name(arch: str, activation: str, template: str) -> str:
    pcn_activation = normalize_activation_for_pcn(activation)
    return template.format(
        arch=arch,
        activation=activation,
        act=activation,
        pcn_activation=pcn_activation,
        pcn_act=pcn_activation,
    )


def load_ffn_seed(seed_path: Path, *, include_input: bool = False) -> list[np.ndarray]:
    """Load one FFN seed's activation matrices in layer order."""
    if not seed_path.exists():
        raise FileNotFoundError(seed_path)

    files = sorted(seed_path.glob("*.pkl"), key=layer_sort_key)
    if not include_input:
        files = [path for path in files if path.name != "input_layer.pkl"]
    if not files:
        raise FileNotFoundError(f"No activation .pkl files found in {seed_path}")

    activations = []
    for path in files:
        arr = np.asarray(load_pickle_or_dill(path))
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2:
            raise ValueError(f"{path} is not a 2D activation matrix: shape={arr.shape}")
        activations.append(arr)
    return activations


def finite_point_cloud(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D point cloud, got shape {arr.shape}")
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] < 3:
        raise ValueError(f"Need at least 3 finite points, got {arr.shape[0]}")
    return arr


def knn_graph_distances(points: np.ndarray, k: int, max_distance: float) -> np.ndarray:
    """Return unweighted shortest-path distances on an undirected k-NN graph."""
    arr = finite_point_cloud(points)
    n_neighbors = min(k, arr.shape[0] - 1)
    if n_neighbors < 1:
        raise ValueError("Need at least 2 points for a k-NN graph")

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


def activation_to_diagram(
    activation: np.ndarray,
    *,
    k: int,
    maxdim: int,
    thresh: float,
) -> list[np.ndarray]:
    distances = knn_graph_distances(activation, k=k, max_distance=float(thresh) + 1.0)
    return ripser(
        distances,
        distance_matrix=True,
        maxdim=maxdim,
        thresh=float(thresh),
    )["dgms"]


def activations_to_diagrams(
    activations: list[np.ndarray],
    *,
    k: int = 14,
    maxdim: int = 1,
    thresh: float = 2.5,
) -> list[list[np.ndarray]]:
    """Convert one seed's activation matrices to layer-wise persistence diagrams."""
    return [
        activation_to_diagram(layer, k=k, maxdim=maxdim, thresh=thresh)
        for layer in activations
    ]


def looks_like_single_layer_diagram(obj: Any) -> bool:
    """True for [array((n, 2)), array((m, 2)), ...]."""
    if not isinstance(obj, (list, tuple)) or not obj:
        return False
    for item in obj:
        try:
            arr = np.asarray(item)
        except ValueError:
            return False
        if arr.ndim != 2 or arr.shape[1] != 2:
            return False
    return True


def normalize_diagram_object(obj: Any) -> list[list[np.ndarray]]:
    """Normalize supported diagram formats to list[layer][homology_dim]."""
    if looks_like_single_layer_diagram(obj):
        return [[np.asarray(arr, dtype=float) for arr in obj]]

    if isinstance(obj, (list, tuple)):
        layers = []
        for layer in obj:
            if looks_like_single_layer_diagram(layer):
                layers.append([np.asarray(arr, dtype=float) for arr in layer])
            elif isinstance(layer, dict) and "dgms" in layer:
                layers.append([np.asarray(arr, dtype=float) for arr in layer["dgms"]])
            else:
                raise ValueError(
                    "Unsupported diagram layer format. Expected a list of arrays "
                    "with shape (num_points, 2)."
                )
        return layers

    if isinstance(obj, dict) and "dgms" in obj:
        return [[np.asarray(arr, dtype=float) for arr in obj["dgms"]]]

    raise ValueError("Object does not look like persistence diagrams.")


def betti_at_eta(diagrams: list[np.ndarray], dim: int, eta: float) -> int:
    if dim >= len(diagrams):
        return 0
    arr = np.asarray(diagrams[dim], dtype=float)
    if arr.size == 0:
        return 0
    births = arr[:, 0]
    deaths = arr[:, 1]
    return int(np.count_nonzero((births <= eta) & (eta < deaths)))


def compute_com_from_diagrams(
    dgms: Any,
    *,
    eta: float = 2.5,
    dims: tuple[int, ...] = (0, 1),
    use_running_min: bool = True,
    no_drop_value: float = math.nan,
) -> float:
    """Compute one COM value from layer-wise persistence diagrams.

    Parameters
    ----------
    dgms:
        Diagram object in the format already used by PCN files:
        [ [H0_array, H1_array, ...], [H0_array, H1_array, ...], ... ].
        Each Hk array must have shape (num_points, 2).
    eta:
        Filtration value at which Betti numbers are counted.
    dims:
        Homology dimensions to sum into one topology-complexity curve.
    use_running_min:
        If true, compute COM from the running minimum Betti curve, matching the
        paper's irreversible simplification convention.
    """
    layer_diagrams = normalize_diagram_object(dgms)
    betti_curve = np.asarray(
        [
            sum(betti_at_eta(layer, dim, eta) for dim in dims)
            for layer in layer_diagrams
        ],
        dtype=float,
    )

    if betti_curve.size < 2:
        return float(no_drop_value)

    beta = np.minimum.accumulate(betti_curve) if use_running_min else betti_curve
    drops = beta[:-1] - beta[1:]
    if not use_running_min:
        drops = np.maximum(drops, 0.0)

    total_drop = float(np.sum(drops))
    if total_drop <= 0.0:
        return float(no_drop_value)

    transition_index = np.arange(1, betti_curve.size, dtype=float)
    return float(np.dot(transition_index, drops) / total_drop)


def load_pcn_model(dill_path: Path) -> list[list[np.ndarray]]:
    """Load one PCN model_*.dill file and return normalized diagrams."""
    obj = load_pickle_or_dill(dill_path)
    return normalize_diagram_object(obj)


def ffn_diagram_cache_path(
    output_root: Path,
    arch: str,
    activation: str,
    seed: int,
    *,
    k: int,
    maxdim: int,
    thresh: float,
    include_input: bool,
) -> Path:
    layer_scope = "with_input" if include_input else "hidden_only"
    return (
        output_root
        / "ffn_diagram_cache"
        / f"k{k}_maxdim{maxdim}_thresh{thresh:g}_{layer_scope}"
        / arch
        / activation
        / f"seed_{seed}.dill"
    )


def ffn_seed_com(
    *,
    seed_path: Path,
    cache_path: Path,
    k: int,
    maxdim: int,
    thresh: float,
    eta: float,
    dims: tuple[int, ...],
    force_ripser: bool,
    use_running_min: bool,
    include_input: bool,
) -> tuple[float, list[list[np.ndarray]]]:
    if cache_path.exists() and not force_ripser:
        dgms = load_pcn_model(cache_path)
    else:
        activations = load_ffn_seed(seed_path, include_input=include_input)
        dgms = activations_to_diagrams(activations, k=k, maxdim=maxdim, thresh=thresh)
        save_dill(dgms, cache_path)

    return (
        compute_com_from_diagrams(
            dgms,
            eta=eta,
            dims=dims,
            use_running_min=use_running_min,
        ),
        dgms,
    )


def pcn_model_files(
    pcn_study_root: Path,
    *,
    model_selection: str,
    num_seeds: int,
) -> list[Path]:
    ripser_files = sorted(
        pcn_study_root.glob("model_*.dill"),
        key=model_id_from_path,
    )
    if model_selection == "range":
        selected = [pcn_study_root / f"model_{i}.dill" for i in range(num_seeds)]
    elif model_selection == "first-n":
        selected = ripser_files[:num_seeds]
    else:
        raise ValueError(f"Unknown model selection mode: {model_selection}")

    missing = [p for p in selected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing PCN model files: " + ", ".join(str(p) for p in missing[:5])
        )
    return selected


def build_seed_level_com_table(
    *,
    ffn_activations_root: Path,
    pcn_results_root: Path,
    output_root: Path,
    architectures: list[str],
    activations: list[str],
    num_seeds: int,
    pcn_study_template: str,
    pcn_dir_name: str,
    pcn_model_selection: str,
    k: int,
    maxdim: int,
    thresh: float,
    eta: float,
    dims: tuple[int, ...],
    use_running_min: bool,
    force_ripser: bool,
    include_ffn_input: bool,
) -> pd.DataFrame:
    rows = []
    for arch in architectures:
        for activation in activations:
            print(f"\n=== {arch} / {activation} ===")

            ffn_coms = []
            for seed in tqdm(range(num_seeds), desc="FFN seeds"):
                seed_path = ffn_activations_root / arch / activation / f"seed_{seed}"
                cache_path = ffn_diagram_cache_path(
                    output_root,
                    arch,
                    activation,
                    seed,
                    k=k,
                    maxdim=maxdim,
                    thresh=thresh,
                    include_input=include_ffn_input,
                )
                com, _ = ffn_seed_com(
                    seed_path=seed_path,
                    cache_path=cache_path,
                    k=k,
                    maxdim=maxdim,
                    thresh=thresh,
                    eta=eta,
                    dims=dims,
                    force_ripser=force_ripser,
                    use_running_min=use_running_min,
                    include_input=include_ffn_input,
                )
                ffn_coms.append(com)
                rows.append(
                    {
                        "network": "ffn",
                        "arch": arch,
                        "activation": activation,
                        "seed_index": seed,
                        "source_id": f"seed_{seed}",
                        "com": com,
                        "valid_com": bool(np.isfinite(com)),
                    }
                )

            pcn_study = pcn_study_name(arch, activation, pcn_study_template)
            pcn_ripser_dir = pcn_results_root / pcn_study / pcn_dir_name
            pcn_files = pcn_model_files(
                pcn_ripser_dir,
                model_selection=pcn_model_selection,
                num_seeds=num_seeds,
            )

            pcn_coms = []
            for seed_index, model_path in enumerate(tqdm(pcn_files, desc="PCN models")):
                dgms = load_pcn_model(model_path)
                com = compute_com_from_diagrams(
                    dgms,
                    eta=eta,
                    dims=dims,
                    use_running_min=use_running_min,
                )
                pcn_coms.append(com)
                rows.append(
                    {
                        "network": "pcn",
                        "arch": arch,
                        "activation": activation,
                        "seed_index": seed_index,
                        "source_id": model_path.stem,
                        "com": com,
                        "valid_com": bool(np.isfinite(com)),
                    }
                )

            check_exactly_30(arch, activation, ffn_coms, pcn_coms, expected=num_seeds)

    return pd.DataFrame(rows)


def check_exactly_30(
    arch: str,
    activation: str,
    ffn_coms: list[float],
    pcn_coms: list[float],
    *,
    expected: int,
) -> None:
    ffn_valid = np.asarray(ffn_coms, dtype=float)
    pcn_valid = np.asarray(pcn_coms, dtype=float)
    ffn_valid = ffn_valid[np.isfinite(ffn_valid)]
    pcn_valid = pcn_valid[np.isfinite(pcn_valid)]

    if len(ffn_valid) != expected or len(pcn_valid) != expected:
        raise RuntimeError(
            f"{arch}/{activation} failed sanity check: "
            f"FFN valid COM count={len(ffn_valid)}, PCN valid COM count={len(pcn_valid)}, "
            f"expected {expected} each. No bootstrap comparison was run for this group."
        )


def summarize_bootstrap_trials(trials: pd.DataFrame) -> dict[str, float | str]:
    diff = trials["diff_pcn_minus_ffn"].to_numpy(dtype=float)
    ffn_means = trials["ffn_mean"].to_numpy(dtype=float)
    pcn_means = trials["pcn_mean"].to_numpy(dtype=float)
    ci_low, ci_high = np.quantile(diff, [0.025, 0.975])
    if ci_low > 0:
        decision = "pcn_larger"
    elif ci_high < 0:
        decision = "ffn_larger"
    else:
        decision = "unclear"

    return {
        "ffn_bootstrap_mean": float(np.mean(ffn_means)),
        "pcn_bootstrap_mean": float(np.mean(pcn_means)),
        "diff_mean_pcn_minus_ffn": float(np.mean(diff)),
        "diff_ci_low_95": float(ci_low),
        "diff_ci_high_95": float(ci_high),
        "fraction_pcn_gt_ffn": float(np.mean(diff > 0)),
        "decision": decision,
    }


def bootstrap_compare(
    ffn_coms,
    pcn_coms,
    B: int = 10_000,
    seed: int = 42,
    sample_size: int = 30,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Bootstrap one architecture/activation at the seed level."""
    ffn = np.asarray(ffn_coms, dtype=float)
    pcn = np.asarray(pcn_coms, dtype=float)
    ffn = ffn[np.isfinite(ffn)]
    pcn = pcn[np.isfinite(pcn)]

    if ffn.size != sample_size or pcn.size != sample_size:
        raise ValueError(
            f"Expected exactly {sample_size} FFN and {sample_size} PCN COMs, "
            f"got {ffn.size}, {pcn.size}"
        )

    rng = np.random.default_rng(seed)
    ffn_means = np.empty(B, dtype=float)
    pcn_means = np.empty(B, dtype=float)
    for b in range(B):
        ffn_means[b] = np.mean(rng.choice(ffn, size=sample_size, replace=True))
        pcn_means[b] = np.mean(rng.choice(pcn, size=sample_size, replace=True))

    trials = pd.DataFrame(
        {
            "trial": np.arange(B, dtype=int),
            "ffn_mean": ffn_means,
            "pcn_mean": pcn_means,
            "diff_pcn_minus_ffn": pcn_means - ffn_means,
        }
    )
    summary = summarize_bootstrap_trials(trials)
    summary.update(
        {
            "ffn_observed_mean": float(np.mean(ffn)),
            "pcn_observed_mean": float(np.mean(pcn)),
        }
    )
    return summary, trials


def build_bootstrap_table(
    seed_com_df: pd.DataFrame,
    *,
    architectures: list[str],
    activations: list[str],
    B: int,
    seed: int,
    sample_size: int,
) -> pd.DataFrame:
    rows = []
    bootstrap_dir_rows = []
    for arch in architectures:
        for activation in activations:
            group = seed_com_df[
                (seed_com_df["arch"] == arch)
                & (seed_com_df["activation"] == activation)
                & (seed_com_df["valid_com"])
            ]
            ffn_coms = group[group["network"] == "ffn"].sort_values("seed_index")["com"].to_numpy()
            pcn_coms = group[group["network"] == "pcn"].sort_values("seed_index")["com"].to_numpy()
            summary, trials = bootstrap_compare(
                ffn_coms,
                pcn_coms,
                B=B,
                seed=seed,
                sample_size=sample_size,
            )
            summary.update(
                {
                    "arch": arch,
                    "activation": activation,
                    "n_ffn": int(len(ffn_coms)),
                    "n_pcn": int(len(pcn_coms)),
                }
            )
            rows.append(summary)
            trials.insert(0, "activation", activation)
            trials.insert(0, "arch", arch)
            bootstrap_dir_rows.extend(trials.to_dict("records"))

    return pd.DataFrame(rows), pd.DataFrame(bootstrap_dir_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare seed-level COM between FFNs and PCNs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ffn-activations-root", type=Path, required=True)
    parser.add_argument("--pcn-results-root", type=Path, required=True)
    parser.add_argument("--pcn-study-template", default="{arch}_{pcn_activation}")
    parser.add_argument("--pcn-dir-name", default="ripser_only_0_k14")
    parser.add_argument(
        "--pcn-model-selection",
        choices=("first-n", "range"),
        default="first-n",
        help=(
            "'first-n' uses the first 30 model_*.dill files sorted by numeric model id. "
            "'range' requires model_0.dill ... model_29.dill exactly."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("com_comparison_results"))
    parser.add_argument("--architectures", default=",".join(DEFAULT_ARCHITECTURES))
    parser.add_argument("--activations", default=",".join(DEFAULT_ACTIVATIONS))
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--k", type=int, default=14)
    parser.add_argument("--eta", type=float, default=2.5)
    parser.add_argument("--maxdim", type=int, default=1)
    parser.add_argument("--dims", default="0,1")
    parser.add_argument("--no-running-min", action="store_true")
    parser.add_argument(
        "--include-ffn-input",
        action="store_true",
        help=(
            "Include input_layer.pkl in FFN COM. Default skips it so FFN COM "
            "matches PCN model_*.dill files that contain only layer diagrams."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--force-ripser", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    architectures = parse_csv(args.architectures, DEFAULT_ARCHITECTURES)
    activations = parse_csv(args.activations, DEFAULT_ACTIVATIONS)
    dims = tuple(int(x) for x in parse_csv(args.dims, ("0", "1")))
    maxdim = max(args.maxdim, max(dims))

    args.output_root.mkdir(parents=True, exist_ok=True)

    seed_com_df = build_seed_level_com_table(
        ffn_activations_root=args.ffn_activations_root,
        pcn_results_root=args.pcn_results_root,
        output_root=args.output_root,
        architectures=architectures,
        activations=activations,
        num_seeds=args.num_seeds,
        pcn_study_template=args.pcn_study_template,
        pcn_dir_name=args.pcn_dir_name,
        pcn_model_selection=args.pcn_model_selection,
        k=args.k,
        maxdim=maxdim,
        thresh=args.eta,
        eta=args.eta,
        dims=dims,
        use_running_min=not args.no_running_min,
        force_ripser=args.force_ripser,
        include_ffn_input=args.include_ffn_input,
    )

    seed_com_path = args.output_root / "seed_level_com.csv"
    seed_com_df.to_csv(seed_com_path, index=False)

    summary_df, bootstrap_df = build_bootstrap_table(
        seed_com_df,
        architectures=architectures,
        activations=activations,
        B=args.n_bootstrap,
        seed=args.bootstrap_seed,
        sample_size=args.num_seeds,
    )
    summary_path = args.output_root / "bootstrap_summary.csv"
    bootstrap_path = args.output_root / "bootstrap_trials.csv"
    summary_df.to_csv(summary_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)

    config_path = args.output_root / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str))

    print(f"\nSaved per-seed COM values: {seed_com_path}")
    print(f"Saved bootstrap summary: {summary_path}")
    print(f"Saved bootstrap trials: {bootstrap_path}")
    print("\nSummary:")
    print(
        summary_df[
            [
                "arch",
                "activation",
                "ffn_bootstrap_mean",
                "pcn_bootstrap_mean",
                "diff_mean_pcn_minus_ffn",
                "diff_ci_low_95",
                "diff_ci_high_95",
                "fraction_pcn_gt_ffn",
                "decision",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
