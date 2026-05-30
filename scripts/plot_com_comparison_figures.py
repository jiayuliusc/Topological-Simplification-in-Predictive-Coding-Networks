#!/usr/bin/env python3
"""Create publication-ready figures for FFN vs PCN COM comparison.

Recommended use:

    python scripts/plot_com_comparison_figures.py \
      --results-root /path/to/com_comparison_results \
      --output-dir /path/to/com_comparison_results/figures

The main paper figure is the forest plot of bootstrap mean differences
(`PCN - FFN`) with 95% percentile bootstrap confidence intervals. Supporting
figures show seed-level COM distributions and bootstrap difference densities.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - script works without seaborn.
    sns = None


ARCH_ORDER = ["18x8", "24x8", "30x4_12x4", "30x4_18x4", "30x4_24x4", "30x8"]
ACTIVATION_ORDER = ["leaky_relu", "relu", "tanh"]
ACTIVATION_LABELS = {
    "leaky_relu": "Leaky ReLU",
    "relu": "ReLU",
    "tanh": "Tanh",
}
NETWORK_LABELS = {"ffn": "FFN", "pcn": "PCN"}
PALETTE = {
    "leaky_relu": "#0072B2",
    "relu": "#009E73",
    "tanh": "#D55E00",
    "ffn": "#4D4D4D",
    "pcn": "#CC6677",
}


def set_style() -> None:
    if sns is not None:
        sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "font.size": 9,
        }
    )


def load_results(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(results_root / "bootstrap_summary.csv")
    seed_level = pd.read_csv(results_root / "seed_level_com.csv")
    trials = pd.read_csv(results_root / "bootstrap_trials.csv")
    return summary, seed_level, trials


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


def bootstrap_from_seed_level(
    ffn_coms: np.ndarray,
    pcn_coms: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ffn_coms = np.asarray(ffn_coms, dtype=float)
    pcn_coms = np.asarray(pcn_coms, dtype=float)
    ffn_coms = ffn_coms[np.isfinite(ffn_coms)]
    pcn_coms = pcn_coms[np.isfinite(pcn_coms)]
    if len(ffn_coms) == 0 or len(pcn_coms) == 0:
        raise ValueError("Need at least one FFN and one PCN COM value.")

    ffn_means = np.empty(n_bootstrap, dtype=float)
    pcn_means = np.empty(n_bootstrap, dtype=float)
    for trial in range(n_bootstrap):
        ffn_means[trial] = rng.choice(ffn_coms, size=len(ffn_coms), replace=True).mean()
        pcn_means[trial] = rng.choice(pcn_coms, size=len(pcn_coms), replace=True).mean()

    trials = pd.DataFrame(
        {
            "trial": np.arange(n_bootstrap, dtype=int),
            "ffn_mean": ffn_means,
            "pcn_mean": pcn_means,
            "diff_pcn_minus_ffn": pcn_means - ffn_means,
        }
    )
    summary = summarize_bootstrap_trials(trials)
    summary.update(
        {
            "ffn_observed_mean": float(np.mean(ffn_coms)),
            "pcn_observed_mean": float(np.mean(pcn_coms)),
            "ffn_bootstrap_sample_size": int(len(ffn_coms)),
            "pcn_bootstrap_sample_size": int(len(pcn_coms)),
        }
    )
    return summary, trials


def complete_available_comparisons(
    summary: pd.DataFrame,
    seed_level: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill old skipped rows by bootstrapping available seed-level COM values."""
    summary = summary.copy()
    trials = trials.copy()
    valid_statuses = {"ok", "ok_unequal_n"}

    for idx, row in summary.iterrows():
        if row.get("status") in valid_statuses and np.isfinite(row.get("diff_mean_pcn_minus_ffn", np.nan)):
            continue

        arch = row["arch"]
        activation = row["activation"]
        group = seed_level[
            (seed_level["arch"] == arch)
            & (seed_level["activation"] == activation)
            & (seed_level["valid_com"])
        ]
        ffn_coms = group[group["network"] == "ffn"].sort_values("seed_index")["com"].to_numpy()
        pcn_coms = group[group["network"] == "pcn"].sort_values("seed_index")["com"].to_numpy()
        if len(ffn_coms) == 0 or len(pcn_coms) == 0:
            continue

        replacement_summary, replacement_trials = bootstrap_from_seed_level(
            ffn_coms,
            pcn_coms,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        for key, value in replacement_summary.items():
            summary.loc[idx, key] = value
        summary.loc[idx, "n_ffn"] = int(len(ffn_coms))
        summary.loc[idx, "n_pcn"] = int(len(pcn_coms))
        summary.loc[idx, "status"] = "ok" if len(ffn_coms) == len(pcn_coms) else "ok_unequal_n"
        summary.loc[idx, "skip_reason"] = ""

        replacement_trials.insert(0, "activation", activation)
        replacement_trials.insert(0, "arch", arch)
        trials = pd.concat([trials, replacement_trials], ignore_index=True)

    return summary, trials


def ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["arch"] = pd.Categorical(out["arch"], ARCH_ORDER, ordered=True)
    out["activation"] = pd.Categorical(out["activation"], ACTIVATION_ORDER, ordered=True)
    return out.sort_values(["arch", "activation"])


def valid_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = ordered_summary(summary)
    return out[out["status"].isin(["ok", "ok_unequal_n"]) & np.isfinite(out["diff_mean_pcn_minus_ffn"])]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}")
    plt.close(fig)


def plot_difference_forest(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    data = ordered_summary(summary)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    y_positions = {}
    y = 0
    group_gap = 0.62
    activation_offsets = {"leaky_relu": 0.22, "relu": 0.0, "tanh": -0.22}
    arch_centers = {}

    for arch in ARCH_ORDER:
        base = y
        arch_centers[arch] = base
        for activation in ACTIVATION_ORDER:
            y_positions[(arch, activation)] = base + activation_offsets[activation]
        y -= 1.0 + group_gap

    for _, row in data.iterrows():
        arch = str(row["arch"])
        activation = str(row["activation"])
        ypos = y_positions[(arch, activation)]
        color = PALETTE[activation]
        if row.get("status") not in ["ok", "ok_unequal_n"] or not np.isfinite(row["diff_mean_pcn_minus_ffn"]):
            continue

        mean = float(row["diff_mean_pcn_minus_ffn"])
        low = float(row["diff_ci_low_95"])
        high = float(row["diff_ci_high_95"])
        ax.hlines(ypos, low, high, color=color, linewidth=2.0, alpha=0.95)
        ax.scatter(mean, ypos, s=36, color=color, edgecolor="white", linewidth=0.6, zorder=5)

    ax.axvline(0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_yticks([arch_centers[arch] for arch in ARCH_ORDER])
    ax.set_yticklabels(ARCH_ORDER)
    ax.set_xlabel("Mean COM difference, PCN - FFN (95% bootstrap CI)")
    ax.set_ylabel("Architecture")
    ax.set_title("PCNs have larger center-of-mass topology than matched FFNs", pad=28)

    max_high = np.nanmax(data["diff_ci_high_95"].to_numpy(dtype=float))
    ax.set_xlim(-0.35, max_high + 0.55)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.grid(axis="y", visible=False)

    handles = [
        Line2D([0], [0], color=PALETTE[act], marker="o", linewidth=2, markersize=5)
        for act in ACTIVATION_ORDER
    ]
    labels = [ACTIVATION_LABELS[act] for act in ACTIVATION_ORDER]
    ax.legend(
        handles,
        labels,
        title="Activation",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.065),
        ncol=3,
        columnspacing=1.1,
        handlelength=1.6,
    )

    caption = (
        "Each point is the bootstrap mean difference for one matched "
        "architecture/activation pair; horizontal bars show percentile 95% CIs. "
        "Positive values indicate larger COM for PCNs."
    )
    fig.text(0.02, 0.01, textwrap.fill(caption, 118), ha="left", va="bottom", fontsize=7)
    fig.subplots_adjust(top=0.86, bottom=0.13)

    save_figure(fig, output_dir, "figure_1_com_difference_forest", formats)


def plot_seed_level_distributions(
    seed_level: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    data = seed_level[seed_level["valid_com"]].copy()
    data["arch"] = pd.Categorical(data["arch"], ARCH_ORDER, ordered=True)
    data["activation"] = pd.Categorical(data["activation"], ACTIVATION_ORDER, ordered=True)
    data["network_label"] = data["network"].map(NETWORK_LABELS)
    data = data.sort_values(["activation", "arch", "network"])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8), sharey=True)
    rng = np.random.default_rng(9)
    network_offsets = {"ffn": -0.18, "pcn": 0.18}

    for ax, activation in zip(axes, ACTIVATION_ORDER):
        sub = data[data["activation"] == activation]
        for i, arch in enumerate(ARCH_ORDER):
            for network in ["ffn", "pcn"]:
                vals = sub[(sub["arch"] == arch) & (sub["network"] == network)]["com"].to_numpy()
                if vals.size == 0:
                    continue
                xpos = i + network_offsets[network]
                jitter = rng.normal(0, 0.025, size=vals.size)
                ax.scatter(
                    np.full(vals.size, xpos) + jitter,
                    vals,
                    s=11,
                    alpha=0.58,
                    color=PALETTE[network],
                    linewidth=0,
                    zorder=3,
                )
                mean = np.mean(vals)
                sem = np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
                ci = 1.96 * sem
                ax.errorbar(
                    xpos,
                    mean,
                    yerr=ci,
                    fmt="o",
                    color="black",
                    markerfacecolor=PALETTE[network],
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    markersize=4.5,
                    capsize=2.5,
                    linewidth=1.0,
                    zorder=4,
                )

        ax.set_title(ACTIVATION_LABELS[activation])
        ax.set_xticks(range(len(ARCH_ORDER)))
        ax.set_xticklabels(ARCH_ORDER, rotation=35, ha="right")
        ax.set_xlabel("")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Seed-level COM")
    handles = [
        Patch(facecolor=PALETTE["ffn"], edgecolor="none", label="FFN seeds"),
        Patch(facecolor=PALETTE["pcn"], edgecolor="none", label="PCN models"),
        Line2D([0], [0], color="black", marker="o", linestyle="none", markersize=4, label="Mean +/- 95% CI"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Seed-level COM values by architecture and activation", y=1.09, fontsize=10)
    fig.subplots_adjust(top=0.82, wspace=0.08)

    save_figure(fig, output_dir, "figure_2_seed_level_com_distributions", formats)


def plot_bootstrap_density_grid(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    valid = valid_summary(summary)
    summary_lookup = {
        (str(row["arch"]), str(row["activation"])): row
        for _, row in valid.iterrows()
    }
    fig, axes = plt.subplots(6, 3, figsize=(9.0, 10.0), sharex=True, sharey=False)

    for r, arch in enumerate(ARCH_ORDER):
        for c, activation in enumerate(ACTIVATION_ORDER):
            ax = axes[r, c]
            sub = trials[(trials["arch"] == arch) & (trials["activation"] == activation)]
            row = summary_lookup.get((arch, activation))
            if sub.empty or row is None:
                ax.set_axis_off()
                continue

            diff = sub["diff_pcn_minus_ffn"].to_numpy(dtype=float)
            ax.hist(diff, bins=35, density=True, color=PALETTE[activation], alpha=0.76, linewidth=0)
            ax.axvline(0, color="#222222", linestyle="--", linewidth=0.8)
            ax.axvline(row["diff_mean_pcn_minus_ffn"], color="black", linewidth=1.1)
            ax.axvspan(row["diff_ci_low_95"], row["diff_ci_high_95"], color="black", alpha=0.12, linewidth=0)
            ax.set_title(f"{arch} / {ACTIVATION_LABELS[activation]}", fontsize=8)
            ax.set_yticks([])
            ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)

    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("PCN - FFN")
    fig.suptitle("Bootstrap distributions of COM mean differences", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    save_figure(fig, output_dir, "figure_3_bootstrap_difference_distributions", formats)


def plot_difference_heatmap(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    data = ordered_summary(summary)
    matrix = data.pivot(index="activation", columns="arch", values="diff_mean_pcn_minus_ffn")
    matrix = matrix.reindex(index=ACTIVATION_ORDER, columns=ARCH_ORDER)

    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    values = matrix.to_numpy(dtype=float)
    vmax = np.nanmax(values)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=vmax)

    for y, activation in enumerate(ACTIVATION_ORDER):
        for x, arch in enumerate(ARCH_ORDER):
            val = matrix.loc[activation, arch]
            if np.isfinite(val):
                ax.text(x, y, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(ARCH_ORDER)))
    ax.set_xticklabels(ARCH_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(ACTIVATION_ORDER)))
    ax.set_yticklabels([ACTIVATION_LABELS[act] for act in ACTIVATION_ORDER])
    ax.set_title("Mean bootstrap COM difference (PCN - FFN)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("COM difference")
    fig.tight_layout()

    save_figure(fig, output_dir, "figure_4_com_difference_heatmap", formats)


def write_figure_notes(summary: pd.DataFrame, output_dir: Path) -> None:
    valid = valid_summary(summary)
    unequal = valid[valid["status"] == "ok_unequal_n"]
    min_low = valid["diff_ci_low_95"].min()
    max_high = valid["diff_ci_high_95"].max()
    min_fraction = valid["fraction_pcn_gt_ffn"].min()

    lines = [
        "Recommended paper figure: figure_1_com_difference_forest.pdf",
        "",
        "Why this figure:",
        "- It directly shows the estimand used in the experiment: mean COM difference, PCN - FFN.",
        "- It shows uncertainty with 95% bootstrap confidence intervals.",
        "- It makes the zero/no-difference reference explicit.",
        "- It avoids bar-only summaries and keeps all available architecture/activation groups visible.",
        "",
        f"Valid comparisons: {len(valid)} of {len(summary)} architecture/activation groups.",
        f"All valid 95% CIs are above zero: lowest lower CI = {min_low:.3f}, highest upper CI = {max_high:.3f}.",
        f"Minimum bootstrap fraction with PCN > FFN among valid groups = {min_fraction:.3f}.",
    ]
    if not unequal.empty:
        lines.append("")
        lines.append("Unequal-n groups bootstrapped from available seed-level values:")
        for _, row in unequal.iterrows():
            lines.append(f"- {row['arch']} / {row['activation']}: n_ffn={int(row['n_ffn'])}, n_pcn={int(row['n_pcn'])}")

    (output_dir / "figure_notes.txt").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready COM comparison figures from CSV results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--formats",
        default="pdf,svg,png",
        help="Comma-separated output formats supported by matplotlib.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Bootstrap iterations used only when filling older skipped rows from seed_level_com.csv.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed used only when filling older skipped rows from seed_level_com.csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (args.results_root / "figures")
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    set_style()

    summary, seed_level, trials = load_results(args.results_root)
    summary, trials = complete_available_comparisons(
        summary,
        seed_level,
        trials,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_difference_forest(summary, output_dir, formats)
    plot_seed_level_distributions(seed_level, output_dir, formats)
    plot_bootstrap_density_grid(trials, summary, output_dir, formats)
    plot_difference_heatmap(summary, output_dir, formats)
    write_figure_notes(summary, output_dir)

    print(f"Saved COM comparison figures to {output_dir}")
    print("Main paper figure: figure_1_com_difference_forest.pdf")


if __name__ == "__main__":
    main()
