# Scripts

This directory contains the main command-line scripts for the research code and
a few legacy helper scripts retained for provenance.

## Primary Analysis Scripts

- `compare_ffn_pcn_com.py`: computes FFN Ripser diagrams from activation
  matrices, loads PCN diagrams, computes seed-level COM, and performs
  seed-level bootstrap comparisons.
- `plot_com_comparison_figures.py`: generates paper-ready COM comparison
  figures from `seed_level_com.csv`, `bootstrap_summary.csv`, and
  `bootstrap_trials.csv`.
Example:

```bash
python scripts/compare_ffn_pcn_com.py \
  --ffn-activations-root /path/to/ffn/activations \
  --pcn-results-root /path/to/pcn/results/D1 \
  --output-root /path/to/com_comparison_results

python scripts/plot_com_comparison_figures.py \
  --results-root /path/to/com_comparison_results \
  --output-dir /path/to/com_comparison_results/figures
```

## Supporting Analysis Scripts

- `compute_mnist_persistent_homology.py`: computes MNIST class/layer Betti
  summaries from trained PCN models.
- `search_mnist_knn_parameters.py`: sweeps kNN graph parameters for MNIST
  persistent-homology preprocessing.
- `reformat_knn_trial_reports.py`: reformats kNN sweep outputs.
- `generate_mnist_orchard_reconstructions.py`: legacy Orchard/Sun MNIST
  reconstruction wrapper. It requires a helper module named `Trainer_orchard`
  that is not currently part of the public package, so it is retained for
  provenance rather than recommended as a core public entry point.

Examples:

```bash
python scripts/compute_mnist_persistent_homology.py \
  --root /path/to/results/MNIST \
  --data-dir /path/to/results/MNIST/data_by_class \
  --study-name 256x8_ReLU \
  --num-models 30

python scripts/search_mnist_knn_parameters.py \
  --data-root /path/to/data \
  --output-dir /path/to/results/MNIST/knn_parameter_trials \
  --n-trials 30 \
  --classes 0,1,2,3,4,5,6,7,8,9

python scripts/reformat_knn_trial_reports.py \
  --input-dir /path/to/results/MNIST/knn_parameter_trials \
  --output-dir /path/to/results/MNIST/knn_trials_by_class
```

## FFN Baseline Scripts

- `train_single_synthetic_ann_topology.py`: trains one fixed FFN baseline on a
  synthetic dataset and saves a topology plot.
- `train_synthetic_ann_ensemble_topology.py`: trains a 30-seed fixed FFN
  ensemble and saves seed-level topology summaries.

These FFN scripts are intentionally simple baselines. For the paper COM
comparison, use `compare_ffn_pcn_com.py` on the saved FFN activation folders
described in `docs/replication.md`.

Examples:

```bash
python scripts/train_single_synthetic_ann_topology.py \
  --dataset /path/to/full_dataset.npz \
  --output /path/to/results/single_ffn_topology.png

python scripts/train_synthetic_ann_ensemble_topology.py \
  --dataset /path/to/full_dataset.npz \
  --output-dir /path/to/results/ann_ensemble
```
