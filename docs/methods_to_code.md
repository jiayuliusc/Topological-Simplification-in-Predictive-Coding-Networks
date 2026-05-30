# Methods-to-Code Map

This page maps the main reproducible analyses to repository entry points.

## FFN vs PCN COM Comparison

Question: Do PCNs and FFNs differ in the center-of-mass (COM) of topological
simplification?

Use:

- `scripts/compare_ffn_pcn_com.py`
- `scripts/plot_com_comparison_figures.py`

Inputs:

- FFN activation folders described in `docs/replication.md`.
- PCN `model_*.dill` persistence-diagram outputs.

Outputs:

- `seed_level_com.csv`
- `bootstrap_summary.csv`
- `bootstrap_trials.csv`
- `figures/figure_1_com_difference_forest.pdf`

## MNIST Persistent Homology Summaries

Question: How do Betti summaries change across layers/classes for trained MNIST
PCNs?

Use:

- `scripts/compute_mnist_persistent_homology.py`

Inputs:

- Trained MNIST PCN models under `<root>/<study_name>/trained_models/`.
- Class-specific data files such as `label_0_1500.dill`.

Outputs:

- One CSV per class, for example `results_class_0.csv`.

## MNIST kNN Parameter Search

Question: Which kNN graph parameters produce stable class-wise topology
summaries?

Use:

- `scripts/search_mnist_knn_parameters.py`
- `scripts/reformat_knn_trial_reports.py`

Outputs:

- Report-style trial CSV files.
- Reformatted per-digit wide CSV files.

## FFN Synthetic Baselines

Question: How does a fixed FFN baseline simplify the synthetic dataset topology?

Use:

- `scripts/train_single_synthetic_ann_topology.py`
- `scripts/train_synthetic_ann_ensemble_topology.py`

Inputs:

- Synthetic dataset `.npz` with `points` and `labels` arrays.

Outputs:

- Model checkpoints in the selected output directory.
- Beta-0 summary CSVs and topology plots.

## Exploratory Notebooks

The notebooks in `notebooks/` are retained for provenance. They are not the
recommended public code interface because some depend on intermediate files
generated during development.
