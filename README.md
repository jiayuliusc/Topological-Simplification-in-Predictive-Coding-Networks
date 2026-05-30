# Topological Simplification in Predictive Coding Networks

This repository contains research code for studying how predictive coding
networks (PCNs) and feedforward neural networks (FFNs) change the topology of
their layer representations.

The main code path compares PCNs and FFNs by:

1. loading saved layer activations or persistence diagrams,
2. building kNN graph metrics where needed,
3. running persistent homology,
4. computing center-of-mass (COM) summaries, and
5. bootstrapping seed-level COM differences between architectures.

## File structure

```text
.
├── config/
│   └── local_config.example.json
├── docs/
│   ├── artifact_manifest_template.csv
│   ├── data.md
│   ├── methods_to_code.md
│   ├── release_checklist.md
│   └── replication.md
├── notebooks/
│   └── README.md
├── scripts/
│   ├── compare_ffn_pcn_com.py
│   ├── plot_com_comparison_figures.py
│   ├── compute_mnist_persistent_homology.py
│   ├── search_mnist_knn_parameters.py
│   ├── reformat_knn_trial_reports.py
│   └── README.md
├── src/topological_dl/
│   ├── config.py
│   ├── data_loading.py
│   ├── pcn_backend.py
│   ├── pcn_model.py
│   ├── pcn_training.py
│   ├── trainer.py
│   └── trainer_impl.py
└── tests/
```

The `trainer.py` module is kept as a compatibility wrapper. The implementation
is split across smaller files:

- `data_loading.py`: dataset loaders.
- `pcn_backend.py`: PCX/PCX2 backend imports and shared dependencies.
- `pcn_model.py`: PCN model and energy functions.
- `pcn_training.py`: training and evaluation loops.
- `trainer_impl.py`: `Trainer` class methods and analysis utilities.

## Configuration

Copy the example config if you want local paths to be picked up automatically:

```bash
cp config/local_config.example.json config/local_config.json
```

You can also set paths with environment variables:

- `TDL_ROOT_DIR`
- `TDL_DATA_DIR`
- `TDL_RESULTS_DIR`
- `TDL_PCX_DIR`
- `TDL_PCX2_DIR`
- `TDL_RIPSER_PLUSPLUS_DIR`
- `TDL_USE_PCX2`

`config/local_config.json` is ignored by git. Use it for machine-specific paths
such as dataset folders, result folders, and local PCX/PCX2 checkouts.

## Environment

Use Python 3.10 or newer. The dependency extras in `pyproject.toml` are starting
points, not lock files:

```bash
python -m pip install -e ".[analysis]"  # COM comparison and figures
python -m pip install -e ".[mnist]"     # MNIST topology helpers
python -m pip install -e ".[torch]"     # simple FFN baselines
python -m pip install -e ".[pcn]"       # PCN trainer utilities
python -m pip install -e ".[dev]"       # tests and linting
```

PCN training depends on `pcx` or `pcx2`. If those packages are not installed,
set `TDL_PCX_DIR` or `TDL_PCX2_DIR`, or add the path in
`config/local_config.json`.

## COM comparison

Run the FFN-vs-PCN COM comparison with:

```bash
python scripts/compare_ffn_pcn_com.py \
  --ffn-activations-root /path/to/ffn/activations \
  --pcn-results-root /path/to/pcn/results/D1 \
  --pcn-study-template "{arch}_{pcn_activation}" \
  --pcn-dir-name ripser_only_0_k14 \
  --output-root /path/to/com_comparison_results \
  --k 14 \
  --eta 2.5 \
  --dims 0,1 \
  --num-seeds 30
```

The script writes:

- `seed_level_com.csv`
- `bootstrap_summary.csv`
- `bootstrap_trials.csv`
- `run_config.json`
- cached FFN diagrams under `ffn_diagram_cache/`

If a group has unequal sample sizes, for example 30 FFN seeds and 29 PCN
models, the script bootstraps the available seed-level values and records the
actual counts.

## Figures

After the COM comparison finishes, generate figures with:

```bash
python scripts/plot_com_comparison_figures.py \
  --results-root /path/to/com_comparison_results \
  --output-dir /path/to/com_comparison_results/figures
```

The main figure is `figure_1_com_difference_forest.pdf`, which plots the mean
COM difference `PCN - FFN` with 95% bootstrap confidence intervals.

## More notes

- `docs/replication.md` describes the expected input/output layouts.
- `docs/methods_to_code.md` maps analysis questions to scripts.
- `docs/data.md` explains which artifacts should stay outside git.
- `scripts/README.md` gives short examples for each script.
- `notebooks/` contains exploratory notebooks kept for project history.

## Citation and license

`CITATION.cff` is a placeholder. Update it with the final author list and
arXiv/DOI information before public release.

Add a license before making the repository public.
