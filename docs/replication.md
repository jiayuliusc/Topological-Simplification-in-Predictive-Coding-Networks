# Core Workflow Guide

This guide documents the core file interfaces and script workflow used by the
project. It is meant to make the research code readable and adaptable, not to be
a turnkey environment recipe. Use your own environment manager, cluster modules,
or coding assistant to install compatible package versions for your machine.

The notebooks in `notebooks/` are archived exploratory analyses and are useful
for provenance. The scripts are the clearer public entry points for the core
pipeline.

## 1. Configure Local Paths

The `pyproject.toml` extras provide dependency starting points. For example:

```bash
python -m pip install -e ".[analysis]"
```

Use `.[pcn]` when training or loading PCN models through
`topological_dl.trainer`, and `.[mnist]` for the MNIST kNN helper scripts. These
extras are not a strict lock file; adapt them as needed for your local JAX,
PyTorch, CUDA, or cluster setup.

Create a machine-local config:

```bash
cp config/local_config.example.json config/local_config.json
```

Edit `config/local_config.json` so that:

- `data_dir` points to downloaded or generated datasets.
- `results_dir` points to trained models, persistent-homology outputs, and
  derived results.
- `pcx_dir` or `pcx2_dir` points to the corresponding predictive-coding
  library checkout if it is not installed as a package.

Do not commit `config/local_config.json`; it is intentionally ignored by git.

## 2. Expected FFN Activation Layout

The FFN-vs-PCN COM comparison expects FFN activations in this layout:

```text
activations/
  18x8/
    relu/
      seed_0/
        input_layer.pkl
        net_0.pkl
        net_2.pkl
        ...
      seed_1/
      ...
    tanh/
    leaky_relu/
  24x8/
  30x4_12x4/
  30x4_18x4/
  30x4_24x4/
  30x8/
```

Each `seed_*` folder should contain one activation matrix per layer. By
default, `compare_ffn_pcn_com.py` excludes `input_layer.pkl` from the COM
calculation so that FFN and PCN COM values are computed over corresponding
hidden/output layer diagrams.

## 3. Expected PCN Ripser Layout

The default PCN layout is:

```text
results/D1/
  18x8_relu/
    ripser_only_0_k14/
      model_0.dill
      model_1.dill
      ...
  18x8_tanh/
  18x8_leaky/
  ...
```

Each `model_*.dill` file should contain layer-wise persistence diagrams in the
project format:

```python
[
    [H0_array, H1_array, ...],
    [H0_array, H1_array, ...],
    ...
]
```

where each array has shape `(num_points, 2)` with birth/death columns.

## 4. Run FFN vs PCN COM Comparison

```bash
python scripts/compare_ffn_pcn_com.py \
  --ffn-activations-root /path/to/ffn/activations \
  --pcn-results-root /path/to/pcn/results/D1 \
  --output-root /path/to/com_comparison_results \
  --pcn-study-template "{arch}_{pcn_activation}" \
  --pcn-dir-name ripser_only_0_k14 \
  --k 14 \
  --eta 2.5 \
  --dims 0,1 \
  --num-seeds 30
```

Outputs:

- `seed_level_com.csv`: one COM value per model seed.
- `bootstrap_summary.csv`: one row per architecture/activation comparison.
- `bootstrap_trials.csv`: bootstrap mean differences for uncertainty plots.
- `ffn_diagram_cache/`: cached FFN persistence diagrams.
- `run_config.json`: the command configuration used for the run.

The bootstrap is performed at the seed/model level. If a group has unequal
sample sizes, for example 30 FFNs and 29 PCNs, the script bootstraps the
available independent units and records the actual counts.

## 5. Generate Paper Figures

```bash
python scripts/plot_com_comparison_figures.py \
  --results-root /path/to/com_comparison_results \
  --output-dir /path/to/com_comparison_results/figures
```

Recommended main figure:

- `figure_1_com_difference_forest.pdf`: mean COM difference (`PCN - FFN`) with
  95% bootstrap confidence intervals.

Supporting figures:

- `figure_2_seed_level_com_distributions.pdf`
- `figure_3_bootstrap_difference_distributions.pdf`
- `figure_4_com_difference_heatmap.pdf`

## 6. Reproducibility Notes

- Keep raw datasets and trained models outside git.
- Record the exact command used for each run; the COM comparison script writes
  `run_config.json` automatically.
- Use the same `--output-root` to reuse cached FFN diagrams. Add
  `--force-ripser` only when intentionally recomputing diagrams.
- Report the actual seed/model counts from `bootstrap_summary.csv` in the
  paper or supplement when they differ across architectures.
- Environment details are intentionally described at the dependency/interface
  level rather than pinned globally, because local JAX/PyTorch installations are
  platform-specific.
