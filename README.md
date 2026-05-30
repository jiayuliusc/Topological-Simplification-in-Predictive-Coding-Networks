# Topological Simplification in Predictive Coding Networks

This repository contains code for studying how predictive coding networks
(PCNs) and feedforward neural networks (FFNs) transform representation topology
across layers. The experiments train model ensembles, extract layer
representations, compute persistent homology on k-nearest-neighbor graph
metrics, and compare center-of-mass (COM) summaries across architectures.

The repository is being prepared as the public code companion for the paper
*Topological Simplification in Predictive Coding Networks*.

## Repository Layout

- `src/topological_dl/`: reusable configuration, training, topology, and
  reconstruction utilities.
- `scripts/`: command-line workflows for training, persistent homology, COM
  comparison, and figure generation. See [scripts/README.md](scripts/README.md).
- `notebooks/`: archived exploratory notebooks grouped by experiment family.
  Scripts are preferred for reproducible runs. See
  [notebooks/README.md](notebooks/README.md).
- `config/local_config.example.json`: template for local paths and optional
  external dependencies.
- `docs/`: workflow notes, data layout, and public-release checklist.

Large generated files are intentionally ignored by git. Keep datasets, trained
models, Ripser outputs, and generated figures under local `data/`, `results/`,
or user-specified output folders.

## Environment

This repository is research code. It documents the core scripts and expected
file layouts, but it does not try to provide a universal environment lock file
for every machine or cluster. Use Python 3.10 or newer, then adapt the
environment with your preferred tooling. The optional extras in `pyproject.toml`
are intended as starting points:

```bash
git clone https://github.com/jiayuliusc/Topological-Simplification-in-Predictive-Coding-Networks.git
cd Topological-Simplification-in-Predictive-Coding-Networks
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis]"  # COM comparison and figure scripts
```

Other useful extras:

```bash
python -m pip install -e ".[mnist]"  # MNIST kNN / persistent-homology scripts
python -m pip install -e ".[torch]"  # simple FFN baseline scripts
python -m pip install -e ".[pcn]"    # full PCN training utilities
python -m pip install -e ".[dev]"    # tests and linting
```

Some PCN training workflows require `pcx` or `pcx2`. If these packages are not
installed in your environment, point the repository to local checkouts with the
configuration file or environment variables below.

## Configuration

Copy the example config and edit paths for your machine:

```bash
cp config/local_config.example.json config/local_config.json
```

The same values can also be supplied with environment variables:

- `TDL_ROOT_DIR`
- `TDL_DATA_DIR`
- `TDL_RESULTS_DIR`
- `TDL_PCX_DIR`
- `TDL_PCX2_DIR`
- `TDL_RIPSER_PLUSPLUS_DIR`
- `TDL_USE_PCX2`

The trainer does not change directories, load cluster modules, or assume a
specific HPC filesystem path at import time.

## Typical PCN Usage

```python
import jax
from topological_dl.trainer import Trainer

trainer = Trainer(
    dataset="MNIST",
    hidden_dims=[256] * 8,
    act_fn=jax.nn.relu,
    study_name="256x8_relu",
)
```

For scripts run directly from the repository root, the existing scripts add
`src/` to `sys.path`. Installing the package in editable mode is convenient but
not required for reading the core implementation.

## FFN vs PCN COM Comparison

Use `scripts/compare_ffn_pcn_com.py` to compute Ripser diagrams from saved FFN
activations and compare bootstrapped mean COM against PCN results.

Example with FFN activations and PCN `Trainer`-style Ripser output. Replace the
three path arguments with locations on your machine or cluster:

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

By default, `leaky_relu` maps to PCN folders named `*_leaky`, and the script
uses the first 30 `model_*.dill` files sorted by numeric model id. Add
`--pcn-model-selection range` only if every PCN folder has exactly
`model_0.dill` through `model_29.dill`.

If one architecture/activation pair has fewer than 30 valid FFN or PCN COM
values, the script bootstraps the available valid values for that group and
records the actual sample sizes in the summary. Add `--on-incomplete-group skip`
to skip incomplete groups, or `--on-incomplete-group error` to stop immediately.

## Paper Figures

After running the COM comparison, generate publication-ready PDF/SVG/PNG
figures:

```bash
python scripts/plot_com_comparison_figures.py \
  --results-root /path/to/com_comparison_results \
  --output-dir /path/to/com_comparison_results/figures
```

The main recommended panel is
`figure_1_com_difference_forest.pdf`, a forest plot of bootstrap mean
differences (`PCN - FFN`) with 95% confidence intervals.

## Core Workflow Notes

See [docs/replication.md](docs/replication.md) for the expected result
directory layout, the COM comparison workflow, and practical notes for adapting
the analysis on a local machine or cluster. See
[docs/methods_to_code.md](docs/methods_to_code.md) for a paper-methods to
script map, and [docs/data.md](docs/data.md) for the artifact policy.

## Citation

Preliminary citation metadata is available in [CITATION.cff](CITATION.cff).
Update the author list, DOI/arXiv identifier, and release date before the
public release.

## License

Add a license before public release. Until a license is added, the code is not
formally licensed for reuse.
