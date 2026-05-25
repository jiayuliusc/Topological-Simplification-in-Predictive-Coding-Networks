# Topological Simplification in Predictive Coding Networks

This repository contains experiments for studying how predictive coding networks
(PCNs) transform representation topology across layers.

The `topological_simplification_in__predictive_coding_network`,
extends the topological simplification analysis of Naitzat et al. to PCNs. The
main workflow trains PCNs on a synthetic two-class manifold and MNIST, extracts
layer activations, computes persistent homology/Betti numbers, and compares
when architectures simplify topology. The paper's core findings are that
smaller PCNs tend to collapse connected components earlier than larger PCNs, and
that stronger topological simplification correlates with worse generalized
reconstruction.

## Layout

- `src/topological_dl/`: reusable training, topology, reconstruction, and config code.
- `scripts/`: command-line/notebook-friendly analysis scripts.
- `notebooks/`: archived exploratory notebooks, grouped by experiment family.
- `docs/`: paper and project documentation.
- `config/local_config.example.json`: template for machine-specific paths.

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

The trainer no longer changes directories, loads CARC modules, or assumes the
old cluster project path at import time.

## Typical Usage

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
`src/` to `sys.path`. For new code, prefer installing the package in editable
mode once a full dependency environment is available.
