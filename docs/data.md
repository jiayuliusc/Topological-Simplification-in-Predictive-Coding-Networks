# Data and Artifact Policy

The public repository should contain source code, lightweight configuration
templates, and documentation. Large generated artifacts should live outside git.

## Keep Out of Git

- Raw datasets and derived `.npz` files.
- Trained PCN and FFN models.
- Activation dumps.
- Ripser outputs and diagram caches.
- Bootstrap trial CSVs and generated figures.
- Machine-local configuration files.

The `.gitignore` file already excludes the common output locations and binary
artifact extensions used by the project.

## Recommended Local Layout

```text
project-root/
  data/
    MNIST/
    D1/
  results/
    D1/
    MNIST/
  com_comparison_results/
    seed_level_com.csv
    bootstrap_summary.csv
    bootstrap_trials.csv
    figures/
```

For clusters or shared filesystems, the same layout can be located elsewhere.
Point scripts to those locations with command-line arguments or
`config/local_config.json`.

## Sharing Data for Publication

For a public paper release, use a data archive such as Zenodo, OSF, or an
institutional repository for:

- Synthetic datasets needed to reproduce the D1 experiments.
- Trained model ensembles if retraining is expensive.
- Precomputed persistence diagrams if recomputing Ripser is expensive.

Add archive links and checksums here once they are available.
