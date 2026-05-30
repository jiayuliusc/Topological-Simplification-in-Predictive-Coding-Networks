import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_ffn_pcn_com.py"
SPEC = importlib.util.spec_from_file_location("compare_ffn_pcn_com", SCRIPT_PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def test_compute_com_from_layerwise_diagrams():
    diagrams = [
        [np.array([[0.0, 3.0], [0.0, 3.0], [0.0, np.inf]])],
        [np.array([[0.0, 3.0], [0.0, np.inf]])],
        [np.array([[0.0, np.inf]])],
    ]

    com = compare.compute_com_from_diagrams(diagrams, eta=2.5, dims=(0,))

    assert com == 1.5


def test_normalize_pcn_single_dim_layer_format():
    diagrams = [
        [np.array([[0.0, 1.0], [0.0, np.inf]])],
        [np.array([[0.0, np.inf]])],
    ]

    normalized = compare.normalize_diagram_object(diagrams)

    assert len(normalized) == 2
    assert normalized[0][0].shape == (2, 2)


def test_bootstrap_compare_allows_unequal_sample_counts():
    summary, trials = compare.bootstrap_compare(
        [1.0, 2.0, 3.0],
        [5.0, 6.0],
        B=100,
        seed=0,
    )

    assert summary["ffn_bootstrap_sample_size"] == 3
    assert summary["pcn_bootstrap_sample_size"] == 2
    assert summary["fraction_pcn_gt_ffn"] == 1.0
    assert len(trials) == 100
