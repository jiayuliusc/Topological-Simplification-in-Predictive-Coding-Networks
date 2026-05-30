import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_com_comparison_figures.py"
SPEC = importlib.util.spec_from_file_location("plot_com_comparison_figures", SCRIPT_PATH)
plotting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotting)


def test_complete_available_comparisons_backfills_skipped_row():
    summary = pd.DataFrame(
        [
            {
                "arch": "18x8",
                "activation": "tanh",
                "n_ffn": 3,
                "n_pcn": 2,
                "status": "skipped_incomplete",
                "skip_reason": "old skipped row",
                "diff_mean_pcn_minus_ffn": np.nan,
            }
        ]
    )
    seed_level = pd.DataFrame(
        [
            {"network": "ffn", "arch": "18x8", "activation": "tanh", "seed_index": 0, "com": 1.0, "valid_com": True},
            {"network": "ffn", "arch": "18x8", "activation": "tanh", "seed_index": 1, "com": 2.0, "valid_com": True},
            {"network": "ffn", "arch": "18x8", "activation": "tanh", "seed_index": 2, "com": 3.0, "valid_com": True},
            {"network": "pcn", "arch": "18x8", "activation": "tanh", "seed_index": 0, "com": 6.0, "valid_com": True},
            {"network": "pcn", "arch": "18x8", "activation": "tanh", "seed_index": 1, "com": 7.0, "valid_com": True},
        ]
    )

    filled_summary, filled_trials = plotting.complete_available_comparisons(
        summary,
        seed_level,
        pd.DataFrame(),
        n_bootstrap=25,
        bootstrap_seed=0,
    )

    row = filled_summary.iloc[0]
    assert row["status"] == "ok_unequal_n"
    assert row["n_ffn"] == 3
    assert row["n_pcn"] == 2
    assert row["decision"] == "pcn_larger"
    assert len(filled_trials) == 25
