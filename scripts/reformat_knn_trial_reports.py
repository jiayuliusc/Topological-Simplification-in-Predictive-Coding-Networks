import argparse
import re
import glob
from io import StringIO
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from topological_dl.config import CONFIG, dataset_results_dir


def reformat_trial_files(
    input_dir=None,
    output_dir=None,
    trial_glob="trial_*_all_classes.csv",
):
    """
    Parses report-style trial files:
      trial_0_all_classes.csv ... trial_29_all_classes.csv

    Writes ONE CSV per class (digit). In each class CSV:
      - each row corresponds to a (k, trial) pair
      - columns are the B0 and B1 values across eta

    Output columns:
      k, trial, B0_<eta1>, B0_<eta2>, ..., B1_<eta1>, B1_<eta2>, ...
    (eta columns are sorted numerically)

    Notes:
      - We do NOT include the digit/class number as a column in the per-class CSV.
      - Sorting is by k first, then trial (so all k=3 rows are together).
    """

    input_dir = Path(input_dir) if input_dir is not None else CONFIG.root_dir / "mnist_class_trials"
    output_dir = Path(output_dir) if output_dir is not None else dataset_results_dir("MNIST") / "knn_trials_by_class"

    output_dir.mkdir(parents=True, exist_ok=True)

    trial_paths = sorted(glob.glob(os.path.join(input_dir, trial_glob)))
    if not trial_paths:
        raise FileNotFoundError(f"No files matched {os.path.join(input_dir, trial_glob)}")

    def _collect_csv_block(lines, start_idx):
        """
        Collect consecutive CSV lines (non-empty) starting at start_idx.
        Stops on blank line or marker lines like '---', '===', or '#'.
        Returns (block_str, next_idx).
        """
        block = []
        i = start_idx
        while i < len(lines):
            s = lines[i].rstrip("\n")
            if s.strip() == "":
                break
            if s.startswith("---") or s.startswith("===") or s.startswith("#"):
                break
            block.append(s)
            i += 1
        return "\n".join(block), i

    # Accumulate long tidy rows per digit: (k, eta, trial, B0, B1)
    per_digit_long = {d: [] for d in range(10)}

    for path in trial_paths:
        with open(path, "r") as f:
            lines = f.readlines()

        # Trial ID (prefer header; fallback to filename)
        trial_id = None
        for ln in lines[:60]:
            m = re.search(r"Trial ID:\s*(\d+)", ln)
            if m:
                trial_id = int(m.group(1))
                break
        if trial_id is None:
            m = re.search(r"trial_(\d+)_all_classes\.csv$", os.path.basename(path))
            if not m:
                raise ValueError(f"Could not determine trial id for file: {path}")
            trial_id = int(m.group(1))

        i = 0
        while i < len(lines):
            ln = lines[i].rstrip("\n")

            if ln.startswith("=== DIGIT") and "ANALYSIS" in ln:
                md = re.search(r"===\s*DIGIT\s+(\d+)\s+ANALYSIS\s*===", ln)
                if not md:
                    i += 1
                    continue
                digit = int(md.group(1))
                i += 1

                # Find B0 header
                while i < len(lines) and "B0 (Components)" not in lines[i]:
                    i += 1
                if i >= len(lines):
                    break
                i += 1
                b0_block, i = _collect_csv_block(lines, i)

                # Skip blank lines
                while i < len(lines) and lines[i].strip() == "":
                    i += 1

                # Find B1 header
                while i < len(lines) and "B1 (Holes)" not in lines[i]:
                    if lines[i].startswith("=== DIGIT"):
                        break
                    i += 1
                if i >= len(lines) or "B1 (Holes)" not in lines[i]:
                    continue
                i += 1
                b1_block, i = _collect_csv_block(lines, i)

                if not b0_block.strip() or not b1_block.strip():
                    continue

                b0_tbl = pd.read_csv(StringIO(b0_block))
                b1_tbl = pd.read_csv(StringIO(b1_block))

                # Ensure first column is named 'k'
                if b0_tbl.columns[0] != "k":
                    b0_tbl = b0_tbl.rename(columns={b0_tbl.columns[0]: "k"})
                if b1_tbl.columns[0] != "k":
                    b1_tbl = b1_tbl.rename(columns={b1_tbl.columns[0]: "k"})

                # Wide -> long
                b0_long = b0_tbl.melt(id_vars=["k"], var_name="eta", value_name="B0")
                b1_long = b1_tbl.melt(id_vars=["k"], var_name="eta", value_name="B1")
                merged = pd.merge(b0_long, b1_long, on=["k", "eta"], how="inner")

                merged["trial"] = trial_id

                # Coerce numeric types and drop malformed rows
                for col in ["k", "eta", "B0", "B1", "trial"]:
                    merged[col] = pd.to_numeric(merged[col], errors="coerce")
                merged = merged.dropna(subset=["k", "eta", "B0", "B1", "trial"])

                merged["k"] = merged["k"].astype(int)
                merged["eta"] = merged["eta"].astype(int)
                merged["B0"] = merged["B0"].astype(int)
                merged["B1"] = merged["B1"].astype(int)
                merged["trial"] = merged["trial"].astype(int)

                per_digit_long[digit].append(merged[["k", "eta", "trial", "B0", "B1"]])

            i += 1

    wrote_any = False

    for digit in range(10):
        if not per_digit_long[digit]:
            continue

        long_df = pd.concat(per_digit_long[digit], ignore_index=True)

        # (k, trial) rows; eta columns for B0 and B1
        b0_wide = long_df.pivot_table(
            index=["k", "trial"], columns="eta", values="B0", aggfunc="first"
        )
        b1_wide = long_df.pivot_table(
            index=["k", "trial"], columns="eta", values="B1", aggfunc="first"
        )

        # Sort eta columns numerically
        b0_wide = b0_wide.reindex(sorted(b0_wide.columns), axis=1)
        b1_wide = b1_wide.reindex(sorted(b1_wide.columns), axis=1)

        # Rename columns to keep eta visible while distinguishing B0 vs B1
        b0_wide.columns = [f"B0_{int(eta)}" for eta in b0_wide.columns]
        b1_wide.columns = [f"B1_{int(eta)}" for eta in b1_wide.columns]

        wide = pd.concat([b0_wide, b1_wide], axis=1).reset_index()

        # Sort so all k=3 together, then k=4, etc. (trial within k)
        wide = wide.sort_values(["k", "trial"]).reset_index(drop=True)

        out_path = output_dir / f"digit_{digit}_wide_trials_by_k.csv"
        wide.to_csv(out_path, index=False)
        wrote_any = True

    if not wrote_any:
        raise ValueError("No digit files were written. Check input format / paths.")

    print(f"Done. Wrote per-class wide files to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reformat report-style MNIST kNN trial outputs into per-digit wide CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=CONFIG.root_dir / "mnist_class_trials")
    parser.add_argument("--output-dir", type=Path, default=dataset_results_dir("MNIST") / "knn_trials_by_class")
    parser.add_argument("--trial-glob", default="trial_*_all_classes.csv")
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    reformat_trial_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        trial_glob=args.trial_glob,
    )
