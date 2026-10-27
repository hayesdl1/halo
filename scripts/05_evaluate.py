"""M6: evaluate the strategies on pre-holdout data and write results/REPORT.md.

    python scripts/05_evaluate.py

Run after scripts/03_build_panel.py and scripts/04_tune_walkforward.py. Reads the panel,
the tuning results and data/raw/ (factors, rates, and the raw tables for robustness
rebuilds). Writes one CSV per analysis, charts in results/charts/, and REPORT.md built
from the CSVs. Never touches holdout months. Expect 10 to 20 minutes, mostly the
robustness grid (7 panel rebuilds); --no-robustness skips it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from halo import REPO_ROOT
from halo.config import load_config
from halo.eval_steps import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--panel", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "panel.parquet")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--no-robustness", action="store_true")
    parser.add_argument("--bootstraps", type=int, default=500)
    args = parser.parse_args()
    for need in [args.panel, args.results / "weights_path.csv", args.results / "oos_returns.csv"]:
        if not need.exists():
            sys.exit(f"{need} not found. Run scripts/03_build_panel.py and "
                     "scripts/04_tune_walkforward.py first.")
    start = time.time()
    report = run_evaluation(args.raw_dir, args.panel, args.results, load_config(),
                            n_boot=args.bootstraps, robustness=not args.no_robustness)
    print("\n" + "=" * 25 + " PASTE EVERYTHING BELOW THIS LINE " + "=" * 25)
    print(report.read_text())
    print(f"\nFinished in {time.time() - start:.0f}s. Report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
