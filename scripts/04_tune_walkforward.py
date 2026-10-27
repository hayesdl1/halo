"""M5: tune the blend of the six metrics with an expanding-window walk-forward.

    python scripts/04_tune_walkforward.py

Reads data/processed/panel.parquet (from scripts/03_build_panel.py). Never touches
months on or after holdout.start in config/config.yaml. Writes results/weights_path.csv,
oos_returns.csv, oos_summary.csv and appends to trials.csv (results/ is gitignored).
Expect 2 to 5 minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from halo import REPO_ROOT
from halo.config import load_config
from halo.tuning import run_tuning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "panel.parquet")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    if not args.panel.exists():
        sys.exit(f"{args.panel} not found. Run scripts/03_build_panel.py first.")
    start = time.time()
    report = run_tuning(args.panel, args.results, load_config())
    print("\n" + "=" * 25 + " PASTE EVERYTHING BELOW THIS LINE " + "=" * 25)
    print(report)
    print(f"\nFinished in {time.time() - start:.0f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
