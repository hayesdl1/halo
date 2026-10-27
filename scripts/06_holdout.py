"""M7: evaluate the sealed holdout. Run ONCE, after M6, and only after agreeing to.

    python scripts/06_holdout.py

Freezes the weights by tuning on every month before holdout.start, writes them to
results/holdout_weights.csv, then evaluates the holdout months and Goldman's regime
window, writes results/holdout_*.csv and charts, adds a holdout section to REPORT.md,
and creates results/holdout_evaluated.lock. A second run refuses unless --force is
given; a forced rerun is flagged in bold in the report. Expect 3 to 6 minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from halo import REPO_ROOT
from halo.config import load_config
from halo.holdout import HoldoutLocked, run_holdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--panel", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "panel.parquet")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--force", action="store_true",
                        help="rerun even though the holdout was already evaluated (recorded)")
    args = parser.parse_args()
    if not (args.results / "perf_summary.csv").exists():
        sys.exit("Run scripts/05_evaluate.py first; the holdout comes last.")
    if args.force:
        print("WARNING: forcing a rerun of the holdout. This is recorded in REPORT.md.")
    start = time.time()
    try:
        report = run_holdout(args.raw_dir, args.panel, args.results, load_config(),
                             force=args.force)
    except HoldoutLocked as exc:
        sys.exit(str(exc))
    text = report.read_text()
    section = text[text.find("## Holdout result"):text.find("## Data checks")]
    print("\n" + "=" * 25 + " PASTE EVERYTHING BELOW THIS LINE " + "=" * 25)
    print(section)
    print(f"Finished in {time.time() - start:.0f}s. Full report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
