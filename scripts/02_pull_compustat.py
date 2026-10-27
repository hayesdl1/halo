"""M2 step 2: pull Compustat, Fama-French factors and FRED rates, then cross-check.

    python scripts/02_pull_compustat.py

Run after scripts/01_pull_crsp.py: the cross-checks (link coverage of the universe,
fundamentals coverage, the lag rule) read the CRSP files it wrote. FRED is public and
downloaded over HTTPS; use --skip-fred if your network blocks it. Writes parquet files
to data/raw/ (gitignored). Exits 1 if any check FAILs. Expect 2 to 5 minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from halo import REPO_ROOT
from halo.checks import failures
from halo.config import load_config
from halo.pull_steps import step_compustat
from halo.schema import load_schema_map
from halo.wrds_io import connect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--skip-fred", action="store_true", help="do not download FRED rates")
    args = parser.parse_args()
    if not (args.raw_dir / "ccm_link.parquet").exists():
        sys.exit(f"No CRSP pull found in {args.raw_dir}. Run scripts/01_pull_crsp.py first.")
    cfg, smap = load_config(), load_schema_map()
    start = time.time()
    db = connect()
    try:
        report, checks = step_compustat(db, smap, cfg, args.raw_dir, skip_fred=args.skip_fred)
    finally:
        db.close()
    print("\n" + "=" * 25 + " PASTE EVERYTHING BELOW THIS LINE " + "=" * 25)
    print(report)
    print(f"\nFinished in {time.time() - start:.0f}s. Files in {args.raw_dir}.")
    fails = failures(checks)
    if fails:
        print("\nCOMPUSTAT PULL CHECKS FAILED: " + ", ".join(c.id for c in fails))
        return 1
    print("Compustat pull passed. M2 is complete once both reports are pasted back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
