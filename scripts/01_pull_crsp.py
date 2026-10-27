"""M2 step 1: pull CRSP monthly, security info and the CCM link. Run on your machine.

    python scripts/01_pull_crsp.py

Reads table and column names from config/schema_map.yaml and settings from
config/config.yaml. Writes parquet files to data/raw/ (gitignored). Prints a summary
and the spec 3.9 checks; exits 1 if any check FAILs. Expect 5 to 15 minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from halo import REPO_ROOT
from halo.checks import failures
from halo.config import load_config
from halo.pull import today
from halo.pull_steps import step_crsp
from halo.schema import load_schema_map
from halo.wrds_io import connect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--end", default=today(), help="last date to pull (default today)")
    args = parser.parse_args()
    cfg, smap = load_config(), load_schema_map()
    start = time.time()
    db = connect()
    try:
        report, checks = step_crsp(db, smap, cfg, args.raw_dir, args.end)
    finally:
        db.close()
    print("\n" + "=" * 25 + " PASTE EVERYTHING BELOW THIS LINE " + "=" * 25)
    print(report)
    print(f"\nFinished in {time.time() - start:.0f}s. Files in {args.raw_dir}.")
    fails = failures(checks)
    if fails:
        print("\nCRSP PULL CHECKS FAILED: " + ", ".join(c.id for c in fails))
        return 1
    print("CRSP pull passed. Next: python scripts/02_pull_compustat.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
