"""M3: build the monthly panel from the M2 pulls. No WRDS connection needed.

    python scripts/03_build_panel.py

Reads data/raw/, writes data/processed/panel.parquet (gitignored), and prints the
spec 3.9 panel checks including the Goldman Exhibit 6 sanity check. Exits 1 if any
check FAILs. Expect 1 to 3 minutes and a few GB of memory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from halo import REPO_ROOT
from halo.checks import failures
from halo.config import load_config
from halo.panel import run_build


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    args = parser.parse_args()
    start = time.time()
    report, checks = run_build(args.raw_dir, args.out_dir, load_config())
    print("\n" + "=" * 25 + " PASTE EVERYTHING BELOW THIS LINE " + "=" * 25)
    print(report)
    print(f"\nFinished in {time.time() - start:.0f}s.")
    fails = failures(checks)
    if fails:
        print("\nPANEL CHECKS FAILED: " + ", ".join(c.id for c in fails))
        return 1
    print("Panel build passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
