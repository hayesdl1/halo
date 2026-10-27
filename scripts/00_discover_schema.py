"""M1: discover the WRDS schema. Run on your own machine; it needs your WRDS login.

    export WRDS_USERNAME=<your username>
    python scripts/00_discover_schema.py

Writes docs/schema_discovery.md and config/schema_map.yaml (names, types and
aggregates only, no data rows) and prints both. Paste the printed output back, or
commit the two files. Exits with status 1 if a needed table is missing or a critical
check is refuted. Expect a few minutes; most of it is aggregates over the monthly
stock file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from halo import REPO_ROOT
from halo.discovery import run_discovery
from halo.wrds_io import connect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefer-library", default=None,
                        help="try this library first when several match (e.g. crsp_a_stock62)")
    parser.add_argument("--doc", type=Path, default=REPO_ROOT / "docs" / "schema_discovery.md")
    parser.add_argument("--map", type=Path, default=REPO_ROOT / "config" / "schema_map.yaml")
    args = parser.parse_args()

    start = time.time()
    db = connect()
    try:
        d = run_discovery(db, args.doc, args.map, prefer_library=args.prefer_library)
    finally:
        db.close()

    print("\n" + "=" * 25 + " PASTE EVERYTHING BELOW THIS LINE " + "=" * 25)
    print(args.doc.read_text())
    print("---- config/schema_map.yaml ----")
    print(args.map.read_text())
    print(f"Finished in {time.time() - start:.0f}s. Wrote {args.doc} and {args.map}.")
    if d.failures:
        print("\nDISCOVERY FAILED:")
        for f in d.failures:
            print(f"  - {f}")
        return 1
    print("Discovery passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
