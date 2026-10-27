"""Refuse to commit licensed data or credentials.

Fails if any given path is a parquet file, is larger than 5 MB, sits under data/ or
results/, or looks like a credentials file. Used three ways:

    python tools/check_staged_files.py FILE [FILE ...]   # pre-commit passes staged files
    python tools/check_staged_files.py --staged          # check the git index directly
    git ls-files | python tools/check_staged_files.py --stdin   # CI check of the whole tree
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_BYTES = 5 * 1024 * 1024
BLOCKED_SUFFIXES = {".parquet", ".feather", ".h5", ".pkl"}
BLOCKED_NAMES = {".pgpass", ".env"}
BLOCKED_DIRS = {"data", "results"}


def file_size(path: str) -> int | None:
    """Size of the staged blob if the path is in the index, else the size on disk."""
    try:
        out = subprocess.run(
            ["git", "cat-file", "-s", f":{path}"], capture_output=True, text=True, check=True
        )
        return int(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        p = Path(path)
        return p.stat().st_size if p.is_file() else None


def problems_for(path: str) -> list[str]:
    pp = PurePosixPath(path.replace("\\", "/"))
    found = []
    if pp.suffix.lower() in BLOCKED_SUFFIXES:
        found.append(f"{pp.suffix} files are data and must not be committed")
    if pp.name in BLOCKED_NAMES or pp.name.startswith(".env."):
        found.append("credentials file")
    if pp.parts and pp.parts[0] in BLOCKED_DIRS:
        found.append(f"files under {pp.parts[0]}/ are data or results and are gitignored")
    size = file_size(path)
    if size is not None and size > MAX_BYTES:
        found.append(f"{size / 1024 / 1024:.1f} MB exceeds the 5 MB limit")
    return found


def staged_paths() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    parser.add_argument("--staged", action="store_true", help="check files in the git index")
    parser.add_argument("--stdin", action="store_true", help="read paths from stdin")
    args = parser.parse_args(argv)

    paths = list(args.files)
    if args.staged:
        paths += staged_paths()
    if args.stdin:
        paths += [line.strip() for line in sys.stdin if line.strip()]

    failures = {p: probs for p in paths if (probs := problems_for(p))}
    if failures:
        print("BLOCKED: these files must not be committed (WRDS data is licensed):")
        for p, probs in failures.items():
            print(f"  {p}: {'; '.join(probs)}")
        print("Unstage them with: git restore --staged <file>")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
