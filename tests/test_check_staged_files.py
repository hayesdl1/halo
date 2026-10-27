"""The data guard must block parquet files, big files and credentials (spec rule 0.2)."""

import subprocess
import sys

import pytest

from halo import REPO_ROOT

SCRIPT = REPO_ROOT / "tools" / "check_staged_files.py"


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    return tmp_path


def run_staged(repo):
    return subprocess.run([sys.executable, str(SCRIPT), "--staged"], cwd=repo,
                          capture_output=True, text=True)


def test_clean_commit_passes(repo):
    (repo / "a.py").write_text("x = 1\n")
    git(repo, "add", "a.py")
    assert run_staged(repo).returncode == 0


@pytest.mark.parametrize("name", ["panel.parquet", "sub/dir/x.PARQUET", ".pgpass", ".env",
                                  "data/raw/msf.csv", "results/REPORT.md"])
def test_blocked_paths(repo, name):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    git(repo, "add", "-f", name)
    res = run_staged(repo)
    assert res.returncode == 1
    assert "BLOCKED" in res.stdout


def test_large_file_blocked(repo):
    (repo / "big.bin").write_bytes(b"0" * (5 * 1024 * 1024 + 1))
    git(repo, "add", "big.bin")
    res = run_staged(repo)
    assert res.returncode == 1
    assert "5 MB" in res.stdout


def test_file_args_and_stdin(repo):
    ok = subprocess.run([sys.executable, str(SCRIPT), "src/x.py"], cwd=repo)
    assert ok.returncode == 0
    bad = subprocess.run([sys.executable, str(SCRIPT), "--stdin"], cwd=repo,
                         input="README.md\nx.parquet\n", text=True, capture_output=True)
    assert bad.returncode == 1
