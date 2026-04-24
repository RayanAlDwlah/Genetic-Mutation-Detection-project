# Added for Phase 2.1 (S13): regression guard for ESM-2 / HEAD split drift.
"""Wrap scripts/verify_esm2_split_integrity.py into a CI test."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/verify_esm2_split_integrity.py"


@pytest.mark.skipif(not SCRIPT.exists(), reason="integrity script missing")
def test_esm2_split_integrity_passes():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO), "PATH": ""},
    )
    assert result.returncode == 0, (
        f"verify_esm2_split_integrity.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
