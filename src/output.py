"""Shared output helpers for pipeline scripts."""

from __future__ import annotations

from typing import Any


def echo(*args: Any, **kwargs: Any) -> None:
    """Print output consistently from pipeline scripts."""
    print(*args, **kwargs)
