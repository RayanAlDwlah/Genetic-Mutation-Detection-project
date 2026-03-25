"""Shared output helpers for pipeline scripts."""

from __future__ import annotations

from typing import Any


def echo(*_args: Any, **_kwargs: Any) -> None:
    """No-op output helper; notebooks are the source of presentation output."""
    return None
