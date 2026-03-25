"""Shared utility helpers for pipeline scripts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def resolve_path(repo_root: Path, path_str: str | Path) -> Path:
    """Resolve absolute/relative paths against the repository root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def normalize_chromosome(value: Any) -> str | None:
    """Normalize chromosome labels to 1-22/X/Y style without a chr prefix."""
    if value is None:
        return None

    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "NULL", "<NA>", ".", "-"}:
        return None

    if text.lower().startswith("chr"):
        text = text[3:]

    text = text.upper()
    if text == "23":
        return "X"
    if text == "24":
        return "Y"
    if text.isdigit():
        return str(int(text))
    if text in {"X", "Y"}:
        return text
    return text


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file and require a top-level mapping."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit("Missing dependency: pyyaml. Install requirements first.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a top-level mapping")
    return data


def require_file(path: Path, label: str = "") -> Path:
    """Raise a FileNotFoundError with an optional label for clarity."""
    if not path.exists():
        if label:
            raise FileNotFoundError(f"{label.strip()} file not found: {path}")
        raise FileNotFoundError(f"Required file not found: {path}")
    return path
