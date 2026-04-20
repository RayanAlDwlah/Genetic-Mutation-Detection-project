"""Unit tests for `src.utils` — small helpers used everywhere."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.utils import (
    load_yaml_config,
    normalize_chromosome,
    require_file,
    resolve_path,
)


class TestResolvePath:
    def test_absolute_path_passes_through(self, tmp_path: Path) -> None:
        assert resolve_path(tmp_path, "/already/absolute") == Path("/already/absolute")

    def test_relative_resolved_against_root(self, tmp_path: Path) -> None:
        assert resolve_path(tmp_path, "data/splits/train.parquet") == (
            tmp_path / "data/splits/train.parquet"
        )


class TestNormalizeChromosome:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", "1"),
            ("chr1", "1"),
            ("CHR17", "17"),
            ("chrX", "X"),
            ("Y", "Y"),
            ("23", "X"),  # Ensembl legacy encoding
            ("24", "Y"),
        ],
    )
    def test_valid(self, raw: str, expected: str) -> None:
        assert normalize_chromosome(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "NA", ".", "-", "null"])
    def test_null_like(self, raw: object) -> None:
        assert normalize_chromosome(raw) is None

    def test_out_of_range_returns_none(self) -> None:
        assert normalize_chromosome("25") is None
        assert normalize_chromosome("0") is None


class TestRequireFile:
    def test_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "present.txt"
        p.write_text("ok")
        assert require_file(p) == p

    def test_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            require_file(tmp_path / "missing.txt")

    def test_missing_with_label(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="config file not found"):
            require_file(tmp_path / "config.yaml", label="config")


class TestLoadYamlConfig:
    def test_valid_mapping(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.yaml"
        p.write_text("seed: 42\nname: test\n")
        assert load_yaml_config(p) == {"seed": 42, "name": "test"}

    def test_empty_yields_empty_mapping(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.yaml"
        p.write_text("")
        assert load_yaml_config(p) == {}

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.yaml"
        p.write_text("- just_a_list\n- item2\n")
        with pytest.raises(ValueError, match="top-level mapping"):
            load_yaml_config(p)
