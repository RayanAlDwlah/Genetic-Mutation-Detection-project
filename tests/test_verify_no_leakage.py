"""Wrap `src.verify_no_leakage` as pytest so CI can block on leakage.

The original module is CLI-first (`python -m src.verify_no_leakage`). This
file exposes each of the four checks as a separate `test_*` function so the
failure message in CI pinpoints the exact leakage category rather than a
generic "exit code 1".
"""

from __future__ import annotations

from src.verify_no_leakage import (
    check_feature_hygiene,
    check_label_balance,
    check_missense_filter,
    check_split_disjoint,
)


def _fail_if_errors(errors: list[str]) -> None:
    if errors:
        msg = "\n  - " + "\n  - ".join(errors)
        raise AssertionError(f"leakage check produced errors:{msg}")


def test_no_banned_features_in_training_matrix() -> None:
    """`{is_common, chr, ref, alt}` must NEVER appear in the feature list."""
    _fail_if_errors(check_feature_hygiene())


def test_no_non_missense_rows_in_training_split() -> None:
    """Every training row must have both ref_aa AND alt_aa populated."""
    _fail_if_errors(check_missense_filter())


def test_train_and_test_are_family_disjoint() -> None:
    """Gene-level AND gene-family-level disjointness (paralog-aware)."""
    _fail_if_errors(check_split_disjoint())


def test_label_balance_across_splits() -> None:
    """Pathogenic rate gap between splits must stay ≤ 8pp."""
    _fail_if_errors(check_label_balance())
