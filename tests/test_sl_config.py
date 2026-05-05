"""Validation tests for SLConfig.__post_init__.

Regression for keisei-678359b7aa: SLConfig accepted negative lambda_policy /
lambda_value / lambda_score, which silently inverts gradient descent into
gradient ascent for that loss component while per-head metrics still look
positive.
"""

from __future__ import annotations

import math

import pytest

from keisei.sl.trainer import SLConfig


def _config(**overrides: object) -> SLConfig:
    base = dict(data_dir="/tmp/unused")  # noqa: S108 — config validation only
    base.update(overrides)
    return SLConfig(**base)  # type: ignore[arg-type]


class TestSLConfigLambdaValidation:
    def test_defaults_accepted(self) -> None:
        cfg = _config()
        assert cfg.lambda_policy == 1.0
        assert cfg.lambda_value == 1.5
        assert cfg.lambda_score == 0.02

    @pytest.mark.parametrize("field", ["lambda_policy", "lambda_value", "lambda_score"])
    def test_zero_is_legitimate(self, field: str) -> None:
        """Zero disables a head — must remain valid for ablations."""
        cfg = _config(**{field: 0.0})
        assert getattr(cfg, field) == 0.0

    @pytest.mark.parametrize("field", ["lambda_policy", "lambda_value", "lambda_score"])
    def test_negative_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be >= 0"):
            _config(**{field: -1.0})

    @pytest.mark.parametrize("field", ["lambda_policy", "lambda_value", "lambda_score"])
    def test_small_negative_rejected(self, field: str) -> None:
        """Even tiny negatives flip optimisation direction — reject."""
        with pytest.raises(ValueError, match=f"{field} must be >= 0"):
            _config(**{field: -1e-9})

    @pytest.mark.parametrize("field", ["lambda_policy", "lambda_value", "lambda_score"])
    def test_nan_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be finite"):
            _config(**{field: math.nan})

    @pytest.mark.parametrize("field", ["lambda_policy", "lambda_value", "lambda_score"])
    def test_positive_inf_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be finite"):
            _config(**{field: math.inf})

    @pytest.mark.parametrize("field", ["lambda_policy", "lambda_value", "lambda_score"])
    def test_negative_inf_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be finite"):
            _config(**{field: -math.inf})


class TestSLConfigPreexistingValidation:
    """Pre-existing checks must continue to fire — guards against regression in __post_init__ ordering."""

    def test_grad_clip_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="grad_clip must be > 0"):
            _config(grad_clip=0.0)

    def test_negative_total_epochs_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_epochs must be >= 0"):
            _config(total_epochs=-1)

    def test_zero_batch_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            _config(batch_size=0)

    def test_zero_learning_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            _config(learning_rate=0.0)

    def test_negative_num_workers_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_workers must be >= 0"):
            _config(num_workers=-1)
