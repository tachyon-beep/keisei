"""Gap-analysis tests for registries: PPOParams validation, Transformer nhead."""

from __future__ import annotations

import pytest

from keisei.training.algorithm_registry import PPOParams, validate_algorithm_params
from keisei.training.model_registry import build_model, validate_model_params


# ===================================================================
# H6 — PPOParams with extra/unknown kwargs and degenerate values
# ===================================================================


class TestPPOParamsValidation:
    """PPOParams is a frozen dataclass — no range validation.
    These tests document the current contract."""

    def test_extra_kwarg_raises_type_error(self) -> None:
        """Unknown parameters should be rejected by the dataclass."""
        with pytest.raises(TypeError, match="Invalid params"):
            validate_algorithm_params("ppo", {"typo_param": 0.5})

    def test_clip_epsilon_zero_accepted(self) -> None:
        """clip_epsilon=0 makes every ratio clamp to [1,1], effectively killing
        the policy gradient.  Document that the dataclass accepts this."""
        params = validate_algorithm_params("ppo", {"clip_epsilon": 0.0})
        assert isinstance(params, PPOParams)
        assert params.clip_epsilon == 0.0

    def test_clip_epsilon_negative_accepted(self) -> None:
        """Negative clip_epsilon is nonsensical but the dataclass has no guard."""
        params = validate_algorithm_params("ppo", {"clip_epsilon": -0.5})
        assert isinstance(params, PPOParams)
        assert params.clip_epsilon == -0.5

    def test_negative_learning_rate_accepted(self) -> None:
        """Document: no range validation on learning_rate."""
        params = validate_algorithm_params("ppo", {"learning_rate": -1e-3})
        assert isinstance(params, PPOParams)

    def test_valid_params_pass(self) -> None:
        params = validate_algorithm_params("ppo", {
            "learning_rate": 1e-4,
            "gamma": 0.95,
            "clip_epsilon": 0.1,
            "epochs_per_batch": 2,
            "batch_size": 128,
        })
        assert isinstance(params, PPOParams)
        assert params.gamma == 0.95

    def test_default_params(self) -> None:
        """Empty dict should produce defaults."""
        params = validate_algorithm_params("ppo", {})
        assert isinstance(params, PPOParams)
        assert params.learning_rate == 3e-4
        assert params.clip_epsilon == 0.2


# ===================================================================
# M4 — Transformer with incompatible d_model / nhead
# ===================================================================


class TestTransformerNheadValidation:
    """PyTorch requires d_model % nhead == 0.  The registry doesn't pre-check
    this, so the error comes from PyTorch at construction time."""

    def test_incompatible_nhead_raises(self) -> None:
        """d_model=32, nhead=5 → 32 % 5 != 0 → PyTorch raises."""
        with pytest.raises((ValueError, AssertionError)):
            build_model("transformer", {
                "d_model": 32, "nhead": 5, "num_layers": 1,
            })

    def test_compatible_nhead_succeeds(self) -> None:
        """d_model=32, nhead=4 → 32 % 4 == 0 → works."""
        model = build_model("transformer", {
            "d_model": 32, "nhead": 4, "num_layers": 1,
        })
        assert model is not None
