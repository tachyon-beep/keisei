"""Gap-analysis tests for registries: PPOParams validation, Transformer nhead."""

from __future__ import annotations

import pytest

from keisei.training.algorithm_registry import validate_algorithm_params
from keisei.training.model_registry import build_model

# ===================================================================
# H6 — PPOParams with extra/unknown kwargs and degenerate values
# ===================================================================


class TestLegacyPPORejected:
    """Legacy 'ppo' algorithm was removed from the registry.

    It passed config validation but crashed at KataGoTrainingLoop init
    with a TypeError (expected KataGoPPOParams, got PPOParams).
    Now it fails early at config validation time.
    """

    def test_ppo_rejected_at_validation(self) -> None:
        """'ppo' should be rejected with a clear error."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            validate_algorithm_params("ppo", {})


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
