"""Value-head adapters for dual-contract model support.

Encapsulates loss computation differences between scalar-value models
(BaseModel) and multi-head W/D/L models (KataGoBaseModel) so the
unified training loop never branches on model type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class ValueHeadAdapter(ABC):
    """Interface for value-head loss computation and scalar projection."""

    @abstractmethod
    def scalar_value_from_output(self, value_output: torch.Tensor) -> torch.Tensor:
        """Project model's value output to a scalar (batch,) for GAE."""
        ...

    @abstractmethod
    def compute_value_loss(
        self,
        value_output: torch.Tensor,
        returns: torch.Tensor | None,
        value_cats: torch.Tensor | None,
        score_targets: torch.Tensor | None,
        score_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute value loss appropriate for the model contract."""
        ...


class ScalarValueAdapter(ValueHeadAdapter):
    """For BaseModel: tanh-activated scalar value, MSE loss vs returns."""

    def scalar_value_from_output(self, value_output: torch.Tensor) -> torch.Tensor:
        return value_output.squeeze(-1)

    def compute_value_loss(
        self,
        value_output: torch.Tensor,
        returns: torch.Tensor | None,
        value_cats: torch.Tensor | None = None,
        score_targets: torch.Tensor | None = None,
        score_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if returns is None:
            raise ValueError("ScalarValueAdapter requires returns")
        return F.mse_loss(value_output.squeeze(-1), returns)


class MultiHeadValueAdapter(ValueHeadAdapter):
    """For KataGoBaseModel: W/D/L cross-entropy + score MSE."""

    def __init__(self, lambda_value: float = 1.5, lambda_score: float = 0.02) -> None:
        self.lambda_value = lambda_value
        self.lambda_score = lambda_score

    def scalar_value_from_output(self, value_output: torch.Tensor) -> torch.Tensor:
        # value_output is value_logits (batch, 3) — W/D/L
        value_probs = F.softmax(value_output, dim=-1)
        return value_probs[:, 0] - value_probs[:, 2]  # P(W) - P(L)

    def compute_value_loss(
        self,
        value_output: torch.Tensor,
        returns: torch.Tensor | None = None,
        value_cats: torch.Tensor | None = None,
        score_targets: torch.Tensor | None = None,
        score_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if value_cats is None:
            raise ValueError("MultiHeadValueAdapter requires value_cats")
        if score_targets is None:
            raise ValueError("MultiHeadValueAdapter requires score_targets")
        if score_pred is None:
            raise ValueError("MultiHeadValueAdapter requires score_pred")

        # Guard: when all value_cats are -1 (all non-terminal), cross_entropy
        # returns NaN. Use graph-connected zero to preserve backward() graph.
        has_valid = (value_cats >= 0).any()
        if has_valid:
            value_loss = F.cross_entropy(value_output, value_cats, ignore_index=-1)
        else:
            value_loss = value_output.sum() * 0.0

        # Guard: score_targets uses NaN sentinel for non-terminal positions.
        # Filter before MSE to avoid NaN propagation.
        score_valid = ~score_targets.isnan()
        if score_valid.any():
            score_loss = F.mse_loss(
                score_pred.squeeze(-1)[score_valid], score_targets[score_valid],
            )
        else:
            score_loss = score_pred.sum() * 0.0

        return self.lambda_value * value_loss + self.lambda_score * score_loss


def get_value_adapter(
    model_contract: str,
    lambda_value: float = 1.5,
    lambda_score: float = 0.02,
) -> ValueHeadAdapter:
    """Return the appropriate adapter for a model contract type."""
    if model_contract == "scalar":
        return ScalarValueAdapter()
    elif model_contract == "multi_head":
        return MultiHeadValueAdapter(lambda_value=lambda_value, lambda_score=lambda_score)
    else:
        raise ValueError(f"Unknown model contract: {model_contract}")
