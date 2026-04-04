"""Algorithm registry: algorithm name -> params dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from keisei.training.katago_ppo import KataGoPPOParams


@dataclass(frozen=True)
class PPOParams:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5


_PARAM_SCHEMAS: dict[str, type] = {
    "katago_ppo": KataGoPPOParams,
}

VALID_ALGORITHMS = set(_PARAM_SCHEMAS.keys())


def validate_algorithm_params(algorithm: str, params: dict[str, Any]) -> object:
    """Validate and instantiate params for the given algorithm."""
    if algorithm not in _PARAM_SCHEMAS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Valid: {sorted(VALID_ALGORITHMS)}"
        )
    params_cls = _PARAM_SCHEMAS[algorithm]
    try:
        return params_cls(**params)
    except TypeError as e:
        raise TypeError(f"Invalid params for '{algorithm}': {e}") from e
