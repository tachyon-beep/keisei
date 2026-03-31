"""Algorithm registry: algorithm name -> params dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PPOParams:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256


_PARAM_SCHEMAS: dict[str, type] = {
    "ppo": PPOParams,
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
