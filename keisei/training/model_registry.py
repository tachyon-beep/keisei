"""Model registry: architecture name -> (model class, params dataclass)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from keisei.training.models.base import BaseModel
from keisei.training.models.katago_base import KataGoBaseModel
from keisei.training.models.mlp import MLPModel, MLPParams
from keisei.training.models.resnet import ResNetModel, ResNetParams
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
from keisei.training.models.transformer import TransformerModel, TransformerParams

_REGISTRY: dict[str, tuple[type[nn.Module], type]] = {
    "resnet": (ResNetModel, ResNetParams),
    "mlp": (MLPModel, MLPParams),
    "transformer": (TransformerModel, TransformerParams),
    "se_resnet": (SEResNetModel, SEResNetParams),
}

VALID_ARCHITECTURES = set(_REGISTRY.keys())


def validate_model_params(architecture: str, params: dict[str, Any]) -> object:
    """Validate and instantiate params for the given architecture."""
    if architecture not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Valid: {sorted(VALID_ARCHITECTURES)}"
        )
    _, params_cls = _REGISTRY[architecture]
    try:
        return params_cls(**params)
    except TypeError as e:
        raise TypeError(f"Invalid params for '{architecture}': {e}") from e


def build_model(architecture: str, params: dict[str, Any]) -> BaseModel | KataGoBaseModel:
    """Build a model instance for the given architecture and params."""
    validated_params = validate_model_params(architecture, params)
    model_cls, _ = _REGISTRY[architecture]
    return model_cls(validated_params)
