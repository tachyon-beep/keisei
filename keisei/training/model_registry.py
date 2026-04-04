"""Model registry: architecture name -> spec (model class, params, contract, channels)."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch.nn as nn

from keisei.training.models.base import BaseModel
from keisei.training.models.katago_base import KataGoBaseModel
from keisei.training.models.mlp import MLPModel, MLPParams
from keisei.training.models.resnet import ResNetModel, ResNetParams
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
from keisei.training.models.transformer import TransformerModel, TransformerParams


class ArchitectureSpec(NamedTuple):
    model_cls: type[nn.Module]
    params_cls: type
    contract: str  # "scalar" or "multi_head"
    obs_channels: int


_REGISTRY: dict[str, ArchitectureSpec] = {
    "resnet": ArchitectureSpec(ResNetModel, ResNetParams, "scalar", 50),
    "mlp": ArchitectureSpec(MLPModel, MLPParams, "scalar", 50),
    "transformer": ArchitectureSpec(TransformerModel, TransformerParams, "scalar", 50),
    "se_resnet": ArchitectureSpec(SEResNetModel, SEResNetParams, "multi_head", 50),
}

VALID_ARCHITECTURES = set(_REGISTRY.keys())


def _get_spec(architecture: str) -> ArchitectureSpec:
    if architecture not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Valid: {sorted(VALID_ARCHITECTURES)}"
        )
    return _REGISTRY[architecture]


def validate_model_params(architecture: str, params: dict[str, Any]) -> object:
    """Validate and instantiate params for the given architecture."""
    spec = _get_spec(architecture)
    try:
        validated = spec.params_cls(**params)
    except TypeError as e:
        raise TypeError(f"Invalid params for '{architecture}': {e}") from e

    # Architecture-specific semantic validation
    if architecture == "transformer":
        if validated.nhead <= 0:
            raise ValueError(f"transformer: nhead must be > 0, got {validated.nhead}")
        if validated.d_model <= 0:
            raise ValueError(f"transformer: d_model must be > 0, got {validated.d_model}")
        if validated.d_model % validated.nhead != 0:
            raise ValueError(
                f"transformer: d_model ({validated.d_model}) must be divisible "
                f"by nhead ({validated.nhead})"
            )
    elif architecture == "se_resnet":
        if validated.channels <= 0:
            raise ValueError(f"se_resnet: channels must be > 0, got {validated.channels}")
        if validated.se_reduction <= 0:
            raise ValueError(f"se_resnet: se_reduction must be > 0, got {validated.se_reduction}")
        if validated.channels // validated.se_reduction < 1:
            raise ValueError(
                f"se_resnet: channels ({validated.channels}) // se_reduction "
                f"({validated.se_reduction}) must be >= 1"
            )

    return validated


def build_model(architecture: str, params: dict[str, Any]) -> nn.Module:
    """Build a model instance for the given architecture and params."""
    validated_params = validate_model_params(architecture, params)
    spec = _get_spec(architecture)
    return spec.model_cls(validated_params)


def get_model_contract(architecture: str) -> str:
    """Return the value-head contract type for an architecture."""
    return _get_spec(architecture).contract


def get_obs_channels(architecture: str) -> int:
    """Return the expected observation channels for an architecture."""
    return _get_spec(architecture).obs_channels
