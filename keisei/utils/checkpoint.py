"""
checkpoint.py: Model checkpoint migration and compatibility utilities for Keisei Shogi.
"""

import logging
from typing import Any, Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)


def load_checkpoint_with_padding(
    model: nn.Module, checkpoint: Dict[str, Any]
) -> None:
    """
    Loads a checkpoint into the model, zero-padding the first conv layer if input channels have increased.
    Args:
        model: The model instance (should have a .stem Conv2d layer).
        checkpoint: The loaded checkpoint dict (from torch.load(...)). May contain a
            'model_state_dict' key wrapping the actual state dict, or be a raw state dict.
    """
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    # Work on a copy to avoid mutating the caller's dict
    state_dict = dict(state_dict)
    model_state = model.state_dict()
    # Handle first conv layer (stem)
    stem_key = None
    for k in model_state:
        if k.endswith("stem.weight"):
            stem_key = k
            break
    if stem_key is not None:
        old_weight = state_dict[stem_key]
        new_weight = model_state[stem_key]
        if old_weight.shape[1] < new_weight.shape[1]:
            # Zero-pad input channels
            pad_channels = new_weight.shape[1] - old_weight.shape[1]
            logger.warning(
                "Padding stem.weight: checkpoint has %d input channels, "
                "model expects %d (adding %d zero-filled channels)",
                old_weight.shape[1],
                new_weight.shape[1],
                pad_channels,
            )
            pad = torch.zeros(
                (
                    old_weight.shape[0],
                    pad_channels,
                    *old_weight.shape[2:],
                ),
                dtype=old_weight.dtype,
                device=old_weight.device,
            )
            padded_weight = torch.cat([old_weight, pad], dim=1)
            state_dict[stem_key] = padded_weight
        elif old_weight.shape[1] > new_weight.shape[1]:
            # Truncate input channels
            logger.warning(
                "Truncating stem.weight: checkpoint has %d input channels, "
                "model expects %d (discarding %d channels)",
                old_weight.shape[1],
                new_weight.shape[1],
                old_weight.shape[1] - new_weight.shape[1],
            )
            state_dict[stem_key] = old_weight[:, : new_weight.shape[1], :, :]
    model.load_state_dict(state_dict, strict=True)
