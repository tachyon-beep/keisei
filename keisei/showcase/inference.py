"""CPU-only model loading and inference for showcase games."""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import threading

from keisei.training.model_registry import build_model, get_model_contract, get_obs_channels

logger = logging.getLogger(__name__)


def enforce_cpu_only(cpu_threads: int = 2) -> None:
    """Set environment and torch config for CPU-only inference.
    MUST be called before any other torch operations in the process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set — safe to ignore in tests


def load_model_for_showcase(
    checkpoint_path: Path | str,
    architecture: str,
    model_params: dict[str, Any],
) -> nn.Module:
    """Load a model checkpoint for CPU-only showcase inference.
    Raises FileNotFoundError if checkpoint doesn't exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = build_model(architecture, model_params)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if hasattr(model, "configure_amp"):
        model.configure_amp(enabled=False)

    for name, param in model.named_parameters():
        assert param.device == torch.device("cpu"), (
            f"Parameter {name} on {param.device} — GPU leak in showcase"
        )

    return model


def run_inference(
    model: nn.Module,
    obs: np.ndarray,
    architecture: str,
) -> tuple[np.ndarray, float]:
    """Run a single forward pass. Returns (policy_logits, win_probability).

    Handles observation channel mismatch: SpectatorEnv produces 46-channel
    observations but models expect 50 channels. Zero-pads the difference.

    win_probability normalized to [0, 1]:
    - scalar contract: tanh [-1,1] -> [0,1]
    - multi_head contract: softmax(value_logits)[0] = P(win)
    """
    expected_channels = get_obs_channels(architecture)
    actual_channels = obs.shape[0]
    if actual_channels < expected_channels:
        padding = np.zeros((expected_channels - actual_channels, 9, 9), dtype=obs.dtype)
        obs = np.concatenate([obs, padding], axis=0)
    elif actual_channels > expected_channels:
        obs = obs[:expected_channels]

    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()

    with torch.inference_mode():
        output = model(obs_tensor)

    contract = get_model_contract(architecture)

    if contract == "multi_head":
        # NOTE: reshape(-1) flattens (9,9,139) in row-major order.
        # This matches the DefaultActionMapper's flat index convention
        # used by SpectatorEnv. If a spatial action mapper is used instead,
        # the flatten order must be verified against that mapper.
        policy_logits = output.policy_logits.squeeze(0).reshape(-1)
        win_prob = torch.softmax(output.value_logits.squeeze(0), dim=0)[0].item()
    else:
        policy_logits_t, value_tensor = output
        policy_logits = policy_logits_t.squeeze(0)
        win_prob = (value_tensor.squeeze(0).item() + 1.0) / 2.0

    return policy_logits.numpy(), float(win_prob)


class ModelCache:
    """LRU cache for loaded models, keyed on (entry_id, checkpoint_path).

    Thread-safe: all access is guarded by a lock.
    """

    def __init__(self, max_size: int = 2) -> None:
        self._cache: OrderedDict[tuple[str, str], nn.Module] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def get_or_load(
        self,
        entry_id: str,
        checkpoint_path: str,
        architecture: str,
        model_params: dict[str, Any],
    ) -> nn.Module:
        key = (entry_id, checkpoint_path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        # Load outside the lock (slow I/O)
        model = load_model_for_showcase(checkpoint_path, architecture, model_params)

        with self._lock:
            # Check again — another thread may have loaded it
            if key in self._cache:
                return self._cache[key]
            self._cache[key] = model
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Evicted model %s from cache", evicted_key)
        return model
