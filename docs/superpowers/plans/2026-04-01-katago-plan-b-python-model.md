# KataGo Plan B: Python SE-ResNet Model & Multi-Head PPO

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the SE-ResNet model with KataGo-style global pooling bias and a multi-head PPO algorithm supporting W/D/L value + score lead.

**Architecture:** New `KataGoBaseModel` ABC and `SEResNetModel` implementation alongside existing models. New `KataGoPPOAlgorithm` with `KataGoRolloutBuffer`. All new code — no modifications to existing model/PPO classes.

**Tech Stack:** Python 3.13, PyTorch, dataclasses. Tests via `uv run pytest`.

**Spec reference:** `docs/superpowers/specs/2026-04-01-katago-se-resnet-design.md` — Slices 3 & 4.

**Dependency:** Can be built in parallel with Plan A. Does NOT require Rust changes — uses synthetic tensors for testing.

**Verified API:** `compute_gae(rewards, values, dones, next_value, gamma, lam)` at `keisei/training/ppo.py:12` — signature matches the plan's call in Task 8. Takes per-env 1D tensors, returns 1D advantages tensor.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `keisei/training/models/katago_base.py` | `KataGoOutput` dataclass + `KataGoBaseModel` ABC |
| Create | `keisei/training/models/se_resnet.py` | `SEResNetParams`, `GlobalPoolBiasBlock`, `SEResNetModel` |
| Create | `keisei/training/katago_ppo.py` | `KataGoPPOParams`, `KataGoRolloutBuffer`, `KataGoPPOAlgorithm` |
| Modify | `keisei/training/model_registry.py` | Register `"se_resnet"` architecture |
| Modify | `keisei/training/algorithm_registry.py` | Register `"katago_ppo"` algorithm |
| Create | `tests/test_katago_model.py` | Model unit tests |
| Create | `tests/test_katago_ppo.py` | PPO unit tests |

---

### Task 1: KataGoBaseModel and KataGoOutput

**Files:**
- Create: `keisei/training/models/katago_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_katago_model.py`:

```python
# tests/test_katago_model.py
"""Tests for the KataGo model architecture."""

import torch
import torch.nn.functional as F
import pytest

from keisei.training.models.katago_base import KataGoBaseModel, KataGoOutput


def test_katago_output_fields():
    """KataGoOutput should have policy_logits, value_logits, score_lead."""
    output = KataGoOutput(
        policy_logits=torch.randn(2, 9, 9, 139),
        value_logits=torch.randn(2, 3),
        score_lead=torch.randn(2, 1),
    )
    assert output.policy_logits.shape == (2, 9, 9, 139)
    assert output.value_logits.shape == (2, 3)
    assert output.score_lead.shape == (2, 1)


def test_katago_base_model_is_abstract():
    """KataGoBaseModel should not be instantiable directly."""
    with pytest.raises(TypeError):
        KataGoBaseModel()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_model.py::test_katago_output_fields -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'keisei.training.models.katago_base'`

- [ ] **Step 3: Write the implementation**

```python
# keisei/training/models/katago_base.py
"""Abstract base model for KataGo-style multi-head architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class KataGoOutput:
    """Output container for KataGo-style models.

    Fields:
        policy_logits: (batch, 9, 9, 139) — spatial, raw, unmasked
        value_logits:  (batch, 3) — [W, D, L] logits (pre-softmax)
        score_lead:    (batch, 1) — predicted point advantage
    """

    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    score_lead: torch.Tensor


class KataGoBaseModel(ABC, nn.Module):
    """Abstract base for KataGo-style multi-head architectures.

    Contract:
    - Input: observation tensor (batch, obs_channels, 9, 9)
    - Output: KataGoOutput
    """

    BOARD_SIZE = 9
    SPATIAL_MOVE_TYPES = 139
    SPATIAL_ACTION_SPACE = 81 * 139  # 11,259

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> KataGoOutput: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_katago_model.py -v -k "output_fields or is_abstract"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/models/katago_base.py tests/test_katago_model.py
git commit -m "feat: add KataGoBaseModel ABC and KataGoOutput dataclass"
```

---

### Task 2: SE-ResNet — GlobalPoolBiasBlock

**Files:**
- Create: `keisei/training/models/se_resnet.py`
- Modify: `tests/test_katago_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_model.py`:

```python
from keisei.training.models.se_resnet import SEResNetParams, GlobalPoolBiasBlock


class TestGlobalPoolBiasBlock:
    def test_output_shape(self):
        block = GlobalPoolBiasBlock(channels=64, se_reduction=16, global_pool_channels=32)
        x = torch.randn(4, 64, 9, 9)
        out = block(x)
        assert out.shape == (4, 64, 9, 9)

    def test_residual_connection(self):
        """Output should differ from input (block is not identity)."""
        block = GlobalPoolBiasBlock(channels=64, se_reduction=16, global_pool_channels=32)
        x = torch.randn(4, 64, 9, 9)
        out = block(x)
        assert not torch.allclose(out, x), "Block should not be identity"

    def test_gradient_flows(self):
        block = GlobalPoolBiasBlock(channels=64, se_reduction=16, global_pool_channels=32)
        x = torch.randn(4, 64, 9, 9, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_model.py::TestGlobalPoolBiasBlock -v`
Expected: FAIL — import error

- [ ] **Step 3: Write the implementation**

```python
# keisei/training/models/se_resnet.py
"""SE-ResNet architecture with KataGo-style global pooling bias."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .katago_base import KataGoBaseModel, KataGoOutput


@dataclass(frozen=True)
class SEResNetParams:
    num_blocks: int = 40
    channels: int = 256
    se_reduction: int = 16
    global_pool_channels: int = 128
    policy_channels: int = 32
    value_fc_size: int = 256
    score_fc_size: int = 128
    obs_channels: int = 50


class GlobalPoolBiasBlock(nn.Module):
    """SE-ResBlock with global pooling bias (KataGo-style).

    Architecture:
        conv1 -> BN -> ReLU -> add global_pool_bias(block_input) -> conv2 -> BN
        -> SE(scale + shift) -> residual add -> ReLU
    """

    def __init__(self, channels: int, se_reduction: int, global_pool_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Global pooling bias: pools from block INPUT (mean + max + std -> bottleneck -> channels)
        # global_pool_channels controls the bottleneck width (spec: configurable, default 128)
        self.global_fc = nn.Sequential(
            nn.Linear(channels * 3, global_pool_channels),
            nn.ReLU(),
            nn.Linear(global_pool_channels, channels),
        )

        # SE: squeeze-and-excitation with scale + shift
        se_hidden = channels // se_reduction
        self.se_fc1 = nn.Linear(channels, se_hidden)
        self.se_fc2 = nn.Linear(se_hidden, channels * 2)  # scale + shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))

        # Global pool bias from block INPUT x (not post-conv1 out)
        g_mean = x.mean(dim=(-2, -1))                  # (B, C)
        g_max = x.amax(dim=(-2, -1))                    # (B, C)
        g_std = x.std(dim=(-2, -1), correction=0)  # population std          # (B, C) — population std (divides by N, not N-1)
        g = self.global_fc(torch.cat([g_mean, g_max, g_std], dim=-1))  # (B, C)
        out = out + g.unsqueeze(-1).unsqueeze(-1)        # broadcast over 9x9

        out = self.bn2(self.conv2(out))

        # SE attention: pool post-conv2 output
        se_input = out.mean(dim=(-2, -1))                # (B, C)
        se = F.relu(self.se_fc1(se_input))
        se = self.se_fc2(se)                              # (B, 2C)
        scale, shift = se.chunk(2, dim=-1)                # each (B, C)
        out = out * torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1) + \
              shift.unsqueeze(-1).unsqueeze(-1)

        return F.relu(out + residual)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_model.py::TestGlobalPoolBiasBlock -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/models/se_resnet.py tests/test_katago_model.py
git commit -m "feat: add GlobalPoolBiasBlock with SE and global pool bias"
```

---

### Task 3: SE-ResNet — Full Model with Three Heads

**Files:**
- Modify: `keisei/training/models/se_resnet.py`
- Modify: `tests/test_katago_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_model.py`:

```python
from keisei.training.models.se_resnet import SEResNetModel


class TestSEResNetModel:
    @pytest.fixture
    def model(self):
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        return SEResNetModel(params)

    def test_output_types(self, model):
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        assert isinstance(output, KataGoOutput)

    def test_policy_shape(self, model):
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        assert output.policy_logits.shape == (4, 9, 9, 139)

    def test_value_shape(self, model):
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        assert output.value_logits.shape == (4, 3)

    def test_score_shape(self, model):
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        assert output.score_lead.shape == (4, 1)

    def test_value_logits_are_raw(self, model):
        """Value logits should be raw (not softmaxed). Deterministic check."""
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        # Softmax outputs always sum to exactly 1.0 per sample.
        # Raw linear outputs have no such constraint — assert they DON'T sum to 1.
        row_sums = output.value_logits.sum(dim=-1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums)), \
            "Value logits should be raw, not already a probability distribution"

    def test_gradient_through_all_heads(self, model):
        obs = torch.randn(4, 50, 9, 9, requires_grad=True)
        output = model(obs)
        loss = (
            output.policy_logits.sum()
            + output.value_logits.sum()
            + output.score_lead.sum()
        )
        loss.backward()
        assert obs.grad is not None
        assert obs.grad.abs().sum() > 0

    def test_wrong_input_channels_raises(self, model):
        obs = torch.randn(4, 46, 9, 9)  # wrong: 46 instead of 50
        with pytest.raises(ValueError, match="Expected 50 input channels"):
            model(obs)

    def test_batch_size_1(self, model):
        # BatchNorm2d with a single sample in training mode produces undefined
        # variance. Must use eval() mode for batch_size=1.
        model.eval()
        obs = torch.randn(1, 50, 9, 9)
        output = model(obs)
        assert output.policy_logits.shape == (1, 9, 9, 139)
        assert output.value_logits.shape == (1, 3)
        assert output.score_lead.shape == (1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_model.py::TestSEResNetModel -v`
Expected: FAIL — `SEResNetModel` not found

- [ ] **Step 3: Write the implementation**

Add to `keisei/training/models/se_resnet.py`:

```python
def _global_pool(x: torch.Tensor) -> torch.Tensor:
    """Global pool: mean + max + std concatenated. Input (B, C, H, W) -> (B, 3C)."""
    g_mean = x.mean(dim=(-2, -1))
    g_max = x.amax(dim=(-2, -1))
    g_std = x.std(dim=(-2, -1), correction=0)  # population std
    return torch.cat([g_mean, g_max, g_std], dim=-1)


class SEResNetModel(KataGoBaseModel):
    """SE-ResNet with global pooling bias, 3-head output."""

    def __init__(self, params: SEResNetParams) -> None:
        super().__init__()
        self.params = params
        ch = params.channels

        # Input conv
        self.input_conv = nn.Conv2d(params.obs_channels, ch, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(ch)

        # Residual tower
        self.blocks = nn.Sequential(*[
            GlobalPoolBiasBlock(ch, params.se_reduction, params.global_pool_channels)
            for _ in range(params.num_blocks)
        ])

        # Policy head: two conv layers -> (B, 139, 9, 9) -> permute to (B, 9, 9, 139)
        self.policy_conv1 = nn.Conv2d(ch, params.policy_channels, 1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(params.policy_channels)
        self.policy_conv2 = nn.Conv2d(params.policy_channels, self.SPATIAL_MOVE_TYPES, 1)

        # Value head: global pool -> FC -> 3 logits (W/D/L)
        self.value_fc1 = nn.Linear(ch * 3, params.value_fc_size)
        self.value_fc2 = nn.Linear(params.value_fc_size, 3)

        # Score head: global pool -> FC -> 1 scalar
        self.score_fc1 = nn.Linear(ch * 3, params.score_fc_size)
        self.score_fc2 = nn.Linear(params.score_fc_size, 1)

    def forward(self, obs: torch.Tensor) -> KataGoOutput:
        # Always check — one integer comparison is negligible vs a forward pass.
        # Use raise, not assert, so the check survives Python -O mode.
        if obs.shape[1] != self.params.obs_channels:
            raise ValueError(
                f"Expected {self.params.obs_channels} input channels, "
                f"got {obs.shape[1]}"
            )

        # Trunk
        x = F.relu(self.input_bn(self.input_conv(obs)))
        x = self.blocks(x)

        # Policy head
        p = F.relu(self.policy_bn1(self.policy_conv1(x)))
        p = self.policy_conv2(p)                    # (B, 139, 9, 9)
        p = p.permute(0, 2, 3, 1)                   # (B, 9, 9, 139)

        # Global pool shared by value and score heads (computed once)
        pool = _global_pool(x)                       # (B, 3C)

        # Value head
        v = F.relu(self.value_fc1(pool))
        v = self.value_fc2(v)                        # (B, 3) raw logits

        # Score head
        s = F.relu(self.score_fc1(pool))
        s = self.score_fc2(s)                        # (B, 1)

        return KataGoOutput(policy_logits=p, value_logits=v, score_lead=s)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_model.py::TestSEResNetModel -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/models/se_resnet.py tests/test_katago_model.py
git commit -m "feat: add SEResNetModel with policy, W/D/L value, and score heads"
```

---

### Task 4: Model Registry Integration

**Files:**
- Modify: `keisei/training/model_registry.py`

**WARNING:** `keisei/config.py:9-10` has separate allowlists `VALID_ARCHITECTURES` and `VALID_ALGORITHMS` that `load_config()` validates against BEFORE the registries are consulted. Plan B tests call `build_model()` directly (bypassing config), so this doesn't block Plan B. But Plan C MUST add `"se_resnet"` and `"katago_ppo"` to `config.py` — see Plan C Task 1.
- Modify: `tests/test_katago_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_model.py`:

```python
from keisei.training.model_registry import build_model, validate_model_params, VALID_ARCHITECTURES


class TestModelRegistry:
    def test_se_resnet_in_valid_architectures(self):
        assert "se_resnet" in VALID_ARCHITECTURES

    def test_build_se_resnet(self):
        params = {
            "num_blocks": 2, "channels": 32, "se_reduction": 8,
            "global_pool_channels": 16, "policy_channels": 8,
            "value_fc_size": 32, "score_fc_size": 16, "obs_channels": 50,
        }
        model = build_model("se_resnet", params)
        assert isinstance(model, SEResNetModel)

    def test_validate_se_resnet_params(self):
        params = {"num_blocks": 2, "channels": 32}
        validated = validate_model_params("se_resnet", params)
        assert isinstance(validated, SEResNetParams)
        assert validated.num_blocks == 2
        assert validated.channels == 32
        # Defaults should apply
        assert validated.obs_channels == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_model.py::TestModelRegistry -v`
Expected: FAIL — `"se_resnet"` not in `VALID_ARCHITECTURES`

- [ ] **Step 3: Register the architecture**

In `keisei/training/model_registry.py`, add the import:

```python
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
```

And add to `_REGISTRY`:

```python
"se_resnet": (SEResNetModel, SEResNetParams),
```

Note: `build_model` expects `BaseModel` as the return type, but `SEResNetModel` extends `KataGoBaseModel`, not `BaseModel`. The registry's type hint needs to accommodate both without breaking downstream type expectations.

Use `Union[BaseModel, KataGoBaseModel]` for the return type. Both are `nn.Module` subclasses, so runtime behavior is identical. The explicit union preserves type information for downstream code:

```python
from keisei.training.models.katago_base import KataGoBaseModel

_REGISTRY: dict[str, tuple[type[nn.Module], type]] = {
    "resnet": (ResNetModel, ResNetParams),
    "mlp": (MLPModel, MLPParams),
    "transformer": (TransformerModel, TransformerParams),
    "se_resnet": (SEResNetModel, SEResNetParams),
}
```

Update the `build_model` return type:

```python
def build_model(architecture: str, params: dict[str, Any]) -> BaseModel | KataGoBaseModel:
```

The `VALID_ARCHITECTURES` set updates automatically (derived from `_REGISTRY.keys()`).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_model.py::TestModelRegistry -v`
Expected: PASS

- [ ] **Step 5: Verify existing tests still pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add keisei/training/model_registry.py tests/test_katago_model.py
git commit -m "feat: register se_resnet architecture in model registry"
```

---

### Task 5: KataGoPPOParams and Algorithm Registry

**Files:**
- Create: `keisei/training/katago_ppo.py`
- Modify: `keisei/training/algorithm_registry.py`
- Create: `tests/test_katago_ppo.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_katago_ppo.py`:

```python
# tests/test_katago_ppo.py
"""Tests for the KataGo multi-head PPO algorithm."""

import torch
import pytest

from keisei.training.katago_ppo import KataGoPPOParams
from keisei.training.algorithm_registry import validate_algorithm_params, VALID_ALGORITHMS


class TestKataGoPPOParams:
    def test_defaults(self):
        params = KataGoPPOParams()
        assert params.learning_rate == 2e-4
        assert params.gamma == 0.99
        assert params.gae_lambda == 0.95
        assert params.lambda_value == 1.5
        assert params.lambda_score == 0.02
        assert params.lambda_entropy == 0.01
        assert params.score_normalization == 76.0
        assert params.grad_clip == 1.0

    def test_custom_params(self):
        params = KataGoPPOParams(learning_rate=1e-3, gamma=1.0)
        assert params.learning_rate == 1e-3
        assert params.gamma == 1.0


class TestAlgorithmRegistry:
    def test_katago_ppo_in_valid_algorithms(self):
        assert "katago_ppo" in VALID_ALGORITHMS

    def test_validate_katago_ppo_params(self):
        validated = validate_algorithm_params("katago_ppo", {"learning_rate": 1e-3})
        assert isinstance(validated, KataGoPPOParams)
        assert validated.learning_rate == 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_ppo.py -v -k "defaults or registry"`
Expected: FAIL — import errors

- [ ] **Step 3: Write the params and register**

Create `keisei/training/katago_ppo.py`:

```python
# keisei/training/katago_ppo.py
"""KataGo-style multi-head PPO: W/D/L value, score lead, spatial policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from keisei.training.ppo import compute_gae


@dataclass(frozen=True)
class KataGoPPOParams:
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95        # GAE lambda — exposed as config, not hardcoded
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256
    lambda_policy: float = 1.0
    lambda_value: float = 1.5
    lambda_score: float = 0.02
    lambda_entropy: float = 0.01
    score_normalization: float = 76.0  # used by KataGoTrainingLoop to normalize targets at buffer level
    grad_clip: float = 1.0
```

In `keisei/training/algorithm_registry.py`, add:

```python
from keisei.training.katago_ppo import KataGoPPOParams
```

And add to `_PARAM_SCHEMAS`:

```python
"katago_ppo": KataGoPPOParams,
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_ppo.py -v -k "defaults or registry"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py keisei/training/algorithm_registry.py tests/test_katago_ppo.py
git commit -m "feat: add KataGoPPOParams and register katago_ppo algorithm"
```

---

### Task 6: KataGoRolloutBuffer

**Files:**
- Modify: `keisei/training/katago_ppo.py`
- Modify: `tests/test_katago_ppo.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_ppo.py`:

```python
from keisei.training.katago_ppo import KataGoRolloutBuffer


class TestKataGoRolloutBuffer:
    def test_add_and_size(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        obs = torch.randn(2, 50, 9, 9)
        actions = torch.tensor([100, 200])
        log_probs = torch.tensor([0.5, 0.6])
        values = torch.tensor([0.1, 0.2])
        rewards = torch.tensor([0.0, 0.0])
        dones = torch.tensor([False, False])
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)
        value_cats = torch.tensor([0, 2])  # W, L
        score_targets = torch.tensor([0.5, -0.3])

        buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks,
                value_cats, score_targets)
        assert buf.size == 1

    def test_flatten(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(3):
            buf.add(
                torch.randn(2, 50, 9, 9),
                torch.randint(0, 11259, (2,)),
                torch.randn(2),
                torch.randn(2),
                torch.randn(2),
                torch.zeros(2, dtype=torch.bool),
                torch.ones(2, 11259, dtype=torch.bool),
                torch.randint(0, 3, (2,)),
                torch.randn(2),
            )
        data = buf.flatten()
        assert data["observations"].shape == (6, 50, 9, 9)
        assert data["actions"].shape == (6,)
        assert data["value_categories"].shape == (6,)
        assert data["score_targets"].shape == (6,)

    def test_clear(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        buf.add(
            torch.randn(2, 50, 9, 9), torch.zeros(2, dtype=torch.long),
            torch.zeros(2), torch.zeros(2), torch.zeros(2),
            torch.zeros(2, dtype=torch.bool), torch.ones(2, 11259, dtype=torch.bool),
            torch.zeros(2, dtype=torch.long), torch.zeros(2),
        )
        assert buf.size == 1
        buf.clear()
        assert buf.size == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_ppo.py::TestKataGoRolloutBuffer -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

Add to `keisei/training/katago_ppo.py`:

```python
# NOTE: Buffer memory at scale (128 steps × 512 envs):
# - legal_masks: 128 × 512 × 11259 × 1 byte = ~740 MB
# - observations: 128 × 512 × 50 × 9 × 9 × 4 bytes = ~1060 MB
# - other fields (7 × ~33 MB each): ~230 MB
# Total: ~2 GB CPU RAM. If memory becomes the binding constraint,
# consider sparse legal_mask storage or regenerating masks from
# game state during update. For now, keep it simple and dense.

class KataGoRolloutBuffer:
    def __init__(self, num_envs: int, obs_shape: tuple[int, ...], action_space: int) -> None:
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.clear()

    def clear(self) -> None:
        self.observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.legal_masks: list[torch.Tensor] = []
        self.value_categories: list[torch.Tensor] = []
        self.score_targets: list[torch.Tensor] = []

    @property
    def size(self) -> int:
        return len(self.observations)

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        legal_masks: torch.Tensor,
        value_categories: torch.Tensor,
        score_targets: torch.Tensor,
    ) -> None:
        """Add a timestep to the buffer.

        Args:
            score_targets: Pre-normalized score estimates in [-1, 1]. The caller
                (KataGoTrainingLoop) divides raw material difference by
                KataGoPPOParams.score_normalization before storing here.
                Raw scores can range from -200 to +200; without normalization,
                the MSE loss would dominate all other loss terms.
        """
        # Guard against unnormalized score targets (catches Plan C integration bugs)
        if score_targets.abs().max() > 2.0:
            raise ValueError(
                f"score_targets appear unnormalized: max abs value = "
                f"{score_targets.abs().max().item():.1f}, expected <= 1.0. "
                f"Divide by score_normalization before storing."
            )
        self.observations.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.legal_masks.append(legal_masks)
        self.value_categories.append(value_categories)
        self.score_targets.append(score_targets)

    def flatten(self) -> dict[str, torch.Tensor]:
        return {
            "observations": torch.stack(self.observations).reshape(-1, *self.obs_shape),
            "actions": torch.stack(self.actions).reshape(-1),
            "log_probs": torch.stack(self.log_probs).reshape(-1),
            "values": torch.stack(self.values).reshape(-1),
            "rewards": torch.stack(self.rewards).reshape(-1),
            "dones": torch.stack(self.dones).reshape(-1),
            "legal_masks": torch.stack(self.legal_masks).reshape(-1, self.action_space),
            "value_categories": torch.stack(self.value_categories).reshape(-1),
            "score_targets": torch.stack(self.score_targets).reshape(-1),
        }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_ppo.py::TestKataGoRolloutBuffer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_katago_ppo.py
git commit -m "feat: add KataGoRolloutBuffer with value_categories and score_targets"
```

---

### Task 7: KataGoPPOAlgorithm — Action Selection

**Files:**
- Modify: `keisei/training/katago_ppo.py`
- Modify: `tests/test_katago_ppo.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_ppo.py`:

```python
from keisei.training.katago_ppo import KataGoPPOAlgorithm
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


@pytest.fixture
def small_model():
    params = SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    )
    return SEResNetModel(params)


@pytest.fixture
def ppo(small_model):
    params = KataGoPPOParams()
    return KataGoPPOAlgorithm(params, small_model)


class TestKataGoPPOActionSelection:
    def test_select_actions_shapes(self, ppo):
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_select_actions_values_bounded(self, ppo):
        """Scalar value P(W) - P(L) should be in [-1, 1]."""
        obs = torch.randn(8, 50, 9, 9)
        legal_masks = torch.ones(8, 11259, dtype=torch.bool)
        _, _, values = ppo.select_actions(obs, legal_masks)
        assert values.min() >= -1.0
        assert values.max() <= 1.0

    def test_select_actions_all_illegal_raises(self, ppo):
        """All-False legal mask should raise, not produce NaN."""
        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.zeros(2, 11259, dtype=torch.bool)  # all illegal
        with pytest.raises(RuntimeError, match="zero legal actions"):
            ppo.select_actions(obs, legal_masks)

    def test_select_actions_respects_mask(self, ppo):
        """Actions should only be sampled from legal positions (20 trials)."""
        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.zeros(2, 11259, dtype=torch.bool)
        # Only allow action 0 and action 1000
        legal_masks[:, 0] = True
        legal_masks[:, 1000] = True
        # Match existing PPO test pattern: repeat to avoid single-trial flakes
        for _ in range(20):
            actions, _, _ = ppo.select_actions(obs, legal_masks)
            for a in actions.tolist():
                assert a in (0, 1000), f"Action {a} should be 0 or 1000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_ppo.py::TestKataGoPPOActionSelection -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

Add to `keisei/training/katago_ppo.py`:

```python
from keisei.training.models.katago_base import KataGoBaseModel
from keisei.training.ppo import compute_gae


class KataGoPPOAlgorithm:
    def __init__(
        self,
        params: KataGoPPOParams,
        model: KataGoBaseModel,
        forward_model: torch.nn.Module | None = None,
    ) -> None:
        self.params = params
        self.model = model
        self.forward_model = forward_model or model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        self.current_entropy_coeff = params.lambda_entropy  # mutable; Plan D warmup updates this

    @staticmethod
    def scalar_value(value_logits: torch.Tensor) -> torch.Tensor:
        """Project W/D/L logits to scalar value: P(W) - P(L).

        Used by both select_actions (rollout) and bootstrap (GAE).
        Centralised here so the formula can't diverge between the two call sites.
        """
        value_probs = F.softmax(value_logits, dim=-1)
        return value_probs[:, 0] - value_probs[:, 2]

    @torch.no_grad()
    def select_actions(
        self, obs: torch.Tensor, legal_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.forward_model.eval()
        output = self.forward_model(obs)

        # Guard: no env should have zero legal actions
        legal_counts = legal_masks.sum(dim=-1)
        if (legal_counts == 0).any():
            zero_envs = (legal_counts == 0).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"Environments {zero_envs} have zero legal actions — "
                f"all-False legal mask would produce NaN"
            )

        # Flatten spatial policy to (B, 11259), apply mask
        flat_logits = output.policy_logits.reshape(obs.shape[0], -1)
        masked_logits = flat_logits.masked_fill(~legal_masks, float("-inf"))

        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Scalar value for GAE — uses shared projection method
        scalar_values = self.scalar_value(output.value_logits)

        return actions, log_probs, scalar_values
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_ppo.py::TestKataGoPPOActionSelection -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_katago_ppo.py
git commit -m "feat: add KataGoPPOAlgorithm.select_actions with P(W)-P(L) scalar value"
```

---

### Task 8: KataGoPPOAlgorithm — Update Step

**Files:**
- Modify: `keisei/training/katago_ppo.py`
- Modify: `tests/test_katago_ppo.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_ppo.py`:

```python
from keisei.training.ppo import compute_gae


class TestKataGoPPOUpdate:
    def test_update_returns_metrics(self, ppo):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), torch.randn(2),
            )
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "score_loss" in losses
        assert "entropy" in losses
        assert "gradient_norm" in losses

    def test_update_loss_values_are_finite(self, ppo):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.randn(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), torch.randn(2),
            )
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        for key, val in losses.items():
            assert not torch.tensor(val).isnan(), f"{key} is NaN"
            assert not torch.tensor(val).isinf(), f"{key} is inf"

    def test_update_clears_buffer(self, ppo):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.zeros(2, dtype=torch.long), torch.zeros(2),
            )
        ppo.update(buf, torch.zeros(2))
        assert buf.size == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_ppo.py::TestKataGoPPOUpdate -v`
Expected: FAIL — `update` method not found

- [ ] **Step 3: Write the implementation**

Add to `KataGoPPOAlgorithm` in `keisei/training/katago_ppo.py`:

```python
    def update(self, buffer: KataGoRolloutBuffer, next_values: torch.Tensor) -> dict[str, float]:

        self.forward_model.train()
        data = buffer.flatten()
        T = buffer.size
        N = buffer.num_envs

        # GAE computation (uses scalar P(W)-P(L) values)
        rewards_2d = data["rewards"].reshape(T, N)
        values_2d = data["values"].reshape(T, N)
        dones_2d = data["dones"].reshape(T, N)

        all_advantages = torch.zeros(T, N, device=data["rewards"].device)
        for env_i in range(N):
            all_advantages[:, env_i] = compute_gae(
                rewards_2d[:, env_i], values_2d[:, env_i], dones_2d[:, env_i],
                next_values[env_i], gamma=self.params.gamma, lam=self.params.gae_lambda,
            )

        advantages = all_advantages.reshape(-1)
        # Note: no `returns` variable — KataGoPPO uses cross-entropy on W/D/L
        # categories for value loss, not MSE on scalar returns.
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = T * N
        batch_size = min(self.params.batch_size, total_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_score_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        num_updates = 0

        for _ in range(self.params.epochs_per_batch):
            indices = torch.randperm(total_samples, device=data["rewards"].device)
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                idx = indices[start:end]

                batch_obs = data["observations"][idx]
                batch_actions = data["actions"][idx]
                batch_old_log_probs = data["log_probs"][idx]
                batch_advantages = advantages[idx]
                batch_legal_masks = data["legal_masks"][idx]
                batch_value_cats = data["value_categories"][idx]
                batch_score_targets = data["score_targets"][idx]

                output = self.forward_model(batch_obs)

                # Policy loss (clipped surrogate)
                flat_logits = output.policy_logits.reshape(batch_obs.shape[0], -1)

                # NaN guard: check raw model output BEFORE masking.
                # After masking, -inf is expected (not NaN). NaN in flat_logits
                # means the model produced garbage — catch it before it propagates.
                if flat_logits.isnan().any():
                    raise RuntimeError("NaN in raw policy logits from model forward pass")

                masked_logits = flat_logits.masked_fill(~batch_legal_masks, float("-inf"))
                log_probs_all = F.log_softmax(masked_logits, dim=-1)
                new_log_probs = log_probs_all.gather(
                    1, batch_actions.unsqueeze(1)
                ).squeeze(1)

                ratio = (new_log_probs - batch_old_log_probs).exp()
                clip = self.params.clip_epsilon
                surr1 = ratio * batch_advantages
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy over legal actions only.
                # softmax(-inf) = 0 for illegal actions, log_softmax(-inf) = -inf.
                # Product 0 * -inf = NaN in IEEE 754, so we replace -inf with 0.0
                # in log_probs BEFORE multiplication. Result: 0 * 0 = 0 for illegals.
                # In float32, softmax may produce tiny non-zero values (~1e-38) for
                # masked positions, but multiplied by 0.0 they contribute nothing.
                probs = F.softmax(masked_logits, dim=-1)
                safe_log_probs = log_probs_all.masked_fill(~batch_legal_masks, 0.0)
                entropy = -(probs * safe_log_probs).sum(dim=-1).mean()

                # Value loss (cross-entropy on W/D/L)
                # ignore_index=-1: non-terminal steps have value_cats=-1 and are
                # excluded from the loss. Only terminal steps with known outcomes
                # (0=W, 1=D, 2=L) contribute to the value gradient.
                value_loss = F.cross_entropy(
                    output.value_logits, batch_value_cats, ignore_index=-1
                )

                # Score loss (MSE on normalized score)
                score_loss = F.mse_loss(output.score_lead.squeeze(-1), batch_score_targets)

                # Combined loss (no value clipping — documented as intentional)
                loss = (
                    self.params.lambda_policy * policy_loss
                    + self.params.lambda_value * value_loss
                    + self.params.lambda_score * score_loss
                    - self.current_entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params.grad_clip
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_score_loss += score_loss.item()
                total_entropy += entropy.item()
                total_grad_norm += float(grad_norm)
                num_updates += 1

        buffer.clear()

        denom = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "score_loss": total_score_loss / denom,
            "entropy": total_entropy / denom,
            "gradient_norm": total_grad_norm / denom,
        }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_ppo.py::TestKataGoPPOUpdate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_katago_ppo.py
git commit -m "feat: add KataGoPPOAlgorithm.update with multi-head loss"
```

---

### Task 9: Value Prediction Metrics

**Files:**
- Modify: `tests/test_katago_ppo.py`

- [ ] **Step 1: Write test for prediction breakdown metrics**

Add to `tests/test_katago_ppo.py`:

```python
class TestValueMetrics:
    def test_compute_value_metrics(self):
        """Verify the metrics helper computes frac_predicted_win/draw/loss correctly."""
        from keisei.training.katago_ppo import compute_value_metrics

        # 4 predictions: 2 predict W, 1 predicts D, 1 predicts L
        value_logits = torch.tensor([
            [2.0, 0.0, 0.0],  # predicts W
            [0.0, 2.0, 0.0],  # predicts D
            [0.0, 0.0, 2.0],  # predicts L
            [1.5, 0.0, 0.0],  # predicts W
        ])
        value_targets = torch.tensor([0, 1, 2, 0])  # W, D, L, W

        metrics = compute_value_metrics(value_logits, value_targets)
        assert metrics["value_accuracy"] == 1.0  # all correct
        assert metrics["frac_predicted_win"] == 0.5
        assert metrics["frac_predicted_draw"] == 0.25
        assert metrics["frac_predicted_loss"] == 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_ppo.py::TestValueMetrics -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

Add to `keisei/training/katago_ppo.py`:

```python
def compute_value_metrics(
    value_logits: torch.Tensor, value_targets: torch.Tensor,
) -> dict[str, float]:
    """Compute value prediction metrics for monitoring.

    Args:
        value_logits: (N, 3) raw W/D/L logits
        value_targets: (N,) int targets {0=W, 1=D, 2=L}

    Returns:
        Dict with value_accuracy, frac_predicted_win/draw/loss
    """
    predictions = value_logits.argmax(dim=-1)
    return {
        "value_accuracy": (predictions == value_targets).float().mean().item(),
        "frac_predicted_win": (predictions == 0).float().mean().item(),
        "frac_predicted_draw": (predictions == 1).float().mean().item(),
        "frac_predicted_loss": (predictions == 2).float().mean().item(),
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_ppo.py::TestValueMetrics -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_katago_ppo.py
git commit -m "feat: add compute_value_metrics for W/D/L prediction monitoring"
```

---

### Task 10: Full Test Suite Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all Python tests**

Run: `uv run pytest -v`
Expected: All tests PASS, including existing `test_models.py`, `test_ppo.py`, `test_registries.py` (no regressions).

- [ ] **Step 2: Verify model parameter counts**

Run: `uv run python -c "
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
for name, blocks, ch in [('b20c128', 20, 128), ('b40c256', 40, 256)]:
    p = SEResNetParams(num_blocks=blocks, channels=ch)
    m = SEResNetModel(p)
    n = sum(x.numel() for x in m.parameters())
    print(f'{name}: {n:,} params')
"`
Expected: b20c128 ~5M, b40c256 ~25M (approximate — exact counts will vary).

- [ ] **Step 3: Commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: address issues found in full test suite verification"
```
