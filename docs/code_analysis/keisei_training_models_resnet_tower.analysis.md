# Code Analysis: `keisei/training/models/resnet_tower.py`

## 1. Purpose & Role

This module implements the primary neural network architecture for the Keisei Shogi RL system: a residual tower with optional Squeeze-and-Excitation (SE) blocks, split into separate policy and value heads. It is the only concrete model in the codebase and is instantiated exclusively through `model_factory` in `training/models/__init__.py`. The class `ActorCriticResTower` extends `BaseActorCriticModel`, which provides `get_action_and_value` and `evaluate_actions` implementations, so this file only needs to define the architecture and `forward` method.

## 2. Interface Contracts

### Imports
- `torch`, `torch.nn`, `torch.nn.functional` -- PyTorch primitives.
- `keisei.constants.SHOGI_BOARD_SIZE`, `SHOGI_BOARD_SQUARES` -- board geometry constants (9 and 81).
- `keisei.core.base_actor_critic.BaseActorCriticModel` -- abstract base class providing protocol compliance.

### `SqueezeExcitation` (lines 15-26)
- **Constructor**: `channels` (int), `se_ratio` (float, default 0.25).
- **Forward**: Takes tensor `x` of shape `(B, C, H, W)`, returns tensor of same shape with channel-wise attention applied.
- **Hidden dimension**: `max(1, int(channels * se_ratio))` (line 18), ensuring at least 1 hidden unit.

### `ResidualBlock` (lines 29-44)
- **Constructor**: `channels` (int), `se_ratio` (Optional[float], default None).
- **Forward**: Standard pre-activation residual block: `conv -> bn -> relu -> conv -> bn -> [SE] -> add residual -> relu`.
- **SE integration**: Applied only if `se_ratio` is truthy (line 41).

### `ActorCriticResTower` (lines 47-84)
- **Constructor**: `input_channels` (int), `num_actions_total` (int), `tower_depth` (int, default 9), `tower_width` (int, default 256), `se_ratio` (Optional[float], default None).
- **Forward**: Returns `(policy_logits, value)` where `policy_logits` has shape `(B, num_actions_total)` and `value` has shape `(B,)`.

## 3. Correctness Analysis

- **Residual block architecture (lines 38-44)**: The block applies `conv1 -> bn1 -> relu -> conv2 -> bn2 -> [SE] -> residual add -> relu`. This is correct standard ResNet post-activation ordering. Both convolutions use `padding=1` with `kernel_size=3`, preserving spatial dimensions.
- **SE block (lines 22-26)**: Uses global average pooling (`adaptive_avg_pool2d(x, 1)`) followed by two 1x1 convolutions with ReLU and sigmoid activations. This is the standard SE implementation. The multiplicative attention `x * s` on line 26 is correct.
- **SE hidden dimension (line 18)**: `max(1, int(channels * se_ratio))` -- the `int()` truncates rather than rounds. For `channels=3, se_ratio=0.25`, hidden = `max(1, 0)` = 1. The `max(1, ...)` guard prevents a zero-width hidden layer, which is correct.
- **Policy head (lines 63-69)**: Reduces tower width to 2 channels via 1x1 conv, then flattens to `2 * 81 = 162` features, then projects to `num_actions_total` (typically 13527). This is a slim head design following AlphaZero conventions.
- **Value head (lines 71-77)**: Same 2-channel reduction, flattening to 162 features, projecting to scalar (1). The `.squeeze(-1)` on line 83 removes the trailing dimension, producing shape `(B,)`.
- **Spatial dimension assumption**: The linear layers in both heads hardcode `2 * SHOGI_BOARD_SQUARES` (= 162). This assumes the input spatial dimensions are exactly 9x9. If an input with different spatial dimensions is provided, the `nn.Linear` will fail with a dimension mismatch. This is correct for the Shogi domain but makes the model non-reusable for other board sizes.
- **Value squeeze (line 83)**: `value_head(x).squeeze(-1)` produces shape `(B,)`. The base class `evaluate_actions` also has a squeeze guard (line 181 of `base_actor_critic.py`), making the double-squeeze safe (squeezing a `(B,)` tensor is a no-op).

## 4. Robustness & Error Handling

- **No input validation**: The constructor does not validate that `input_channels > 0`, `tower_depth >= 0`, `tower_width > 0`, or `num_actions_total > 0`. Invalid values would produce PyTorch construction errors.
- **`tower_depth=0` edge case**: If `tower_depth=0`, `self.res_blocks` becomes an empty `nn.Sequential`, which acts as an identity pass-through. The model would still function but with no residual blocks -- just the stem, policy head, and value head. This is a valid degenerate case.
- **`se_ratio=0.0` treated as falsy**: On line 36, `if se_ratio` evaluates to `False` for `se_ratio=0.0`. A caller passing `se_ratio=0.0` would get no SE block, which is semantically correct (a zero ratio means no squeeze-excitation). However, this conflates `None` (disabled) with `0.0` (zero ratio).
- **No NaN/Inf guards**: The forward pass does not check for NaN or Inf in intermediate tensors. This is standard practice for PyTorch models -- such checks are typically done at higher levels.
- **`pylint: disable=not-callable` on line 42**: Suppresses a false positive from pylint about calling `self.se` when it is an `nn.Module` instance. This is a correct suppression.

## 5. Performance & Scalability

- **Standard ResNet efficiency**: The architecture uses standard PyTorch operations (Conv2d, BatchNorm2d, ReLU) that are well-optimized on GPU. The SE block adds minimal overhead (global average pool + two 1x1 convolutions).
- **Memory**: Each residual block has 2 convolutional layers and 2 batch norm layers. With default `tower_depth=9` and `tower_width=256`, this is `9 * (2 * 256 * 256 * 3 * 3 + 2 * 256 * 4)` = approximately 10.6M parameters in the tower alone.
- **Policy head bottleneck**: The slim 2-channel policy head reduces parameters compared to a full-width head, but the `Linear(162, 13527)` layer still has ~2.2M parameters.
- **No `torch.compile` directives**: The model does not use `@torch.compile` or other JIT hints, but compilation is handled externally by `ModelManager`.

## 6. Security & Safety

- No file I/O, network access, or dynamic code execution.
- No user-controlled strings used in operations.
- All tensor operations are standard PyTorch with known numerical properties.

## 7. Maintainability

- **84 lines, 3 classes**: Well-structured with each class having a single responsibility.
- **Clear naming**: `SqueezeExcitation`, `ResidualBlock`, `ActorCriticResTower` are descriptive.
- **Docstring**: Module-level docstring on line 2. No per-class or per-method docstrings.
- **Hardcoded board size**: `SHOGI_BOARD_SQUARES` is imported from constants rather than hardcoded as a magic number, which is good practice.
- **No `__all__` export**: The module exports all three classes by default, though only `ActorCriticResTower` is imported by `__init__.py`.
- **Inheritance chain**: `ActorCriticResTower -> BaseActorCriticModel -> nn.Module + ActorCriticProtocol`. The inheritance is clean and the only requirement on subclasses is implementing `forward`.

## 8. Verdict

**SOUND**

The neural network architecture is a correct, standard implementation of a ResNet tower with SE blocks and separate policy/value heads for AlphaZero-style Shogi. The spatial dimension assumption (9x9) is domain-appropriate. The `se_ratio=0.0` treated as falsy is a minor semantic ambiguity but not a functional bug. The code is concise, well-structured, and follows established deep learning patterns.
