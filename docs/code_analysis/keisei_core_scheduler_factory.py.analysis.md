# Analysis: keisei/core/scheduler_factory.py

**Lines:** 110
**Role:** Factory class that creates PyTorch learning rate schedulers based on configuration. Supports linear decay (via LambdaLR), cosine annealing, exponential decay, and step decay. Created during PPOAgent initialization and stepped either per-epoch or per-minibatch-update as configured.
**Key dependencies:** Imports `torch`, `LambdaLR`, `CosineAnnealingLR`, `ExponentialLR`, `StepLR` from PyTorch. Imported only by `keisei/core/ppo_agent.py`.
**Analysis depth:** FULL

## Summary

This is a small, clean factory class with no critical issues. The scheduler implementations are straightforward wrappers around PyTorch's built-in schedulers. The main concern is a subtle numerical issue in the linear scheduler where `total_steps=0` is guarded against but `step > total_steps` silently clamps, and the interaction between `total_steps` calculation in `PPOAgent._calculate_total_scheduler_steps()` and the actual number of scheduler steps can lead to misaligned schedules if configuration values don't divide evenly. Confidence is HIGH.

## Warnings

### [66-74] Linear decay lambda captures total_steps by closure, clamping behavior may cause premature LR floor

**What:** The `linear_decay` lambda clamps `step` to `total_steps` when `step > total_steps`. This means if the scheduler is stepped more times than `total_steps`, the learning rate stays at the `final_lr_fraction` floor rather than decaying further.

**Why it matters:** The `total_steps` value is calculated by `PPOAgent._calculate_total_scheduler_steps()` using integer division of `total_timesteps // steps_per_epoch`. If `total_timesteps` is not evenly divisible by `steps_per_epoch`, the calculated `total_steps` may be fewer than the actual number of scheduler steps taken. For example, with `total_timesteps=10000`, `steps_per_epoch=2048`, the number of epochs is `10000 // 2048 = 4` (8192 timesteps, not 10000). The remaining 1808 timesteps still produce a partial epoch where the scheduler has already reached its floor. This is likely the intended behavior (floor rather than overshoot), but the truncation means up to `steps_per_epoch - 1` timesteps may train at the minimum learning rate unexpectedly.

**Evidence:**
```python
def linear_decay(step: int) -> float:
    if step > total_steps:
        current_step = total_steps
    else:
        current_step = step
    progress = current_step / total_steps
    return (1.0 - progress) * (1.0 - final_lr_fraction) + final_lr_fraction
```

And in ppo_agent.py:
```python
def _calculate_total_scheduler_steps(self, config: AppConfig) -> int:
    if config.training.lr_schedule_step_on == "epoch":
        return (config.training.total_timesteps // config.training.steps_per_epoch) * config.training.ppo_epochs
    else:
        updates_per_epoch = (config.training.steps_per_epoch // config.training.minibatch_size) * config.training.ppo_epochs
        num_epochs = (config.training.total_timesteps // config.training.steps_per_epoch)
        return num_epochs * updates_per_epoch
```

Note: When `lr_schedule_step_on == "epoch"`, the total steps include multiplication by `ppo_epochs`. Looking at the PPOAgent.learn() method, the scheduler is stepped once per PPO epoch (line 433-434) when `step_on == "epoch"`. The loop runs `ppo_epochs` times per learn() call, so the total is `num_learn_calls * ppo_epochs`, which matches. This is correct.

When `lr_schedule_step_on == "update"`, the scheduler steps once per minibatch (line 423-424). The total updates per epoch is `(steps_per_epoch // minibatch_size) * ppo_epochs`, which also accounts for the truncation from integer division of `steps_per_epoch // minibatch_size`. If `steps_per_epoch` is not a multiple of `minibatch_size`, the last partial minibatch in each inner loop still runs (NumPy slicing doesn't truncate), but the total calculated steps won't count it. This creates a slight mismatch.

### [87-91] Cosine scheduler eta_min depends on initial optimizer LR, not config LR

**What:** The cosine scheduler computes `eta_min = initial_lr * eta_min_fraction` where `initial_lr = optimizer.param_groups[0]["lr"]`.

**Why it matters:** If the optimizer's learning rate has already been modified (e.g., by loading a checkpoint where the optimizer state includes a different LR, or by a previous scheduler), `initial_lr` will not be the configured learning rate. This could cause the cosine schedule to use an unexpected minimum LR. In practice, the scheduler is created during `__init__` before any training, so `initial_lr` should match `config.training.learning_rate` unless the optimizer fallback (line 78) was triggered.

**Evidence:**
```python
initial_lr = optimizer.param_groups[0]["lr"]
eta_min = initial_lr * eta_min_fraction
```

### [98-100] Exponential scheduler default gamma of 0.995 may be too aggressive or too mild depending on step frequency

**What:** The exponential scheduler uses a default `gamma=0.995` per step. The effective LR decay depends entirely on whether the scheduler steps per epoch or per minibatch update.

**Why it matters:** With `step_on="epoch"` and 10 PPO epochs per learn() call, the LR after 244 learn calls (100 epochs at 2048 steps/epoch with 500K total timesteps) would be `0.995^2440 = ~5e-6` times the initial LR. With `step_on="update"` and 320 updates per epoch (2048/64 * 10), the LR would decay astronomically faster. The default value is reasonable only for specific step frequencies, and without documentation about which `step_on` mode it's designed for, users may get unexpectedly fast or slow decay. The config documentation for `lr_schedule_kwargs` should specify that `gamma` must be tuned relative to the `step_on` choice.

**Evidence:**
```python
gamma = kwargs.get("gamma", 0.995)
return ExponentialLR(optimizer, gamma=gamma)
```

## Observations

### [33-34] None schedule type returns None cleanly

**What:** `if schedule_type is None: return None` provides a clean "no scheduler" path.

**Why it matters:** This is good defensive design. The PPOAgent correctly checks `if self.scheduler is not None` before calling `self.scheduler.step()`.

### [59-62, 83-86] Validation of total_steps for linear and cosine schedulers

**What:** Both linear and cosine schedulers validate that `total_steps > 0`, raising `ValueError` if not.

**Why it matters:** Good -- prevents division by zero in the linear lambda and invalid `T_max` in cosine. The error will surface clearly during initialization rather than silently during training.

### [51-52] Unsupported scheduler type raises ValueError

**What:** An unknown `schedule_type` raises `ValueError`.

**Why it matters:** Good -- fails fast with a clear message. The `config_schema.py` validator at line 171-175 also validates the allowed types, providing a double safety net.

### General: No state management complexity

**What:** The factory is purely a creation utility with no mutable state. All schedulers are standard PyTorch objects.

**Why it matters:** Low risk of bugs. The complexity lies in the interaction between the scheduler and the `PPOAgent.learn()` stepping logic, which is in `ppo_agent.py`, not here.

## Verdict

**Status:** SOUND
**Recommended action:** (1) Document the interaction between `step_on` mode and scheduler hyperparameters (especially for exponential decay). (2) Consider adding a validation or warning when `steps_per_epoch` is not evenly divisible by `minibatch_size` in the `update` step mode, as this creates a mismatch between calculated and actual total steps. (3) The integer division truncation in `_calculate_total_scheduler_steps` (which lives in `ppo_agent.py`) should be flagged there, but it directly affects scheduler behavior.
**Confidence:** HIGH -- The code is short, uses standard PyTorch APIs, and has minimal room for subtle bugs. The concerns are about parameter tuning and edge cases rather than correctness.
