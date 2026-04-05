## Summary

`DynamicTrainer` computes PPO `old_log_probs` after forcing the model into training mode, so BatchNorm/other train-mode behavior can change logits from the rollout policy and corrupt the PPO ratio baseline.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `/home/john/keisei/keisei/training/dynamic_trainer.py:280-281` sets training mode before baseline computation:
```python
model = self.store.load_opponent(entry, device)
model.train()
```
- `/home/john/keisei/keisei/training/dynamic_trainer.py:311-319` computes `old_log_probs` under that mode:
```python
with torch.no_grad():
    output = model(all_obs)
    ...
    old_log_probs = F.log_softmax(masked, dim=-1).gather(...)
```
- Rollout data for Dynamic training is collected with inference models in eval mode:
  - `/home/john/keisei/keisei/training/concurrent_matches.py:511-512`
  - `/home/john/keisei/keisei/training/match_utils.py:122-123`
- Common model used in this system contains BatchNorm layers (mode-sensitive behavior):
  - `/home/john/keisei/keisei/training/models/se_resnet.py:111,121`
  - `/home/john/keisei/keisei/training/models/resnet.py:46,51,58`

## Root Cause Hypothesis

The code reuses a freshly loaded opponent model (initially eval in store loader) but immediately switches it to `train()` before deriving `old_log_probs`. For architectures with BatchNorm (and any train/eval-sensitive layers), the baseline log-probabilities no longer match the behavior policy that generated actions, so PPO importance ratios become biased/noisy. This is triggered whenever Dynamic entries use BN-based architectures (which they do in this repo).

## Suggested Fix

Compute `old_log_probs` in eval mode, then switch to train mode for optimization steps. Example change in `dynamic_trainer.py`:

1. After loading model, keep/effectively set `model.eval()` for baseline pass.
2. Run the `with torch.no_grad(): ... old_log_probs ...` block in eval mode.
3. Call `model.train()` immediately before the epoch update loop.

Concretely, move `model.train()` from before baseline computation to just before `for _ in range(self.config.update_epochs_per_batch):`.
