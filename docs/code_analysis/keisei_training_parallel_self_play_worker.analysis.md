# Code Analysis: `keisei/training/parallel/self_play_worker.py`

## 1. Purpose & Role

`SelfPlayWorker` is a `multiprocessing.Process` subclass that runs an independent self-play loop in a worker process. Each worker maintains its own Shogi game environment and a local copy of the neural network model, collecting experience tuples that are batched and sent to the main process via a queue. This is the workhorse of the parallel system -- all game simulation and inference happen here.

## 2. Interface Contracts

### Constructor (`__init__`, lines 35-85)
- **Inputs**: `worker_id`, `env_config`, `model_config`, `parallel_config`, three `mp.Queue` instances, `seed_offset`.
- **Expects** `env_config` to contain `"seed"` key (line 71).
- **State**: Initializes in parent process; `game`, `model`, `policy_mapper` are `None` until `run()`.

### `run` (lines 87-95)
- **Contract**: Calls `_setup_worker()` then `_worker_loop()`. Cleanup runs in `finally`.

### `_setup_worker` (lines 97-135)
- **Contract**: Sets random seeds, creates `ShogiGame`, `PolicyOutputMapper`, and model via `model_factory`. Uses late import for `model_factory` (line 112).

### `_collect_single_experience` (lines 172-268)
- **Returns**: `Optional[Experience]`
- **Contract**: Runs one step of self-play: get observation, forward through model, mask illegal actions, sample action, execute move, create `Experience`.

### `_send_experience_batch` (lines 270-308)
- **Contract**: Converts experiences to batched tensor format and puts on the experience queue.

### `_update_model` (lines 395-435)
- **Contract**: Reconstructs state_dict from received model data and loads into the local model.

## 3. Correctness Analysis

- **Self-play is single-player, not two-player (lines 172-268)**: The worker plays moves sequentially in the game, always acting as the current player. Since Shogi is a two-player game, the worker effectively plays both sides. The observation from `game.reset()` and `game.get_observation()` is from the perspective of the current player. After a move, the game switches the current player internally. This means experiences alternate between black and white perspectives. For self-play training with PPO, this is standard practice -- both players share the same policy -- so this is correct.

- **`game.reset()` return value (line 185)**: `_current_obs` is assigned `self.game.reset()` which returns `np.ndarray`. Then on line 188, `torch.from_numpy(self._current_obs).float()` is called. This is correct.

- **`game.make_move` return value (line 237)**: The call `next_obs, reward, done, _ = self.game.make_move(selected_move)` destructures into 4 values. `make_move` returns `Tuple[np.ndarray, float, bool, Dict[str, Any]]` when `is_simulation=False`. This matches. However, `make_move` can also return a `Dict[str, Any]` when `is_simulation=True`. Since `is_simulation` defaults to False and the worker never passes True, this is safe.

- **Legal move masking correctness (lines 196-213)**: Legal moves are obtained from the game, converted to a boolean mask via `PolicyOutputMapper.get_legal_mask`, then applied multiplicatively to action probabilities. The normalization `masked_probs / (masked_probs.sum() + 1e-8)` prevents division by zero. However, if **all** legal moves map to zero probability (extremely unlikely but theoretically possible with floating point precision), the epsilon-normalized distribution would be nearly uniform over the zero entries, potentially sampling an illegal action. The `policy_index_to_shogi_move` call (line 219) would then either succeed with a technically-legal-but-zero-probability move (since the mask zeros out illegal moves, the remaining moves are legal) or fail. Since the mask guarantees only legal actions have nonzero probability, this edge case cannot produce illegal actions.

- **Model not yet synced at first experience (lines 152-155)**: The worker waits in a sleep loop (`time.sleep(0.1)`) if `self.model is None`. However, `_setup_worker` creates the model via `model_factory` (lines 119-126), so `self.model` is non-None after setup. The initial model has random weights. The first model update from the main process arrives asynchronously. Between setup and the first model update, the worker collects experiences using random-weight inference. For training purposes, these experiences are low quality but not incorrect.

- **`_update_model` format handling (lines 395-435)**: The method handles two formats: dict-with-`"data"`-key (lines 416-425) and raw numpy array (line 427). The dict format checks for `compressed` flag. This matches the output of `WorkerCommunicator._prepare_model_data`, where each key maps to a dict containing `"data"`, `"shape"`, `"dtype"`, and `"compressed"` fields. The `else` branch on line 427 (`torch.from_numpy(data)`) handles the case where `data` is a raw numpy array, which could occur if the model data format changes. However, `_prepare_model_data` always produces dicts, so the `else` branch is defensive code that may never execute.

- **`_experiences_to_batch` legal_mask stacking (line 339)**: `torch.stack([exp.legal_mask for exp in experiences], dim=0)` assumes all `legal_mask` tensors have the same shape. Since all workers use the same `PolicyOutputMapper` with 13,527 actions, this is guaranteed.

- **Observation tensor shape (line 188)**: `obs_tensor = torch.from_numpy(self._current_obs).float().to(self.device)` followed by `obs_tensor.unsqueeze(0)` (line 192) for the model forward pass. The observation is `(46, 9, 9)` from `ShogiGame`, and unsqueeze produces `(1, 46, 9, 9)`. The `Experience` stores `obs_tensor.cpu()` (line 241) which is the un-batched `(46, 9, 9)` tensor. This is correct.

- **No `game.reset()` after `game_over` (lines 251-260)**: When `done=True`, `_current_obs` is set to `None` (line 252), causing the next call to `_collect_single_experience` to call `self.game.reset()` (line 185). This correctly handles episode boundaries.

## 4. Robustness & Error Handling

- **Broad exception handling in `run` (lines 87-95)**: Catches `(ValueError, RuntimeError, ImportError, OSError)`. This covers most failure modes but does not catch `KeyError` (which could occur from dict access in `_update_model`), `TypeError`, or `torch` specific exceptions not derived from these bases.

- **Experience collection failure (lines 264-268)**: If any step in `_collect_single_experience` fails, the method returns `None` and the experience is simply skipped. The error is logged. This is resilient but could mask systematic issues.

- **Queue full handling (lines 301-304)**: When the experience queue is full, the batch is dropped with a warning. This is a data loss scenario. If the main process is slow to consume, workers lose experiences permanently. There is no backpressure mechanism.

- **Model update failure (lines 434-435)**: Failed model updates are logged and the worker continues with the old model weights. This is correct resilience behavior.

- **`_check_control_commands` and `_check_model_updates` (lines 342-368)**: Both use the same pattern of draining the queue with `get_nowait()`. If multiple model updates are queued, all are applied sequentially (line 361-362). Each update overwrites the previous, so only the latest matters. This is correct but wastes decompression work for intermediate updates.

- **Pause command (lines 387-389)**: `time.sleep(pause_duration)` blocks the entire worker process. During this time, control commands and model updates are not processed. A long pause could cause the worker to miss stop commands.

## 5. Performance & Scalability

- **CPU-only inference (line 80)**: Workers use `torch.device("cpu")`. For models like ResNet with SE blocks, CPU inference is significantly slower than GPU. This is a deliberate design choice to avoid GPU memory contention, but it limits per-worker throughput.

- **Late import (line 112)**: `from keisei.training.models import model_factory` is imported inside `_setup_worker` to avoid importing heavy modules in the parent process before fork. This is correct practice for multiprocessing.

- **Model initialization per worker (lines 119-130)**: Each worker creates its own model instance. For N workers, this means N copies of model parameters in memory. For a ResNet tower with ~5M parameters in float32, this is ~20MB per worker -- manageable.

- **Batch sending efficiency (lines 270-308)**: Experiences are accumulated into a list and batch-converted to tensors via `_experiences_to_batch`. This avoids per-experience queue overhead. The batch size is controlled by `parallel_config["batch_size"]`.

- **Tight loop without sleep (lines 142-166)**: The main worker loop has no sleep between iterations. When there are legal moves and the model is ready, the loop runs as fast as inference allows. This maximizes throughput but uses 100% CPU per worker.

## 6. Security & Safety

- **`model_factory` dynamic import (line 112)**: The import is from a known internal module path, not from user-configurable input. No injection risk.
- **Pickle-based IPC**: Experience batches (containing torch tensors) are pickled for queue transmission. Standard practice for multiprocessing, acceptable for trusted local processes.
- **No user input handling**: The worker does not accept external input beyond configuration and queue messages from the parent process.

## 7. Maintainability

- **Clear structure**: The worker follows a clean setup -> loop -> cleanup lifecycle with well-separated concerns.
- **Duplicate null checks for `policy_mapper`**: Lines 197-205 and 217-234 both check `if self.policy_mapper is not None`. Since `policy_mapper` is set in `_setup_worker` and never cleared, these checks are defensive but redundant after successful setup.
- **`get_worker_stats` (lines 450-463)**: Returns stats dict, but this method can only be called in the parent process (where the stats are stale -- they were initialized before `start()` and not updated across the process boundary). The actual stats are sent via batch messages. This method is effectively useless after `start()`.
- **Type annotations**: Well-typed with Optional types used appropriately for post-init fields.
- **Logging**: Consistent logging at appropriate levels throughout the worker lifecycle.

## 8. Verdict

**NEEDS_ATTENTION** -- The worker is the most complex and critical component in the parallel system. It is functionally correct for its core self-play and experience collection loop. Key concerns are: (1) dropped batches when experience queue is full with no backpressure mechanism, (2) `get_worker_stats` is misleading since it returns parent-process-side values that are never updated, (3) the pause command blocks all processing including stop commands, and (4) the worker collects experiences with random-weight model before the first sync, producing low-quality data that enters the training buffer.
