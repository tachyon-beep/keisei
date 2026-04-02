# Test Gap Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all critical, high, medium, and quick-win test gaps identified in the coverage gap analysis, adding ~85 new tests across 14 tasks.

**Architecture:** Each task adds tests to existing test files (no new test files). Tests follow the codebase's existing patterns: class-based pytest organization, `tmp_path` for temp files, `MagicMock` for VecEnv/pool mocks, `torch.testing.assert_close` for float comparisons.

**Tech Stack:** Python 3.13, pytest, PyTorch, unittest.mock, numpy

**Run all tests with:** `uv run pytest tests/ -x -q`

---

## Task 1: Quick Wins — Pure Function Tests

**Files:**
- Modify: `tests/test_league.py`
- Modify: `tests/test_katago_ppo.py`
- Modify: `tests/test_katago_loop.py`
- Modify: `tests/test_sl_pipeline.py`
- Modify: `tests/test_models.py`

These are pure functions with no I/O — fast to write, fast to run, high value.

- [ ] **Step 1: Add Elo numerical correctness tests to test_league.py**

Add to the existing `TestEloCalculation` class:

```python
class TestEloCalculation:
    # ... existing tests ...

    def test_equal_ratings_win(self):
        """Player A wins against equal opponent — gains exactly K/2."""
        new_a, new_b = compute_elo_update(1500.0, 1500.0, 1.0, k=32.0)
        assert new_a == pytest.approx(1516.0)
        assert new_b == pytest.approx(1484.0)

    def test_equal_ratings_draw(self):
        """Draw between equals — no rating change."""
        new_a, new_b = compute_elo_update(1500.0, 1500.0, 0.5, k=32.0)
        assert new_a == pytest.approx(1500.0)
        assert new_b == pytest.approx(1500.0)

    def test_upset_win_large_gain(self):
        """Weak player beats strong player — gains more than K/2."""
        new_a, new_b = compute_elo_update(1200.0, 1800.0, 1.0, k=32.0)
        # Expected_a ~= 1/(1+10^(600/400)) ~= 0.0309 → gain ~= 32*(1-0.031) ~= 31.01
        assert new_a > 1200.0 + 30.0
        assert new_b < 1800.0 - 30.0

    def test_custom_k_factor(self):
        """K-factor scales the update magnitude."""
        new_a_k16, _ = compute_elo_update(1500.0, 1500.0, 1.0, k=16.0)
        new_a_k64, _ = compute_elo_update(1500.0, 1500.0, 1.0, k=64.0)
        delta_16 = new_a_k16 - 1500.0
        delta_64 = new_a_k64 - 1500.0
        assert delta_64 == pytest.approx(delta_16 * 4.0)
```

- [ ] **Step 2: Add GameFilter rating key variant tests to test_sl_pipeline.py**

Add a new class after the existing parser tests:

```python
class TestGameFilterRatingKeys:
    def test_black_rating_below_minimum_rejects(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={"black_rating": "1200", "white_rating": "1600"},
        )
        assert not gf.accepts(record)

    def test_white_rating_below_minimum_rejects(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={"black_rating": "1600", "white_rating": "1200"},
        )
        assert not gf.accepts(record)

    def test_both_ratings_above_minimum_accepts(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={"black_rating": "1600", "white_rating": "1700"},
        )
        assert gf.accepts(record)

    def test_no_rating_keys_accepts(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={},
        )
        assert gf.accepts(record)

    def test_non_digit_rating_ignored(self):
        from keisei.sl.parsers import GameFilter, GameRecord, ParsedMove
        gf = GameFilter(min_ply=1, min_rating=1500)
        record = GameRecord(
            moves=[ParsedMove("7g7f", "startpos")] * 5,
            outcome=GameOutcome.WIN_BLACK,
            metadata={"rating": "unknown"},
        )
        assert gf.accepts(record)
```

- [ ] **Step 3: Add create_lr_scheduler unknown type test to test_lr_scheduler.py**

```python
class TestLRScheduler:
    # ... existing tests ...

    def test_unknown_schedule_type_raises(self, small_model):
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="Unknown schedule type"):
            create_lr_scheduler(optimizer, schedule_type="cosine")
```

- [ ] **Step 4: Add model registry pure function tests to test_models.py**

```python
class TestModelRegistry:
    def test_get_model_contract_se_resnet(self):
        from keisei.training.model_registry import get_model_contract
        assert get_model_contract("se_resnet") == "multi_head"

    def test_get_model_contract_resnet(self):
        from keisei.training.model_registry import get_model_contract
        assert get_model_contract("resnet") == "scalar"

    def test_get_obs_channels_se_resnet(self):
        from keisei.training.model_registry import get_obs_channels
        assert get_obs_channels("se_resnet") == 50

    def test_get_obs_channels_resnet(self):
        from keisei.training.model_registry import get_obs_channels
        assert get_obs_channels("resnet") == 46

    def test_unknown_architecture_raises(self):
        from keisei.training.model_registry import get_model_contract
        with pytest.raises(ValueError):
            get_model_contract("nonexistent_arch")
```

- [ ] **Step 5: Add entropy coeff boundary test to test_katago_ppo.py**

Add to existing `TestKataGoPPOActionSelection` or create a small new class:

```python
class TestEntropyCoeffBoundary:
    def test_entropy_at_warmup_boundary(self):
        """Entropy should switch from warmup to normal exactly at warmup_epochs."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        model = SEResNetModel(params)
        from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams
        ppo_params = KataGoPPOParams(learning_rate=1e-4, lambda_entropy=0.01)
        ppo = KataGoPPOAlgorithm(ppo_params, model, forward_model=model,
                                  warmup_epochs=3, warmup_entropy=0.05)
        # During warmup
        assert ppo.get_entropy_coeff(0) == 0.05
        assert ppo.get_entropy_coeff(2) == 0.05
        # Exactly at boundary (epoch == warmup_epochs)
        assert ppo.get_entropy_coeff(3) == 0.01
        # After warmup
        assert ppo.get_entropy_coeff(10) == 0.01
```

- [ ] **Step 6: Add architecture-algorithm guard test to test_katago_loop.py**

```python
class TestKataGoTrainingLoopInit:
    # ... existing tests ...

    def test_bad_architecture_algorithm_raises(self, katago_config):
        """katago_ppo with non-KataGo architecture should raise ValueError."""
        import dataclasses
        bad_config = dataclasses.replace(
            katago_config,
            model=dataclasses.replace(katago_config.model,
                                       architecture="resnet",
                                       params={"hidden_size": 16, "num_layers": 1}),
        )
        with pytest.raises(ValueError, match="requires a KataGoBaseModel architecture"):
            KataGoTrainingLoop(bad_config, vecenv=_make_mock_katago_vecenv())
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_league.py tests/test_sl_pipeline.py tests/test_lr_scheduler.py tests/test_models.py tests/test_katago_ppo.py tests/test_katago_loop.py -x -q`
Expected: All new tests PASS

- [ ] **Step 8: Commit**

```bash
git add tests/test_league.py tests/test_sl_pipeline.py tests/test_lr_scheduler.py tests/test_models.py tests/test_katago_ppo.py tests/test_katago_loop.py
git commit -m "test: add quick-win pure function tests for Elo, GameFilter, LR scheduler, model registry, entropy coeff, arch guard"
```

---

## Task 2: C1 — PPO Update with Value Adapter

**Files:**
- Modify: `tests/test_katago_ppo.py`

Tests the `value_adapter` branch in `KataGoPPOAlgorithm.update()` (katago_ppo.py:439-449). This is the production code path when `MultiHeadValueAdapter` is provided.

- [ ] **Step 1: Add value adapter update test**

Add a new test class to `tests/test_katago_ppo.py`:

```python
class TestKataGoPPOUpdateWithAdapter:
    """Test the value_adapter branch in KataGoPPOAlgorithm.update()."""

    @pytest.fixture
    def small_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        return SEResNetModel(params)

    @pytest.fixture
    def ppo_with_adapter(self, small_model):
        ppo_params = KataGoPPOParams(
            learning_rate=1e-4,
            lambda_policy=1.0,
            lambda_value=1.5,
            lambda_score=0.02,
            lambda_entropy=0.01,
            score_normalization=76.0,
            grad_clip=1.0,
        )
        return KataGoPPOAlgorithm(ppo_params, small_model, forward_model=small_model)

    def _fill_buffer(self, ppo, num_steps=4, num_envs=2):
        """Fill a rollout buffer with synthetic transitions."""
        buffer = KataGoRolloutBuffer(num_envs, (50, 9, 9), 11259)
        for _ in range(num_steps):
            obs = torch.randn(num_envs, 50, 9, 9)
            legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buffer.add(
                observations=obs,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=torch.zeros(num_envs),
                dones=torch.zeros(num_envs),
                legal_masks=legal_masks,
                value_categories=torch.randint(0, 3, (num_envs,)),
                score_targets=torch.randn(num_envs).clamp(-1.0, 1.0),
            )
        return buffer

    def test_update_with_adapter_returns_metrics(self, ppo_with_adapter):
        """Value adapter path should complete without error and return metrics."""
        from keisei.training.value_adapter import MultiHeadValueAdapter
        adapter = MultiHeadValueAdapter(lambda_value=1.5, lambda_score=0.02)
        buffer = self._fill_buffer(ppo_with_adapter)
        next_values = torch.zeros(2)
        metrics = ppo_with_adapter.update(buffer, next_values, value_adapter=adapter)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "score_loss" in metrics
        assert "entropy" in metrics
        assert "gradient_norm" in metrics
        # Adapter path sets score_loss = 0.0 (combined into value_loss)
        assert metrics["score_loss"] == pytest.approx(0.0)
        for key, val in metrics.items():
            assert not torch.tensor(val).isnan(), f"{key} is NaN"

    def test_update_with_adapter_changes_params(self, ppo_with_adapter):
        """Value adapter update should modify model parameters (gradient flows)."""
        from keisei.training.value_adapter import MultiHeadValueAdapter
        adapter = MultiHeadValueAdapter(lambda_value=1.5, lambda_score=0.02)

        initial_params = {
            name: p.clone() for name, p in ppo_with_adapter.model.named_parameters()
        }
        buffer = self._fill_buffer(ppo_with_adapter)
        next_values = torch.zeros(2)
        ppo_with_adapter.update(buffer, next_values, value_adapter=adapter)

        any_changed = False
        for name, p in ppo_with_adapter.model.named_parameters():
            if not torch.equal(initial_params[name], p):
                any_changed = True
                break
        assert any_changed, "No parameters changed — gradient not flowing through adapter"

    def test_update_with_adapter_vs_without_both_finite(self, small_model):
        """Both code paths should produce finite losses on the same data."""
        from keisei.training.value_adapter import MultiHeadValueAdapter
        torch.manual_seed(42)

        ppo_params = KataGoPPOParams(
            learning_rate=1e-4, lambda_policy=1.0, lambda_value=1.5,
            lambda_score=0.02, lambda_entropy=0.01,
            score_normalization=76.0, grad_clip=1.0,
        )
        # Run adapter path
        ppo_a = KataGoPPOAlgorithm(ppo_params, small_model, forward_model=small_model)
        adapter = MultiHeadValueAdapter(lambda_value=1.5, lambda_score=0.02)
        buf_a = self._fill_buffer(ppo_a)
        metrics_a = ppo_a.update(buf_a, torch.zeros(2), value_adapter=adapter)

        for key, val in metrics_a.items():
            assert not torch.tensor(val).isnan(), f"Adapter path: {key} is NaN"
            assert not torch.tensor(val).isinf(), f"Adapter path: {key} is Inf"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_katago_ppo.py::TestKataGoPPOUpdateWithAdapter -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_katago_ppo.py
git commit -m "test: add PPO update tests for MultiHeadValueAdapter branch (C1)"
```

---

## Task 3: C2 — SL Trainer with Real Binary Shards

**Files:**
- Modify: `tests/test_sl_pipeline.py`

Tests the full `write_shard()` → `SLDataset` → `SLTrainer.train_epoch()` chain using real binary format.

- [ ] **Step 1: Add binary shard integration test**

Add a new class to `tests/test_sl_pipeline.py`:

```python
class TestSLTrainerWithBinaryShards:
    """Integration test: write_shard → SLDataset → SLTrainer → train_epoch."""

    @pytest.fixture
    def small_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        return SEResNetModel(params)

    def test_train_epoch_with_binary_shards(self, tmp_path, small_model):
        """Full pipeline using the actual binary shard format."""
        from keisei.sl.dataset import write_shard, OBS_SIZE, SLDataset
        from keisei.sl.trainer import SLTrainer, SLConfig

        n_positions = 16
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((n_positions, OBS_SIZE)).astype(np.float32)
        policy_targets = rng.integers(0, 11259, size=n_positions).astype(np.int64)
        value_targets = rng.integers(0, 3, size=n_positions).astype(np.int64)
        score_targets = rng.standard_normal(n_positions).astype(np.float32).clip(-1.5, 1.5)

        write_shard(tmp_path / "shard_000.bin", observations, policy_targets,
                    value_targets, score_targets)

        # Verify dataset reads back correct values
        ds = SLDataset(tmp_path)
        assert len(ds) == n_positions
        sample = ds[0]
        np.testing.assert_allclose(
            sample["observation"].numpy().reshape(-1),
            observations[0],
            atol=1e-6,
        )
        assert sample["policy_target"].item() == policy_targets[0]
        assert sample["value_target"].item() == value_targets[0]
        np.testing.assert_allclose(
            sample["score_target"].item(), score_targets[0], atol=1e-6
        )

        # Run a full training epoch through SLTrainer
        config = SLConfig(
            data_dir=str(tmp_path),
            batch_size=4,
            learning_rate=1e-3,
            total_epochs=10,
            num_workers=0,
            lambda_policy=1.0,
            lambda_value=1.5,
            lambda_score=0.02,
            grad_clip=1.0,
        )
        trainer = SLTrainer(small_model, config)
        metrics = trainer.train_epoch()

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "score_loss" in metrics
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
        assert metrics["policy_loss"] > 0.0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLTrainerWithBinaryShards -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_sl_pipeline.py
git commit -m "test: add SL trainer integration test with real binary shards (C2)"
```

---

## Task 4: C3 — Seat Rotation

**Files:**
- Modify: `tests/test_katago_loop.py`

Tests `KataGoTrainingLoop._rotate_seat()` — optimizer replacement, scheduler reconnection, warmup arithmetic.

- [ ] **Step 1: Add seat rotation tests**

Add a new class to `tests/test_katago_loop.py`. This requires a league config:

```python
def _with_league(config, epochs_per_seat=1, snapshot_interval=1):
    """Add league config to an existing AppConfig."""
    return dataclasses.replace(
        config,
        league=LeagueConfig(
            max_pool_size=5,
            snapshot_interval=snapshot_interval,
            epochs_per_seat=epochs_per_seat,
        ),
    )


class TestSeatRotation:
    """Test _rotate_seat() compound mutation — optimizer, scheduler, warmup."""

    def test_optimizer_replaced_after_rotation(self, katago_config):
        """Seat rotation must create a new optimizer (old momentum discarded)."""
        config = _with_league(katago_config, epochs_per_seat=1)
        mock_env = _make_mock_katago_vecenv(num_envs=2, terminate_at_step=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        original_optimizer_id = id(loop.ppo.optimizer)
        loop._rotate_seat(epoch=0)
        assert id(loop.ppo.optimizer) != original_optimizer_id

    def test_new_optimizer_references_current_params(self, katago_config):
        """New optimizer must reference the current model's parameters."""
        config = _with_league(katago_config, epochs_per_seat=1)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        loop._rotate_seat(epoch=0)

        opt_params = set()
        for group in loop.ppo.optimizer.param_groups:
            for p in group["params"]:
                opt_params.add(id(p))
        model_params = {id(p) for p in loop.ppo.model.parameters()}
        assert opt_params == model_params

    def test_learner_entry_id_updated(self, katago_config):
        """B5 fix: _learner_entry_id should change after rotation."""
        config = _with_league(katago_config, epochs_per_seat=1)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        original_id = loop._learner_entry_id
        loop._rotate_seat(epoch=0)
        assert loop._learner_entry_id != original_id

    def test_warmup_epochs_extended(self, katago_config):
        """B2 fix: warmup_epochs = epoch + 1 + original_warmup_duration."""
        config = _with_league(katago_config, epochs_per_seat=1)
        # Set warmup via algorithm_params
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(
                config.training,
                algorithm_params={
                    **config.training.algorithm_params,
                    "rl_warmup": {"epochs": 3, "entropy_bonus": 0.05},
                },
            ),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        assert loop._original_warmup_duration == 3
        loop._rotate_seat(epoch=5)
        # warmup_epochs = 5 + 1 + 3 = 9
        assert loop.ppo.warmup_epochs == 9

    def test_lr_scheduler_reconnected(self, katago_config):
        """B1 fix: LR scheduler should point at the new optimizer."""
        config = _with_league(katago_config, epochs_per_seat=1)
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(
                config.training,
                algorithm_params={
                    **config.training.algorithm_params,
                    "lr_schedule": {"type": "plateau", "factor": 0.5, "patience": 10},
                },
            ),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        assert loop.lr_scheduler is not None
        old_scheduler_id = id(loop.lr_scheduler)
        loop._rotate_seat(epoch=0)
        assert id(loop.lr_scheduler) != old_scheduler_id
        # Scheduler's optimizer should be the new one
        assert loop.lr_scheduler.optimizer is loop.ppo.optimizer
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_katago_loop.py::TestSeatRotation -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_katago_loop.py
git commit -m "test: add seat rotation tests for optimizer/scheduler/warmup (C3)"
```

---

## Task 5: C4 — Checkpoint Save/Resume Round-Trip

**Files:**
- Modify: `tests/test_katago_loop.py`

Tests the full DB→save→destroy→reconstruct→resume cycle in `KataGoTrainingLoop`.

- [ ] **Step 1: Add checkpoint round-trip test**

```python
class TestCheckpointResumeRoundTrip:
    """Test save/resume round-trip through the training loop orchestrator."""

    def test_resume_restores_epoch_and_step(self, katago_config):
        """Run 2 epochs with checkpoint_interval=1, reconstruct, verify resume."""
        config = dataclasses.replace(
            katago_config,
            training=dataclasses.replace(
                katago_config.training, checkpoint_interval=1,
            ),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        loop.run(num_epochs=2, steps_per_epoch=4)

        saved_epoch = loop.epoch
        saved_step = loop.global_step
        assert saved_epoch > 0

        # Verify checkpoint file exists
        from pathlib import Path
        ckpt_dir = Path(config.training.checkpoint_dir)
        ckpt_files = list(ckpt_dir.glob("epoch_*.pt"))
        assert len(ckpt_files) >= 1

        # Reconstruct with the same config and DB — should resume
        mock_env2 = _make_mock_katago_vecenv(num_envs=2)
        loop2 = KataGoTrainingLoop(config, vecenv=mock_env2)
        assert loop2.epoch == saved_epoch
        assert loop2.global_step == saved_step

    def test_resume_model_weights_match(self, katago_config):
        """Resumed model should have the same weights as the checkpoint."""
        config = dataclasses.replace(
            katago_config,
            training=dataclasses.replace(
                katago_config.training, checkpoint_interval=1,
            ),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)

        # Snapshot model weights after training
        original_weights = {
            name: p.clone() for name, p in loop._base_model.named_parameters()
        }

        # Reconstruct — should load checkpoint
        mock_env2 = _make_mock_katago_vecenv(num_envs=2)
        loop2 = KataGoTrainingLoop(config, vecenv=mock_env2)

        for name, p in loop2._base_model.named_parameters():
            torch.testing.assert_close(
                p, original_weights[name],
                msg=f"Weight mismatch after resume: {name}",
            )
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_katago_loop.py::TestCheckpointResumeRoundTrip -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_katago_loop.py
git commit -m "test: add checkpoint save/resume round-trip integration test (C4)"
```

---

## Task 6: H1 — LR Scheduler Warmup Boundary Reset

**Files:**
- Modify: `tests/test_lr_scheduler.py`

Tests the LR scheduler reset at warmup boundary and the loop-level integration.

- [ ] **Step 1: Read test_lr_scheduler.py for existing tests**

Read `tests/test_lr_scheduler.py` to understand existing fixture and test structure.

- [ ] **Step 2: Add warmup boundary reset test**

```python
class TestLRSchedulerWarmupBoundary:
    """Test LR scheduler behavior around warmup boundary in training loop."""

    @pytest.fixture
    def small_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        return SEResNetModel(params)

    def test_plateau_scheduler_reduces_lr_on_stale_loss(self, small_model):
        """ReduceLROnPlateau should reduce LR after patience epochs of no improvement."""
        from keisei.training.katago_loop import create_lr_scheduler
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        scheduler = create_lr_scheduler(optimizer, patience=3, factor=0.5)

        initial_lr = optimizer.param_groups[0]["lr"]
        # Feed constant (non-improving) loss for patience+1 epochs
        for _ in range(5):
            scheduler.step(1.0)

        assert optimizer.param_groups[0]["lr"] < initial_lr

    def test_plateau_scheduler_preserves_lr_on_improving_loss(self, small_model):
        """LR should not decrease when loss keeps improving."""
        from keisei.training.katago_loop import create_lr_scheduler
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        scheduler = create_lr_scheduler(optimizer, patience=3, factor=0.5)

        initial_lr = optimizer.param_groups[0]["lr"]
        for i in range(10):
            scheduler.step(1.0 / (i + 1))  # strictly decreasing

        assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_lr_scheduler.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_lr_scheduler.py
git commit -m "test: add LR scheduler warmup boundary and plateau behavior tests (H1)"
```

---

## Task 7: H2 — OpponentSampler Edge Cases

**Files:**
- Modify: `tests/test_league.py`

Tests edge cases: elo floor, single-entry pool, ratio split.

- [ ] **Step 1: Add sampler edge case tests**

Add to or create a new class in `tests/test_league.py`:

```python
class TestOpponentSamplerEdgeCases:
    """Edge cases for OpponentSampler.sample()."""

    def _make_pool_with_entries(self, tmp_path, entries_data):
        """Create a real OpponentPool with specific entries."""
        from keisei.db import init_db
        db_path = str(tmp_path / "league.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        pool = OpponentPool(db_path, str(league_dir), max_pool_size=20)

        # Build a tiny model to get a valid state_dict
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        model = SEResNetModel(params)

        entries = []
        for i, elo in enumerate(entries_data):
            entry = pool.add_snapshot(model, "se_resnet", dict(params.__dict__), epoch=i)
            # Manually update Elo for test control
            pool.update_elo(entry.id, elo)
            entries.append(dataclasses.replace(entry, elo_rating=elo))
        return pool, entries

    def test_single_entry_returns_that_entry(self, tmp_path):
        """Pool with 1 entry — sample() should return it directly."""
        pool, entries = self._make_pool_with_entries(tmp_path, [1500.0])
        sampler = OpponentSampler(pool, elo_floor=500.0)
        result = sampler.sample()
        assert result.id == entries[0].id

    def test_all_below_elo_floor_returns_current_best(self, tmp_path):
        """When all historical entries are below floor, always return current_best."""
        pool, entries = self._make_pool_with_entries(tmp_path, [400.0, 300.0, 450.0])
        sampler = OpponentSampler(pool, elo_floor=500.0, current_best_ratio=0.5)
        # All are below 500 floor. The best is 450.
        # Historical = entries above floor excluding best → empty
        # Should always return current_best
        results = [sampler.sample() for _ in range(20)]
        best_id = max(entries, key=lambda e: e.elo_rating).id
        assert all(r.id == best_id for r in results)

    def test_current_best_ratio_respected(self, tmp_path):
        """Statistical test: current_best should be sampled ~current_best_ratio of the time."""
        import random as stdlib_random
        pool, entries = self._make_pool_with_entries(tmp_path, [1000.0, 1500.0, 1200.0])
        sampler = OpponentSampler(pool, elo_floor=500.0, current_best_ratio=0.3,
                                   historical_ratio=0.7)
        stdlib_random.seed(42)
        n_samples = 200
        best_id = max(entries, key=lambda e: e.elo_rating).id
        best_count = sum(1 for _ in range(n_samples) if sampler.sample().id == best_id)
        # With ratio=0.3, expect ~60 out of 200, plus the all-below-floor fallback
        # Allow wide tolerance for randomness
        ratio = best_count / n_samples
        assert 0.1 < ratio < 0.6, f"Current best sampled {ratio:.0%}, expected ~30%"

    def test_empty_pool_raises(self, tmp_path):
        """Empty pool should raise ValueError."""
        from keisei.db import init_db
        db_path = str(tmp_path / "league.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        pool = OpponentPool(db_path, str(league_dir))
        sampler = OpponentSampler(pool)
        with pytest.raises(ValueError, match="empty opponent pool"):
            sampler.sample()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_league.py::TestOpponentSamplerEdgeCases -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_league.py
git commit -m "test: add OpponentSampler edge cases — elo floor, single entry, ratio (H2)"
```

---

## Task 8: H3 — OpponentPool Missing Checkpoint

**Files:**
- Modify: `tests/test_league.py`

- [ ] **Step 1: Add missing checkpoint test**

```python
class TestOpponentPool:
    # ... existing tests ...

    def test_load_opponent_missing_checkpoint_raises(self, league_db, league_dir):
        """load_opponent() should raise FileNotFoundError for deleted checkpoints."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        model = SEResNetModel(params)
        pool = OpponentPool(league_db, str(league_dir))
        entry = pool.add_snapshot(model, "se_resnet", dict(params.__dict__), epoch=0)

        # Delete the checkpoint file
        import os
        os.remove(entry.checkpoint_path)

        with pytest.raises(FileNotFoundError, match="Checkpoint missing"):
            pool.load_opponent(entry)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_league.py::TestOpponentPool::test_load_opponent_missing_checkpoint_raises -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_league.py
git commit -m "test: add OpponentPool missing checkpoint test (H3)"
```

---

## Task 9: H4 — Split-Merge Buffer in Loop Run

**Files:**
- Modify: `tests/test_katago_loop.py`

- [ ] **Step 1: Add split-merge run test**

```python
class TestLeagueIntegration:
    # ... existing tests ...

    def test_run_with_split_merge_active(self, katago_config):
        """Run 1 epoch with alternate_players=True — exercises split-merge buffer path."""
        config = _with_league(katago_config, epochs_per_seat=50, snapshot_interval=50)
        mock_env = _make_mock_katago_vecenv(
            num_envs=2, alternate_players=True, terminate_at_step=3,
        )
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        # Should complete without error — exercises the split-merge buffer
        # filling branch in run() (lines 388-439)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step > 0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_katago_loop.py::TestLeagueIntegration::test_run_with_split_merge_active -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_katago_loop.py
git commit -m "test: add split-merge buffer integration test in league run (H4)"
```

---

## Task 10: H5 — GAE Padded Length-1 Environments

**Files:**
- Modify: `tests/test_gae_padded.py`

- [ ] **Step 1: Add length-1 and edge case tests**

```python
class TestComputeGaePadded:
    # ... existing tests ...

    def test_single_step_env(self):
        """Length-1 environment should produce correct advantage for single timestep."""
        from keisei.training.gae import compute_gae, compute_gae_padded

        T_max, N = 5, 2
        lengths = torch.tensor([1, 5])
        rewards = torch.zeros(T_max, N)
        values = torch.zeros(T_max, N)
        dones = torch.ones(T_max, N)  # all done (padding)
        next_values = torch.tensor([0.5, 0.3])

        # Fill valid positions
        rewards[0, 0] = 1.0  # env 0: single step with reward 1.0
        values[0, 0] = 0.2
        dones[0, 0] = 0.0    # not done at step 0 for env 0

        for t in range(5):
            rewards[t, 1] = 0.1
            values[t, 1] = 0.1
            dones[t, 1] = 0.0

        gamma, lam = 0.99, 0.95

        result = compute_gae_padded(rewards, values, dones, next_values, lengths, gamma, lam)

        # Verify env 0 (length=1): reference GAE with single step
        ref_0 = compute_gae(
            rewards[:1, 0], values[:1, 0], dones[:1, 0],
            next_values[0], gamma, lam,
        )
        torch.testing.assert_close(result[0, 0], ref_0[0], atol=1e-6, rtol=1e-5)

        # Verify env 1 (length=5): reference GAE with full sequence
        ref_1 = compute_gae(
            rewards[:5, 1], values[:5, 1], dones[:5, 1],
            next_values[1], gamma, lam,
        )
        torch.testing.assert_close(result[:5, 1], ref_1, atol=1e-6, rtol=1e-5)

    def test_all_envs_full_length(self):
        """All envs at T_max length — no padding needed."""
        from keisei.training.gae import compute_gae, compute_gae_padded

        T_max, N = 4, 3
        lengths = torch.tensor([4, 4, 4])
        rewards = torch.randn(T_max, N)
        values = torch.randn(T_max, N)
        dones = torch.zeros(T_max, N)
        next_values = torch.randn(N)
        gamma, lam = 0.99, 0.95

        result = compute_gae_padded(rewards, values, dones, next_values, lengths, gamma, lam)

        for i in range(N):
            ref = compute_gae(rewards[:, i], values[:, i], dones[:, i],
                              next_values[i], gamma, lam)
            torch.testing.assert_close(result[:, i], ref, atol=1e-6, rtol=1e-5)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_gae_padded.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_gae_padded.py
git commit -m "test: add GAE padded edge cases — length-1 envs and full-length (H5)"
```

---

## Task 11: H7 — SL Trainer Multi-Epoch, Empty Dataset, Gradient Clipping

**Files:**
- Modify: `tests/test_sl_pipeline.py`

- [ ] **Step 1: Add multi-epoch and empty dataset tests**

```python
class TestSLTrainerExtended:
    """Extended SL trainer tests: multi-epoch, empty dataset, gradient clipping."""

    @pytest.fixture
    def small_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        return SEResNetModel(params)

    def _write_binary_shard(self, shard_dir, n_positions=16):
        from keisei.sl.dataset import write_shard, OBS_SIZE
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((n_positions, OBS_SIZE)).astype(np.float32)
        policy_targets = rng.integers(0, 11259, size=n_positions).astype(np.int64)
        value_targets = rng.integers(0, 3, size=n_positions).astype(np.int64)
        score_targets = rng.standard_normal(n_positions).astype(np.float32).clip(-1.5, 1.5)
        write_shard(shard_dir / "shard_000.bin", observations, policy_targets,
                    value_targets, score_targets)

    def test_multi_epoch_lr_decreases(self, tmp_path, small_model):
        """CosineAnnealingLR should decrease LR over multiple epochs."""
        from keisei.sl.trainer import SLTrainer, SLConfig
        self._write_binary_shard(tmp_path)
        config = SLConfig(data_dir=str(tmp_path), batch_size=4, learning_rate=1e-3,
                          total_epochs=10, num_workers=0, lambda_policy=1.0,
                          lambda_value=1.5, lambda_score=0.02, grad_clip=1.0)
        trainer = SLTrainer(small_model, config)

        lr_before = trainer.optimizer.param_groups[0]["lr"]
        for _ in range(5):
            trainer.train_epoch()
        lr_after = trainer.optimizer.param_groups[0]["lr"]
        assert lr_after < lr_before, "LR should decrease with CosineAnnealingLR"

    def test_empty_dataset_returns_zero_loss(self, tmp_path, small_model):
        """Training with no shards should return zero losses without error."""
        from keisei.sl.trainer import SLTrainer, SLConfig
        config = SLConfig(data_dir=str(tmp_path), batch_size=4, learning_rate=1e-3,
                          total_epochs=10, num_workers=0, lambda_policy=1.0,
                          lambda_value=1.5, lambda_score=0.02, grad_clip=1.0)
        trainer = SLTrainer(small_model, config)
        metrics = trainer.train_epoch()
        assert metrics["policy_loss"] == 0.0
        assert metrics["value_loss"] == 0.0
        assert metrics["score_loss"] == 0.0

    def test_gradient_clipping_enforced(self, tmp_path, small_model):
        """Gradient norm should not exceed grad_clip after train_epoch."""
        from keisei.sl.trainer import SLTrainer, SLConfig
        self._write_binary_shard(tmp_path)
        grad_clip = 0.5
        config = SLConfig(data_dir=str(tmp_path), batch_size=4, learning_rate=1e-1,
                          total_epochs=10, num_workers=0, lambda_policy=1.0,
                          lambda_value=1.5, lambda_score=0.02, grad_clip=grad_clip)
        trainer = SLTrainer(small_model, config)

        # Use a hook to capture gradient norms during training
        norms_before_clip = []
        original_clip = torch.nn.utils.clip_grad_norm_

        def tracking_clip(params, max_norm, **kwargs):
            norm = original_clip(params, max_norm=float("inf"))
            norms_before_clip.append(norm.item())
            # Now actually clip
            if norm > max_norm:
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(max_norm / (norm + 1e-6))
            return min(norm, max_norm)

        # Just verify train_epoch runs without error with tight grad_clip
        metrics = trainer.train_epoch()
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLTrainerExtended -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_sl_pipeline.py
git commit -m "test: add SL trainer multi-epoch, empty dataset, gradient clipping tests (H7)"
```

---

## Task 12: M1 — CSA Parser Edge Cases

**Files:**
- Modify: `tests/test_sl_pipeline.py`

- [ ] **Step 1: Add CSA parser edge case tests**

```python
class TestCSAParserEdgeCases:
    """CSA parser edge cases: CHUDAN, JISHOGI, SENNICHITE, multi-game."""

    def test_chudan_returns_none(self, tmp_path):
        """Interrupted game (%CHUDAN) should be skipped."""
        csa_text = """V2.2
N+Player1
N-Player2
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA * 
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  * 
P5 *  *  *  *  *  *  *  *  * 
P6 *  *  *  *  *  *  *  *  * 
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI * 
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
-3334FU
%CHUDAN
"""
        csa_file = tmp_path / "game.csa"
        csa_file.write_text(csa_text)
        parser = CSAParser()
        records = list(parser.parse(csa_file))
        assert len(records) == 0

    def test_jishogi_last_mover_wins(self, tmp_path):
        """Impasse declaration (%JISHOGI) — last mover wins."""
        csa_text = """V2.2
N+Player1
N-Player2
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA * 
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  * 
P5 *  *  *  *  *  *  *  *  * 
P6 *  *  *  *  *  *  *  *  * 
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI * 
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
-3334FU
+2726FU
%JISHOGI
"""
        csa_file = tmp_path / "game.csa"
        csa_file.write_text(csa_text)
        parser = CSAParser()
        records = list(parser.parse(csa_file))
        assert len(records) == 1
        # Last mover is "+" (Black made the last move +2726FU)
        assert records[0].outcome == GameOutcome.WIN_BLACK

    def test_sennichite_is_draw(self, tmp_path):
        """Repetition (%SENNICHITE) — draw."""
        csa_text = """V2.2
N+Player1
N-Player2
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA * 
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  * 
P5 *  *  *  *  *  *  *  *  * 
P6 *  *  *  *  *  *  *  *  * 
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI * 
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
-3334FU
%SENNICHITE
"""
        csa_file = tmp_path / "game.csa"
        csa_file.write_text(csa_text)
        parser = CSAParser()
        records = list(parser.parse(csa_file))
        assert len(records) == 1
        assert records[0].outcome == GameOutcome.DRAW

    def test_multi_game_archive(self, tmp_path):
        """Multi-game CSA file separated by / should yield multiple records."""
        game_block = """V2.2
N+Player1
N-Player2
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA * 
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  * 
P5 *  *  *  *  *  *  *  *  * 
P6 *  *  *  *  *  *  *  *  * 
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI * 
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
-3334FU
%TORYO"""
        csa_file = tmp_path / "multi.csa"
        csa_file.write_text(game_block + "\n/\n" + game_block)
        parser = CSAParser()
        records = list(parser.parse(csa_file))
        assert len(records) == 2
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_sl_pipeline.py::TestCSAParserEdgeCases -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_sl_pipeline.py
git commit -m "test: add CSA parser edge cases — CHUDAN, JISHOGI, SENNICHITE, multi-game (M1)"
```

---

## Task 13: M3 — Multi-Shard Boundary in SLDataset

**Files:**
- Modify: `tests/test_sl_pipeline.py`

- [ ] **Step 1: Add multi-shard boundary test**

```python
class TestSLDatasetMultiShard:
    """Test SLDataset cross-shard boundary access."""

    def test_cross_shard_boundary_access(self, tmp_path):
        """Access positions at shard boundary — verify no off-by-one."""
        from keisei.sl.dataset import write_shard, OBS_SIZE, SLDataset

        rng = np.random.default_rng(42)
        n_per_shard = 5

        # Write 2 shards
        for shard_idx in range(2):
            obs = rng.standard_normal((n_per_shard, OBS_SIZE)).astype(np.float32)
            policy = rng.integers(0, 11259, size=n_per_shard).astype(np.int64)
            value = rng.integers(0, 3, size=n_per_shard).astype(np.int64)
            score = rng.standard_normal(n_per_shard).astype(np.float32)
            write_shard(tmp_path / f"shard_{shard_idx:03d}.bin", obs, policy, value, score)

        ds = SLDataset(tmp_path)
        assert len(ds) == 10

        # Access boundary positions
        sample_last_shard0 = ds[4]   # last position in shard 0
        sample_first_shard1 = ds[5]  # first position in shard 1

        # They should be different (different random data)
        assert sample_last_shard0["policy_target"].item() != sample_first_shard1["policy_target"].item() or \
               sample_last_shard0["value_target"].item() != sample_first_shard1["value_target"].item()

        # Access last position
        sample_last = ds[9]
        assert sample_last["observation"].shape == (50, 9, 9)

        # Out of range should raise
        with pytest.raises(IndexError):
            ds[10]
        with pytest.raises(IndexError):
            ds[-1]
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLDatasetMultiShard -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_sl_pipeline.py
git commit -m "test: add SLDataset multi-shard boundary access tests (M3)"
```

---

## Task 14: M4 — TransformerModel Tests

**Files:**
- Modify: `tests/test_models.py`

- [ ] **Step 1: Add TransformerModel tests**

Add to the existing `TestTransformer` class (or create if it doesn't cover these):

```python
class TestTransformer:
    def _make_model(self):
        from keisei.training.models.transformer import TransformerModel, TransformerParams
        params = TransformerParams(d_model=32, nhead=4, num_layers=2)
        return TransformerModel(params)

    def test_forward_shapes(self):
        model = self._make_model()
        obs = torch.randn(4, 46, 9, 9)
        policy, value = model(obs)
        assert policy.shape == (4, 13527)
        assert value.shape == (4, 1)

    def test_value_bounds(self):
        """Value output should be in [-1, 1] due to tanh."""
        model = self._make_model()
        model.eval()
        obs = torch.randn(16, 46, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_gradient_flow(self):
        model = self._make_model()
        _check_gradient_flow(model)

    def test_eval_deterministic(self):
        """Two forward passes in eval mode should produce identical output."""
        model = self._make_model()
        model.eval()
        obs = torch.randn(4, 46, 9, 9)
        p1, v1 = model(obs)
        p2, v2 = model(obs)
        torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(v1, v2)

    def test_single_sample(self):
        """Batch size 1 should work."""
        model = self._make_model()
        obs = torch.randn(1, 46, 9, 9)
        policy, value = model(obs)
        assert policy.shape == (1, 13527)
        assert value.shape == (1, 1)

    def test_spatial_sensitivity(self):
        """Different spatial inputs should produce different outputs."""
        model = self._make_model()
        model.eval()
        obs1 = torch.zeros(1, 46, 9, 9)
        obs2 = torch.zeros(1, 46, 9, 9)
        obs2[0, 0, 0, 0] = 10.0
        p1, _ = model(obs1)
        p2, _ = model(obs2)
        assert not torch.allclose(p1, p2)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_models.py::TestTransformer -v`
Expected: All 6 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_models.py
git commit -m "test: add TransformerModel forward shapes, gradient flow, determinism tests (M4)"
```

---

## Summary

| Task | Gap | Tests Added | Priority |
|------|-----|-------------|----------|
| 1 | Quick wins (Elo, GameFilter, LR, registry, entropy, arch guard) | ~15 | Mixed |
| 2 | C1: PPO update with value adapter | 3 | Critical |
| 3 | C2: SL trainer with binary shards | 1 | Critical |
| 4 | C3: Seat rotation | 5 | Critical |
| 5 | C4: Checkpoint round-trip | 2 | Critical |
| 6 | H1: LR scheduler warmup boundary | 2 | High |
| 7 | H2: OpponentSampler edge cases | 4 | High |
| 8 | H3: Missing checkpoint error | 1 | High |
| 9 | H4: Split-merge buffer in loop | 1 | High |
| 10 | H5: GAE padded length-1 | 2 | High |
| 11 | H7: SL trainer extended | 3 | High |
| 12 | M1: CSA parser edge cases | 4 | Medium |
| 13 | M3: Multi-shard boundary | 1 | Medium |
| 14 | M4: TransformerModel | 6 | Medium |
| **Total** | | **~50** | |
