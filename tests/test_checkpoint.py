from pathlib import Path

import pytest
import torch

from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.models.resnet import ResNetModel, ResNetParams


@pytest.fixture
def model() -> ResNetModel:
    return ResNetModel(ResNetParams(hidden_size=16, num_layers=1))


def test_save_and_load_round_trip(tmp_path: Path, model: ResNetModel) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, epoch=10, step=1000)
    assert path.exists()

    loaded = load_checkpoint(path, model, optimizer)
    assert loaded["epoch"] == 10
    assert loaded["step"] == 1000


def test_model_weights_preserved(tmp_path: Path, model: ResNetModel) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    obs = torch.randn(1, 50, 9, 9)
    with torch.no_grad():
        original_policy, original_value = model(obs)

    save_checkpoint(path, model, optimizer, epoch=1, step=100)

    for p in model.parameters():
        p.data.add_(torch.randn_like(p))

    load_checkpoint(path, model, optimizer)

    with torch.no_grad():
        restored_policy, restored_value = model(obs)

    assert torch.allclose(original_policy, restored_policy, atol=1e-6)
    assert torch.allclose(original_value, restored_value, atol=1e-6)


def test_load_nonexistent_raises(tmp_path: Path, model: ResNetModel) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model, optimizer)


# ---------------------------------------------------------------------------
# High gap: cross-architecture checkpoint loading
# ---------------------------------------------------------------------------


def test_load_checkpoint_wrong_architecture_raises(tmp_path: Path) -> None:
    """Loading a checkpoint saved from one architecture into another should fail."""
    from keisei.training.models.mlp import MLPModel, MLPParams

    resnet = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
    optimizer_resnet = torch.optim.Adam(resnet.parameters(), lr=1e-3)
    path = tmp_path / "resnet.pt"
    save_checkpoint(path, resnet, optimizer_resnet, epoch=1, step=100)

    mlp = MLPModel(MLPParams(hidden_sizes=[16]))
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError):
        load_checkpoint(path, mlp, optimizer_mlp)


def test_load_corrupted_checkpoint_raises(tmp_path: Path, model: ResNetModel) -> None:
    """A truncated/corrupted checkpoint file should raise an error."""
    path = tmp_path / "corrupted.pt"
    path.write_bytes(b"not a valid checkpoint file")

    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(Exception):
        load_checkpoint(path, model, optimizer)


# ---------------------------------------------------------------------------
# T4 — SE-ResNet checkpoint round-trip (production architecture)
# ---------------------------------------------------------------------------


class TestSEResNetCheckpointRoundTrip:
    """Checkpoint round-trip with the actual production architecture."""

    @pytest.fixture
    def se_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        return SEResNetModel(params)

    def test_se_resnet_weights_preserved(self, tmp_path: Path, se_model) -> None:
        """Save SE-ResNet checkpoint, load into fresh model, verify outputs match."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        optimizer = torch.optim.Adam(se_model.parameters(), lr=2e-4)
        # Run a training step to populate optimizer state
        obs = torch.randn(2, 50, 9, 9)
        output = se_model(obs)
        loss = output.policy_logits.sum() + output.value_logits.sum() + output.score_lead.sum()
        loss.backward()
        optimizer.step()

        # Capture outputs after training step
        se_model.eval()
        with torch.no_grad():
            ref_output = se_model(obs)

        path = tmp_path / "se_resnet.pt"
        save_checkpoint(path, se_model, optimizer, epoch=5, step=500,
                        architecture="se_resnet")

        # Load into fresh model
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        fresh = SEResNetModel(params)
        fresh_opt = torch.optim.Adam(fresh.parameters(), lr=2e-4)
        meta = load_checkpoint(path, fresh, fresh_opt, expected_architecture="se_resnet")

        assert meta["epoch"] == 5
        assert meta["step"] == 500

        fresh.eval()
        with torch.no_grad():
            fresh_output = fresh(obs)

        assert torch.allclose(ref_output.policy_logits, fresh_output.policy_logits, atol=1e-6)
        assert torch.allclose(ref_output.value_logits, fresh_output.value_logits, atol=1e-6)
        assert torch.allclose(ref_output.score_lead, fresh_output.score_lead, atol=1e-6)

    def test_se_resnet_optimizer_state_restored(self, tmp_path: Path, se_model) -> None:
        """Adam momentum buffers should be restored for SE-ResNet."""
        optimizer = torch.optim.Adam(se_model.parameters(), lr=2e-4)

        # Generate momentum
        for _ in range(3):
            obs = torch.randn(2, 50, 9, 9)
            output = se_model(obs)
            loss = output.policy_logits.sum() + output.value_logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        first_key = list(optimizer.state.keys())[0]
        original_step = optimizer.state[first_key]["step"].item()

        path = tmp_path / "se_resnet_opt.pt"
        save_checkpoint(path, se_model, optimizer, epoch=3, step=30)

        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        fresh = SEResNetModel(SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        ))
        fresh_opt = torch.optim.Adam(fresh.parameters(), lr=2e-4)
        load_checkpoint(path, fresh, fresh_opt)

        first_key2 = list(fresh_opt.state.keys())[0]
        assert fresh_opt.state[first_key2]["step"].item() == original_step

    def test_architecture_mismatch_detected(self, tmp_path: Path, se_model) -> None:
        """Loading SE-ResNet checkpoint with wrong expected_architecture should raise."""
        optimizer = torch.optim.Adam(se_model.parameters())
        path = tmp_path / "se_arch_check.pt"
        save_checkpoint(path, se_model, optimizer, epoch=1, step=0,
                        architecture="se_resnet")

        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        fresh = SEResNetModel(SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        ))
        fresh_opt = torch.optim.Adam(fresh.parameters())
        with pytest.raises(ValueError, match="architecture mismatch"):
            load_checkpoint(path, fresh, fresh_opt, expected_architecture="resnet")


# ---------------------------------------------------------------------------
# skip_optimizer parameter
# ---------------------------------------------------------------------------


def test_skip_optimizer_leaves_optimizer_fresh(tmp_path: Path, model: ResNetModel) -> None:
    """load_checkpoint(skip_optimizer=True) should NOT load optimizer state."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run a training step to populate optimizer momentum buffers
    obs = torch.randn(1, 50, 9, 9)
    policy, value = model(obs)
    loss = policy.sum() + value.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Save checkpoint (now contains non-empty optimizer state)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=5, step=500)

    # Create fresh model and optimizer
    fresh_model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)

    # Verify fresh optimizer has no state
    assert len(fresh_optimizer.state) == 0

    # Load with skip_optimizer=True
    meta = load_checkpoint(path, fresh_model, fresh_optimizer, skip_optimizer=True)

    # Optimizer state should still be empty
    assert len(fresh_optimizer.state) == 0
    assert meta["epoch"] == 5
    assert meta["step"] == 500


def test_skip_optimizer_still_loads_model_weights(tmp_path: Path, model: ResNetModel) -> None:
    """skip_optimizer=True should still restore model weights correctly."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs = torch.randn(1, 50, 9, 9)
    with torch.no_grad():
        original_policy, original_value = model(obs)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1, step=100)

    # Corrupt model weights
    for p in model.parameters():
        p.data.add_(torch.randn_like(p))

    fresh_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    load_checkpoint(path, model, fresh_optimizer, skip_optimizer=True)

    with torch.no_grad():
        restored_policy, restored_value = model(obs)

    assert torch.allclose(original_policy, restored_policy, atol=1e-6)
    assert torch.allclose(original_value, restored_value, atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end SL→RL checkpoint transition
# ---------------------------------------------------------------------------


class TestSLtoRLCheckpointTransition:
    """End-to-end: SL checkpoint loaded into RL training has fresh optimizer."""

    @pytest.fixture
    def se_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        return SEResNetModel(params)

    def test_sl_checkpoint_loaded_for_rl_has_fresh_optimizer(
        self, tmp_path: Path, se_model
    ) -> None:
        """Save SL checkpoint with warm optimizer, load for RL, verify optimizer is fresh."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        sl_optimizer = torch.optim.Adam(se_model.parameters(), lr=1e-3)

        # Train a few steps to populate Adam momentum buffers
        for _ in range(3):
            obs = torch.randn(2, 50, 9, 9)
            output = se_model(obs)
            loss = output.policy_logits.sum() + output.value_logits.sum()
            loss.backward()
            sl_optimizer.step()
            sl_optimizer.zero_grad()

        # Confirm SL optimizer has populated state
        assert len(sl_optimizer.state) > 0

        # Save SL checkpoint (includes optimizer state)
        ckpt_path = tmp_path / "sl_checkpoint.pt"
        save_checkpoint(
            ckpt_path, se_model, sl_optimizer, epoch=30, step=9000,
            architecture="se_resnet",
        )

        # Create fresh RL model and optimizer (simulating what KataGoTrainingLoop does)
        rl_params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        rl_model = SEResNetModel(rl_params)
        rl_optimizer = torch.optim.Adam(rl_model.parameters(), lr=3e-4)

        # Load with skip_optimizer=True (SL→RL transition)
        meta = load_checkpoint(
            ckpt_path, rl_model, rl_optimizer,
            expected_architecture="se_resnet",
            skip_optimizer=True,
        )

        # Model weights should be loaded
        se_model.eval()
        rl_model.eval()
        test_obs = torch.randn(1, 50, 9, 9)
        with torch.no_grad():
            sl_out = se_model(test_obs)
            rl_out = rl_model(test_obs)
        assert torch.allclose(sl_out.policy_logits, rl_out.policy_logits, atol=1e-6)

        # Optimizer should be EMPTY (fresh) — no momentum from SL training
        assert len(rl_optimizer.state) == 0

        # Training metadata should be carried over
        assert meta["epoch"] == 30
        assert meta["step"] == 9000
