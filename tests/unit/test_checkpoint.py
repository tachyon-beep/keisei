"""Unit tests for keisei.utils.checkpoint: load_checkpoint_with_padding."""

import copy

import pytest
import torch
import torch.nn as nn

from keisei.core.neural_network import ActorCritic
from keisei.training.models.resnet_tower import ActorCriticResTower
from keisei.utils.checkpoint import load_checkpoint_with_padding

# ---------------------------------------------------------------------------
# Constants -- small sizes for fast tests
# ---------------------------------------------------------------------------
NUM_ACTIONS = 64  # small action space to keep models tiny
BOARD_SIZE = 9
CHANNELS_46 = 46
CHANNELS_51 = 51
TOWER_DEPTH = 1
TOWER_WIDTH = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_resnet(input_channels: int) -> ActorCriticResTower:
    """Build a minimal ResNet model (has .stem layer)."""
    return ActorCriticResTower(
        input_channels=input_channels,
        num_actions_total=NUM_ACTIONS,
        tower_depth=TOWER_DEPTH,
        tower_width=TOWER_WIDTH,
    )


def _make_cnn(input_channels: int) -> ActorCritic:
    """Build a minimal CNN model (no .stem layer)."""
    return ActorCritic(
        input_channels=input_channels,
        num_actions_total=NUM_ACTIONS,
    )


def _dummy_input(input_channels: int) -> torch.Tensor:
    """Create a deterministic observation tensor for prediction comparison."""
    torch.manual_seed(0)
    return torch.randn(1, input_channels, BOARD_SIZE, BOARD_SIZE)


# ---------------------------------------------------------------------------
# 1. Normal case -- matching channel count
# ---------------------------------------------------------------------------
class TestMatchingChannels:
    """Loading a checkpoint whose channel count matches the model exactly."""

    def test_load_matching_channels_weights_identical(self):
        """Weights are identical after loading a checkpoint with the same channel count."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_46)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        for key in model_src.state_dict():
            assert torch.equal(
                model_dst.state_dict()[key], model_src.state_dict()[key]
            ), f"Weight mismatch on key '{key}' after loading matching-channel checkpoint"

    def test_load_matching_channels_predictions_match(self):
        """Predictions are identical after loading a checkpoint with matching channels."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_46)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        obs = _dummy_input(CHANNELS_46)
        model_src.eval()
        model_dst.eval()
        with torch.no_grad():
            logits_src, val_src = model_src(obs)
            logits_dst, val_dst = model_dst(obs)

        assert torch.allclose(logits_src, logits_dst, atol=1e-6), (
            "Policy logits differ after loading matching-channel checkpoint"
        )
        assert torch.allclose(val_src, val_dst, atol=1e-6), (
            "Value predictions differ after loading matching-channel checkpoint"
        )


# ---------------------------------------------------------------------------
# 2. Padding case -- 46-channel checkpoint into 51-channel model
# ---------------------------------------------------------------------------
class TestPaddingChannels:
    """Loading a smaller checkpoint into a model with more input channels."""

    def test_padding_first_channels_preserved(self):
        """First 46 input channels of stem.weight are preserved exactly after padding."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_51)

        src_stem = model_src.state_dict()["stem.weight"].clone()
        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_51)

        dst_stem = model_dst.state_dict()["stem.weight"]
        assert torch.equal(dst_stem[:, :CHANNELS_46, :, :], src_stem), (
            "First 46 channels should be identical to the source checkpoint"
        )

    def test_padding_extra_channels_are_zero(self):
        """Extra 5 channels (indices 46..50) are zero-filled after padding."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_51)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_51)

        dst_stem = model_dst.state_dict()["stem.weight"]
        extra = dst_stem[:, CHANNELS_46:, :, :]
        assert torch.all(extra == 0.0), (
            f"Padded channels should be zero; got max abs value {extra.abs().max().item()}"
        )

    def test_padding_stem_shape_correct(self):
        """After padding, stem.weight has the full expected shape."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_51)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_51)

        dst_stem = model_dst.state_dict()["stem.weight"]
        assert dst_stem.shape[1] == CHANNELS_51, (
            f"Expected {CHANNELS_51} input channels, got {dst_stem.shape[1]}"
        )


# ---------------------------------------------------------------------------
# 3. Truncation case -- 51-channel checkpoint into 46-channel model
# ---------------------------------------------------------------------------
class TestTruncationChannels:
    """Loading a larger checkpoint into a model with fewer input channels."""

    def test_truncation_first_channels_preserved(self):
        """First 46 channels of the 51-channel checkpoint are preserved after truncation."""
        model_src = _make_resnet(CHANNELS_51)
        model_dst = _make_resnet(CHANNELS_46)

        src_stem = model_src.state_dict()["stem.weight"].clone()
        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        dst_stem = model_dst.state_dict()["stem.weight"]
        assert torch.equal(dst_stem, src_stem[:, :CHANNELS_46, :, :]), (
            "Truncated stem should match the first 46 channels of the source"
        )

    def test_truncation_stem_shape_correct(self):
        """After truncation, stem.weight has the correct smaller shape."""
        model_src = _make_resnet(CHANNELS_51)
        model_dst = _make_resnet(CHANNELS_46)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        dst_stem = model_dst.state_dict()["stem.weight"]
        assert dst_stem.shape[1] == CHANNELS_46, (
            f"Expected {CHANNELS_46} input channels after truncation, got {dst_stem.shape[1]}"
        )


# ---------------------------------------------------------------------------
# 4. No stem layer -- model without .stem.weight
# ---------------------------------------------------------------------------
class TestNoStemLayer:
    """Models that lack a .stem.weight key should still load successfully."""

    def test_cnn_model_loads_without_stem(self):
        """ActorCritic (CNN) has no .stem layer; loading should not raise."""
        model_src = _make_cnn(CHANNELS_46)
        model_dst = _make_cnn(CHANNELS_46)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        # Should not raise
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

    def test_cnn_model_weights_loaded_correctly(self):
        """CNN model weights are loaded correctly even without a stem layer."""
        model_src = _make_cnn(CHANNELS_46)
        model_dst = _make_cnn(CHANNELS_46)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        for key in model_src.state_dict():
            assert torch.equal(
                model_dst.state_dict()[key], model_src.state_dict()[key]
            ), f"Weight mismatch on key '{key}' for CNN model"

    def test_cnn_model_predictions_match_after_load(self):
        """CNN predictions are identical after loading checkpoint."""
        model_src = _make_cnn(CHANNELS_46)
        model_dst = _make_cnn(CHANNELS_46)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        obs = _dummy_input(CHANNELS_46)
        model_src.eval()
        model_dst.eval()
        with torch.no_grad():
            logits_src, val_src = model_src(obs)
            logits_dst, val_dst = model_dst(obs)

        assert torch.allclose(logits_src, logits_dst, atol=1e-6)
        assert torch.allclose(val_src, val_dst, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Full state_dict format -- checkpoint IS the raw state_dict (no wrapper)
# ---------------------------------------------------------------------------
class TestRawStateDictFormat:
    """Checkpoint that is a raw state_dict without a 'model_state_dict' wrapper."""

    def test_raw_state_dict_loads_successfully(self):
        """A raw state_dict (no 'model_state_dict' key) loads without error."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_46)

        # Pass the raw state_dict directly -- no "model_state_dict" key
        checkpoint = dict(model_src.state_dict())
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

    def test_raw_state_dict_weights_loaded_correctly(self):
        """Weights are correct when loaded from a raw state_dict."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_46)

        checkpoint = dict(model_src.state_dict())
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        for key in model_src.state_dict():
            assert torch.equal(
                model_dst.state_dict()[key], model_src.state_dict()[key]
            ), f"Weight mismatch on key '{key}' with raw state_dict format"


# ---------------------------------------------------------------------------
# 6. Wrapped format -- checkpoint has "model_state_dict" key
# ---------------------------------------------------------------------------
class TestWrappedStateDictFormat:
    """Checkpoint that wraps the state_dict under 'model_state_dict'."""

    def test_wrapped_format_loads_successfully(self):
        """A wrapped checkpoint with 'model_state_dict' loads without error."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_46)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

    def test_wrapped_format_weights_loaded_correctly(self):
        """Weights are correct when loaded from a wrapped checkpoint."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_46)

        checkpoint = {"model_state_dict": model_src.state_dict()}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        for key in model_src.state_dict():
            assert torch.equal(
                model_dst.state_dict()[key], model_src.state_dict()[key]
            ), f"Weight mismatch on key '{key}' with wrapped format"

    def test_wrapped_format_extra_metadata_ignored(self):
        """Extra metadata keys alongside 'model_state_dict' are ignored."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_46)

        checkpoint = {
            "model_state_dict": model_src.state_dict(),
            "optimizer_state_dict": {"some": "data"},
            "global_timestep": 500,
        }
        # Should not raise despite extra keys
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        for key in model_src.state_dict():
            assert torch.equal(
                model_dst.state_dict()[key], model_src.state_dict()[key]
            ), f"Weight mismatch on key '{key}' with extra metadata in checkpoint"


# ---------------------------------------------------------------------------
# 7. Weight integrity -- non-stem weights unchanged after padding/truncation
# ---------------------------------------------------------------------------
class TestNonStemWeightIntegrity:
    """After padding or truncation, all non-stem weights must be loaded correctly."""

    def test_padding_nonstem_weights_unchanged(self):
        """Non-stem weights are identical to source after padding the stem."""
        model_src = _make_resnet(CHANNELS_46)
        model_dst = _make_resnet(CHANNELS_51)

        src_state = model_src.state_dict()
        checkpoint = {"model_state_dict": copy.deepcopy(src_state)}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_51)

        dst_state = model_dst.state_dict()
        for key in src_state:
            if key == "stem.weight":
                continue  # stem is expected to differ in shape
            assert torch.equal(dst_state[key], src_state[key]), (
                f"Non-stem weight '{key}' changed after padding"
            )

    def test_truncation_nonstem_weights_unchanged(self):
        """Non-stem weights are identical to source after truncating the stem."""
        model_src = _make_resnet(CHANNELS_51)
        model_dst = _make_resnet(CHANNELS_46)

        src_state = model_src.state_dict()
        checkpoint = {"model_state_dict": copy.deepcopy(src_state)}
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)

        dst_state = model_dst.state_dict()
        for key in src_state:
            if key == "stem.weight":
                continue  # stem is expected to differ in shape
            assert torch.equal(dst_state[key], src_state[key]), (
                f"Non-stem weight '{key}' changed after truncation"
            )


# ---------------------------------------------------------------------------
# 8. Roundtrip -- save -> load_with_padding -> predictions match
# ---------------------------------------------------------------------------
class TestRoundtrip:
    """Save a model to disk, then load via load_checkpoint_with_padding."""

    def test_roundtrip_matching_channels_predictions_identical(self, tmp_path):
        """save -> load_with_padding with matching channels gives identical predictions."""
        model_src = _make_resnet(CHANNELS_46)
        model_src.eval()

        obs = _dummy_input(CHANNELS_46)
        with torch.no_grad():
            logits_before, val_before = model_src(obs)

        # Save to disk
        ckpt_path = str(tmp_path / "roundtrip.pt")
        torch.save({"model_state_dict": model_src.state_dict()}, ckpt_path)

        # Load into a fresh model
        model_dst = _make_resnet(CHANNELS_46)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)
        model_dst.eval()

        with torch.no_grad():
            logits_after, val_after = model_dst(obs)

        assert torch.allclose(logits_before, logits_after, atol=1e-6), (
            "Policy logits differ after roundtrip save/load"
        )
        assert torch.allclose(val_before, val_after, atol=1e-6), (
            "Value predictions differ after roundtrip save/load"
        )

    def test_roundtrip_raw_state_dict_predictions_identical(self, tmp_path):
        """save raw state_dict -> load_with_padding gives identical predictions."""
        model_src = _make_resnet(CHANNELS_46)
        model_src.eval()

        obs = _dummy_input(CHANNELS_46)
        with torch.no_grad():
            logits_before, val_before = model_src(obs)

        # Save raw state_dict (no wrapper)
        ckpt_path = str(tmp_path / "roundtrip_raw.pt")
        torch.save(model_src.state_dict(), ckpt_path)

        model_dst = _make_resnet(CHANNELS_46)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)
        model_dst.eval()

        with torch.no_grad():
            logits_after, val_after = model_dst(obs)

        assert torch.allclose(logits_before, logits_after, atol=1e-6), (
            "Policy logits differ after raw-state-dict roundtrip"
        )
        assert torch.allclose(val_before, val_after, atol=1e-6), (
            "Value predictions differ after raw-state-dict roundtrip"
        )

    def test_roundtrip_cnn_model_predictions_identical(self, tmp_path):
        """Roundtrip with CNN model (no stem) gives identical predictions."""
        model_src = _make_cnn(CHANNELS_46)
        model_src.eval()

        obs = _dummy_input(CHANNELS_46)
        with torch.no_grad():
            logits_before, val_before = model_src(obs)

        ckpt_path = str(tmp_path / "roundtrip_cnn.pt")
        torch.save({"model_state_dict": model_src.state_dict()}, ckpt_path)

        model_dst = _make_cnn(CHANNELS_46)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        load_checkpoint_with_padding(model_dst, checkpoint, input_channels=CHANNELS_46)
        model_dst.eval()

        with torch.no_grad():
            logits_after, val_after = model_dst(obs)

        assert torch.allclose(logits_before, logits_after, atol=1e-6), (
            "CNN policy logits differ after roundtrip"
        )
        assert torch.allclose(val_before, val_after, atol=1e-6), (
            "CNN value predictions differ after roundtrip"
        )
