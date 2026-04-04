import pytest
import torch

from keisei.training.algorithm_registry import (
    VALID_ALGORITHMS,
    validate_algorithm_params,
)
from keisei.training.model_registry import (
    VALID_ARCHITECTURES,
    build_model,
    get_model_contract,
    get_obs_channels,
    validate_model_params,
)


class TestModelRegistry:
    def test_valid_architectures_match_config(self) -> None:
        assert "resnet" in VALID_ARCHITECTURES
        assert "mlp" in VALID_ARCHITECTURES
        assert "transformer" in VALID_ARCHITECTURES

    def test_build_resnet(self) -> None:
        model = build_model("resnet", {"hidden_size": 32, "num_layers": 2})
        obs = torch.randn(2, 50, 9, 9)
        p, v = model(obs)
        assert p.shape == (2, 11259)
        assert v.shape == (2, 1)

    def test_build_mlp(self) -> None:
        model = build_model("mlp", {"hidden_sizes": [128, 64]})
        obs = torch.randn(2, 50, 9, 9)
        p, v = model(obs)
        assert p.shape == (2, 11259)

    def test_build_transformer(self) -> None:
        model = build_model("transformer", {"d_model": 32, "nhead": 4, "num_layers": 2})
        obs = torch.randn(2, 50, 9, 9)
        p, v = model(obs)
        assert p.shape == (2, 11259)

    def test_unknown_architecture_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("nonexistent", {})

    def test_bad_params_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            validate_model_params("resnet", {"bad_key": 99})

    def test_missing_params_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            validate_model_params("resnet", {})


class TestModelContractTypes:
    def test_resnet_is_scalar(self):
        assert get_model_contract("resnet") == "scalar"

    def test_mlp_is_scalar(self):
        assert get_model_contract("mlp") == "scalar"

    def test_transformer_is_scalar(self):
        assert get_model_contract("transformer") == "scalar"

    def test_se_resnet_is_multi_head(self):
        assert get_model_contract("se_resnet") == "multi_head"

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError):
            get_model_contract("nonexistent")

    def test_obs_channels_scalar(self):
        assert get_obs_channels("resnet") == 50

    def test_obs_channels_multi_head(self):
        assert get_obs_channels("se_resnet") == 50


class TestAlgorithmRegistry:
    def test_katago_ppo_in_registry(self) -> None:
        assert "katago_ppo" in VALID_ALGORITHMS

    def test_legacy_ppo_not_in_registry(self) -> None:
        """Legacy 'ppo' was removed — it passed validation but crashed at loop init."""
        assert "ppo" not in VALID_ALGORITHMS

    def test_unknown_algorithm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            validate_algorithm_params("nonexistent", {})
