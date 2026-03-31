import pytest
import torch

from keisei.training.algorithm_registry import (
    VALID_ALGORITHMS,
    validate_algorithm_params,
)
from keisei.training.model_registry import (
    VALID_ARCHITECTURES,
    build_model,
    validate_model_params,
)


class TestModelRegistry:
    def test_valid_architectures_match_config(self) -> None:
        assert "resnet" in VALID_ARCHITECTURES
        assert "mlp" in VALID_ARCHITECTURES
        assert "transformer" in VALID_ARCHITECTURES

    def test_build_resnet(self) -> None:
        model = build_model("resnet", {"hidden_size": 32, "num_layers": 2})
        obs = torch.randn(2, 46, 9, 9)
        p, v = model(obs)
        assert p.shape == (2, 13527)
        assert v.shape == (2, 1)

    def test_build_mlp(self) -> None:
        model = build_model("mlp", {"hidden_sizes": [128, 64]})
        obs = torch.randn(2, 46, 9, 9)
        p, v = model(obs)
        assert p.shape == (2, 13527)

    def test_build_transformer(self) -> None:
        model = build_model("transformer", {"d_model": 32, "nhead": 4, "num_layers": 2})
        obs = torch.randn(2, 46, 9, 9)
        p, v = model(obs)
        assert p.shape == (2, 13527)

    def test_unknown_architecture_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("nonexistent", {})

    def test_bad_params_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            validate_model_params("resnet", {"bad_key": 99})

    def test_missing_params_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            validate_model_params("resnet", {})


class TestAlgorithmRegistry:
    def test_ppo_in_registry(self) -> None:
        assert "ppo" in VALID_ALGORITHMS

    def test_unknown_algorithm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            validate_algorithm_params("nonexistent", {})
