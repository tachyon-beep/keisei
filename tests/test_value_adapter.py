"""Tests for value-head adapters (scalar vs multi-head)."""

import torch
import pytest

from keisei.training.value_adapter import (
    MultiHeadValueAdapter,
    ScalarValueAdapter,
    get_value_adapter,
)


class TestScalarValueAdapter:
    def test_scalar_value_output(self):
        adapter = ScalarValueAdapter()
        value = torch.tensor([[0.5], [-0.3], [0.8]])
        scalar = adapter.scalar_value_from_output(value)
        assert scalar.shape == (3,)
        assert torch.allclose(scalar, torch.tensor([0.5, -0.3, 0.8]))

    def test_scalar_value_loss(self):
        adapter = ScalarValueAdapter()
        value = torch.tensor([[0.5], [-0.3]], requires_grad=True)
        returns = torch.tensor([0.8, -0.1])
        loss = adapter.compute_value_loss(value, returns, value_cats=None, score_targets=None)
        assert loss.item() > 0
        loss.backward()
        assert value.grad is not None

    def test_scalar_requires_returns(self):
        adapter = ScalarValueAdapter()
        value = torch.tensor([[0.5]])
        with pytest.raises(ValueError, match="requires returns"):
            adapter.compute_value_loss(value, returns=None, value_cats=None, score_targets=None)


class TestMultiHeadValueAdapter:
    def test_scalar_value_output(self):
        adapter = MultiHeadValueAdapter()
        value_logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        scalar = adapter.scalar_value_from_output(value_logits)
        assert scalar.shape == (2,)
        assert scalar[0] > 0  # P(W) > P(L)
        assert scalar[1] < 0  # P(L) > P(W)

    def test_multi_head_value_loss(self):
        adapter = MultiHeadValueAdapter()
        value_logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=True)
        value_cats = torch.tensor([0, 2])
        score_pred = torch.tensor([[0.5], [-0.3]], requires_grad=True)
        score_targets = torch.tensor([0.013, -0.013])
        loss = adapter.compute_value_loss(
            value_logits, returns=None,
            value_cats=value_cats, score_targets=score_targets,
            score_pred=score_pred,
        )
        assert loss.item() > 0
        loss.backward()
        assert value_logits.grad is not None

    def test_ignore_index_for_non_terminal(self):
        adapter = MultiHeadValueAdapter()
        value_logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
        value_cats = torch.tensor([-1, 1])
        score_pred = torch.tensor([[0.0], [0.0]])
        score_targets = torch.tensor([0.0, 0.0])
        loss = adapter.compute_value_loss(
            value_logits, returns=None,
            value_cats=value_cats, score_targets=score_targets,
            score_pred=score_pred,
        )
        assert loss.item() > 0


class TestGetValueAdapter:
    def test_returns_scalar(self):
        adapter = get_value_adapter(model_contract="scalar")
        assert isinstance(adapter, ScalarValueAdapter)

    def test_returns_multi_head(self):
        adapter = get_value_adapter(model_contract="multi_head")
        assert isinstance(adapter, MultiHeadValueAdapter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model contract"):
            get_value_adapter(model_contract="nonexistent")
