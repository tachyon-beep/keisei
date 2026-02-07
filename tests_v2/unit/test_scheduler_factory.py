"""
Tests for SchedulerFactory learning rate scheduler creation.

Validates that all supported scheduler types are correctly instantiated
and that stepping them modifies the optimizer's learning rate as expected.
"""

import pytest
import torch

from keisei.core.scheduler_factory import SchedulerFactory


@pytest.fixture
def optimizer():
    """Create a simple optimizer for scheduler tests."""
    model = torch.nn.Linear(10, 10)
    return torch.optim.Adam(model.parameters(), lr=0.001)


# ---------------------------------------------------------------------------
# Scheduler creation
# ---------------------------------------------------------------------------

class TestSchedulerCreation:
    def test_none_schedule_returns_none(self, optimizer):
        """schedule_type=None should return None (scheduling disabled)."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer, schedule_type=None, total_steps=100
        )
        assert scheduler is None

    def test_linear_creates_steppable_scheduler(self, optimizer):
        """'linear' should create a scheduler that can be stepped."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer, schedule_type="linear", total_steps=100
        )
        assert scheduler is not None
        scheduler.step()  # Should not raise

    def test_cosine_creates_steppable_scheduler(self, optimizer):
        """'cosine' should create a scheduler that can be stepped."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer, schedule_type="cosine", total_steps=100
        )
        assert scheduler is not None
        scheduler.step()

    def test_exponential_creates_steppable_scheduler(self, optimizer):
        """'exponential' should create a scheduler that can be stepped."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer, schedule_type="exponential", total_steps=100
        )
        assert scheduler is not None
        scheduler.step()

    def test_step_creates_steppable_scheduler(self, optimizer):
        """'step' should create a StepLR scheduler that can be stepped."""
        scheduler = SchedulerFactory.create_scheduler(
            optimizer, schedule_type="step", total_steps=100
        )
        assert scheduler is not None
        scheduler.step()

    def test_invalid_schedule_type_raises_value_error(self, optimizer):
        """An unsupported schedule type must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            SchedulerFactory.create_scheduler(
                optimizer, schedule_type="invalid", total_steps=100
            )


# ---------------------------------------------------------------------------
# Scheduler behaviour
# ---------------------------------------------------------------------------

class TestSchedulerBehaviour:
    def test_linear_scheduler_decreases_learning_rate(self, optimizer):
        """Linear scheduler should reduce the effective LR over many steps."""
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler = SchedulerFactory.create_scheduler(
            optimizer, schedule_type="linear", total_steps=100
        )
        for _ in range(50):
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr < initial_lr

    def test_stepping_changes_optimizer_lr(self, optimizer):
        """After stepping any scheduler, the optimizer LR should have changed."""
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler = SchedulerFactory.create_scheduler(
            optimizer, schedule_type="exponential", total_steps=100
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr != initial_lr
