from pathlib import Path
from unittest.mock import MagicMock

from keisei.config_schema import AppConfig
from keisei.evaluation.opponents import OpponentPool
from keisei.evaluation.opponents.elo_registry import EloRegistry
from keisei.training.callbacks import EvaluationCallback
from keisei.training.metrics_manager import MetricsManager
from keisei.utils import PolicyOutputMapper
from tests.evaluation.conftest import make_test_config


class DummyAgent:
    def __init__(self):
        self.model = MagicMock()

    def save_model(self, path, *_args):
        Path(path).write_text("dummy")


class DummyTrainer:
    def __init__(self, tmp_path):
        self.config: AppConfig = make_test_config()
        self.config.evaluation.enable_periodic_evaluation = True
        self.config.evaluation.previous_model_pool_size = 2
        self.config.evaluation.elo_registry_path = str(tmp_path / "elo.json")
        self.policy_output_mapper = PolicyOutputMapper()
        self.model_dir = str(tmp_path)
        self.run_name = "test"
        self.global_timestep = 0
        self.total_episodes_completed = 0
        self.agent = DummyAgent()
        self.agent.model = MagicMock()  # Add model attribute to agent
        self.metrics_manager = MetricsManager()
        self.evaluation_manager = MagicMock()
        self.evaluation_manager.opponent_pool = OpponentPool(
            pool_size=2, elo_registry_path=self.config.evaluation.elo_registry_path
        )
        self.evaluation_manager.evaluate_current_agent = self.fake_eval_current
        self.evaluation_elo_snapshot = None
        self.log_both = lambda *a, **kw: None
        # Add one previous checkpoint
        ck = Path(tmp_path / "old.pth")
        ck.write_text("a")
        self.evaluation_manager.opponent_pool.add_checkpoint(ck)

    def fake_eval_current(self, agent):
        """Simulate evaluation and update Elo ratings."""
        registry = EloRegistry(Path(self.config.evaluation.elo_registry_path))
        agent_id = self.run_name or "test"  # Ensure non-None value

        # Use the opponent from the pool (simulated)
        opponent_ckpt = self.evaluation_manager.opponent_pool.sample()
        if opponent_ckpt:
            opponent_id = Path(opponent_ckpt).name
        else:
            opponent_id = "opponent"

        registry.update_ratings(agent_id, opponent_id, ["agent_win"])
        registry.save()

        # Create a mock evaluation result that matches expected interface
        mock_result = MagicMock()
        mock_result.summary_stats = MagicMock()
        mock_result.summary_stats.win_rate = 1.0
        mock_result.summary_stats.loss_rate = 0.0
        return mock_result


def test_evaluation_callback_updates_elo(tmp_path):
    trainer = DummyTrainer(tmp_path)
    cb = EvaluationCallback(trainer.config.evaluation, interval=1)
    # Type ignore since DummyTrainer is a test mock
    cb.on_step_end(trainer)  # type: ignore
    reg = EloRegistry(Path(trainer.config.evaluation.elo_registry_path))
    assert trainer.evaluation_elo_snapshot is not None
    assert trainer.evaluation_elo_snapshot["current_rating"] != 1500.0
    assert len(reg.ratings) == 2  # agent and opponent


def test_elo_tracking_works_without_logging(tmp_path):
    """Regression test for P1-13: Elo tracking must work when log_both is None.

    Previously, the Elo registry update was nested inside an
    `if trainer.log_both is not None:` guard, so disabling logging
    silently stopped all Elo tracking.
    """
    trainer = DummyTrainer(tmp_path)
    trainer.log_both = None  # Simulate logging unavailable
    cb = EvaluationCallback(trainer.config.evaluation, interval=1)
    cb.on_step_end(trainer)  # type: ignore
    reg = EloRegistry(Path(trainer.config.evaluation.elo_registry_path))
    assert trainer.evaluation_elo_snapshot is not None
    assert trainer.evaluation_elo_snapshot["current_rating"] != 1500.0
    assert len(reg.ratings) == 2
