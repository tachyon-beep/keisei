"""
Tests for Pydantic config schema validation.

Validates field constraints, custom validators, and model-level settings
defined in keisei.config_schema.
"""

import pytest
from pydantic import ValidationError

from keisei.config_schema import (
    AppConfig,
    DisplayConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
    WebUIConfig,
)


def make_app_config(**overrides):
    """Build a minimal valid AppConfig, with optional section overrides."""
    defaults = dict(
        env=EnvConfig(),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(),
        logging=LoggingConfig(),
        wandb=WandBConfig(),
        parallel=ParallelConfig(),
    )
    defaults.update(overrides)
    return AppConfig(**defaults)


# ---------------------------------------------------------------------------
# TrainingConfig defaults
# ---------------------------------------------------------------------------

class TestTrainingConfigDefaults:
    def test_default_training_config_creates_successfully(self):
        """Default TrainingConfig should instantiate without errors."""
        config = TrainingConfig()
        assert config.learning_rate == 3e-4
        assert config.minibatch_size == 64
        assert config.lr_schedule_type is None

    def test_default_training_config_has_expected_ppo_defaults(self):
        """Verify PPO-related defaults are sensible."""
        config = TrainingConfig()
        assert config.ppo_epochs == 10
        assert config.clip_epsilon == 0.2
        assert config.gamma == 0.99


# ---------------------------------------------------------------------------
# EvaluationConfig defaults
# ---------------------------------------------------------------------------

class TestEvaluationConfigDefaults:
    def test_default_evaluation_config_creates_successfully(self):
        """Default EvaluationConfig should instantiate without errors."""
        config = EvaluationConfig()
        assert config.num_games == 20
        assert config.strategy == "single_opponent"
        assert config.log_level == "INFO"


# ---------------------------------------------------------------------------
# AppConfig defaults
# ---------------------------------------------------------------------------

class TestAppConfigDefaults:
    def test_default_app_config_creates_successfully(self):
        """AppConfig with all required sections should create without errors."""
        config = make_app_config()
        assert config.env.device == "cpu"
        assert config.training.learning_rate == 3e-4
        assert config.evaluation.num_games == 20


# ---------------------------------------------------------------------------
# learning_rate validator
# ---------------------------------------------------------------------------

class TestLearningRateValidator:
    def test_learning_rate_zero_raises_validation_error(self):
        """learning_rate=0 must be rejected (must be positive)."""
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0)

    def test_learning_rate_negative_raises_validation_error(self):
        """learning_rate=-1 must be rejected."""
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-1)

    def test_learning_rate_valid_value_works(self):
        """A valid positive learning rate should be accepted."""
        config = TrainingConfig(learning_rate=1e-4)
        assert config.learning_rate == 1e-4


# ---------------------------------------------------------------------------
# lr_schedule_type validator
# ---------------------------------------------------------------------------

class TestLrScheduleTypeValidator:
    def test_lr_schedule_type_invalid_raises_validation_error(self):
        """An unsupported schedule type must be rejected."""
        with pytest.raises(ValidationError, match="lr_schedule_type must be one of"):
            TrainingConfig(lr_schedule_type="invalid")

    def test_lr_schedule_type_none_works(self):
        """None disables scheduling and should be accepted."""
        config = TrainingConfig(lr_schedule_type=None)
        assert config.lr_schedule_type is None

    def test_lr_schedule_type_cosine_works(self):
        """'cosine' is a valid schedule type."""
        config = TrainingConfig(lr_schedule_type="cosine")
        assert config.lr_schedule_type == "cosine"


# ---------------------------------------------------------------------------
# lr_schedule_step_on validator
# ---------------------------------------------------------------------------

class TestLrScheduleStepOnValidator:
    def test_lr_schedule_step_on_invalid_raises_validation_error(self):
        """An unsupported step_on value must be rejected."""
        with pytest.raises(
            ValidationError, match="lr_schedule_step_on must be 'epoch' or 'update'"
        ):
            TrainingConfig(lr_schedule_step_on="invalid")


# ---------------------------------------------------------------------------
# minibatch_size constraint (gt=1)
# ---------------------------------------------------------------------------

class TestMinibatchSizeConstraint:
    def test_minibatch_size_one_raises_validation_error(self):
        """minibatch_size=1 violates the gt=1 constraint."""
        with pytest.raises(ValidationError):
            TrainingConfig(minibatch_size=1)

    def test_minibatch_size_zero_raises_validation_error(self):
        """minibatch_size=0 violates the gt=1 constraint."""
        with pytest.raises(ValidationError):
            TrainingConfig(minibatch_size=0)


# ---------------------------------------------------------------------------
# EvaluationConfig validators
# ---------------------------------------------------------------------------

class TestEvaluationValidators:
    def test_num_games_zero_raises_validation_error(self):
        """num_games=0 must be rejected (must be positive)."""
        with pytest.raises(ValidationError, match="num_games must be positive"):
            EvaluationConfig(num_games=0)

    def test_num_games_negative_raises_validation_error(self):
        """num_games=-1 must be rejected."""
        with pytest.raises(ValidationError, match="num_games must be positive"):
            EvaluationConfig(num_games=-1)

    def test_strategy_invalid_raises_validation_error(self):
        """An unrecognised strategy must be rejected."""
        with pytest.raises(ValidationError):
            EvaluationConfig(strategy="invalid_strategy")

    def test_strategy_single_opponent_works(self):
        """'single_opponent' is a valid strategy."""
        config = EvaluationConfig(strategy="single_opponent")
        assert config.strategy == "single_opponent"

    def test_log_level_invalid_raises_validation_error(self):
        """An invalid log level must be rejected."""
        with pytest.raises(ValidationError, match="log_level must be one of"):
            EvaluationConfig(log_level="invalid")

    def test_log_level_lowercase_is_normalised_to_uppercase(self):
        """The log_level validator should normalise 'debug' to 'DEBUG'."""
        config = EvaluationConfig(log_level="debug")
        assert config.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# AppConfig extra fields
# ---------------------------------------------------------------------------

class TestAppConfigExtraFields:
    def test_app_config_rejects_unknown_extra_fields(self):
        """AppConfig has extra='forbid', so unknown keys must raise."""
        with pytest.raises(ValidationError):
            AppConfig(
                env=EnvConfig(),
                training=TrainingConfig(),
                evaluation=EvaluationConfig(),
                logging=LoggingConfig(),
                wandb=WandBConfig(),
                parallel=ParallelConfig(),
                nonexistent_section="should_fail",
            )


# ---------------------------------------------------------------------------
# torch_compile_mode validator
# ---------------------------------------------------------------------------

class TestTorchCompileModeValidator:
    def test_torch_compile_mode_invalid_raises_validation_error(self):
        """An unsupported compile mode must be rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(torch_compile_mode="invalid")
