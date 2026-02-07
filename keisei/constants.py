"""
constants.py: Application-wide constants for the Keisei Shogi RL system.

This module centralizes magic numbers and hardcoded values used throughout the codebase
to improve maintainability and readability. Constants are organized by functional area.
"""

# Shogi game constants
SHOGI_BOARD_SIZE = 9
SHOGI_BOARD_SQUARES = SHOGI_BOARD_SIZE * SHOGI_BOARD_SIZE  # 81
MOVE_COUNT_NORMALIZATION_FACTOR = 512.0

# Action space and observation constants
ALTERNATIVE_ACTION_SPACE = 6480  # Alternative/reduced action space
FULL_ACTION_SPACE = 13527  # Full Shogi action space (12960 + 567)
CORE_OBSERVATION_CHANNELS = 46  # Standard 46-channel observation
EXTENDED_OBSERVATION_CHANNELS = 51  # Extended observation with additional features

# Observation channel layout constants (OBS_CURR_*, OBS_OPP_*, etc.)
# are defined in shogi.shogi_core_definitions — import from there.

# Training constants

# GAE and advantage computation

# Model architecture defaults

# Rendering and display defaults

# Edge case test constants
TEST_CONFIG_STEPS_PER_EPOCH = 32
TEST_CONFIG_TOWER_DEPTH = 3
TEST_CONFIG_TOWER_WIDTH = 64


class GameTerminationReason:
    """Enumeration of common game termination reasons."""

    INVALID_MOVE = "invalid_move"
    STALEMATE = "stalemate"
    POLICY_ERROR = "policy_error"


# Parallel training defaults

# Timeout and retry constants

# Communication and buffer sizes

# Display update frequencies

# Logging and formatting

# Default evaluation parameters

# Test-specific magic numbers
TEST_THRESHOLD_HIGH = 0.9
TEST_THRESHOLD_MID = 0.5
TEST_THRESHOLD_LOW = 0.1
TEST_EPSILON = 0.01
TEST_MAX_DEPENDENCY_ISSUES = 20  # Updated from magic number 15

# Common test dimensions
TEST_BUFFER_SIZE = 4  # Small buffer size for testing
TEST_SMALL_BUFFER = 8
TEST_BATCH_SIZE = 32

# Default seed values
SEED_OFFSET_MULTIPLIER = 1000

# Default computation settings

# File patterns and extensions
CHECKPOINT_FILE_PATTERN = "*.pt"
CONFIG_FILE_EXTENSION = ".yaml"
LOG_FILE_EXTENSION = ".log"

# Directory and path defaults

# Common mathematical values used in RL
EPSILON_SMALL = 1e-8  # Small epsilon for numerical stability
EPSILON_MEDIUM = 1e-6  # Medium epsilon for comparisons
EPSILON_LARGE = 1e-4  # Large epsilon for loose comparisons

# Normalization and scaling factors
PERCENTAGE_SCALE = 100.0
PROBABILITY_THRESHOLD = 0.5

# Resource limits
MAX_MEMORY_WARNING_THRESHOLD = 0.9  # 90% memory usage warning
MAX_QUEUE_WAIT_TIME = 30.0  # Maximum time to wait for queue operations
MAX_FILE_RETRY_ATTEMPTS = 5  # Maximum file operation retries

# Error thresholds
MAX_CONSECUTIVE_FAILURES = 10  # Maximum consecutive operation failures
MIN_SUCCESS_RATE = 0.8  # Minimum acceptable success rate

# Test constants
TEST_MEDIUM_BUFFER_SIZE = 8  # Medium buffer size for testing
TEST_ODD_BUFFER_SIZE = 5  # Odd buffer size for testing uneven splits
TEST_UNEVEN_MINIBATCH_SIZE = 3  # Minibatch size that doesn't divide evenly
TEST_WIN_RATE_THRESHOLD = 0.9  # Win rate threshold for tests
TEST_LOSS_THRESHOLD = 0.1  # Loss threshold for tests
TEST_ACCURACY_THRESHOLD = 0.01  # Accuracy threshold for tests
TEST_LOG_PROB_MULTIPLIER = 0.1  # Log probability multiplier for test data
TEST_LAST_VALUE = 1.2  # Last value for advantage computation
TEST_PPO_EPOCHS = 2  # PPO epochs for testing
TEST_HIGH_LEARNING_RATE = 1.0  # High learning rate for testing
TEST_GRADIENT_CLIP_NORM = 0.5  # Gradient clipping norm for testing
TEST_SMALL_MINIBATCH = 2  # Small minibatch size for testing
TEST_ADVANTAGE_STD_THRESHOLD = 10.0  # Threshold for advantage std dev testing
TEST_KL_DIVERGENCE_THRESHOLD = 10.0  # KL divergence threshold for testing
TEST_VALUE_DEFAULT = 0.5  # Default test value for experience buffer

# Learning rate scheduler test constants
TEST_SCHEDULER_LEARNING_RATE = 0.001  # Initial learning rate for scheduler tests
TEST_SCHEDULER_FINAL_FRACTION = 0.1  # Final learning rate fraction
TEST_SCHEDULER_TOTAL_TIMESTEPS = 1000  # Total timesteps for scheduler test
TEST_SCHEDULER_STEPS_PER_EPOCH = 100  # Steps per epoch for scheduler test

# Test action indices and edge case constants
TEST_SINGLE_LEGAL_ACTION_INDEX = 42  # Single legal action index for testing
TEST_VERY_SMALL_LEARNING_RATE = 1e-10  # Very small learning rate for testing
TEST_TINY_LEARNING_RATE = 1e-8  # Tiny learning rate for extreme tests

# Test reward and value data (for deterministic testing)
TEST_HIGH_VARIANCE_REWARDS = [100.0, -50.0, 75.0, -25.0, 50.0, -75.0, 25.0, -100.0]
TEST_HIGH_VARIANCE_VALUES = [10.0, 5.0, 8.0, 3.0, 6.0, 2.0, 4.0, 1.0]
TEST_MIXED_REWARDS = [1.0, -0.5, 2.0, 0.0, 1.5, -1.0, 0.5, 2.5]
TEST_MIXED_VALUES = [0.8, 0.2, 1.5, 0.1, 1.0, -0.3, 0.3, 2.0]
TEST_EXTREME_REWARDS = [
    100.0,
    -100.0,
    50.0,
    -50.0,
]  # Extreme rewards for gradient testing

# Edge case test constants
TEST_PARAMETER_FILL_VALUE = 999.0  # Fill value for model parameter testing
TEST_TIMEOUT_SECONDS = 10.0  # Timeout for async operations
TEST_NUM_WORKERS = 4  # Number of workers for parallel tests
TEST_SYNC_INTERVAL = 100  # Sync interval for distributed training
TEST_DEMO_MODE_DELAY = 0.5  # Delay for demo mode testing
TEST_WATCH_LOG_FREQ = 1000  # Logging frequency for wandb tests
TEST_EVALUATION_INTERVAL = 1000  # Evaluation interval for training tests
TEST_NEGATIVE_LEARNING_RATE = -1.0  # Invalid negative learning rate
TEST_ZERO_LEARNING_RATE = 0.0  # Invalid zero learning rate
TEST_NEGATIVE_GAMMA = -0.5  # Invalid negative gamma
TEST_GAMMA_GREATER_THAN_ONE = 1.5  # Invalid gamma > 1
TEST_NEGATIVE_CLIP_EPSILON = -0.1  # Invalid negative clip epsilon
TEST_ZERO_CLIP_EPSILON = 0.0  # Invalid zero clip epsilon
TEST_LARGE_INITIAL_LR = 1.0  # Large initial learning rate for scheduler tests
TEST_SMALL_MASK_SIZE = 100  # Small mask size for dimension mismatch tests
TEST_SINGLE_EPOCH = 1  # Single epoch for minimal training tests
TEST_SINGLE_GAME = 1  # Single game for evaluation tests
TEST_WEIGHT_DECAY_ZERO = 0.0  # Zero weight decay for optimizer tests
TEST_LOG_PROB_VALUE = 0.1  # Test log probability value
TEST_VALUE_HALF = 0.5  # Half value commonly used in tests
TEST_GAE_LAMBDA_DEFAULT = 0.95  # Default GAE lambda value for tests
TEST_GAMMA_NINE_TENTHS = 0.9  # Gamma value of 0.9 for testing
TEST_SCHEDULER_STEP_SIZE = 1  # Step size for step scheduler
TEST_SCHEDULER_GAMMA = 0.1  # Gamma for step scheduler
TEST_ETA_MIN_FRACTION = 0.1  # Eta min fraction for cosine scheduler
TEST_T_MAX = 10  # T max for cosine scheduler
TEST_REWARD_VALUE = 1.0  # Default reward value for testing
TEST_ADVANTAGE_GAMMA_ZERO = 0.0  # Zero gamma for advantage computation testing
TEST_GLOBAL_TIMESTEP_ZERO = 0  # Zero global timestep for testing
TEST_GLOBAL_TIMESTEP_NEGATIVE = -1  # Negative global timestep for error checking

# Additional edge case constants that were missing
TEST_LARGE_MASK_SIZE = 20000  # Large mask size for dimension mismatch tests
TEST_STEP_THREE_DONE = 3  # Step 3 is done (for range(4) with 0-indexing)
TEST_MINIMAL_BUFFER_SIZE = 2  # Minimal buffer size for edge case testing
