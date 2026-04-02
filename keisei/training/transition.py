"""SL→RL transition orchestrator.

Wires SLTrainer output into KataGoTrainingLoop input — the glue code
that automates the most important seam in the training pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from keisei.config import (
    AppConfig,
    DisplayConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)
from keisei.db import init_db, write_training_state
from keisei.sl.trainer import SLConfig, SLTrainer
from keisei.training.checkpoint import save_checkpoint
from keisei.training.katago_loop import KataGoTrainingLoop
from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


def sl_to_rl(
    *,
    sl_data_dir: Path,
    sl_epochs: int,
    sl_batch_size: int = 4096,
    checkpoint_dir: Path,
    rl_config_path: Path | None = None,
    architecture: str = "se_resnet",
    model_params: dict[str, Any] | None = None,
    vecenv: Any = None,
    db_path: str = "keisei.db",
    sl_learning_rate: float = 1e-3,
    sl_use_amp: bool = False,
) -> KataGoTrainingLoop:
    """Run SL training, save checkpoint, and return a configured RL training loop.

    This is the orchestrator that bridges supervised pre-training and
    reinforcement learning fine-tuning. It:

    1. Builds a model and runs SL training for ``sl_epochs`` epochs.
    2. Saves the SL checkpoint (model weights + metadata, optimizer state
       included in the file but will be skipped on RL load).
    3. Writes training state to the DB so ``KataGoTrainingLoop._check_resume()``
       finds the checkpoint.
    4. Returns a ``KataGoTrainingLoop`` with ``resume_mode="sl"`` — when the
       caller invokes ``loop.run()``, it loads model weights but discards the
       SL optimizer state.

    Returns:
        A configured ``KataGoTrainingLoop`` ready for ``loop.run()``.
    """
    # --- Phase 1: SL Training ---
    model = build_model(architecture, model_params or {})
    sl_config = SLConfig(
        data_dir=str(sl_data_dir),
        batch_size=sl_batch_size,
        learning_rate=sl_learning_rate,
        total_epochs=sl_epochs,
        use_amp=sl_use_amp,
    )
    trainer = SLTrainer(model, sl_config)

    logger.info("Starting SL training: %d epochs, batch_size=%d", sl_epochs, sl_batch_size)
    for epoch in range(sl_epochs):
        metrics = trainer.train_epoch()
        logger.info("SL epoch %d/%d complete: %s", epoch + 1, sl_epochs, metrics)

    # --- Phase 2: Save SL Checkpoint ---
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "sl_final.pt"
    save_checkpoint(
        ckpt_path,
        model,
        trainer.optimizer,
        epoch=sl_epochs,
        step=0,
        architecture=architecture,
        scheduler=trainer.scheduler,
        grad_scaler=trainer.scaler,
    )
    logger.info("SL checkpoint saved: %s", ckpt_path)

    # --- Phase 3: Write DB state for _check_resume() ---
    init_db(db_path)
    write_training_state(
        db_path,
        {
            "config_json": "{}",
            "display_name": "SL→RL",
            "model_arch": architecture,
            "algorithm_name": "katago_ppo",
            "started_at": "",
            "current_epoch": sl_epochs,
            "current_step": 0,
            "checkpoint_path": str(ckpt_path),
        },
    )

    # --- Phase 4: Build RL Training Loop ---
    if rl_config_path is not None:
        rl_config = load_config(rl_config_path)
    else:
        rl_config = AppConfig(
            training=TrainingConfig(
                num_games=8,
                max_ply=500,
                algorithm="katago_ppo",
                checkpoint_interval=50,
                checkpoint_dir=str(checkpoint_dir),
                algorithm_params={
                    "learning_rate": 2e-4,
                    "gamma": 0.99,
                    "lambda_policy": 1.0,
                    "lambda_value": 1.5,
                    "lambda_score": 0.02,
                    "lambda_entropy": 0.01,
                    "score_normalization": 76.0,
                    "grad_clip": 1.0,
                },
                use_amp=sl_use_amp,
            ),
            display=DisplayConfig(moves_per_minute=30, db_path=db_path),
            model=ModelConfig(
                display_name="SL→RL",
                architecture=architecture,
                params=model_params or {},
            ),
        )

    loop = KataGoTrainingLoop(rl_config, vecenv=vecenv, resume_mode="sl")
    logger.info("RL training loop ready (resume_mode=sl)")
    return loop
