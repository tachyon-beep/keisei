"""
step_manager.py: Manages individual training steps and episode lifecycle.

This module encapsulates the logic for executing single training steps,
handling episode boundaries, and managing the interaction between the agent
and the environment during training.
"""

import time  # Added import
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np  # Added import
import torch  # Added import

from keisei.config_schema import AppConfig
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi import Color, ShogiGame  # Added Color import
from keisei.utils import PolicyOutputMapper, format_move_with_description_enhanced


@dataclass
class EpisodeState:
    """Represents the current state of a training episode."""

    current_obs: np.ndarray
    current_obs_tensor: torch.Tensor
    episode_reward: float
    episode_length: int


@dataclass
class StepResult:
    """Result of executing a single training step."""

    next_obs: np.ndarray
    next_obs_tensor: torch.Tensor
    reward: float
    done: bool
    info: Dict[str, Any]
    selected_move: Optional[Tuple]
    policy_index: int
    log_prob: float
    value_pred: float
    success: bool = True
    error_message: Optional[str] = None


class StepManager:
    """
    Manages individual training step execution and episode lifecycle.

    This class encapsulates the logic for:
    - Executing single training steps
    - Handling episode boundaries and resets
    - Managing demo mode interactions
    - Error handling and recovery during steps
    """

    def __init__(
        self,
        config: AppConfig,
        game: ShogiGame,
        agent: PPOAgent,
        policy_mapper: PolicyOutputMapper,
        experience_buffer: ExperienceBuffer,
    ):
        """
        Initialize the StepManager.

        Args:
            config: Application configuration
            game: Shogi game environment
            agent: PPO agent for action selection
            policy_mapper: Maps between game moves and policy outputs
            experience_buffer: Buffer for storing training experiences
        """
        self.config = config
        self.game = game
        self.agent = agent
        self.policy_mapper = policy_mapper
        self.experience_buffer = experience_buffer
        self.device = torch.device(config.env.device)
        self.move_history: List[Tuple] = []
        self.move_log: List[str] = []
        self.sente_best_capture: Optional[str] = None
        self.sente_best_capture_value: int = 0
        self.gote_best_capture: Optional[str] = None
        self.gote_best_capture_value: int = 0
        self.sente_capture_count: int = 0
        self.gote_capture_count: int = 0
        self.sente_drop_count: int = 0
        self.gote_drop_count: int = 0
        self.sente_promo_count: int = 0
        self.gote_promo_count: int = 0

    def execute_step(
        self,
        episode_state: EpisodeState,
        global_timestep: int,
        logger_func: Callable[[str, bool, Optional[Dict], str], None],
    ) -> StepResult:
        """
        Execute a single training step.

        Args:
            episode_state: Current state of the episode
            global_timestep: Current global training timestep
            logger_func: Function for logging messages

        Returns:
            StepResult containing the outcome of the step
        """
        try:
            # Get legal moves for current position
            legal_shogi_moves = self.game.get_legal_moves()

            # Check for no legal moves condition (terminal state)
            if not legal_shogi_moves:
                error_msg = (
                    f"No legal moves available at timestep {global_timestep}. "
                    f"Game should be in terminal state (checkmate/stalemate)."
                )
                logger_func(
                    f"TERMINAL: {error_msg} Resetting episode.",
                    True,  # also_to_wandb
                    None,  # wandb_data
                    "info",  # log_level
                )

                # Reset game and return failure result
                reset_obs = self.game.reset()
                reset_tensor = torch.tensor(
                    reset_obs,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                return StepResult(
                    next_obs=reset_obs,
                    next_obs_tensor=reset_tensor,
                    reward=0.0,
                    done=True,  # Terminal state
                    info={"terminal_reason": "no_legal_moves"},
                    selected_move=None,
                    policy_index=0,
                    log_prob=0.0,
                    value_pred=0.0,
                    success=False,
                    error_message=error_msg,
                )

            legal_mask_tensor = self.policy_mapper.get_legal_mask(
                legal_shogi_moves, device=self.device
            )

            # Agent action selection
            selected_shogi_move, policy_index, log_prob, value_pred = (
                self.agent.select_action(
                    episode_state.current_obs, legal_mask_tensor, is_training=True
                )
            )

            # Check if agent failed to select a move
            if selected_shogi_move is None:
                error_msg = (
                    f"Agent failed to select a move at timestep {global_timestep}"
                )
                logger_func(
                    f"CRITICAL: {error_msg}. Resetting episode.",
                    True,  # also_to_wandb
                    None,  # wandb_data
                    "error",  # log_level
                )

                # Reset game and return failure result
                reset_obs = self.game.reset()
                reset_tensor = torch.tensor(
                    reset_obs,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                return StepResult(
                    next_obs=reset_obs,
                    next_obs_tensor=reset_tensor,
                    reward=0.0,
                    done=False,
                    info={},
                    selected_move=None,
                    policy_index=0,
                    log_prob=0.0,
                    value_pred=0.0,
                    success=False,
                    error_message=error_msg,
                )

            # --- START: CORRECTED LOGGING LOGIC ---
            # Get piece info for the ACTUAL selected move (not the first legal move)
            piece_info_for_demo = None
            if self.config.display.display_moves:
                try:
                    # Check if it's a board move (not a drop) to get piece from square
                    if (
                        selected_shogi_move[0] is not None
                        and selected_shogi_move[1] is not None
                    ):
                        from_r, from_c = selected_shogi_move[0], selected_shogi_move[1]
                        piece_info_for_demo = self.game.get_piece(from_r, from_c)
                except (AttributeError, IndexError, ValueError):
                    pass  # Silently ignore errors, will fall back to a generic name
            # --- END: CORRECTED LOGGING LOGIC ---

            # Handle display move logging and delay
            if self.config.display.display_moves:
                self._handle_demo_mode(
                    selected_shogi_move,
                    episode_state.episode_length,
                    piece_info_for_demo,
                )

            # Execute the move in the environment
            moving_player = self.game.current_player
            is_drop = selected_shogi_move[0] is None and selected_shogi_move[1] is None
            is_promotion = (
                isinstance(selected_shogi_move[4], bool) and selected_shogi_move[4]
            )
            move_result = self.game.make_move(selected_shogi_move)
            if not (isinstance(move_result, tuple) and len(move_result) == 4):
                raise ValueError(f"Invalid move result: {type(move_result)}")

            next_obs_np, reward, done, info = move_result
            captured_name = info.get("captured_piece_type")
            if captured_name:
                if moving_player == Color.BLACK:
                    self.sente_capture_count += 1
                else:
                    self.gote_capture_count += 1
                value_map = {
                    "PAWN": 1,
                    "LANCE": 3,
                    "KNIGHT": 3,
                    "SILVER": 5,
                    "GOLD": 6,
                    "BISHOP": 8,
                    "ROOK": 10,
                }
                base_name = captured_name.replace("PROMOTED_", "")
                captured_value = value_map.get(base_name, 0)
                if moving_player == Color.BLACK:
                    if captured_value > self.sente_best_capture_value:
                        self.sente_best_capture = base_name.title()
                        self.sente_best_capture_value = captured_value
                else:
                    if captured_value > self.gote_best_capture_value:
                        self.gote_best_capture = base_name.title()
                        self.gote_best_capture_value = captured_value

            if is_drop:
                if moving_player == Color.BLACK:
                    self.sente_drop_count += 1
                else:
                    self.gote_drop_count += 1

            if is_promotion:
                if moving_player == Color.BLACK:
                    self.sente_promo_count += 1
                else:
                    self.gote_promo_count += 1

            # Add experience to buffer
            self.experience_buffer.add(
                episode_state.current_obs_tensor.squeeze(0),
                policy_index,
                reward,
                log_prob,
                value_pred,
                done,
                legal_mask_tensor,
            )

            # Create next observation tensor
            next_obs_tensor = torch.tensor(
                next_obs_np,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            return StepResult(
                next_obs=next_obs_np,
                next_obs_tensor=next_obs_tensor,
                reward=reward,
                done=done,
                info=info,
                selected_move=selected_shogi_move,
                policy_index=policy_index,
                log_prob=log_prob,
                value_pred=value_pred,
                success=True,
            )

        except ValueError as e:
            error_msg = f"Error during training step: {e}"
            logger_func(
                f"CRITICAL: {error_msg}. Resetting episode.",
                True,  # also_to_wandb
                None,  # wandb_data
                "error",  # log_level
            )

            # Reset game and return failure result
            try:
                reset_obs = self.game.reset()
                reset_tensor = torch.tensor(
                    reset_obs,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                return StepResult(
                    next_obs=reset_obs,
                    next_obs_tensor=reset_tensor,
                    reward=0.0,
                    done=False,
                    info={},
                    selected_move=None,
                    policy_index=0,
                    log_prob=0.0,
                    value_pred=0.0,
                    success=False,
                    error_message=error_msg,
                )
            except Exception as reset_error:
                # If reset also fails, return a minimal failure result
                return StepResult(
                    next_obs=episode_state.current_obs,
                    next_obs_tensor=episode_state.current_obs_tensor,
                    reward=0.0,
                    done=True,  # Force episode end
                    info={},
                    selected_move=None,
                    policy_index=0,
                    log_prob=0.0,
                    value_pred=0.0,
                    success=False,
                    error_message=f"{error_msg}; Reset also failed: {reset_error}",
                )

    def handle_episode_end(
        self,
        episode_state: EpisodeState,
        step_result: StepResult,
        game_stats: Dict[str, int],  # Read-only, not modified
        total_episodes_completed: int,
        logger_func: Callable[..., None],
    ) -> Tuple[EpisodeState, Optional[str]]:  # Return tuple
        """
        Handle the end of an episode, prepare for the next one, and log results.

        Args:
            episode_state: Current episode state
            step_result: Result from the final step of the episode
            game_stats: Running game statistics (e.g., black_wins, white_wins, draws) - READ ONLY (not modified)
            total_episodes_completed: Total episodes completed so far (before this one)
            logger_func: Function for logging messages

        Returns:
            Tuple[New EpisodeState for the next episode, final_winner_color (str or None)]
        """
        final_winner_color, reason_from_info = self._determine_winner_and_reason(
            step_result.info
        )
        game_outcome_message = self._format_game_outcome_message(
            final_winner_color, reason_from_info
        )

        # Fix B2: Don't modify game_stats in place to avoid double counting
        # Create a temporary copy for win rate calculations only
        temp_game_stats = game_stats.copy()
        if final_winner_color == "black":
            temp_game_stats["black_wins"] += 1
        elif final_winner_color == "white":
            temp_game_stats["white_wins"] += 1
        elif final_winner_color is None:  # Draw
            temp_game_stats["draws"] += 1

        # Calculate win rates for logging using the temporary game_stats
        updated_total_games = (
            temp_game_stats["black_wins"]
            + temp_game_stats["white_wins"]
            + temp_game_stats["draws"]
        )

        updated_black_win_rate = (
            temp_game_stats["black_wins"] / updated_total_games
            if updated_total_games > 0
            else 0.0
        )
        updated_white_win_rate = (
            temp_game_stats["white_wins"] / updated_total_games
            if updated_total_games > 0
            else 0.0
        )
        updated_draw_rate = (
            temp_game_stats["draws"] / updated_total_games
            if updated_total_games > 0
            else 0.0
        )

        # Log episode completion
        logger_func(
            f"Episode {total_episodes_completed + 1} finished. Length: {episode_state.episode_length}, "
            f"Reward: {episode_state.episode_reward:.2f}. {game_outcome_message}",
            also_to_wandb=True,
            wandb_data={
                "episode_reward": episode_state.episode_reward,
                "episode_length": episode_state.episode_length,
                "game_outcome": final_winner_color,
                "game_reason": reason_from_info,
                "black_wins_total": temp_game_stats[
                    "black_wins"
                ],  # Use updated totals for logging
                "white_wins_total": temp_game_stats[
                    "white_wins"
                ],  # Use updated totals for logging
                "draws_total": temp_game_stats[
                    "draws"
                ],  # Use updated totals for logging
                "black_win_rate": updated_black_win_rate,  # Corrected key
                "white_win_rate": updated_white_win_rate,  # Corrected key
                "draw_rate": updated_draw_rate,  # Corrected key
            },
            log_level="info",
        )

        # Reset game for next episode
        try:
            reset_result = self.game.reset()
            if not isinstance(reset_result, np.ndarray):
                raise RuntimeError("Game reset failed after episode end")

            reset_obs_tensor = torch.tensor(
                reset_result,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            self.move_history.clear()
            self.move_log.clear()
            self.sente_best_capture = None
            self.sente_best_capture_value = 0
            self.gote_best_capture = None
            self.gote_best_capture_value = 0
            self.sente_capture_count = 0
            self.gote_capture_count = 0
            self.sente_drop_count = 0
            self.gote_drop_count = 0
            self.sente_promo_count = 0
            self.gote_promo_count = 0

            new_episode_state = EpisodeState(
                current_obs=reset_result,
                current_obs_tensor=reset_obs_tensor,
                episode_reward=0.0,
                episode_length=0,
            )
            return new_episode_state, final_winner_color

        except (RuntimeError, ValueError, OSError) as e:
            logger_func(
                f"CRITICAL: Game reset failed after episode end: {e}",
                True,
                None,
                "error",
            )
            # Return current state to allow caller to handle the error
            return (
                episode_state,
                final_winner_color,
            )  # Return winner color even on reset failure

    def reset_episode(self) -> EpisodeState:
        """
        Reset the game and create a new episode state.

        Returns:
            New EpisodeState for a fresh episode
        """
        reset_obs = self.game.reset()
        reset_tensor = torch.tensor(
            reset_obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self.move_history.clear()
        self.move_log.clear()
        self.sente_best_capture = None
        self.sente_best_capture_value = 0
        self.gote_best_capture = None
        self.gote_best_capture_value = 0
        self.sente_capture_count = 0
        self.gote_capture_count = 0
        self.sente_drop_count = 0
        self.gote_drop_count = 0
        self.sente_promo_count = 0
        self.gote_promo_count = 0

        return EpisodeState(
            current_obs=reset_obs,
            current_obs_tensor=reset_tensor,
            episode_reward=0.0,
            episode_length=0,
        )

    def update_episode_state(
        self, episode_state: EpisodeState, step_result: StepResult
    ) -> EpisodeState:
        """
        Update episode state with the results of a step.

        Args:
            episode_state: Current episode state
            step_result: Result from the executed step

        Returns:
            Updated EpisodeState
        """
        return EpisodeState(
            current_obs=step_result.next_obs,
            current_obs_tensor=step_result.next_obs_tensor,
            episode_reward=episode_state.episode_reward + step_result.reward,
            episode_length=episode_state.episode_length + 1,
        )

    def _handle_demo_mode(
        self,
        selected_move: Tuple,
        episode_length: int,
        piece_info_for_demo: Optional[Any],
    ) -> None:
        """
        Handle demo mode logging and delay.

        Args:
            selected_move: The move selected by the agent
            episode_length: Current episode length
            piece_info_for_demo: Piece information for demo display
        """
        # Get current player name
        current_player_name = (
            getattr(
                self.game.current_player,
                "name",
                str(self.game.current_player),
            )
            if hasattr(self.game, "current_player")
            else "Unknown"
        )

        # Format move description
        move_str = format_move_with_description_enhanced(
            selected_move,
            self.policy_mapper,
            piece_info_for_demo,
        )

        if current_player_name.upper() == "BLACK":
            player_display = "Sente"
        elif current_player_name.upper() == "WHITE":
            player_display = "Gote"
        else:
            player_display = current_player_name
        log_msg = f"Move {episode_length + 1} ({player_display}): {move_str}"
        self.move_log.append(log_msg)
        self.move_history.append(selected_move)

        # Add delay for easier observation
        demo_delay = self.config.display.turn_tick
        if demo_delay > 0:
            time.sleep(demo_delay)

    def _determine_winner_and_reason(
        self, step_info: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[str], str]:
        """Helper to determine winner and reason from step_info and game state."""
        winner_from_info = None
        reason_from_info = "Unknown"

        if step_info:
            winner_from_info = step_info.get("winner")
            reason_from_info = step_info.get("reason", "Unknown")

        # Normalize winner color to lowercase for consistency
        final_winner_color = winner_from_info
        if winner_from_info and isinstance(winner_from_info, str):
            final_winner_color = winner_from_info.lower()

        if reason_from_info == "Tsumi" and winner_from_info is None:
            if hasattr(self.game, "winner") and self.game.winner is not None:
                game_winner_enum = self.game.winner
                if game_winner_enum == Color.BLACK:
                    final_winner_color = "black"
                elif game_winner_enum == Color.WHITE:
                    final_winner_color = "white"

        return final_winner_color, reason_from_info

    def _format_game_outcome_message(self, winner: Optional[str], reason: str) -> str:
        """Helper to format the game outcome message."""
        if winner == "black":
            return f"Sente wins by {reason}."
        if winner == "white":
            return f"Gote wins by {reason}."
        if winner is None:
            return f"Draw by {reason}."
        return f"Game ended: {winner} by {reason}."  # Should ideally not be reached with current logic
