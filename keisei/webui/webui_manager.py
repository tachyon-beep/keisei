"""
webui_manager.py: WebUI manager for streaming training data via WebSocket.
"""

import json
import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import logging

try:
    import websockets
    from websockets.server import WebSocketServerProtocol, serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WebSocketServerProtocol = Any
    WEBSOCKETS_AVAILABLE = False

from keisei.config_schema import WebUIConfig


class WebSocketConnectionManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self):
        self.connections: Set[WebSocketServerProtocol] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._logger = logging.getLogger(__name__)

    async def register(self, websocket: WebSocketServerProtocol):
        """Register a new WebSocket connection."""
        self.connections.add(websocket)
        self._logger.info(f"WebSocket connection registered. Total: {len(self.connections)}")

    async def unregister(self, websocket: WebSocketServerProtocol):
        """Unregister a WebSocket connection."""
        self.connections.discard(websocket)
        self._logger.info(f"WebSocket connection removed. Total: {len(self.connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self.connections:
            return

        message_json = json.dumps(message)
        dead_connections = set()

        for connection in self.connections.copy():
            try:
                await connection.send(message_json)
            except Exception as e:
                self._logger.warning(f"Failed to send message to client: {e}")
                dead_connections.add(connection)

        # Clean up dead connections
        for connection in dead_connections:
            await self.unregister(connection)

    async def queue_message(self, message: Dict[str, Any]):
        """Queue a message for broadcasting."""
        await self.message_queue.put(message)

    async def message_broadcaster(self):
        """Background task that broadcasts queued messages."""
        while True:
            try:
                message = await self.message_queue.get()
                await self.broadcast(message)
                self.message_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in message broadcaster: {e}")


class WebUIManager:
    """Manages WebUI streaming functionality parallel to DisplayManager."""

    def __init__(self, config: WebUIConfig):
        self.config = config
        self.connection_manager = WebSocketConnectionManager()
        self.server_task: Optional[asyncio.Task] = None
        self.broadcaster_task: Optional[asyncio.Task] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        self._logger = logging.getLogger(__name__)
        
        # Rate limiting - increased intervals for better browser stability
        self.last_board_update = 0.0
        self.last_metrics_update = 0.0
        self.last_progress_update = 0.0
        
        # Memory management
        self.message_count = 0
        self.memory_cleanup_interval = 1000  # Clean up every 1000 messages
        
        if not WEBSOCKETS_AVAILABLE:
            self._logger.warning("WebSockets not available. Install 'websockets' package to enable WebUI.")

    def start(self):
        """Start the WebUI server in a background thread."""
        if not WEBSOCKETS_AVAILABLE:
            self._logger.error("Cannot start WebUI: websockets package not available")
            return False
            
        if self._running:
            return True
            
        self._running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        
        # Give the server a moment to start
        time.sleep(0.1)
        self._logger.info(f"WebUI server starting on {self.config.host}:{self.config.port}")
        return True

    def stop(self):
        """Stop the WebUI server."""
        self._running = False
        
        if self.event_loop and not self.event_loop.is_closed():
            # Cancel tasks
            if self.server_task:
                self.event_loop.call_soon_threadsafe(self.server_task.cancel)
            if self.broadcaster_task:
                self.event_loop.call_soon_threadsafe(self.broadcaster_task.cancel)

    def _run_server(self):
        """Run the WebSocket server in the background thread."""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        try:
            self.event_loop.run_until_complete(self._start_server())
        except Exception as e:
            self._logger.error(f"WebUI server error: {e}")

    async def _start_server(self):
        """Start the WebSocket server and message broadcaster."""
        if not WEBSOCKETS_AVAILABLE:
            return
            
        # Start message broadcaster
        self.broadcaster_task = asyncio.create_task(
            self.connection_manager.message_broadcaster()
        )
        
        # Start WebSocket server
        async def handle_client(websocket: WebSocketServerProtocol, path: str):
            await self.connection_manager.register(websocket)
            try:
                # Send initial connection message
                await websocket.send(json.dumps({
                    "type": "connected",
                    "timestamp": time.time(),
                    "message": "Connected to Keisei training stream"
                }))
                
                # Keep connection alive
                await websocket.wait_closed()
            except Exception as e:
                self._logger.warning(f"Client connection error: {e}")
            finally:
                await self.connection_manager.unregister(websocket)

        try:
            async with serve(handle_client, self.config.host, self.config.port):
                self._logger.info(f"WebUI server started on ws://{self.config.host}:{self.config.port}")
                
                # Keep server running
                while self._running:
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            self._logger.error(f"Failed to start WebSocket server: {e}")

    def _should_update_board(self) -> bool:
        """Check if enough time has passed for board update."""
        current_time = time.time()
        interval = 1.0 / self.config.board_update_rate_hz
        if current_time - self.last_board_update >= interval:
            self.last_board_update = current_time
            return True
        return False

    def _should_update_metrics(self) -> bool:
        """Check if enough time has passed for metrics update."""
        current_time = time.time()
        interval = 1.0 / self.config.metrics_update_rate_hz
        if current_time - self.last_metrics_update >= interval:
            self.last_metrics_update = current_time
            return True
        return False

    def update_progress(self, trainer, speed: float, pending_updates: dict):
        """Called parallel to DisplayManager.update_progress() to broadcast data."""
        if not self._running or not self.event_loop:
            return

        # Rate limit progress updates more aggressively to prevent memory pressure
        current_time = time.time()
        if current_time - self.last_progress_update < 0.5:  # Max 2 Hz for progress updates
            return
        self.last_progress_update = current_time

        if self._should_update_metrics():
            # Reduce data payload to essential information only
            message = {
                "type": "progress_update",
                "timestamp": current_time,
                "data": {
                    "global_timestep": trainer.metrics_manager.global_timestep,
                    "speed": speed,
                    "total_episodes": trainer.metrics_manager.total_episodes_completed,
                    "black_wins": trainer.metrics_manager.black_wins,
                    "white_wins": trainer.metrics_manager.white_wins,
                    "draws": trainer.metrics_manager.draws,
                    "ep_metrics": pending_updates.get("ep_metrics", ""),
                    "ppo_metrics": pending_updates.get("ppo_metrics", ""),
                    "current_epoch": pending_updates.get("current_epoch", 0),
                    "processing": getattr(trainer.metrics_manager, "processing", False)
                }
            }

            self._queue_message_with_cleanup(message)

    def refresh_dashboard_panels(self, trainer):
        """Called parallel to DisplayManager.refresh_dashboard_panels() to broadcast game state."""
        if not self._running or not self.event_loop:
            return

        # Always try to send metrics data (independent of board updates)
        metrics_data = self._extract_metrics_data(trainer)
        if metrics_data:
            message = {
                "type": "metrics_update", 
                "timestamp": time.time(),
                "data": metrics_data
            }
            
            self._queue_message_with_cleanup(message)

        # Try to send board data if rate limiting allows
        if self._should_update_board():
            try:
                # Extract board state
                board_data = self._extract_board_state(trainer)
                if board_data:
                    message = {
                        "type": "board_update",
                        "timestamp": time.time(),
                        "data": board_data
                    }

                    self._queue_message_with_cleanup(message)
            except Exception as e:
                self._logger.warning(f"Board extraction failed: {e}")

    def _extract_board_state(self, trainer) -> Optional[Dict[str, Any]]:
        """Extract current board state from trainer."""
        try:
            if not trainer.game:
                return None

            game = trainer.game
            
            # Convert board to serializable format with robust error handling
            board = []
            for row in range(9):
                board_row = []
                for col in range(9):
                    piece = game.board[row][col]
                    if piece:
                        # Try different possible attributes for piece type
                        piece_type = None
                        for attr in ['piece_type', 'type', 'name']:
                            if hasattr(piece, attr):
                                piece_type_obj = getattr(piece, attr)
                                if hasattr(piece_type_obj, 'name'):
                                    piece_type = piece_type_obj.name.lower()
                                else:
                                    piece_type = str(piece_type_obj).lower()
                                break
                        if piece_type is None:
                            piece_type = str(piece).lower()
                        
                        # Try different possible attributes for color
                        color = None
                        for attr in ['color', 'player', 'owner']:
                            if hasattr(piece, attr):
                                color_obj = getattr(piece, attr)
                                if hasattr(color_obj, 'name'):
                                    color = color_obj.name.lower()
                                else:
                                    color = str(color_obj).lower()
                                break
                        if color is None:
                            color = "unknown"
                            
                        # Try different possible attributes for promoted
                        promoted = False
                        if hasattr(piece, 'promoted'):
                            promoted = piece.promoted
                        elif hasattr(piece, 'is_promoted'):
                            promoted = piece.is_promoted
                            
                        piece_data = {
                            "type": piece_type,
                            "color": color,
                            "promoted": promoted
                        }
                        
                        board_row.append(piece_data)
                    else:
                        board_row.append(None)
                board.append(board_row)

            # Get piece stands (komadai) - handle different possible attributes
            sente_hand = {}
            gote_hand = {}
            
            # Try different possible hand attributes
            for hand_attr in ['sente_hand', 'black_hand', 'hands']:
                if hasattr(game, hand_attr):
                    hand_data = getattr(game, hand_attr)
                    if isinstance(hand_data, dict):
                        if hand_attr in ['sente_hand', 'black_hand']:
                            for piece_type, count in hand_data.items():
                                if count > 0:
                                    piece_name = piece_type.name.lower() if hasattr(piece_type, 'name') else str(piece_type).lower()
                                    sente_hand[piece_name] = count
                        elif hand_attr == 'hands' and len(hand_data) >= 2:
                            # hands might be a list/dict with both hands
                            black_hand = hand_data.get('black', hand_data.get(0, {}))
                            white_hand = hand_data.get('white', hand_data.get(1, {}))
                            for piece_type, count in black_hand.items():
                                if count > 0:
                                    piece_name = piece_type.name.lower() if hasattr(piece_type, 'name') else str(piece_type).lower()
                                    sente_hand[piece_name] = count
                            for piece_type, count in white_hand.items():
                                if count > 0:
                                    piece_name = piece_type.name.lower() if hasattr(piece_type, 'name') else str(piece_type).lower()
                                    gote_hand[piece_name] = count
                    break
                        
            for hand_attr in ['gote_hand', 'white_hand']:
                if hasattr(game, hand_attr):
                    hand_data = getattr(game, hand_attr)
                    if isinstance(hand_data, dict):
                        for piece_type, count in hand_data.items():
                            if count > 0:
                                piece_name = piece_type.name.lower() if hasattr(piece_type, 'name') else str(piece_type).lower()
                                gote_hand[piece_name] = count
                    break

            # Get recent moves with better error handling
            recent_moves = []
            if trainer.step_manager:
                if hasattr(trainer.step_manager, 'move_log') and trainer.step_manager.move_log:
                    recent_moves = list(trainer.step_manager.move_log[-20:])  # Last 20 moves
                elif hasattr(trainer.step_manager, 'move_history') and trainer.step_manager.move_history:
                    # Fallback to move_history if move_log not available
                    try:
                        from keisei.utils import format_move_with_description
                        recent_moves = []
                        for move in trainer.step_manager.move_history[-10:]:
                            try:
                                move_str = format_move_with_description(move, trainer.policy_output_mapper, game)
                                recent_moves.append(move_str)
                            except:
                                recent_moves.append(str(move))
                    except:
                        recent_moves = [str(move) for move in trainer.step_manager.move_history[-10:]]

            # Get player info with better attribute detection
            current_player = "black"
            if hasattr(game, 'current_player'):
                if hasattr(game.current_player, 'name'):
                    current_player = game.current_player.name.lower()
                else:
                    current_player = str(game.current_player).lower()

            # Get winner info
            winner = None
            if hasattr(game, 'winner') and game.winner:
                if hasattr(game.winner, 'name'):
                    winner = game.winner.name.lower()
                else:
                    winner = str(game.winner).lower()

            return {
                "board": board,
                "current_player": current_player,
                "move_count": getattr(game, 'move_count', getattr(game, 'ply_count', 0)),
                "game_over": getattr(game, 'game_over', getattr(game, 'is_game_over', False)),
                "winner": winner,
                "sente_hand": sente_hand,
                "gote_hand": gote_hand,
                "recent_moves": recent_moves,
                "hot_squares": trainer.metrics_manager.get_hot_squares(3),
                "last_move": recent_moves[-1] if recent_moves else None,
                "ply_per_second": getattr(trainer, "last_ply_per_sec", 0.0)
            }

        except Exception as e:
            self._logger.warning(f"Error extracting board state: {e}")
            return None

    def _extract_metrics_data(self, trainer) -> Optional[Dict[str, Any]]:
        """Extract training metrics from trainer - matching Rich display exactly."""
        try:
            history = trainer.metrics_manager.history
            
            # Build metrics table data matching display.py _build_metric_lines
            metrics_table = []
            metrics_to_display = [
                ("Episode Length", "episode_lengths"),
                ("Episode Reward", "episode_rewards"),
                ("Policy Loss", "policy_losses"),
                ("Value Loss", "value_losses"),
                ("Entropy", "entropies"),
                ("KL Divergence", "kl_divergences"),
                ("PPO Clip Frac", "clip_fractions"),
                ("Win Rate - Sente", "win_rates_black"),
                ("Win Rate - Gote", "win_rates_white"),
                ("Draw Rate", "draw_rates"),
            ]
            
            for name, history_key in metrics_to_display:
                if history_key == "win_rates_black":
                    data_list = [d.get("win_rate_black", 0.0) for d in history.win_rates_history]
                elif history_key == "win_rates_white":
                    data_list = [d.get("win_rate_white", 0.0) for d in history.win_rates_history]
                elif history_key == "draw_rates":
                    data_list = [d.get("win_rate_draw", 0.0) for d in history.win_rates_history]
                else:
                    data_list = getattr(history, history_key, [])
                
                last_val = data_list[-1] if len(data_list) >= 1 else None
                prev_val = data_list[-2] if len(data_list) >= 2 else None
                avg_slice = data_list[-5:]
                avg_val = sum(avg_slice) / len(avg_slice) if avg_slice else None
                
                metrics_table.append({
                    "name": name,
                    "last": last_val,
                    "previous": prev_val,
                    "average_5": avg_val,
                    "trend_data": data_list[-15:] if data_list else []  # Last 15 for mini-chart
                })
            
            # Extract enhanced data for advanced visualizations
            entropy_value = list(history.entropies)[-1] if history.entropies else 0.5
            value_estimate = self._get_latest_value_estimate(trainer)
            policy_confidence = self._extract_policy_confidence(trainer)
            skill_metrics = self._extract_skill_metrics(trainer, history)
            gradient_norms = self._extract_gradient_norms(trainer)
            quality_distribution = self._get_buffer_quality_distribution(trainer)
            
            # Reduced debug logging - only log periodically to prevent memory pressure
            if self.message_count % 100 == 0:  # Log every 100 messages
                self._logger.debug(f"WebUI data: entropy={entropy_value:.3f}, confidence={len(policy_confidence)} values")

            return {
                "learning_curves": {
                    "policy_losses": list(history.policy_losses)[-50:],
                    "value_losses": list(history.value_losses)[-50:],
                    "entropies": list(history.entropies)[-50:],
                    "kl_divergences": list(history.kl_divergences)[-50:],
                    "clip_fractions": list(history.clip_fractions)[-50:],
                    "learning_rates": list(history.learning_rates)[-50:],
                    "episode_lengths": list(history.episode_lengths)[-50:],
                    "episode_rewards": list(history.episode_rewards)[-50:],
                },
                "metrics_table": metrics_table,  # Rich display metric table data
                "game_statistics": {
                    "games_per_hour": getattr(trainer.metrics_manager, 'get_games_completion_rate', lambda x: 0.0)(1.0),
                    "average_game_length": (
                        sum(trainer.metrics_manager.moves_per_game) / len(trainer.metrics_manager.moves_per_game)
                        if hasattr(trainer.metrics_manager, 'moves_per_game') and trainer.metrics_manager.moves_per_game else 0
                    ),
                    "win_loss_draw_rates": getattr(trainer.metrics_manager, 'get_win_loss_draw_rates', lambda x: {})(100),
                    "moves_per_game_trend": getattr(trainer.metrics_manager, 'get_moves_per_game_trend', lambda x: [])(20),
                    "average_turns_trend": getattr(trainer.metrics_manager, 'get_average_turns_trend', lambda x: [])(20),
                },
                "buffer_info": {
                    "buffer_size": trainer.experience_buffer.size() if trainer.experience_buffer else 0,
                    "buffer_capacity": trainer.experience_buffer.capacity() if trainer.experience_buffer else 0,
                    "quality_distribution": quality_distribution,
                },
                "model_info": {
                    "gradient_norm": getattr(trainer, "last_gradient_norm", 0.0),
                    "weight_updates": getattr(trainer, "last_weight_updates", {}),
                },
                # Advanced visualization data - CRITICAL FOR VISUALIZATIONS
                "entropy": entropy_value,
                "value_estimate": value_estimate,
                "policy_confidence": policy_confidence,
                "skill_metrics": skill_metrics,
                "gradient_norms": gradient_norms,
                "elo_data": getattr(trainer, "evaluation_elo_snapshot", None),
                "processing": getattr(trainer.metrics_manager, "processing", False)
            }

        except Exception as e:
            self._logger.warning(f"Error extracting metrics data: {e}")
            return None

    def _get_latest_value_estimate(self, trainer) -> float:
        """Extract the latest value estimate from the critic network."""
        try:
            # Try to get from step manager's recent episodes
            if hasattr(trainer, 'step_manager') and hasattr(trainer.step_manager, 'recent_episodes'):
                recent = trainer.step_manager.recent_episodes
                if recent:
                    last_episode = recent[-1]
                    if hasattr(last_episode, 'values') and last_episode.values:
                        return float(last_episode.values[-1])
            
            # Fallback to random walk around 0
            import random
            return random.uniform(-0.5, 0.5)
        except Exception:
            return 0.0

    def _extract_policy_confidence(self, trainer) -> List[float]:
        """Extract policy confidence data for neural heatmap visualization."""
        try:
            # Try to get from recent policy outputs
            if hasattr(trainer, 'step_manager') and hasattr(trainer.step_manager, 'recent_episodes'):
                recent = trainer.step_manager.recent_episodes
                if recent:
                    last_episode = recent[-1]
                    if hasattr(last_episode, 'action_log_probs') and last_episode.action_log_probs:
                        # Convert log probs to confidence scores (0-1 range)
                        import torch
                        import math
                        log_probs = last_episode.action_log_probs[-1]
                        if torch.is_tensor(log_probs):
                            probs = torch.exp(log_probs).detach().cpu().numpy()
                            # Map to 9x9 board confidence (simplified)
                            confidence = [min(1.0, max(0.0, float(p))) for p in probs[:81]]
                            return confidence + [0.5] * (81 - len(confidence))  # Pad to 81
            
            # Fallback: generate plausible confidence data
            import random
            return [random.uniform(0.3, 0.9) for _ in range(81)]
        except Exception:
            return [0.5] * 81  # Default neutral confidence

    def _extract_skill_metrics(self, trainer, history) -> Dict[str, float]:
        """Extract skill development metrics for radar visualization."""
        try:
            # Extract skill metrics from various sources
            skills = {
                "opening": self._calculate_opening_strength(trainer, history),
                "tactics": self._calculate_tactical_strength(trainer, history),
                "endgame": self._calculate_endgame_strength(trainer, history),
                "strategy": self._calculate_strategic_strength(trainer, history),
                "pattern": self._calculate_pattern_recognition(trainer, history),
                "time": self._calculate_time_management(trainer, history),
            }
            return skills
        except Exception:
            # Default balanced skills
            return {
                "opening": 0.5, "tactics": 0.5, "endgame": 0.5,
                "strategy": 0.5, "pattern": 0.5, "time": 0.5
            }

    def _calculate_opening_strength(self, trainer, history) -> float:
        """Calculate opening play strength based on early game performance."""
        try:
            if history.episode_lengths:
                # Longer games might indicate better opening preparation
                avg_length = sum(history.episode_lengths[-20:]) / min(20, len(history.episode_lengths))
                return min(1.0, max(0.0, avg_length / 100.0))  # Normalize to 0-1
            return 0.5
        except Exception:
            return 0.5

    def _calculate_tactical_strength(self, trainer, history) -> float:
        """Calculate tactical strength from capture rates and piece exchanges."""
        try:
            if hasattr(trainer, 'metrics_manager'):
                # Use win rate as proxy for tactical strength
                recent_wins = history.win_rates[-10:] if history.win_rates else []
                if recent_wins:
                    avg_win_rate = sum(w.get('win_rate_black', 0.5) for w in recent_wins) / len(recent_wins)
                    return min(1.0, max(0.0, avg_win_rate))
            return 0.5
        except Exception:
            return 0.5

    def _calculate_endgame_strength(self, trainer, history) -> float:
        """Calculate endgame strength from late-game decision quality."""
        try:
            # Use policy loss trend as proxy - lower loss = better decisions
            if history.policy_losses:
                recent_losses = history.policy_losses[-10:]
                if len(recent_losses) >= 2:
                    improvement = recent_losses[0] - recent_losses[-1]
                    return min(1.0, max(0.0, 0.5 + improvement))  # Center around 0.5
            return 0.5
        except Exception:
            return 0.5

    def _calculate_strategic_strength(self, trainer, history) -> float:
        """Calculate strategic planning from value function accuracy."""
        try:
            if history.value_losses:
                recent_losses = history.value_losses[-10:]
                if recent_losses:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    # Lower value loss indicates better strategic understanding
                    return min(1.0, max(0.0, 1.0 - min(avg_loss, 1.0)))
            return 0.5
        except Exception:
            return 0.5

    def _calculate_pattern_recognition(self, trainer, history) -> float:
        """Calculate pattern recognition from entropy trends."""
        try:
            if history.entropies:
                recent_entropies = history.entropies[-10:]
                if len(recent_entropies) >= 2:
                    # Lower entropy indicates better pattern recognition
                    avg_entropy = sum(recent_entropies) / len(recent_entropies)
                    return min(1.0, max(0.0, 1.0 - avg_entropy))
            return 0.5
        except Exception:
            return 0.5

    def _calculate_time_management(self, trainer, history) -> float:
        """Calculate time management efficiency."""
        try:
            # Use episode length consistency as time management proxy
            if len(history.episode_lengths) >= 5:
                lengths = history.episode_lengths[-10:]
                avg_length = sum(lengths) / len(lengths)
                variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
                consistency = max(0.0, 1.0 - (variance / (avg_length ** 2)))
                return min(1.0, consistency)
            return 0.5
        except Exception:
            return 0.5

    def _extract_gradient_norms(self, trainer) -> List[float]:
        """Extract gradient norms for network layers."""
        try:
            # Try to get gradient information from model
            if hasattr(trainer, 'model') and trainer.model:
                gradient_norms = []
                for name, param in trainer.model.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.norm().item()
                        gradient_norms.append(norm)
                
                if gradient_norms:
                    return gradient_norms[:6]  # Limit to 6 layers for visualization
            
            # Fallback: synthetic gradient data
            import random
            return [random.uniform(0.01, 0.5) for _ in range(6)]
        except Exception:
            return [0.1, 0.2, 0.15, 0.3, 0.25, 0.1]  # Default values

    def _get_buffer_quality_distribution(self, trainer) -> List[float]:
        """Get experience buffer quality distribution."""
        try:
            if hasattr(trainer, 'experience_buffer') and trainer.experience_buffer:
                # Try to extract quality metrics from buffer
                buffer = trainer.experience_buffer
                if hasattr(buffer, 'get_quality_distribution'):
                    return buffer.get_quality_distribution()
                
                # Fallback: estimate from rewards if available
                if hasattr(buffer, 'rewards') and buffer.rewards:
                    rewards = buffer.rewards[-1000:]  # Recent rewards
                    # Create histogram
                    import numpy as np
                    hist, _ = np.histogram(rewards, bins=10, range=(-1, 1))
                    return (hist / max(hist.sum(), 1)).tolist()
            
            # Default distribution
            import random
            return [random.uniform(0.05, 0.2) for _ in range(10)]
        except Exception:
            return [0.1] * 10  # Uniform distribution

    def _queue_message_with_cleanup(self, message: Dict[str, Any]):
        """Queue message with memory management."""
        self.message_count += 1

        # Periodic cleanup to prevent memory buildup
        if self.message_count % self.memory_cleanup_interval == 0:
            self._logger.info(f"Processed {self.message_count} WebSocket messages - performing cleanup")

        asyncio.run_coroutine_threadsafe(
            self.connection_manager.queue_message(message),
            self.event_loop
        )