# Perspective-Action Alignment Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the observation-action coordinate mismatch that causes asymmetric learning between Black and White in self-play training.

**Architecture:** Add a `flip_move` static method to `PolicyOutputMapper` and two perspective-aware wrappers (`get_legal_mask_perspective`, `perspective_index_to_absolute_move`). Update all call sites (training step manager, parallel worker, evaluation strategies, scheduler) to use the perspective-aware methods, passing `is_white` based on `game.current_player`. The observation encoder already flips the board for White — this fix makes the action space match.

**Tech Stack:** Python, PyTorch, numpy, pytest

**Root Cause:** The observation in `shogi_game_io.py` flips board coordinates 180° for White's perspective (`r → 8-r, c → 8-c`), but `get_legal_moves()` and `PolicyOutputMapper` operate in absolute coordinates. This means the CNN's spatial features don't align with the action indices when playing as White, making it much harder for the network to learn a coherent White policy.

---

### Task 1: Add perspective-aware methods to PolicyOutputMapper

**Files:**
- Modify: `keisei/utils/utils.py:186-342` (PolicyOutputMapper class)
- Test: `tests/unit/test_policy_mapper.py`

- [ ] **Step 1: Write failing tests for flip_move and perspective methods**

Add to `tests/unit/test_policy_mapper.py`:

```python
class TestPerspectiveAlignment:
    """Tests for perspective-aware action mapping."""

    def setup_method(self):
        self.mapper = PolicyOutputMapper()

    def test_flip_board_move(self):
        """Flipping a board move rotates coordinates 180 degrees."""
        original = (6, 0, 5, 0, False)  # Black pawn push from (6,0) to (5,0)
        flipped = PolicyOutputMapper.flip_move(original)
        assert flipped == (2, 8, 3, 8, False)  # 8-6=2, 8-0=8, 8-5=3, 8-0=8

    def test_flip_board_move_with_promotion(self):
        """Promotion flag is preserved through flip."""
        original = (6, 2, 2, 2, True)
        flipped = PolicyOutputMapper.flip_move(original)
        assert flipped == (2, 6, 6, 6, True)
        assert flipped[4] is True

    def test_flip_drop_move(self):
        """Drop move flips only the target square, preserves piece type."""
        from keisei.shogi.shogi_core_definitions import PieceType
        original = (None, None, 2, 0, PieceType.PAWN)
        flipped = PolicyOutputMapper.flip_move(original)
        assert flipped == (None, None, 6, 8, PieceType.PAWN)

    def test_flip_is_involution(self):
        """Flipping twice returns the original move."""
        board_move = (3, 5, 7, 1, False)
        assert PolicyOutputMapper.flip_move(PolicyOutputMapper.flip_move(board_move)) == board_move

        from keisei.shogi.shogi_core_definitions import PieceType
        drop_move = (None, None, 4, 6, PieceType.SILVER)
        assert PolicyOutputMapper.flip_move(PolicyOutputMapper.flip_move(drop_move)) == drop_move

    def test_get_legal_mask_perspective_black_unchanged(self):
        """For Black (is_white=False), mask is identical to non-perspective version."""
        import torch
        from keisei.shogi import ShogiGame
        game = ShogiGame()
        legal_moves = game.get_legal_moves()
        device = torch.device("cpu")
        mask_normal = self.mapper.get_legal_mask(legal_moves, device)
        mask_perspective = self.mapper.get_legal_mask_perspective(legal_moves, device, is_white=False)
        assert torch.equal(mask_normal, mask_perspective)

    def test_get_legal_mask_perspective_white_differs(self):
        """For White, the perspective mask differs from the absolute mask."""
        import torch
        from keisei.shogi import ShogiGame
        game = ShogiGame()
        game.make_move((6, 6, 5, 6, False))  # Black moves, now White's turn
        legal_moves = game.get_legal_moves()
        device = torch.device("cpu")
        mask_absolute = self.mapper.get_legal_mask(legal_moves, device)
        mask_perspective = self.mapper.get_legal_mask_perspective(legal_moves, device, is_white=True)
        assert not torch.equal(mask_absolute, mask_perspective)

    def test_perspective_roundtrip_white(self):
        """Flipping legal moves, picking an index, and flipping back produces a valid absolute move."""
        import torch
        from keisei.shogi import ShogiGame
        game = ShogiGame()
        game.make_move((6, 6, 5, 6, False))  # Now White's turn
        legal_moves = game.get_legal_moves()
        device = torch.device("cpu")

        # Get perspective mask
        mask = self.mapper.get_legal_mask_perspective(legal_moves, device, is_white=True)

        # Pick first legal action from the perspective mask
        legal_indices = torch.where(mask)[0]
        assert len(legal_indices) > 0
        idx = int(legal_indices[0].item())

        # Convert back to absolute move
        absolute_move = self.mapper.perspective_index_to_absolute_move(idx, is_white=True)

        # The absolute move must be in the original legal moves list
        assert absolute_move in legal_moves, (
            f"Roundtrip move {absolute_move} not in legal_moves"
        )

    def test_perspective_roundtrip_all_white_moves(self):
        """Every legal move for White survives the perspective roundtrip."""
        import torch
        from keisei.shogi import ShogiGame
        game = ShogiGame()
        game.make_move((6, 6, 5, 6, False))  # Now White's turn
        legal_moves = game.get_legal_moves()
        device = torch.device("cpu")

        mask = self.mapper.get_legal_mask_perspective(legal_moves, device, is_white=True)
        legal_indices = torch.where(mask)[0]

        # Number of legal indices should equal number of legal moves
        assert len(legal_indices) == len(legal_moves)

        # Every index should roundtrip to a legal move
        recovered_moves = set()
        for idx in legal_indices:
            move = self.mapper.perspective_index_to_absolute_move(int(idx.item()), is_white=True)
            assert move in legal_moves, f"Roundtripped move {move} not in legal_moves"
            recovered_moves.add(move)

        # Should recover all legal moves (no duplicates lost)
        assert len(recovered_moves) == len(legal_moves)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_policy_mapper.py::TestPerspectiveAlignment -v`
Expected: FAIL — `flip_move`, `get_legal_mask_perspective`, and `perspective_index_to_absolute_move` do not exist.

- [ ] **Step 3: Implement the three methods in PolicyOutputMapper**

Add to `PolicyOutputMapper` class in `keisei/utils/utils.py`, after the existing `get_legal_mask` method (around line 342):

```python
@staticmethod
def flip_move(move: "MoveTuple") -> "MoveTuple":
    """Flip move coordinates 180° for perspective transformation.

    Board moves: (r, c) -> (8-r, 8-c) for both source and destination.
    Drop moves: only the target square is flipped.
    The promotion flag / piece type (element 4) is preserved.
    """
    if move[0] is None:  # Drop move: (None, None, to_r, to_c, piece_type)
        return (None, None, 8 - move[2], 8 - move[3], move[4])
    # Board move: (from_r, from_c, to_r, to_c, promote)
    return (8 - move[0], 8 - move[1], 8 - move[2], 8 - move[3], move[4])

def get_legal_mask_perspective(
    self,
    legal_shogi_moves: List["MoveTuple"],
    device: torch.device,
    is_white: bool,
) -> torch.Tensor:
    """Create a legal move mask in the current player's perspective space.

    When is_white=True, legal moves (in absolute coordinates) are flipped
    to perspective coordinates before index lookup, so the mask aligns with
    the 180°-rotated observation that White receives.

    When is_white=False, this is identical to get_legal_mask().
    """
    if not is_white:
        return self.get_legal_mask(legal_shogi_moves, device)
    flipped_moves = [self.flip_move(m) for m in legal_shogi_moves]
    return self.get_legal_mask(flipped_moves, device)

def perspective_index_to_absolute_move(
    self, idx: int, is_white: bool
) -> "MoveTuple":
    """Convert a policy index (in perspective space) back to an absolute move.

    When is_white=True, the move stored at idx is in perspective coordinates,
    so it is flipped back to absolute coordinates for the game engine.

    When is_white=False, this is identical to policy_index_to_shogi_move().
    """
    move = self.policy_index_to_shogi_move(idx)
    if is_white:
        return self.flip_move(move)
    return move
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_policy_mapper.py::TestPerspectiveAlignment -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/utils/utils.py tests/unit/test_policy_mapper.py
git commit -m "feat: add perspective-aware action mapping to PolicyOutputMapper

Adds flip_move(), get_legal_mask_perspective(), and
perspective_index_to_absolute_move() to fix the observation-action
coordinate mismatch for White's perspective in self-play."
```

---

### Task 2: Update training StepManager to use perspective-aware mapping

**Files:**
- Modify: `keisei/training/step_manager.py:225-345` (execute_step method)
- Test: `tests/unit/test_step_manager_perspective.py`

- [ ] **Step 1: Write failing test for perspective-aware step execution**

Create `tests/unit/test_step_manager_perspective.py`:

```python
"""Tests that StepManager uses perspective-aware action mapping."""

import numpy as np
import pytest
import torch

from keisei.shogi import Color, ShogiGame
from keisei.utils.utils import PolicyOutputMapper


class TestStepManagerPerspective:
    """Verify that legal mask and move conversion respect player perspective."""

    def setup_method(self):
        self.mapper = PolicyOutputMapper()

    def test_observation_action_alignment_initial_position(self):
        """At the start position, Black's obs and action space should be spatially aligned.

        Black's pawn at absolute (6,0) appears at obs position (6,0).
        The action for pushing that pawn should use obs-space coordinates.
        """
        game = ShogiGame()
        legal_moves = game.get_legal_moves()
        is_white = game.current_player == Color.WHITE
        mask = self.mapper.get_legal_mask_perspective(
            legal_moves, torch.device("cpu"), is_white=is_white
        )

        # Black's pawn push (6,0)->(5,0): in Black's obs, pawn is at (6,0)
        pawn_push = (6, 0, 5, 0, False)
        idx = self.mapper.shogi_move_to_policy_index(pawn_push)
        assert mask[idx], "Black's pawn push should be legal in perspective mask"

    def test_observation_action_alignment_white_turn(self):
        """After Black moves, White's obs is flipped. The perspective mask should
        map White's legal moves into the same flipped coordinate space.

        White's pawn at absolute (2,0) appears at obs position (6,8).
        The perspective mask should have a 1 at the index corresponding to
        the perspective-space move (6,8)->(5,8), NOT the absolute (2,0)->(3,0).
        """
        game = ShogiGame()
        game.make_move((6, 6, 5, 6, False))  # Black moves, now White's turn
        assert game.current_player == Color.WHITE

        legal_moves = game.get_legal_moves()
        is_white = True
        mask = self.mapper.get_legal_mask_perspective(
            legal_moves, torch.device("cpu"), is_white=is_white
        )

        # White's pawn at absolute (2,0), flipped to (6,8) in obs
        # Push to absolute (3,0), flipped to (5,8) in obs
        # Perspective-space move: (6,8)->(5,8)
        perspective_move = (6, 8, 5, 8, False)
        idx = self.mapper.shogi_move_to_policy_index(perspective_move)
        assert mask[idx], (
            "White's pawn push should be legal at perspective-space index"
        )

        # The absolute-space index should NOT be set (that's the old broken behavior)
        absolute_move = (2, 0, 3, 0, False)
        abs_idx = self.mapper.shogi_move_to_policy_index(absolute_move)
        assert not mask[abs_idx], (
            "Absolute-space index should NOT be set in perspective mask"
        )

    def test_perspective_move_roundtrip_produces_valid_game_move(self):
        """Selecting from the perspective mask and converting back gives a valid game move."""
        game = ShogiGame()
        game.make_move((6, 6, 5, 6, False))  # Now White's turn

        legal_moves = game.get_legal_moves()
        is_white = True
        mask = self.mapper.get_legal_mask_perspective(
            legal_moves, torch.device("cpu"), is_white=is_white
        )

        # Pick every legal perspective index and verify roundtrip
        legal_indices = torch.where(mask)[0]
        for idx in legal_indices:
            absolute_move = self.mapper.perspective_index_to_absolute_move(
                int(idx.item()), is_white=is_white
            )
            assert absolute_move in legal_moves, (
                f"Roundtripped move {absolute_move} (from perspective idx {idx}) "
                f"not in legal_moves"
            )
```

- [ ] **Step 2: Run tests to verify they pass** (these only test the mapper methods from Task 1)

Run: `uv run pytest tests/unit/test_step_manager_perspective.py -v`
Expected: All 3 tests PASS (they test the mapper, not step_manager internals yet).

- [ ] **Step 3: Update step_manager.py execute_step to use perspective-aware methods**

In `keisei/training/step_manager.py`, modify `execute_step()`. Change the legal mask creation and add move flipping after action selection:

Replace lines 264-266 (legal mask creation):
```python
            legal_mask_tensor = self.policy_mapper.get_legal_mask(
                legal_shogi_moves, device=self.device
            )
```
with:
```python
            is_white = self.game.current_player == Color.WHITE
            legal_mask_tensor = self.policy_mapper.get_legal_mask_perspective(
                legal_shogi_moves, device=self.device, is_white=is_white
            )
```

Then after the agent selects an action (after line 277, the closing paren of `self.agent.select_action`), add perspective-to-absolute conversion. Replace lines 273-277:
```python
            selected_shogi_move, policy_index, log_prob, value_pred = (
                self.agent.select_action(
                    episode_state.current_obs, legal_mask_tensor, is_training=True
                )
            )
```
with:
```python
            selected_shogi_move, policy_index, log_prob, value_pred = (
                self.agent.select_action(
                    episode_state.current_obs, legal_mask_tensor, is_training=True
                )
            )

            # Convert perspective-space move back to absolute coordinates
            if selected_shogi_move is not None and is_white:
                selected_shogi_move = self.policy_mapper.flip_move(
                    selected_shogi_move
                )
```

- [ ] **Step 4: Run existing tests to check for regressions**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All existing tests PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/step_manager.py tests/unit/test_step_manager_perspective.py
git commit -m "fix: use perspective-aware action mapping in StepManager

When White plays, legal moves are now flipped to perspective coordinates
for the legal mask, and the selected move is flipped back to absolute
coordinates before being applied to the game. This aligns the action
space with the 180-rotated observation White receives."
```

---

### Task 3: Update parallel self-play worker

**Files:**
- Modify: `keisei/training/parallel/self_play_worker.py:190-250`

- [ ] **Step 1: Update self_play_worker.py to use perspective-aware methods**

In `keisei/training/parallel/self_play_worker.py`, add import for Color at the top (near line 16):

```python
from keisei.shogi.shogi_core_definitions import Color
```

Then modify the action selection block (around lines 197-239). Replace lines 198-202:
```python
                legal_moves = self.game.get_legal_moves()
                if self.policy_mapper is not None:
                    legal_mask = self.policy_mapper.get_legal_mask(
                        legal_moves, self.device
                    )
```
with:
```python
                legal_moves = self.game.get_legal_moves()
                is_white = self.game.current_player == Color.WHITE
                if self.policy_mapper is not None:
                    legal_mask = self.policy_mapper.get_legal_mask_perspective(
                        legal_moves, self.device, is_white=is_white
                    )
```

Then replace lines 219-231 (action-to-move conversion):
```python
            if self.policy_mapper is not None:
                try:
                    selected_move = self.policy_mapper.policy_index_to_shogi_move(
                        int(action.item())
                    )
                except (IndexError, ValueError) as e:
                    logger.error(
                        "Worker %d invalid action %d: %s",
                        self.worker_id,
                        int(action.item()),
                        str(e),
                    )
                    return None
```
with:
```python
            if self.policy_mapper is not None:
                try:
                    selected_move = self.policy_mapper.perspective_index_to_absolute_move(
                        int(action.item()), is_white=is_white
                    )
                except (IndexError, ValueError) as e:
                    logger.error(
                        "Worker %d invalid action %d: %s",
                        self.worker_id,
                        int(action.item()),
                        str(e),
                    )
                    return None
```

- [ ] **Step 2: Run existing tests**

Run: `uv run pytest tests/ -x -q`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add keisei/training/parallel/self_play_worker.py
git commit -m "fix: use perspective-aware action mapping in self-play worker

Parallel self-play workers now flip legal moves and selected actions
for White's perspective, matching the fix in StepManager."
```

---

### Task 4: Update evaluation strategies

All four evaluation strategies (single_opponent, tournament, ladder, benchmark) and the scheduler follow the same pattern: get legal moves, create mask, get action, validate move, make move. Each needs two changes: (1) perspective-aware mask, (2) flip the move back before validation/execution.

**Files:**
- Modify: `keisei/evaluation/strategies/single_opponent.py`
- Modify: `keisei/evaluation/strategies/ladder.py`
- Modify: `keisei/evaluation/strategies/benchmark.py`
- Modify: `keisei/evaluation/strategies/tournament.py`
- Modify: `keisei/evaluation/scheduler.py`

- [ ] **Step 1: Update single_opponent.py**

In `_get_player_action` (line 97), the method receives a pre-built `legal_mask` and passes it to `select_action`. The move returned by `select_action` is in perspective space and must be flipped back. But heuristic opponents (`select_move`) don't use the mask at all — they get moves directly from the game, so they're fine.

Modify `_run_game_loop` (around line 197). Replace:
```python
            legal_mask = self.policy_mapper.get_legal_mask(legal_moves, device_obj)
```
with:
```python
            is_white = game.current_player == Color.WHITE
            legal_mask = self.policy_mapper.get_legal_mask_perspective(
                legal_moves, device_obj, is_white=is_white
            )
```

Then modify `_get_player_action` to accept and use `is_white`. Change the method signature and the `select_action` branch (lines 97-118). Replace:
```python
    async def _get_player_action(
        self, player_entity: Any, game: ShogiGame, legal_mask: Any
    ) -> Any:
        """Gets an action from the player entity (agent or opponent)."""
        move = None
        if hasattr(player_entity, "select_action"):  # PPOAgent-like
            move_tuple = player_entity.select_action(
                game.get_observation(),
                legal_mask,
                is_training=False,
            )
            if move_tuple is not None:
                move = move_tuple[0] if isinstance(move_tuple, tuple) else move_tuple
        elif hasattr(player_entity, "select_move"):  # Heuristic or other BaseOpponent
            move = player_entity.select_move(game)
        else:
            logger.error(
                f"Player entity of type {type(player_entity)} does not have a recognized action selection method."
            )
            raise TypeError(f"Unsupported player entity type: {type(player_entity)}")
        return move
```
with:
```python
    async def _get_player_action(
        self, player_entity: Any, game: ShogiGame, legal_mask: Any,
        is_white: bool = False,
    ) -> Any:
        """Gets an action from the player entity (agent or opponent)."""
        move = None
        if hasattr(player_entity, "select_action"):  # PPOAgent-like
            move_tuple = player_entity.select_action(
                game.get_observation(),
                legal_mask,
                is_training=False,
            )
            if move_tuple is not None:
                move = move_tuple[0] if isinstance(move_tuple, tuple) else move_tuple
            # Convert perspective-space move back to absolute coordinates
            if move is not None and is_white:
                move = self.policy_mapper.flip_move(move)
        elif hasattr(player_entity, "select_move"):  # Heuristic or other BaseOpponent
            move = player_entity.select_move(game)
        else:
            logger.error(
                f"Player entity of type {type(player_entity)} does not have a recognized action selection method."
            )
            raise TypeError(f"Unsupported player entity type: {type(player_entity)}")
        return move
```

Then update the call in `_run_game_loop` (around line 201):
```python
                move = await self._get_player_action(
                    current_player_entity, game, legal_mask
                )
```
to:
```python
                move = await self._get_player_action(
                    current_player_entity, game, legal_mask, is_white=is_white
                )
```

- [ ] **Step 2: Update ladder.py**

Same pattern. In `_game_process_one_turn` (line 280), replace:
```python
        legal_mask = self.policy_mapper.get_legal_mask(legal_moves, device_obj)
```
with:
```python
        is_white = game.current_player == Color.WHITE
        legal_mask = self.policy_mapper.get_legal_mask_perspective(
            legal_moves, device_obj, is_white=is_white
        )
```

Update `_game_get_player_action` signature and body (lines 157-178). Replace:
```python
    async def _game_get_player_action(
        self, player_entity: Any, game: ShogiGame, legal_mask: Any
    ) -> Any:
        """Gets an action from the player entity (agent or opponent)."""
        move = None
        if hasattr(player_entity, "select_action"):  # PPOAgent-like
            move_tuple = player_entity.select_action(
                game.get_observation(),
                legal_mask,
                is_training=False,
            )
            if move_tuple is not None:
                move = move_tuple[0] if isinstance(move_tuple, tuple) else move_tuple
        elif hasattr(player_entity, "select_move"):  # Heuristic or other BaseOpponent
            move = player_entity.select_move(game)
        else:
            self.logger.error(
                "Player entity of type %s does not have a recognized action selection method.",
                type(player_entity).__name__,
            )
            raise TypeError(f"Unsupported player entity type: {type(player_entity)}")
        return move
```
with:
```python
    async def _game_get_player_action(
        self, player_entity: Any, game: ShogiGame, legal_mask: Any,
        is_white: bool = False,
    ) -> Any:
        """Gets an action from the player entity (agent or opponent)."""
        move = None
        if hasattr(player_entity, "select_action"):  # PPOAgent-like
            move_tuple = player_entity.select_action(
                game.get_observation(),
                legal_mask,
                is_training=False,
            )
            if move_tuple is not None:
                move = move_tuple[0] if isinstance(move_tuple, tuple) else move_tuple
            # Convert perspective-space move back to absolute coordinates
            if move is not None and is_white:
                move = self.policy_mapper.flip_move(move)
        elif hasattr(player_entity, "select_move"):  # Heuristic or other BaseOpponent
            move = player_entity.select_move(game)
        else:
            self.logger.error(
                "Player entity of type %s does not have a recognized action selection method.",
                type(player_entity).__name__,
            )
            raise TypeError(f"Unsupported player entity type: {type(player_entity)}")
        return move
```

Update the call in `_game_process_one_turn` (around line 283):
```python
            move = await self._game_get_player_action(player_entity, game, legal_mask)
```
to:
```python
            move = await self._game_get_player_action(
                player_entity, game, legal_mask, is_white=is_white
            )
```

- [ ] **Step 3: Update benchmark.py**

Same pattern as ladder.py. In `_game_process_one_turn` (line 349), replace:
```python
        legal_mask = self.policy_mapper.get_legal_mask(legal_moves, device_obj)
```
with:
```python
        is_white = game.current_player == Color.WHITE
        legal_mask = self.policy_mapper.get_legal_mask_perspective(
            legal_moves, device_obj, is_white=is_white
        )
```

Update `_game_get_player_action` (lines 228-249) — same replacement as ladder.py:
```python
    async def _game_get_player_action(
        self, player_entity: Any, game: ShogiGame, legal_mask: Any,
        is_white: bool = False,
    ) -> Any:
        """Gets an action from the player entity (agent or opponent)."""
        move = None
        if hasattr(player_entity, "select_action"):  # PPOAgent-like
            move_tuple = player_entity.select_action(
                game.get_observation(),
                legal_mask,
                is_training=False,
            )
            if move_tuple is not None:
                move = move_tuple[0] if isinstance(move_tuple, tuple) else move_tuple
            # Convert perspective-space move back to absolute coordinates
            if move is not None and is_white:
                move = self.policy_mapper.flip_move(move)
        elif hasattr(player_entity, "select_move"):  # Heuristic or other BaseOpponent
            move = player_entity.select_move(game)
        else:
            self.logger.error(
                "Player entity of type %s does not have a recognized action selection method.",
                type(player_entity).__name__,
            )
            raise TypeError(f"Unsupported player entity type: {type(player_entity)}")
        return move
```

Update the call in `_game_process_one_turn`:
```python
            move = await self._game_get_player_action(player_entity, game, legal_mask)
```
to:
```python
            move = await self._game_get_player_action(
                player_entity, game, legal_mask, is_white=is_white
            )
```

- [ ] **Step 4: Update tournament.py**

In the game loop (around line 246), replace:
```python
                    legal_mask = self.policy_mapper.get_legal_mask(
                        legal_moves, current_player.device
                    )
```
with:
```python
                    is_white = game.current_player == Color.WHITE
                    legal_mask = self.policy_mapper.get_legal_mask_perspective(
                        legal_moves, current_player.device, is_white=is_white
                    )
```

Tournament has a `_get_player_action` method that also needs updating. Find it and add the `is_white` parameter and flip logic following the same pattern as the other strategies. The tournament `_get_player_action` likely has the same structure — add `is_white: bool = False` parameter, flip the move when `is_white` and the entity is a `select_action` type.

Then update the call site to pass `is_white=is_white`.

- [ ] **Step 5: Update scheduler.py**

In `scheduler.py` (around line 376), replace:
```python
            legal_mask = self._policy_mapper.get_legal_mask(
                legal_moves, device=agent_device
            )
```
with:
```python
            is_white = game.current_player == Color.WHITE
            legal_mask = self._policy_mapper.get_legal_mask_perspective(
                legal_moves, device=agent_device, is_white=is_white
            )
```

The scheduler uses `agent.select_action` directly (line 387). After the move is returned, flip it back. Find where `selected_move` is used after `select_action` returns, and add:
```python
            # Convert perspective-space move back to absolute coordinates
            if selected_move is not None and is_white:
                selected_move = self._policy_mapper.flip_move(selected_move)
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add keisei/evaluation/strategies/single_opponent.py \
       keisei/evaluation/strategies/ladder.py \
       keisei/evaluation/strategies/benchmark.py \
       keisei/evaluation/strategies/tournament.py \
       keisei/evaluation/scheduler.py
git commit -m "fix: use perspective-aware action mapping in all evaluation strategies

Updates single_opponent, ladder, benchmark, tournament strategies and
the scheduler to flip legal moves for White's perspective mask and
convert selected moves back to absolute coordinates."
```

---

### Task 5: Add observation-action symmetry regression test

**Files:**
- Create: `tests/unit/test_observation_action_symmetry.py`

- [ ] **Step 1: Write the symmetry regression test**

Create `tests/unit/test_observation_action_symmetry.py`:

```python
"""
Regression test for observation-action perspective alignment.

Verifies that the spatial structure of the observation matches the spatial
structure of the action space for both Black and White. This test would have
caught the original bug where the observation was flipped for White but the
action space was not.

See: docs/superpowers/specs/2026-03-31-perspective-action-alignment-design.md
"""

import numpy as np
import pytest
import torch

from keisei.shogi import Color, ShogiGame
from keisei.utils.utils import PolicyOutputMapper


class TestObservationActionSymmetry:
    """The observation and action space must be spatially consistent for both colors."""

    def setup_method(self):
        self.mapper = PolicyOutputMapper()

    def _get_pawn_obs_positions(self, obs: np.ndarray) -> list:
        """Get (row, col) positions where own-pawn plane (ch 0) has a 1."""
        positions = []
        for r in range(9):
            for c in range(9):
                if obs[0, r, c] == 1.0:
                    positions.append((r, c))
        return positions

    def test_black_pawn_push_spatial_alignment(self):
        """Black's pawn in obs is at the same coords as the action that moves it."""
        game = ShogiGame()
        assert game.current_player == Color.BLACK
        obs = game.get_observation()

        # Black's pawns in obs should be at row 6
        pawn_positions = self._get_pawn_obs_positions(obs)
        assert all(r == 6 for r, c in pawn_positions)

        # The pawn push action for each pawn should use the same obs-space coords
        legal_moves = game.get_legal_moves()
        is_white = False
        mask = self.mapper.get_legal_mask_perspective(
            legal_moves, torch.device("cpu"), is_white=is_white
        )

        for r, c in pawn_positions:
            # Pawn push: (r, c) -> (r-1, c) in obs space
            push_move = (r, c, r - 1, c, False)
            idx = self.mapper.shogi_move_to_policy_index(push_move)
            assert mask[idx], f"Black pawn push from obs ({r},{c}) should be legal"

    def test_white_pawn_push_spatial_alignment(self):
        """White's pawn in obs is at the same coords as the action that moves it.

        This is the key regression test. Previously, White's obs was flipped but
        the action space was not, so the pawn at obs position (6,8) had its action
        at absolute-space index for (2,0)->(3,0) instead of obs-space (6,8)->(5,8).
        """
        game = ShogiGame()
        game.make_move((6, 6, 5, 6, False))  # Black moves, now White's turn
        assert game.current_player == Color.WHITE
        obs = game.get_observation()

        # White's pawns in obs should be at row 6 (flipped from absolute row 2)
        pawn_positions = self._get_pawn_obs_positions(obs)
        assert all(r == 6 for r, c in pawn_positions), (
            f"White's pawns should appear at row 6 in flipped obs, got {pawn_positions}"
        )

        legal_moves = game.get_legal_moves()
        is_white = True
        mask = self.mapper.get_legal_mask_perspective(
            legal_moves, torch.device("cpu"), is_white=is_white
        )

        for r, c in pawn_positions:
            # Pawn push in obs space: (r, c) -> (r-1, c)
            push_move = (r, c, r - 1, c, False)
            idx = self.mapper.shogi_move_to_policy_index(push_move)
            assert mask[idx], (
                f"White pawn push from obs ({r},{c}) should be legal in perspective mask"
            )

    def test_mirror_symmetry_at_start(self):
        """Black's observation at move 1 and White's observation at move 2
        should have identical own-piece planes (both see their army at bottom).

        This verifies the observation encoder flips correctly AND that the
        perspective-aware action mapping preserves this symmetry.
        """
        game = ShogiGame()
        obs_black = game.get_observation().copy()

        # Make a neutral move (Black pawn push)
        game.make_move((6, 6, 5, 6, False))
        obs_white = game.get_observation().copy()

        # Own king plane (ch 7): both should have king at (8, 4) in obs space
        assert obs_black[7, 8, 4] == 1.0, "Black's king should be at obs (8,4)"
        assert obs_white[7, 8, 4] == 1.0, "White's king should be at obs (8,4)"

        # Own pawn plane (ch 0): both should have pawns at row 6
        # (White's pawns are at absolute row 2, flipped to row 6)
        black_pawn_row_sum = obs_black[0, 6, :].sum()
        white_pawn_row_sum = obs_white[0, 6, :].sum()
        assert black_pawn_row_sum == 9.0, "Black should have 9 pawns at row 6"
        assert white_pawn_row_sum == 9.0, "White should have 9 pawns at row 6"

        # Opponent king plane (ch 21): both should have opponent king at (0, 4)
        assert obs_black[21, 0, 4] == 1.0, "White king at obs (0,4) from Black's view"
        assert obs_white[21, 0, 4] == 1.0, "Black king at obs (0,4) from White's view"

    def test_all_white_legal_moves_roundtrip(self):
        """Every White legal move must survive the perspective flip roundtrip."""
        game = ShogiGame()
        game.make_move((6, 6, 5, 6, False))  # Now White's turn

        legal_moves = game.get_legal_moves()
        mask = self.mapper.get_legal_mask_perspective(
            legal_moves, torch.device("cpu"), is_white=True
        )

        legal_indices = torch.where(mask)[0]
        assert len(legal_indices) == len(legal_moves), (
            f"Perspective mask has {len(legal_indices)} legal actions but "
            f"game has {len(legal_moves)} legal moves"
        )

        for idx in legal_indices:
            absolute_move = self.mapper.perspective_index_to_absolute_move(
                int(idx.item()), is_white=True
            )
            assert absolute_move in legal_moves
```

- [ ] **Step 2: Run the symmetry tests**

Run: `uv run pytest tests/unit/test_observation_action_symmetry.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_observation_action_symmetry.py
git commit -m "test: add observation-action symmetry regression tests

Verifies that the spatial structure of the observation matches the
action space for both Black and White. These tests would have caught
the original perspective mismatch bug."
```

---

### Task 6: Final integration verification

- [ ] **Step 1: Run the empirical verification script**

```bash
uv run python3 -c "
import numpy as np
import torch
from keisei.shogi import ShogiGame, Color
from keisei.utils.utils import PolicyOutputMapper

mapper = PolicyOutputMapper()
game = ShogiGame()
game.make_move((6, 6, 5, 6, False))  # Now White's turn
obs = game.get_observation()

# White's pawn at absolute (2,0) appears at obs (6,8)
assert obs[0, 6, 8] == 1.0, 'White pawn should be at obs (6,8)'

# Get perspective mask
legal_moves = game.get_legal_moves()
mask = mapper.get_legal_mask_perspective(legal_moves, torch.device('cpu'), is_white=True)

# Perspective-space move (6,8)->(5,8) should be legal
persp_move = (6, 8, 5, 8, False)
idx = mapper.shogi_move_to_policy_index(persp_move)
assert mask[idx], 'Perspective pawn push should be legal'

# Roundtrip back to absolute
abs_move = mapper.perspective_index_to_absolute_move(idx, is_white=True)
assert abs_move == (2, 0, 3, 0, False), f'Expected (2,0,3,0,False), got {abs_move}'
assert abs_move in legal_moves, 'Roundtripped move must be in legal moves'

print('All perspective alignment checks passed!')
"
```
Expected: "All perspective alignment checks passed!"

- [ ] **Step 2: Run full test suite one final time**

Run: `uv run pytest tests/ -q`
Expected: All tests PASS.

- [ ] **Step 3: Final commit (if any remaining changes)**

Only if there are unstaged changes from fixes during verification.
