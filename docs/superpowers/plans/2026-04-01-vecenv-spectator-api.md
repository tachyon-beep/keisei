# VecEnv Spectator & Stats API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add spectator data extraction, SFEN access, and episode-level stats to the shogi-gym Rust crate, eliminating Python-side state mirroring.

**Architecture:** Extract shared dict-building logic into `spectator_data.rs`, add methods to `VecEnv` and `SpectatorEnv` that call it. Add one new atomic counter to VecEnv for mean episode length tracking. All changes are within the `shogi-gym` crate.

**Tech Stack:** Rust, PyO3 0.23, numpy 0.23, Python 3.13

**Spec:** `docs/superpowers/specs/2026-04-01-vecenv-spectator-api-design.md`

**Test commands:**
- Rust unit tests: `cd shogi-engine && cargo test -p shogi-gym`
- Python integration tests: `cd shogi-engine/crates/shogi-gym && maturin develop --release && python -m pytest tests/ -v`

---

### Task 1: Extract shared `build_spectator_dict()` into `spectator_data.rs`

**Files:**
- Create: `shogi-engine/crates/shogi-gym/src/spectator_data.rs`
- Modify: `shogi-engine/crates/shogi-gym/src/lib.rs`
- Modify: `shogi-engine/crates/shogi-gym/src/spectator.rs`
- Test: `shogi-engine/crates/shogi-gym/tests/test_spectator.py`

This task extracts the dict-building logic from `SpectatorEnv::to_dict()` into a shared free function, then refactors `SpectatorEnv::to_dict()` to call it. No new features yet — just the refactor. Existing tests must still pass afterward.

- [ ] **Step 1: Create `spectator_data.rs` with shared helpers and `build_spectator_dict()`**

Move the helper functions (`piece_type_name`, `color_name`, `game_result_str`) and the dict-building logic from `spectator.rs` into a new file. The function builds the same dict as `SpectatorEnv::to_dict()` minus `move_history`.

```rust
// shogi-engine/crates/shogi-gym/src/spectator_data.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use shogi_core::{Color, GameResult, GameState, HandPieceType, PieceType, Square};

// ---------------------------------------------------------------------------
// Helpers (moved from spectator.rs)
// ---------------------------------------------------------------------------

pub fn piece_type_name(pt: PieceType) -> &'static str {
    match pt {
        PieceType::Pawn   => "pawn",
        PieceType::Lance  => "lance",
        PieceType::Knight => "knight",
        PieceType::Silver => "silver",
        PieceType::Gold   => "gold",
        PieceType::Bishop => "bishop",
        PieceType::Rook   => "rook",
        PieceType::King   => "king",
    }
}

pub fn color_name(c: Color) -> &'static str {
    match c {
        Color::Black => "black",
        Color::White => "white",
    }
}

pub fn game_result_str(r: &GameResult) -> &'static str {
    match r {
        GameResult::InProgress        => "in_progress",
        GameResult::Checkmate { .. }  => "checkmate",
        GameResult::Repetition        => "repetition",
        GameResult::PerpetualCheck { .. } => "perpetual_check",
        GameResult::Impasse { .. }    => "impasse",
        GameResult::MaxMoves          => "max_moves",
    }
}

// ---------------------------------------------------------------------------
// Shared dict builder
// ---------------------------------------------------------------------------

/// Build a spectator-format Python dict from a GameState.
///
/// Returns dict with keys: board, hands, current_player, ply, is_over,
/// result, sfen, in_check. Does NOT include move_history — the caller
/// appends that if available (SpectatorEnv has it, VecEnv does not).
pub fn build_spectator_dict(py: Python<'_>, game: &GameState) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);

    // -- board: list of 81 elements (None or piece dict) --
    let board_list = PyList::empty(py);
    for idx in 0..81usize {
        let sq = Square::new_unchecked(idx as u8);
        match game.position.piece_at(sq) {
            None => board_list.append(py.None())?,
            Some(piece) => {
                let pd = PyDict::new(py);
                pd.set_item("type", piece_type_name(piece.piece_type()))?;
                pd.set_item("color", color_name(piece.color()))?;
                pd.set_item("promoted", piece.is_promoted())?;
                pd.set_item("row", sq.row() as i64)?;
                pd.set_item("col", sq.col() as i64)?;
                board_list.append(pd)?;
            }
        }
    }
    d.set_item("board", board_list)?;

    // -- hands --
    let hands_dict = PyDict::new(py);
    for &color in &[Color::Black, Color::White] {
        let hand_dict = PyDict::new(py);
        for &hpt in &HandPieceType::ALL {
            let count = game.position.hand_count(color, hpt) as i64;
            hand_dict.set_item(piece_type_name(hpt.to_piece_type()), count)?;
        }
        hands_dict.set_item(color_name(color), hand_dict)?;
    }
    d.set_item("hands", hands_dict)?;

    // -- scalar fields --
    d.set_item("current_player", color_name(game.position.current_player))?;
    d.set_item("ply", game.ply as i64)?;
    d.set_item("is_over", game.result.is_terminal())?;
    d.set_item("result", game_result_str(&game.result))?;
    d.set_item("sfen", game.position.to_sfen())?;
    d.set_item("in_check", game.is_in_check())?;

    Ok(d.into())
}
```

- [ ] **Step 2: Register the new module in `lib.rs`**

Add `mod spectator_data;` to `shogi-engine/crates/shogi-gym/src/lib.rs`:

```rust
mod action_mapper;
mod observation;
mod spectator_data;
mod step_result;
mod vec_env;
mod spectator;
```

- [ ] **Step 3: Refactor `spectator.rs` to use shared builder**

Replace the helper functions and inline dict-building in `spectator.rs` with calls to `spectator_data`. The `hand_piece_char` and `move_notation` helpers stay in `spectator.rs` since they're only used for move history (which only SpectatorEnv tracks).

Replace the entire `spectator.rs` with:

```rust
use numpy::{PyArray3, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use shogi_core::{GameState, HandPieceType, Move};

use crate::action_mapper::{ActionMapper, DefaultActionMapper};
use crate::observation::{DefaultObservationGenerator, ObservationGenerator, BUFFER_LEN, NUM_CHANNELS};
use crate::spectator_data::{build_spectator_dict, color_name};

// ---------------------------------------------------------------------------
// Move notation helpers (SpectatorEnv only — VecEnv doesn't track moves)
// ---------------------------------------------------------------------------

/// Encode a drop piece char: P, L, N, S, G, B, R
fn hand_piece_char(hpt: HandPieceType) -> char {
    match hpt {
        HandPieceType::Pawn   => 'P',
        HandPieceType::Lance  => 'L',
        HandPieceType::Knight => 'N',
        HandPieceType::Silver => 'S',
        HandPieceType::Gold   => 'G',
        HandPieceType::Bishop => 'B',
        HandPieceType::Rook   => 'R',
    }
}

/// Build move notation string from a Move.
fn move_notation(mv: Move) -> String {
    match mv {
        Move::Board { from, to, promote } => {
            let from_col_shogi = 9 - from.col();
            let from_row_char = (b'a' + from.row()) as char;
            let to_col_shogi = 9 - to.col();
            let to_row_char = (b'a' + to.row()) as char;
            let promo_str = if promote { "+" } else { "" };
            format!(
                "{}{}→{}{}{}",
                from_col_shogi, from_row_char,
                to_col_shogi, to_row_char,
                promo_str
            )
        }
        Move::Drop { to, piece_type } => {
            let piece_char = hand_piece_char(piece_type);
            let to_col_shogi = 9 - to.col();
            let to_row_char = (b'a' + to.row()) as char;
            format!("{}*{}{}", piece_char, to_col_shogi, to_row_char)
        }
    }
}

// ---------------------------------------------------------------------------
// SpectatorEnv
// ---------------------------------------------------------------------------

/// Single-game environment for spectator/display use.
///
/// Key differences from VecEnv:
/// - Returns rich Python dicts (acceptable — not on the hot path)
/// - Does NOT auto-reset on game end — stays ended until explicitly `reset()`
/// - Provides `to_dict()` for JSON serialization (dashboard)
/// - `legal_actions()` returns list of valid action indices
#[pyclass]
pub struct SpectatorEnv {
    game: GameState,
    max_ply: u32,
    mapper: DefaultActionMapper,
    obs_gen: DefaultObservationGenerator,
    move_history: Vec<(usize, String)>,
}

#[pymethods]
impl SpectatorEnv {
    #[new]
    #[pyo3(signature = (max_ply = 500))]
    pub fn new(max_ply: u32) -> Self {
        SpectatorEnv {
            game: GameState::with_max_ply(max_ply),
            max_ply,
            mapper: DefaultActionMapper,
            obs_gen: DefaultObservationGenerator::new(),
            move_history: Vec::new(),
        }
    }

    /// Reset game to startpos, clear move history, return state dict.
    pub fn reset(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.game = GameState::with_max_ply(self.max_ply);
        self.move_history.clear();
        self.to_dict(py)
    }

    /// Apply an action to the game.
    pub fn step(&mut self, py: Python<'_>, action: usize) -> PyResult<Py<PyDict>> {
        if self.game.result.is_terminal() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot step: game is already over. Call reset() to start a new game.",
            ));
        }

        let perspective = self.game.position.current_player;
        let mv = <DefaultActionMapper as ActionMapper>::decode(&self.mapper, action, perspective)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        let notation = move_notation(mv);
        self.move_history.push((action, notation));

        self.game.make_move(mv);
        self.game.check_termination();

        self.to_dict(py)
    }

    /// Return current state as a rich Python dict.
    ///
    /// Uses shared `build_spectator_dict()` for board/hands/sfen/etc.,
    /// then appends move_history (which only SpectatorEnv tracks).
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d_bound = build_spectator_dict(py, &self.game)?;
        let d = d_bound.bind(py);

        // Append move_history (SpectatorEnv-only)
        let history_list = PyList::empty(py);
        for (action_idx, notation) in &self.move_history {
            let hd = PyDict::new(py);
            hd.set_item("action", *action_idx as i64)?;
            hd.set_item("notation", notation.as_str())?;
            history_list.append(hd)?;
        }
        d.set_item("move_history", history_list)?;

        Ok(d_bound)
    }

    /// Serialize current position to SFEN string.
    pub fn to_sfen(&self) -> String {
        self.game.position.to_sfen()
    }

    /// Return the observation as a shaped (46, 9, 9) numpy array.
    pub fn get_observation<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray3<f32>>> {
        let mut buffer = vec![0.0_f32; BUFFER_LEN];
        let perspective = self.game.position.current_player;
        self.obs_gen.generate(&self.game, perspective, &mut buffer);
        let array = buffer.to_pyarray(py);
        let shaped = array
            .reshape([NUM_CHANNELS, 9, 9])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(shaped.unbind())
    }

    /// Return a list of legal action indices for the current position.
    pub fn legal_actions(&mut self) -> Vec<usize> {
        let perspective = self.game.position.current_player;
        let moves = self.game.legal_moves();
        moves
            .into_iter()
            .map(|mv| self.mapper.encode(mv, perspective))
            .collect()
    }

    #[getter]
    pub fn is_over(&self) -> bool {
        self.game.result.is_terminal()
    }

    #[getter]
    pub fn current_player(&self) -> &str {
        color_name(self.game.position.current_player)
    }

    #[getter]
    pub fn ply(&self) -> u32 {
        self.game.ply
    }

    #[getter]
    pub fn action_space_size(&self) -> usize {
        self.mapper.action_space_size()
    }
}
```

- [ ] **Step 4: Build and run existing tests to verify refactor is clean**

```bash
cd shogi-engine && cargo test -p shogi-gym
```

Expected: All Rust unit tests pass (the refactor is behavior-preserving).

```bash
cd shogi-engine/crates/shogi-gym && maturin develop --release && python -m pytest tests/test_spectator.py -v
```

Expected: All 7 existing SpectatorEnv tests pass. No behavior change.

- [ ] **Step 5: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/spectator_data.rs \
       shogi-engine/crates/shogi-gym/src/spectator.rs \
       shogi-engine/crates/shogi-gym/src/lib.rs
git commit -m "refactor: extract shared build_spectator_dict() into spectator_data.rs"
```

---

### Task 2: Add `VecEnv.get_spectator_data()`

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/vec_env.rs`
- Test: `shogi-engine/crates/shogi-gym/tests/test_vec_env.py`

- [ ] **Step 1: Write failing Python tests for `get_spectator_data()`**

Add to `tests/test_vec_env.py`:

```python
class TestVecEnvSpectatorData:
    def test_get_spectator_data_returns_list(self):
        """get_spectator_data() returns a list of dicts, one per env."""
        env = VecEnv(num_envs=3, max_ply=100)
        env.reset()
        data = env.get_spectator_data()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_get_spectator_data_dict_keys(self):
        """Each dict has the expected keys (no move_history)."""
        env = VecEnv(num_envs=1, max_ply=100)
        env.reset()
        data = env.get_spectator_data()
        d = data[0]
        expected_keys = {"board", "hands", "current_player", "ply", "is_over", "result", "sfen", "in_check"}
        assert set(d.keys()) == expected_keys
        assert "move_history" not in d

    def test_get_spectator_data_startpos_values(self):
        """Verify startpos dict values are correct."""
        env = VecEnv(num_envs=1, max_ply=100)
        env.reset()
        d = env.get_spectator_data()[0]
        assert d["current_player"] == "black"
        assert d["ply"] == 0
        assert d["is_over"] is False
        assert d["result"] == "in_progress"
        assert d["in_check"] is False
        assert len(d["board"]) == 81
        assert "lnsgkgsnl" in d["sfen"].lower()

    def test_get_spectator_data_hands_structure(self):
        """Verify hands dict has correct structure."""
        env = VecEnv(num_envs=1, max_ply=100)
        env.reset()
        d = env.get_spectator_data()[0]
        assert "black" in d["hands"]
        assert "white" in d["hands"]
        assert "pawn" in d["hands"]["black"]
        assert d["hands"]["black"]["pawn"] == 0  # startpos: no captured pieces

    def test_get_spectator_data_after_step(self):
        """Verify dict updates after stepping."""
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        action = [int(np.where(masks[0])[0][0])]
        env.step(action)
        d = env.get_spectator_data()[0]
        assert d["ply"] == 1
        assert d["current_player"] == "white"

    def test_get_spectator_data_matches_spectator_env(self):
        """VecEnv and SpectatorEnv dicts should match (minus move_history)."""
        from shogi_gym import SpectatorEnv

        vec_env = VecEnv(num_envs=1, max_ply=100)
        spec_env = SpectatorEnv(max_ply=100)

        vec_env.reset()
        spec_env.reset()

        vec_d = vec_env.get_spectator_data()[0]
        spec_d = spec_env.to_dict()

        # Compare all shared keys
        for key in ("board", "hands", "current_player", "ply", "is_over", "result", "sfen", "in_check"):
            assert vec_d[key] == spec_d[key], f"mismatch on key '{key}': {vec_d[key]} != {spec_d[key]}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd shogi-engine/crates/shogi-gym && python -m pytest tests/test_vec_env.py::TestVecEnvSpectatorData -v
```

Expected: All 6 tests FAIL with `AttributeError: 'VecEnv' object has no attribute 'get_spectator_data'`

- [ ] **Step 3: Implement `get_spectator_data()` on VecEnv**

Add to `shogi-engine/crates/shogi-gym/src/vec_env.rs`. First, add the import at the top of the file:

```rust
use crate::spectator_data::build_spectator_dict;
```

Then add this method inside the `#[pymethods] impl VecEnv` block, after the `reset_stats` method (before the closing `}`):

```rust
    /// Return spectator-format dicts for all games.
    ///
    /// Each dict contains: board, hands, current_player, ply, is_over,
    /// result, sfen, in_check. Does NOT include move_history.
    pub fn get_spectator_data(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let mut result = Vec::with_capacity(self.num_envs);
        for game in &self.games {
            result.push(build_spectator_dict(py, game)?);
        }
        Ok(result)
    }
```

- [ ] **Step 4: Build and run tests**

```bash
cd shogi-engine/crates/shogi-gym && maturin develop --release && python -m pytest tests/test_vec_env.py::TestVecEnvSpectatorData -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
cd shogi-engine && cargo test -p shogi-gym && cd crates/shogi-gym && python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/vec_env.rs \
       shogi-engine/crates/shogi-gym/tests/test_vec_env.py
git commit -m "feat: add VecEnv.get_spectator_data() using shared dict builder"
```

---

### Task 3: Add `SpectatorEnv.from_sfen()`

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/spectator.rs`
- Test: `shogi-engine/crates/shogi-gym/tests/test_spectator.py`

- [ ] **Step 1: Write failing Python tests for `from_sfen()`**

Add to `tests/test_spectator.py`:

```python
class TestSpectatorFromSfen:
    STARTPOS_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

    def test_from_sfen_creates_env(self):
        """from_sfen returns a playable SpectatorEnv."""
        env = SpectatorEnv.from_sfen(self.STARTPOS_SFEN)
        assert env.current_player == "black"
        assert env.ply == 0
        assert not env.is_over
        assert env.action_space_size == 13_527

    def test_from_sfen_roundtrip(self):
        """to_sfen -> from_sfen -> to_sfen should produce identical SFEN."""
        env1 = SpectatorEnv()
        env1.reset()
        legal = env1.legal_actions()
        env1.step(legal[0])
        sfen = env1.to_sfen()

        env2 = SpectatorEnv.from_sfen(sfen)
        assert env2.to_sfen() == sfen

    def test_from_sfen_empty_move_history(self):
        """from_sfen creates env with empty move history."""
        env = SpectatorEnv.from_sfen(self.STARTPOS_SFEN)
        d = env.to_dict()
        assert len(d["move_history"]) == 0

    def test_from_sfen_playable(self):
        """Env created from SFEN can be stepped with legal actions."""
        env = SpectatorEnv.from_sfen(self.STARTPOS_SFEN)
        legal = env.legal_actions()
        assert len(legal) == 30  # startpos
        state = env.step(legal[0])
        assert state["ply"] == 1

    def test_from_sfen_custom_max_ply(self):
        """from_sfen respects custom max_ply."""
        env = SpectatorEnv.from_sfen(self.STARTPOS_SFEN, max_ply=1)
        legal = env.legal_actions()
        env.step(legal[0])
        assert env.is_over  # truncated at ply 1

    def test_from_sfen_invalid_raises(self):
        """Invalid SFEN should raise ValueError."""
        with pytest.raises(ValueError):
            SpectatorEnv.from_sfen("not a valid sfen")

    def test_from_sfen_with_hands(self):
        """SFEN with hand pieces produces correct state."""
        sfen_with_hands = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b 2P 1"
        env = SpectatorEnv.from_sfen(sfen_with_hands)
        d = env.to_dict()
        assert d["hands"]["black"]["pawn"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd shogi-engine/crates/shogi-gym && python -m pytest tests/test_spectator.py::TestSpectatorFromSfen -v
```

Expected: All 7 tests FAIL with `TypeError: SpectatorEnv.from_sfen() ...` (method doesn't exist)

- [ ] **Step 3: Implement `from_sfen()` on SpectatorEnv**

Add this method to the `#[pymethods] impl SpectatorEnv` block in `spectator.rs`, after the `new` method:

```rust
    /// Create a SpectatorEnv from a SFEN string.
    ///
    /// Args:
    ///     sfen: SFEN position string.
    ///     max_ply: Maximum plies before truncation (default 500).
    ///
    /// Raises ValueError if the SFEN is invalid.
    #[staticmethod]
    #[pyo3(signature = (sfen, max_ply = None))]
    pub fn from_sfen(sfen: &str, max_ply: Option<u32>) -> PyResult<Self> {
        let max_ply = max_ply.unwrap_or(500);
        let game = GameState::from_sfen(sfen, max_ply)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid SFEN: {e}")))?;
        Ok(SpectatorEnv {
            game,
            max_ply,
            mapper: DefaultActionMapper,
            obs_gen: DefaultObservationGenerator::new(),
            move_history: Vec::new(),
        })
    }
```

Note: This requires adding `GameState` to the existing import from `shogi_core` at the top of `spectator.rs`. The current import line is:

```rust
use shogi_core::{GameState, HandPieceType, Move};
```

`GameState` is already imported, so no change needed.

- [ ] **Step 4: Build and run tests**

```bash
cd shogi-engine/crates/shogi-gym && maturin develop --release && python -m pytest tests/test_spectator.py::TestSpectatorFromSfen -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
cd shogi-engine && cargo test -p shogi-gym && cd crates/shogi-gym && python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/spectator.rs \
       shogi-engine/crates/shogi-gym/tests/test_spectator.py
git commit -m "feat: add SpectatorEnv.from_sfen() static constructor"
```

---

### Task 4: Add `VecEnv.get_sfen()` and `VecEnv.get_sfens()`

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/vec_env.rs`
- Test: `shogi-engine/crates/shogi-gym/tests/test_vec_env.py`

- [ ] **Step 1: Write failing Python tests**

Add to `tests/test_vec_env.py`:

```python
class TestVecEnvSfen:
    def test_get_sfens_returns_list(self):
        """get_sfens() returns a list of strings."""
        env = VecEnv(num_envs=3, max_ply=100)
        env.reset()
        sfens = env.get_sfens()
        assert isinstance(sfens, list)
        assert len(sfens) == 3
        for s in sfens:
            assert isinstance(s, str)

    def test_get_sfens_startpos(self):
        """All SFENs at startpos should be identical."""
        env = VecEnv(num_envs=2, max_ply=100)
        env.reset()
        sfens = env.get_sfens()
        assert sfens[0] == sfens[1]
        assert "lnsgkgsnl" in sfens[0].lower()

    def test_get_sfen_single(self):
        """get_sfen(i) returns the same as get_sfens()[i]."""
        env = VecEnv(num_envs=2, max_ply=100)
        env.reset()
        sfens = env.get_sfens()
        for i in range(2):
            assert env.get_sfen(i) == sfens[i]

    def test_get_sfen_out_of_bounds(self):
        """get_sfen() with invalid index raises IndexError."""
        env = VecEnv(num_envs=2, max_ply=100)
        env.reset()
        with pytest.raises(IndexError):
            env.get_sfen(2)
        with pytest.raises(IndexError):
            env.get_sfen(100)

    def test_get_sfen_changes_after_step(self):
        """SFEN should change after stepping."""
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        sfen_before = env.get_sfen(0)
        masks = np.asarray(result.legal_masks)
        action = [int(np.where(masks[0])[0][0])]
        env.step(action)
        sfen_after = env.get_sfen(0)
        assert sfen_before != sfen_after

    def test_get_sfen_matches_spectator_data(self):
        """SFEN from get_sfen() should match sfen in get_spectator_data()."""
        env = VecEnv(num_envs=2, max_ply=100)
        env.reset()
        sfens = env.get_sfens()
        data = env.get_spectator_data()
        for i in range(2):
            assert sfens[i] == data[i]["sfen"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd shogi-engine/crates/shogi-gym && python -m pytest tests/test_vec_env.py::TestVecEnvSfen -v
```

Expected: All 6 tests FAIL with `AttributeError`

- [ ] **Step 3: Implement `get_sfen()` and `get_sfens()` on VecEnv**

Add these methods inside the `#[pymethods] impl VecEnv` block in `vec_env.rs`, after `get_spectator_data`:

```rust
    /// Get the SFEN string for a single game by index.
    ///
    /// Raises IndexError if game_id >= num_envs.
    pub fn get_sfen(&self, game_id: usize) -> PyResult<String> {
        if game_id >= self.num_envs {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "game_id {} out of range for {} environments",
                game_id, self.num_envs
            )));
        }
        Ok(self.games[game_id].position.to_sfen())
    }

    /// Get SFEN strings for all games.
    pub fn get_sfens(&self) -> Vec<String> {
        self.games.iter().map(|g| g.position.to_sfen()).collect()
    }
```

- [ ] **Step 4: Build and run tests**

```bash
cd shogi-engine/crates/shogi-gym && maturin develop --release && python -m pytest tests/test_vec_env.py::TestVecEnvSfen -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
cd shogi-engine && cargo test -p shogi-gym && cd crates/shogi-gym && python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/vec_env.rs \
       shogi-engine/crates/shogi-gym/tests/test_vec_env.py
git commit -m "feat: add VecEnv.get_sfen() and get_sfens() for SFEN extraction"
```

---

### Task 5: Add episode-level stats (`mean_episode_length`, `truncation_rate`)

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/vec_env.rs`
- Test: `shogi-engine/crates/shogi-gym/tests/test_vec_env.py`

- [ ] **Step 1: Write failing Python tests**

Add to `tests/test_vec_env.py`:

```python
class TestVecEnvEpisodeStats:
    def test_mean_episode_length_zero_before_completion(self):
        """mean_episode_length is 0.0 when no episodes completed."""
        env = VecEnv(num_envs=1, max_ply=100)
        env.reset()
        assert env.mean_episode_length == 0.0

    def test_truncation_rate_zero_before_completion(self):
        """truncation_rate is 0.0 when no episodes completed."""
        env = VecEnv(num_envs=1, max_ply=100)
        env.reset()
        assert env.truncation_rate == 0.0

    def test_truncation_rate_after_max_ply(self):
        """All episodes truncated at max_ply=1 should give truncation_rate=1.0."""
        env = VecEnv(num_envs=2, max_ply=1)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        actions = [int(np.where(masks[i])[0][0]) for i in range(2)]
        env.step(actions)
        assert env.episodes_completed == 2
        assert env.truncation_rate == 1.0

    def test_mean_episode_length_after_truncation(self):
        """After truncation at max_ply=1, mean_episode_length should be 1.0."""
        env = VecEnv(num_envs=2, max_ply=1)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        actions = [int(np.where(masks[i])[0][0]) for i in range(2)]
        env.step(actions)
        assert env.mean_episode_length == 1.0

    def test_mean_episode_length_accumulates(self):
        """Mean episode length accumulates across multiple truncations."""
        env = VecEnv(num_envs=1, max_ply=1)
        result = env.reset()
        for _ in range(5):
            masks = np.asarray(result.legal_masks)
            action = [int(np.where(masks[0])[0][0])]
            result = env.step(action)
        # 5 episodes, each length 1 => mean = 1.0
        assert env.episodes_completed == 5
        assert env.mean_episode_length == 1.0

    def test_reset_stats_clears_episode_length(self):
        """reset_stats() should zero out mean_episode_length."""
        env = VecEnv(num_envs=1, max_ply=1)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        action = [int(np.where(masks[0])[0][0])]
        env.step(action)
        assert env.mean_episode_length == 1.0
        env.reset_stats()
        assert env.mean_episode_length == 0.0
        assert env.truncation_rate == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd shogi-engine/crates/shogi-gym && python -m pytest tests/test_vec_env.py::TestVecEnvEpisodeStats -v
```

Expected: All 6 tests FAIL with `AttributeError: 'VecEnv' object has no attribute 'mean_episode_length'`

- [ ] **Step 3: Add `total_episode_ply` atomic to VecEnv struct**

In `vec_env.rs`, add the new field to the `VecEnv` struct definition, after `episodes_truncated`:

```rust
    episodes_truncated: AtomicU64, // MaxMoves
    total_episode_ply: AtomicU64,  // sum of ply at episode end (for mean_episode_length)
```

And initialize it in `VecEnv::new()`, after `episodes_truncated: AtomicU64::new(0),`:

```rust
            episodes_truncated: AtomicU64::new(0),
            total_episode_ply: AtomicU64::new(0),
```

- [ ] **Step 4: Accumulate ply in the auto-reset path**

In the `step()` method's `process_env` closure, add the `total_episode_ply` reference alongside the other episode counters:

```rust
            let ep_total_ply = &self.total_episode_ply;
```

Then inside the `if terminated || truncated` block, after the existing `ep_completed.fetch_add(1, ...)` line, add:

```rust
                        ep_total_ply.fetch_add(game.ply as u64, Ordering::Relaxed);
```

The full block becomes:

```rust
                    if terminated || truncated {
                        // Update episode counters
                        ep_completed.fetch_add(1, Ordering::Relaxed);
                        ep_total_ply.fetch_add(game.ply as u64, Ordering::Relaxed);
                        match result {
                            GameResult::Repetition
                            | GameResult::Impasse { winner: None } => {
                                ep_drawn.fetch_add(1, Ordering::Relaxed);
                            }
                            GameResult::MaxMoves => {
                                ep_truncated.fetch_add(1, Ordering::Relaxed);
                            }
                            _ => {}
                        }
```

- [ ] **Step 5: Add computed properties**

Add these methods to the `#[pymethods] impl VecEnv` block, after `draw_rate`:

```rust
    /// Mean episode length across all completed episodes since last reset_stats().
    /// Returns 0.0 if no episodes completed yet.
    #[getter]
    pub fn mean_episode_length(&self) -> f64 {
        let completed = self.episodes_completed.load(Ordering::Relaxed);
        if completed == 0 {
            0.0
        } else {
            self.total_episode_ply.load(Ordering::Relaxed) as f64 / completed as f64
        }
    }

    /// Fraction of completed episodes that were truncated (hit max_ply).
    /// Returns 0.0 if no episodes completed yet.
    #[getter]
    pub fn truncation_rate(&self) -> f64 {
        let completed = self.episodes_completed.load(Ordering::Relaxed);
        if completed == 0 {
            0.0
        } else {
            self.episodes_truncated.load(Ordering::Relaxed) as f64 / completed as f64
        }
    }
```

- [ ] **Step 6: Update `reset_stats()` to clear the new counter**

Replace the existing `reset_stats` method:

```rust
    /// Reset episode counters to zero.
    pub fn reset_stats(&self) {
        self.episodes_completed.store(0, Ordering::Relaxed);
        self.episodes_drawn.store(0, Ordering::Relaxed);
        self.episodes_truncated.store(0, Ordering::Relaxed);
        self.total_episode_ply.store(0, Ordering::Relaxed);
    }
```

- [ ] **Step 7: Build and run tests**

```bash
cd shogi-engine/crates/shogi-gym && maturin develop --release && python -m pytest tests/test_vec_env.py::TestVecEnvEpisodeStats -v
```

Expected: All 6 tests PASS.

- [ ] **Step 8: Run full test suite**

```bash
cd shogi-engine && cargo test -p shogi-gym && cd crates/shogi-gym && python -m pytest tests/ -v
```

Expected: All tests pass (including existing tests — the `buffer_sizes` test doesn't check the new atomic fields).

- [ ] **Step 9: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/vec_env.rs \
       shogi-engine/crates/shogi-gym/tests/test_vec_env.py
git commit -m "feat: add VecEnv.mean_episode_length and truncation_rate properties"
```

---

### Task 6: Final integration verification

**Files:** None (read-only verification)

- [ ] **Step 1: Run complete Rust test suite**

```bash
cd shogi-engine && cargo test --all
```

Expected: All tests pass.

- [ ] **Step 2: Run complete Python test suite**

```bash
cd shogi-engine/crates/shogi-gym && maturin develop --release && python -m pytest tests/ -v
```

Expected: All tests pass, including all new tests from Tasks 2–5.

- [ ] **Step 3: Verify the four new API surfaces are accessible from Python**

```bash
cd shogi-engine/crates/shogi-gym && python -c "
from shogi_gym import VecEnv, SpectatorEnv

# Change 1: get_spectator_data
v = VecEnv(num_envs=2, max_ply=100)
v.reset()
data = v.get_spectator_data()
assert len(data) == 2 and 'board' in data[0]
print('OK: VecEnv.get_spectator_data()')

# Change 2: from_sfen
s = SpectatorEnv.from_sfen('lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1')
assert s.current_player == 'black'
print('OK: SpectatorEnv.from_sfen()')

# Change 3: get_sfen / get_sfens
sfen = v.get_sfen(0)
sfens = v.get_sfens()
assert isinstance(sfen, str) and len(sfens) == 2
print('OK: VecEnv.get_sfen() / get_sfens()')

# Change 4: episode stats
assert v.mean_episode_length == 0.0
assert v.truncation_rate == 0.0
print('OK: VecEnv.mean_episode_length / truncation_rate')

print('All 4 API extensions verified.')
"
```

Expected: All assertions pass, "All 4 API extensions verified." printed.

- [ ] **Step 4: Commit any stragglers (if any files were missed)**

Only if needed. All files should already be committed from Tasks 1–5.
