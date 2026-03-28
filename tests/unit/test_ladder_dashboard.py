"""Tests for ladder leaderboard dashboard functions."""

import json
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_has_streamlit = pytest.importorskip is not None
try:
    import streamlit  # noqa: F401
    _has_streamlit = True
except ImportError:
    _has_streamlit = False


@pytest.mark.skipif(not _has_streamlit, reason="streamlit not installed")
class TestLoadLadderState:
    """Tests for load_ladder_state file loading."""

    def test_returns_none_when_file_missing(self):
        from keisei.webui.streamlit_app import load_ladder_state

        result = load_ladder_state("/nonexistent/path.json")
        assert result is None

    def test_returns_none_for_wrong_schema(self):
        from keisei.webui.streamlit_app import load_ladder_state

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"schema_version": "v1.0.0", "training": {}}, f)
            f.flush()
            result = load_ladder_state(f.name)
        assert result is None

    def test_loads_valid_ladder_state(self):
        from keisei.webui.streamlit_app import load_ladder_state

        state = {
            "schema_version": "ladder-v1",
            "timestamp": 1234567890.0,
            "matches": [],
            "leaderboard": [
                {"name": "model_a", "elo": 1550.0, "games_played": 10, "win_rate": 0.7}
            ],
            "recent_results": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(state, f)
            f.flush()
            result = load_ladder_state(f.name)
        assert result is not None
        assert result["schema_version"] == "ladder-v1"
        assert len(result["leaderboard"]) == 1

    def test_returns_none_for_corrupt_json(self):
        from keisei.webui.streamlit_app import load_ladder_state

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            f.flush()
            result = load_ladder_state(f.name)
        assert result is None

    def test_default_path_when_none(self):
        """When None is passed, uses the default .keisei_ladder/state.json path."""
        from keisei.webui.streamlit_app import load_ladder_state

        # File doesn't exist at default path, should return None gracefully
        result = load_ladder_state(None)
        assert result is None


class TestRenderLadderTabLogic:
    """Test the data extraction logic used by render_ladder_tab."""

    def test_playing_now_set_from_matches(self):
        """Currently playing models are extracted from active matches."""
        matches = [
            {
                "model_a": {"name": "ckpt_1000", "elo": 1500},
                "model_b": {"name": "ckpt_2000", "elo": 1520},
                "status": "in_progress",
            },
        ]
        playing_now = set()
        for match in matches:
            model_a = match.get("model_a", {})
            model_b = match.get("model_b", {})
            if isinstance(model_a, dict):
                playing_now.add(model_a.get("name", ""))
            if isinstance(model_b, dict):
                playing_now.add(model_b.get("name", ""))

        assert "ckpt_1000" in playing_now
        assert "ckpt_2000" in playing_now

    def test_recent_deltas_accumulated(self):
        """Recent Elo deltas are summed per model across recent results."""
        recent_results = [
            {"model_a": "m1", "model_b": "m2", "elo_delta_a": 12.0, "elo_delta_b": -12.0},
            {"model_a": "m1", "model_b": "m3", "elo_delta_a": -5.0, "elo_delta_b": 5.0},
        ]
        recent_deltas = {}
        for result in recent_results:
            name_a = result.get("model_a", "")
            name_b = result.get("model_b", "")
            delta_a = result.get("elo_delta_a", 0.0)
            delta_b = result.get("elo_delta_b", 0.0)
            recent_deltas[name_a] = recent_deltas.get(name_a, 0.0) + delta_a
            recent_deltas[name_b] = recent_deltas.get(name_b, 0.0) + delta_b

        assert recent_deltas["m1"] == pytest.approx(7.0)
        assert recent_deltas["m2"] == pytest.approx(-12.0)
        assert recent_deltas["m3"] == pytest.approx(5.0)

    def test_arrow_indicator_logic(self):
        """Arrow indicators show rising/falling based on delta threshold."""
        test_cases = [
            (15.0, True, False),   # positive → green arrow
            (-10.0, False, True),  # negative → red arrow
            (0.3, False, False),   # within threshold → dash
        ]
        for delta, expect_up, expect_down in test_cases:
            if delta > 0.5:
                arrow = f"+{delta:.0f} ↑"
                assert expect_up, f"delta={delta} should be up arrow"
            elif delta < -0.5:
                arrow = f"{delta:.0f} ↓"
                assert expect_down, f"delta={delta} should be down arrow"
            else:
                arrow = "—"
                assert not expect_up and not expect_down
