"""Tests for ladder leaderboard dashboard functions."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper tests (no mocking needed)
# ---------------------------------------------------------------------------


class TestExtractPlayingModels:
    """Tests for extract_playing_models helper."""

    def test_extracts_names_from_matches(self):
        from keisei.webui.streamlit_app import extract_playing_models

        matches = [
            {
                "model_a": {"name": "ckpt_1000", "elo": 1500},
                "model_b": {"name": "ckpt_2000", "elo": 1520},
            },
        ]
        assert extract_playing_models(matches) == {"ckpt_1000", "ckpt_2000"}

    def test_empty_matches_returns_empty_set(self):
        from keisei.webui.streamlit_app import extract_playing_models

        assert extract_playing_models([]) == set()

    def test_handles_non_dict_model_fields(self):
        from keisei.webui.streamlit_app import extract_playing_models

        matches = [{"model_a": "string_name", "model_b": {"name": "ok"}}]
        result = extract_playing_models(matches)
        assert "ok" in result
        assert "string_name" not in result


class TestComputeRecentDeltas:
    """Tests for compute_recent_deltas helper."""

    def test_accumulates_deltas_per_model(self):
        from keisei.webui.streamlit_app import compute_recent_deltas

        results = [
            {"model_a": "m1", "model_b": "m2", "elo_delta_a": 12.0, "elo_delta_b": -12.0},
            {"model_a": "m1", "model_b": "m3", "elo_delta_a": -5.0, "elo_delta_b": 5.0},
        ]
        deltas = compute_recent_deltas(results)
        assert deltas["m1"] == pytest.approx(7.0)
        assert deltas["m2"] == pytest.approx(-12.0)
        assert deltas["m3"] == pytest.approx(5.0)

    def test_empty_results_returns_empty(self):
        from keisei.webui.streamlit_app import compute_recent_deltas

        assert compute_recent_deltas([]) == {}

    def test_limits_to_last_20(self):
        from keisei.webui.streamlit_app import compute_recent_deltas

        results = [
            {"model_a": "a", "model_b": "b", "elo_delta_a": 1.0, "elo_delta_b": -1.0}
        ] * 30
        deltas = compute_recent_deltas(results)
        assert deltas["a"] == pytest.approx(20.0)


class TestFormatEloArrow:
    """Tests for format_elo_arrow helper."""

    def test_positive_delta_shows_green_arrow(self):
        from keisei.webui.streamlit_app import format_elo_arrow

        result = format_elo_arrow(15.0)
        assert "green" in result
        assert "↑" in result

    def test_negative_delta_shows_red_arrow(self):
        from keisei.webui.streamlit_app import format_elo_arrow

        result = format_elo_arrow(-10.0)
        assert "red" in result
        assert "↓" in result

    def test_small_delta_shows_dash(self):
        from keisei.webui.streamlit_app import format_elo_arrow

        assert format_elo_arrow(0.3) == "—"

    def test_boundary_exactly_0_5_shows_dash(self):
        from keisei.webui.streamlit_app import format_elo_arrow

        assert format_elo_arrow(0.5) == "—"

    def test_boundary_exactly_neg_0_5_shows_dash(self):
        from keisei.webui.streamlit_app import format_elo_arrow

        assert format_elo_arrow(-0.5) == "—"


# ---------------------------------------------------------------------------
# load_ladder_state tests
# ---------------------------------------------------------------------------


class TestLoadLadderState:
    """Tests for load_ladder_state file loading."""

    def test_returns_none_when_file_missing(self):
        from keisei.webui.streamlit_app import load_ladder_state

        assert load_ladder_state("/nonexistent/path.json") is None

    def test_returns_none_for_wrong_schema(self, tmp_path):
        from keisei.webui.streamlit_app import load_ladder_state

        f = tmp_path / "state.json"
        f.write_text(json.dumps({"schema_version": "v1.0.0", "training": {}}))
        assert load_ladder_state(str(f)) is None

    def test_returns_none_for_missing_schema_key(self, tmp_path):
        from keisei.webui.streamlit_app import load_ladder_state

        f = tmp_path / "state.json"
        f.write_text(json.dumps({"leaderboard": []}))
        assert load_ladder_state(str(f)) is None

    def test_loads_valid_ladder_state(self, tmp_path):
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
        f = tmp_path / "state.json"
        f.write_text(json.dumps(state))
        result = load_ladder_state(str(f))
        assert result is not None
        assert result["schema_version"] == "ladder-v1"

    def test_returns_none_for_corrupt_json(self, tmp_path):
        from keisei.webui.streamlit_app import load_ladder_state

        f = tmp_path / "corrupt.json"
        f.write_text("not valid json{{{")
        assert load_ladder_state(str(f)) is None

    def test_default_path_when_none(self):
        from keisei.webui.streamlit_app import load_ladder_state

        assert load_ladder_state(None) is None


# ---------------------------------------------------------------------------
# render_ladder_tab tests (mocked streamlit)
# ---------------------------------------------------------------------------


class TestRenderLadderTab:
    """Tests calling render_ladder_tab with mocked st."""

    @patch("keisei.webui.streamlit_app.st")
    def test_none_state_shows_info(self, mock_st):
        from keisei.webui.streamlit_app import render_ladder_tab

        render_ladder_tab(None)
        mock_st.info.assert_called_once()
        assert "No ladder data" in mock_st.info.call_args[0][0]

    @patch("keisei.webui.streamlit_app.st")
    def test_empty_leaderboard_shows_warning(self, mock_st):
        from keisei.webui.streamlit_app import render_ladder_tab

        state = {
            "schema_version": "ladder-v1",
            "leaderboard": [],
            "matches": [],
            "recent_results": [],
        }
        render_ladder_tab(state)
        mock_st.warning.assert_called_once()
        assert "no models" in mock_st.warning.call_args[0][0].lower()

    @patch("keisei.webui.streamlit_app.st")
    def test_renders_leaderboard_rows(self, mock_st):
        from keisei.webui.streamlit_app import render_ladder_tab

        # st.columns(3) returns 3, st.columns([...6...]) returns 6
        mock_st.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]

        state = {
            "schema_version": "ladder-v1",
            "timestamp": time.time(),
            "leaderboard": [
                {"name": "model_a", "elo": 1600.0, "games_played": 20, "win_rate": 0.7},
                {"name": "model_b", "elo": 1400.0, "games_played": 20, "win_rate": 0.3},
            ],
            "matches": [
                {"model_a": {"name": "model_a"}, "model_b": {"name": "model_b"}},
            ],
            "recent_results": [],
        }
        render_ladder_tab(state)

        mock_st.subheader.assert_called_with("Elo Leaderboard")
        # header row + 2 data rows + metrics row = 4 st.columns calls
        assert mock_st.columns.call_count >= 4

    @patch("keisei.webui.streamlit_app.st")
    def test_missing_timestamp_shows_unavailable(self, mock_st):
        from keisei.webui.streamlit_app import render_ladder_tab

        mock_st.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]

        state = {
            "schema_version": "ladder-v1",
            "leaderboard": [
                {"name": "m", "elo": 1500.0, "games_played": 5, "win_rate": 0.5},
            ],
            "matches": [],
            "recent_results": [],
        }
        render_ladder_tab(state)
        mock_st.caption.assert_called_with("Timestamp unavailable")

    @patch("keisei.webui.streamlit_app.st")
    def test_games_played_metric_divides_by_two(self, mock_st):
        from keisei.webui.streamlit_app import render_ladder_tab

        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.side_effect = [
            (mock_col1, mock_col2, mock_col3),  # summary metrics
        ] + [[MagicMock() for _ in range(6)]] * 10  # header + data rows

        state = {
            "schema_version": "ladder-v1",
            "timestamp": time.time(),
            "leaderboard": [
                {"name": "a", "elo": 1500, "games_played": 10, "win_rate": 0.5},
                {"name": "b", "elo": 1500, "games_played": 10, "win_rate": 0.5},
            ],
            "matches": [],
            "recent_results": [],
        }
        render_ladder_tab(state)
        # 10 + 10 = 20 model-games, divided by 2 = 10 unique games
        mock_col2.metric.assert_called_with("Games Played", 10)
