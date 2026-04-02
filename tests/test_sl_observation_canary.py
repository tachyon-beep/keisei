"""Canary test: SL pipeline placeholder observations are all-zero.

This test PASSES today (confirming the placeholder produces zero obs)
and will FAIL once real observation encoding is implemented — at which
point the test should be updated to assert non-zero observations.

See: keisei/sl/prepare.py lines 121-133 (FIXME placeholder)
"""

import numpy as np
import pytest

from keisei.sl.dataset import SLDataset
from keisei.sl.prepare import prepare_sl_data


@pytest.fixture
def canary_dataset(tmp_path):
    """Prepare a minimal SL dataset from a 4-move game."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    sfen_content = "result:win_black\nstartpos\n7g7f\n3c3d\n2g2f\n8c8d\n"
    (games_dir / "test.sfen").write_text(sfen_content)

    output_dir = tmp_path / "processed"
    prepare_sl_data(
        game_sources=[str(games_dir)],
        output_dir=str(output_dir),
        min_ply=1,
    )
    return SLDataset(output_dir)


class TestSLObservationCanary:
    """C1: Detect when SL observations are placeholder zeros."""

    def test_placeholder_observations_are_zero(self, canary_dataset):
        """Current SL pipeline writes zero-tensor observations.

        This is a KNOWN LIMITATION documented in prepare.py.
        This test exists so that when real obs encoding is added,
        this test fails — signaling the canary should be updated.
        """
        for i in range(len(canary_dataset)):
            obs = canary_dataset[i]["observation"].numpy()
            assert np.all(obs == 0.0), (
                f"Position {i}: observation is non-zero — "
                f"if real encoding was added, update this canary test"
            )

    def test_placeholder_policy_targets_are_zero(self, canary_dataset):
        """Current SL pipeline writes policy_target=0 for all positions."""
        for i in range(len(canary_dataset)):
            policy = canary_dataset[i]["policy_target"].item()
            assert policy == 0, (
                f"Position {i}: policy_target is {policy}, not 0 — "
                f"if real encoding was added, update this canary test"
            )
