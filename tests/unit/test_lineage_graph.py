"""
Tests for the lineage graph read model.

Validates deterministic reconstruction from event streams, ancestor/descendant
queries, promotion chain, model profile, and WebUI snapshot generation.
"""

from keisei.lineage.event_schema import make_event, make_model_id
from keisei.lineage.graph import LineageGraph, MatchRecord, ModelNode, PromotionRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(seq, event_type, run_name, model_id, payload):
    """Shorthand for make_event with keyword args."""
    return make_event(
        seq=seq,
        event_type=event_type,
        run_name=run_name,
        model_id=model_id,
        payload=payload,
    )


def _checkpoint_evt(seq, run_name, timestep, parent=None):
    """Create a checkpoint_created event."""
    mid = make_model_id(run_name, timestep)
    return _evt(
        seq,
        "checkpoint_created",
        run_name,
        mid,
        {
            "checkpoint_path": f"/models/{run_name}/checkpoint_ts{timestep}.pth",
            "global_timestep": timestep,
            "total_episodes": timestep // 50,
            "parent_model_id": parent,
        },
    )


def _match_evt(seq, run_name, timestep, opponent_id, result, win_rate):
    """Create a match_completed event."""
    mid = make_model_id(run_name, timestep)
    return _evt(
        seq,
        "match_completed",
        run_name,
        mid,
        {
            "opponent_model_id": opponent_id,
            "result": result,
            "num_games": 20,
            "win_rate": win_rate,
            "agent_rating": 1500.0,
            "opponent_rating": 1500.0,
        },
    )


def _promotion_evt(seq, run_name, timestep, from_r, to_r, reason="elo_improvement"):
    """Create a model_promoted event."""
    mid = make_model_id(run_name, timestep)
    return _evt(
        seq,
        "model_promoted",
        run_name,
        mid,
        {
            "from_rating": from_r,
            "to_rating": to_r,
            "promotion_reason": reason,
        },
    )


def _training_started_evt(seq, run_name, parent=None):
    """Create a training_started event."""
    mid = make_model_id(run_name, 0)
    return _evt(
        seq,
        "training_started",
        run_name,
        mid,
        {"config_snapshot": {"lr": 0.001}, "parent_model_id": parent},
    )


def _training_resumed_evt(seq, run_name, timestep, parent=None):
    """Create a training_resumed event."""
    mid = make_model_id(run_name, timestep)
    return _evt(
        seq,
        "training_resumed",
        run_name,
        mid,
        {
            "resumed_from_checkpoint": f"/models/checkpoint_ts{timestep}.pth",
            "global_timestep_at_resume": timestep,
            "parent_model_id": parent,
        },
    )


# ---------------------------------------------------------------------------
# Construction: from_events
# ---------------------------------------------------------------------------


class TestFromEvents:
    def test_empty_events_produce_empty_graph(self):
        graph = LineageGraph.from_events([])
        assert graph.node_count == 0
        assert graph.all_model_ids == []

    def test_single_checkpoint_creates_one_node(self):
        events = [_checkpoint_evt(0, "run-1", 5000)]
        graph = LineageGraph.from_events(events)
        assert graph.node_count == 1

        node = graph.get_node("run-1::checkpoint_ts5000")
        assert node is not None
        assert node.global_timestep == 5000
        assert node.checkpoint_path == "/models/run-1/checkpoint_ts5000.pth"
        assert node.parent_model_id is None

    def test_checkpoint_with_parent_creates_two_nodes(self):
        parent_id = make_model_id("run-1", 5000)
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(1, "run-1", 10000, parent=parent_id),
        ]
        graph = LineageGraph.from_events(events)
        assert graph.node_count == 2

        child = graph.get_node("run-1::checkpoint_ts10000")
        assert child is not None
        assert child.parent_model_id == parent_id

    def test_parent_node_created_even_without_own_event(self):
        """A parent referenced in payload but never having its own event should still get a node."""
        events = [
            _checkpoint_evt(0, "run-2", 10000, parent="old-run::checkpoint_ts50000"),
        ]
        graph = LineageGraph.from_events(events)
        assert graph.node_count == 2
        assert graph.get_node("old-run::checkpoint_ts50000") is not None

    def test_training_started_sets_parent(self):
        parent_id = "pretrained::checkpoint_ts0"
        events = [_training_started_evt(0, "run-1", parent=parent_id)]
        graph = LineageGraph.from_events(events)

        node = graph.get_node("run-1::checkpoint_ts0")
        assert node is not None
        assert node.parent_model_id == parent_id

    def test_training_resumed_sets_parent(self):
        parent_id = make_model_id("run-1", 5000)
        events = [_training_resumed_evt(0, "run-1", 5000, parent=parent_id)]
        graph = LineageGraph.from_events(events)

        node = graph.get_node("run-1::checkpoint_ts5000")
        assert node is not None
        assert node.parent_model_id == parent_id

    def test_match_completed_adds_record(self):
        events = [_match_evt(0, "run-1", 10000, "opponent_x", "win", 0.7)]
        graph = LineageGraph.from_events(events)

        node = graph.get_node("run-1::checkpoint_ts10000")
        assert node is not None
        assert len(node.matches) == 1
        assert node.matches[0].result == "win"
        assert node.matches[0].opponent_model_id == "opponent_x"
        assert node.matches[0].win_rate == 0.7

    def test_model_promoted_adds_record(self):
        events = [_promotion_evt(0, "run-1", 10000, 1500.0, 1532.0)]
        graph = LineageGraph.from_events(events)

        node = graph.get_node("run-1::checkpoint_ts10000")
        assert node is not None
        assert len(node.promotions) == 1
        assert node.promotions[0].from_rating == 1500.0
        assert node.promotions[0].to_rating == 1532.0
        assert node.latest_rating == 1532.0

    def test_multiple_events_for_same_model(self):
        mid = make_model_id("run-1", 10000)
        events = [
            _checkpoint_evt(0, "run-1", 10000),
            _match_evt(1, "run-1", 10000, "opp-a", "win", 0.6),
            _match_evt(2, "run-1", 10000, "opp-b", "loss", 0.3),
            _promotion_evt(3, "run-1", 10000, 1500.0, 1520.0),
        ]
        graph = LineageGraph.from_events(events)
        assert graph.node_count == 1  # all same model_id

        node = graph.get_node(mid)
        assert node is not None
        assert len(node.matches) == 2
        assert len(node.promotions) == 1
        assert len(node.event_ids) == 4


# ---------------------------------------------------------------------------
# Determinism invariant
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_events_produce_identical_graphs(self):
        """The core invariant: replaying the same events yields the same graph."""
        events = [
            _training_started_evt(0, "run-1"),
            _checkpoint_evt(1, "run-1", 5000),
            _checkpoint_evt(
                2, "run-1", 10000, parent=make_model_id("run-1", 5000)
            ),
            _match_evt(3, "run-1", 10000, "opp", "win", 0.7),
            _promotion_evt(4, "run-1", 10000, 1500.0, 1520.0),
        ]

        graph_a = LineageGraph.from_events(events)
        graph_b = LineageGraph.from_events(events)

        assert graph_a.node_count == graph_b.node_count
        assert graph_a.all_model_ids == graph_b.all_model_ids

        for mid in graph_a.all_model_ids:
            a = graph_a.get_node(mid)
            b = graph_b.get_node(mid)
            assert a is not None and b is not None
            assert a.to_dict() == b.to_dict()

    def test_rebuild_from_snapshot_matches(self):
        """Snapshot output is identical across rebuilds."""
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(
                1, "run-1", 10000, parent=make_model_id("run-1", 5000)
            ),
        ]

        snap_a = LineageGraph.from_events(events).to_snapshot()
        snap_b = LineageGraph.from_events(events).to_snapshot()

        assert snap_a == snap_b


# ---------------------------------------------------------------------------
# Query: ancestors
# ---------------------------------------------------------------------------


class TestAncestors:
    def test_no_parent_returns_empty(self):
        events = [_checkpoint_evt(0, "run-1", 5000)]
        graph = LineageGraph.from_events(events)
        assert graph.ancestors("run-1::checkpoint_ts5000") == []

    def test_single_parent(self):
        parent_id = make_model_id("run-1", 5000)
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(1, "run-1", 10000, parent=parent_id),
        ]
        graph = LineageGraph.from_events(events)
        anc = graph.ancestors("run-1::checkpoint_ts10000")
        assert len(anc) == 1
        assert anc[0].model_id == parent_id

    def test_chain_of_three(self):
        """grandparent → parent → child"""
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(
                1, "run-1", 10000, parent=make_model_id("run-1", 5000)
            ),
            _checkpoint_evt(
                2, "run-1", 15000, parent=make_model_id("run-1", 10000)
            ),
        ]
        graph = LineageGraph.from_events(events)
        anc = graph.ancestors("run-1::checkpoint_ts15000")
        assert len(anc) == 2
        assert anc[0].model_id == "run-1::checkpoint_ts10000"
        assert anc[1].model_id == "run-1::checkpoint_ts5000"

    def test_unknown_model_returns_empty(self):
        graph = LineageGraph.from_events([])
        assert graph.ancestors("nonexistent") == []

    def test_cycle_detection(self):
        """If events somehow create a cycle, ancestors stops."""
        # Manually construct a cyclic graph
        graph = LineageGraph()
        a = ModelNode(model_id="a", parent_model_id="b")
        b = ModelNode(model_id="b", parent_model_id="a")
        graph.nodes["a"] = a
        graph.nodes["b"] = b

        # Should not infinite-loop
        anc = graph.ancestors("a")
        assert len(anc) == 1
        assert anc[0].model_id == "b"


# ---------------------------------------------------------------------------
# Query: descendants
# ---------------------------------------------------------------------------


class TestDescendants:
    def test_no_children_returns_empty(self):
        events = [_checkpoint_evt(0, "run-1", 5000)]
        graph = LineageGraph.from_events(events)
        assert graph.descendants("run-1::checkpoint_ts5000") == []

    def test_single_child(self):
        parent_id = make_model_id("run-1", 5000)
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(1, "run-1", 10000, parent=parent_id),
        ]
        graph = LineageGraph.from_events(events)
        desc = graph.descendants(parent_id)
        assert len(desc) == 1
        assert desc[0].model_id == "run-1::checkpoint_ts10000"

    def test_multiple_children(self):
        """Two checkpoints sharing the same parent."""
        parent_id = make_model_id("run-1", 5000)
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(1, "run-1", 10000, parent=parent_id),
            _checkpoint_evt(2, "run-1", 15000, parent=parent_id),
        ]
        graph = LineageGraph.from_events(events)
        desc = graph.descendants(parent_id)
        assert len(desc) == 2
        desc_ids = {d.model_id for d in desc}
        assert "run-1::checkpoint_ts10000" in desc_ids
        assert "run-1::checkpoint_ts15000" in desc_ids

    def test_grandchildren(self):
        """parent → child → grandchild"""
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(
                1, "run-1", 10000, parent=make_model_id("run-1", 5000)
            ),
            _checkpoint_evt(
                2, "run-1", 15000, parent=make_model_id("run-1", 10000)
            ),
        ]
        graph = LineageGraph.from_events(events)
        desc = graph.descendants("run-1::checkpoint_ts5000")
        assert len(desc) == 2
        desc_ids = [d.model_id for d in desc]
        # BFS order: child first, then grandchild
        assert desc_ids[0] == "run-1::checkpoint_ts10000"
        assert desc_ids[1] == "run-1::checkpoint_ts15000"

    def test_cross_run_descendants(self):
        """A model from run-2 can descend from a run-1 model."""
        parent_id = make_model_id("run-1", 50000)
        events = [
            _checkpoint_evt(0, "run-1", 50000),
            _checkpoint_evt(1, "run-2", 5000, parent=parent_id),
        ]
        graph = LineageGraph.from_events(events)
        desc = graph.descendants(parent_id)
        assert len(desc) == 1
        assert desc[0].run_name == "run-2"


# ---------------------------------------------------------------------------
# Query: promotion chain
# ---------------------------------------------------------------------------


class TestPromotionChain:
    def test_no_promotions_returns_empty(self):
        events = [_checkpoint_evt(0, "run-1", 5000)]
        graph = LineageGraph.from_events(events)
        assert graph.promotion_chain("run-1") == []

    def test_single_promotion(self):
        events = [
            _checkpoint_evt(0, "run-1", 10000),
            _promotion_evt(1, "run-1", 10000, 1500.0, 1520.0),
        ]
        graph = LineageGraph.from_events(events)
        chain = graph.promotion_chain("run-1")
        assert len(chain) == 1
        assert chain[0].model_id == "run-1::checkpoint_ts10000"

    def test_multiple_promotions_in_order(self):
        events = [
            _checkpoint_evt(0, "run-1", 10000),
            _promotion_evt(1, "run-1", 10000, 1500.0, 1520.0),
            _checkpoint_evt(2, "run-1", 20000),
            _promotion_evt(3, "run-1", 20000, 1520.0, 1550.0),
        ]
        graph = LineageGraph.from_events(events)
        chain = graph.promotion_chain("run-1")
        assert len(chain) == 2
        assert chain[0].model_id == "run-1::checkpoint_ts10000"
        assert chain[1].model_id == "run-1::checkpoint_ts20000"

    def test_promotion_chain_filters_by_run(self):
        events = [
            _promotion_evt(0, "run-1", 10000, 1500.0, 1520.0),
            _promotion_evt(1, "run-2", 5000, 1500.0, 1510.0),
        ]
        graph = LineageGraph.from_events(events)
        chain_1 = graph.promotion_chain("run-1")
        chain_2 = graph.promotion_chain("run-2")
        assert len(chain_1) == 1
        assert len(chain_2) == 1
        assert chain_1[0].run_name == "run-1"
        assert chain_2[0].run_name == "run-2"


# ---------------------------------------------------------------------------
# Query: model profile
# ---------------------------------------------------------------------------


class TestModelProfile:
    def test_unknown_model_returns_none(self):
        graph = LineageGraph.from_events([])
        assert graph.model_profile("nonexistent") is None

    def test_basic_profile(self):
        events = [_checkpoint_evt(0, "run-1", 5000)]
        graph = LineageGraph.from_events(events)
        profile = graph.model_profile("run-1::checkpoint_ts5000")
        assert profile is not None
        assert profile["model_id"] == "run-1::checkpoint_ts5000"
        assert profile["global_timestep"] == 5000
        assert profile["ancestor_chain"] == []
        assert profile["descendant_count"] == 0
        assert profile["matches"] == []
        assert profile["promotions"] == []

    def test_profile_with_ancestry(self):
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(
                1, "run-1", 10000, parent=make_model_id("run-1", 5000)
            ),
        ]
        graph = LineageGraph.from_events(events)
        profile = graph.model_profile("run-1::checkpoint_ts10000")
        assert profile is not None
        assert profile["ancestor_chain"] == ["run-1::checkpoint_ts5000"]

    def test_profile_with_matches_and_promotions(self):
        mid = make_model_id("run-1", 10000)
        events = [
            _checkpoint_evt(0, "run-1", 10000),
            _match_evt(1, "run-1", 10000, "opp-1", "win", 0.7),
            _match_evt(2, "run-1", 10000, "opp-2", "loss", 0.3),
            _promotion_evt(3, "run-1", 10000, 1500.0, 1520.0),
        ]
        graph = LineageGraph.from_events(events)
        profile = graph.model_profile(mid)
        assert profile is not None
        assert len(profile["matches"]) == 2
        assert len(profile["promotions"]) == 1
        assert profile["latest_rating"] == 1520.0

    def test_profile_with_descendants(self):
        parent_id = make_model_id("run-1", 5000)
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(1, "run-1", 10000, parent=parent_id),
            _checkpoint_evt(2, "run-1", 15000, parent=parent_id),
        ]
        graph = LineageGraph.from_events(events)
        profile = graph.model_profile(parent_id)
        assert profile is not None
        assert profile["descendant_count"] == 2


# ---------------------------------------------------------------------------
# WebUI snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_graph_snapshot(self):
        graph = LineageGraph.from_events([])
        snap = graph.to_snapshot()
        assert snap["node_count"] == 0
        assert snap["edge_count"] == 0
        assert snap["nodes"] == []
        assert snap["edges"] == []

    def test_snapshot_includes_nodes_and_edges(self):
        parent_id = make_model_id("run-1", 5000)
        events = [
            _checkpoint_evt(0, "run-1", 5000),
            _checkpoint_evt(1, "run-1", 10000, parent=parent_id),
        ]
        graph = LineageGraph.from_events(events)
        snap = graph.to_snapshot()

        assert snap["node_count"] == 2
        assert snap["edge_count"] == 1
        assert snap["edges"][0]["from"] == parent_id
        assert snap["edges"][0]["to"] == "run-1::checkpoint_ts10000"
        assert snap["edges"][0]["type"] == "parent"

    def test_snapshot_node_dicts_have_expected_keys(self):
        events = [_checkpoint_evt(0, "run-1", 5000)]
        graph = LineageGraph.from_events(events)
        snap = graph.to_snapshot()

        node_dict = snap["nodes"][0]
        expected_keys = {
            "model_id",
            "run_name",
            "global_timestep",
            "checkpoint_path",
            "parent_model_id",
            "created_at",
            "latest_rating",
            "num_matches",
            "num_promotions",
        }
        assert set(node_dict.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Integration: full lineage scenario
# ---------------------------------------------------------------------------


class TestFullScenario:
    def test_multi_run_lineage(self):
        """Simulate a realistic training lineage across two runs."""
        run1_parent = make_model_id("run-1", 5000)
        run1_child = make_model_id("run-1", 10000)

        events = [
            # Run 1: fresh start, two checkpoints, one match, one promotion
            _training_started_evt(0, "run-1"),
            _checkpoint_evt(1, "run-1", 5000),
            _match_evt(2, "run-1", 5000, "random", "win", 0.8),
            _promotion_evt(3, "run-1", 5000, 1500.0, 1530.0),
            _checkpoint_evt(4, "run-1", 10000, parent=run1_parent),
            _match_evt(5, "run-1", 10000, run1_parent, "draw", 0.5),
            # Run 2: resumed from run-1 ts10000
            _training_resumed_evt(6, "run-2", 10000, parent=run1_child),
            _checkpoint_evt(7, "run-2", 15000, parent=run1_child),
            _match_evt(8, "run-2", 15000, run1_child, "win", 0.6),
            _promotion_evt(9, "run-2", 15000, 1530.0, 1560.0),
        ]

        graph = LineageGraph.from_events(events)

        # Node count: run-1::ts0, run-1::ts5000, run-1::ts10000,
        #             run-2::ts10000, run-2::ts15000
        assert graph.node_count == 5

        # Ancestor chain from run-2::ts15000
        anc = graph.ancestors("run-2::checkpoint_ts15000")
        anc_ids = [a.model_id for a in anc]
        assert run1_child in anc_ids  # immediate parent

        # Descendants from run-1::ts5000
        desc = graph.descendants(run1_parent)
        desc_ids = {d.model_id for d in desc}
        assert run1_child in desc_ids
        assert "run-2::checkpoint_ts15000" in desc_ids

        # Promotion chain for each run
        assert len(graph.promotion_chain("run-1")) == 1
        assert len(graph.promotion_chain("run-2")) == 1

        # Model profile for promoted checkpoint
        profile = graph.model_profile(run1_parent)
        assert profile is not None
        assert profile["latest_rating"] == 1530.0
        assert len(profile["matches"]) == 1

        # Snapshot is serializable
        snap = graph.to_snapshot()
        assert snap["node_count"] == 5
        assert snap["edge_count"] >= 2  # at least the two parent links
