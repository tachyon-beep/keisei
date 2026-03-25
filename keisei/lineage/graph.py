"""
Lineage graph read model — deterministic projection from the event stream.

The graph is rebuilt from scratch each time by replaying events in order.
This guarantees that the same event log always produces the same graph
(the determinism invariant required by tw5.4).

Usage::

    from keisei.lineage.graph import LineageGraph
    from keisei.lineage.registry import LineageRegistry

    registry = LineageRegistry(Path("lineage.jsonl"))
    graph = LineageGraph.from_events(registry.load_all())
    ancestors = graph.ancestors("my-run::checkpoint_ts50000")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from keisei.lineage.event_schema import LineageEvent


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


@dataclass
class MatchRecord:
    """Summary of a single evaluation match set."""

    opponent_model_id: str
    result: str
    num_games: int
    win_rate: float
    agent_rating: float
    opponent_rating: float
    event_id: str


@dataclass
class PromotionRecord:
    """Summary of a single promotion event."""

    from_rating: float
    to_rating: float
    reason: str
    event_id: str


@dataclass
class ModelNode:
    """A single model in the lineage graph.

    Each unique ``model_id`` becomes one node.  Properties are accumulated
    from events that reference this model.
    """

    model_id: str
    run_name: str = ""
    global_timestep: Optional[int] = None
    checkpoint_path: Optional[str] = None
    parent_model_id: Optional[str] = None
    created_at: Optional[str] = None  # ISO-8601 from first event

    # Accumulated records
    matches: List[MatchRecord] = field(default_factory=list)
    promotions: List[PromotionRecord] = field(default_factory=list)

    # Event IDs that contributed to this node (for provenance)
    event_ids: List[str] = field(default_factory=list)

    @property
    def latest_rating(self) -> Optional[float]:
        """Most recent Elo rating from promotions, or None if never promoted."""
        if self.promotions:
            return self.promotions[-1].to_rating
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebUI/JSON consumption."""
        return {
            "model_id": self.model_id,
            "run_name": self.run_name,
            "global_timestep": self.global_timestep,
            "checkpoint_path": self.checkpoint_path,
            "parent_model_id": self.parent_model_id,
            "created_at": self.created_at,
            "latest_rating": self.latest_rating,
            "num_matches": len(self.matches),
            "num_promotions": len(self.promotions),
        }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass
class LineageGraph:
    """Directed graph of model lineage relationships.

    Nodes are ``ModelNode`` instances keyed by ``model_id``.
    Edges are implicit via ``parent_model_id`` pointers on each node.

    The graph is always built via :meth:`from_events` to guarantee
    deterministic reconstruction.
    """

    nodes: Dict[str, ModelNode] = field(default_factory=dict)

    # ---------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------

    @classmethod
    def from_events(cls, events: Sequence[LineageEvent]) -> LineageGraph:
        """Build a lineage graph by replaying events in order.

        This is the only supported constructor.  It guarantees the
        determinism invariant: same input events → identical graph.
        """
        graph = cls()

        for event in events:
            model_id: str = event["model_id"]
            event_type: str = event["event_type"]
            payload: Dict[str, Any] = event["payload"]

            node = graph._ensure_node(model_id, event)

            if event_type == "checkpoint_created":
                node.checkpoint_path = payload.get("checkpoint_path")
                node.global_timestep = payload.get("global_timestep")
                parent = payload.get("parent_model_id")
                if parent is not None:
                    node.parent_model_id = parent
                    # Ensure parent node exists (it may not have its own
                    # checkpoint_created event if it was from a previous run)
                    graph._ensure_node(parent, event=None)
                node.event_ids.append(event["event_id"])

            elif event_type == "training_started":
                parent = payload.get("parent_model_id")
                if parent is not None:
                    node.parent_model_id = parent
                    graph._ensure_node(parent, event=None)
                node.event_ids.append(event["event_id"])

            elif event_type == "training_resumed":
                parent = payload.get("parent_model_id")
                if parent is not None:
                    node.parent_model_id = parent
                    graph._ensure_node(parent, event=None)
                node.event_ids.append(event["event_id"])

            elif event_type == "match_completed":
                node.matches.append(
                    MatchRecord(
                        opponent_model_id=payload.get("opponent_model_id", ""),
                        result=payload.get("result", ""),
                        num_games=payload.get("num_games", 0),
                        win_rate=payload.get("win_rate", 0.0),
                        agent_rating=payload.get("agent_rating", 0.0),
                        opponent_rating=payload.get("opponent_rating", 0.0),
                        event_id=event["event_id"],
                    )
                )
                node.event_ids.append(event["event_id"])

            elif event_type == "model_promoted":
                node.promotions.append(
                    PromotionRecord(
                        from_rating=payload.get("from_rating", 0.0),
                        to_rating=payload.get("to_rating", 0.0),
                        reason=payload.get("promotion_reason", ""),
                        event_id=event["event_id"],
                    )
                )
                node.event_ids.append(event["event_id"])

        return graph

    def _ensure_node(
        self, model_id: str, event: Optional[LineageEvent]
    ) -> ModelNode:
        """Get or create a node for *model_id*."""
        if model_id not in self.nodes:
            self.nodes[model_id] = ModelNode(model_id=model_id)
        node = self.nodes[model_id]
        if event is not None:
            if not node.run_name:
                node.run_name = event.get("run_name", "")
            if node.created_at is None:
                node.created_at = event.get("emitted_at")
        return node

    # ---------------------------------------------------------------
    # Basic lookups
    # ---------------------------------------------------------------

    def get_node(self, model_id: str) -> Optional[ModelNode]:
        """Return the node for *model_id*, or ``None`` if unknown."""
        return self.nodes.get(model_id)

    @property
    def node_count(self) -> int:
        """Number of unique models in the graph."""
        return len(self.nodes)

    @property
    def all_model_ids(self) -> List[str]:
        """All model IDs in insertion order."""
        return list(self.nodes.keys())

    # ---------------------------------------------------------------
    # Query: ancestors / descendants
    # ---------------------------------------------------------------

    def ancestors(self, model_id: str) -> List[ModelNode]:
        """Walk the parent chain from *model_id* back to the root.

        Returns a list starting with the immediate parent and ending at
        the root (a node with no parent).  Returns ``[]`` if the model
        has no parent or is unknown.

        Cycle detection: stops if a model_id is visited twice.
        """
        result: List[ModelNode] = []
        visited: set[str] = {model_id}
        current = self.nodes.get(model_id)

        while current is not None and current.parent_model_id is not None:
            parent_id = current.parent_model_id
            if parent_id in visited:
                break  # cycle guard
            visited.add(parent_id)
            parent_node = self.nodes.get(parent_id)
            if parent_node is None:
                break
            result.append(parent_node)
            current = parent_node

        return result

    def descendants(self, model_id: str) -> List[ModelNode]:
        """Find all nodes that have *model_id* as an ancestor.

        Uses BFS to collect the full descendant tree.  Returns nodes
        in breadth-first order.
        """
        children_map: Dict[str, List[str]] = {}
        for nid, node in self.nodes.items():
            if node.parent_model_id is not None:
                children_map.setdefault(node.parent_model_id, []).append(nid)

        result: List[ModelNode] = []
        queue = list(children_map.get(model_id, []))
        visited: set[str] = {model_id}

        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            node = self.nodes.get(nid)
            if node is not None:
                result.append(node)
                queue.extend(children_map.get(nid, []))

        return result

    # ---------------------------------------------------------------
    # Query: promotion chain
    # ---------------------------------------------------------------

    def promotion_chain(self, run_name: str) -> List[ModelNode]:
        """Return all promoted models for a run, in promotion order.

        Sorted by the ``emitted_at`` of the first promotion record.
        """
        promoted: List[ModelNode] = []
        for node in self.nodes.values():
            if node.run_name == run_name and node.promotions:
                promoted.append(node)

        # Sort by first promotion event_id (which is monotonic/sortable)
        promoted.sort(
            key=lambda n: n.promotions[0].event_id if n.promotions else ""
        )
        return promoted

    # ---------------------------------------------------------------
    # Query: model profile
    # ---------------------------------------------------------------

    def model_profile(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Build a comprehensive profile for a single model.

        Returns ``None`` if the model is unknown.  The profile includes
        ancestor chain, match history, promotion history, and child count.
        """
        node = self.nodes.get(model_id)
        if node is None:
            return None

        ancestor_ids = [a.model_id for a in self.ancestors(model_id)]
        descendant_ids = [d.model_id for d in self.descendants(model_id)]

        return {
            "model_id": model_id,
            "run_name": node.run_name,
            "global_timestep": node.global_timestep,
            "checkpoint_path": node.checkpoint_path,
            "parent_model_id": node.parent_model_id,
            "created_at": node.created_at,
            "latest_rating": node.latest_rating,
            "ancestor_chain": ancestor_ids,
            "descendant_count": len(descendant_ids),
            "descendant_ids": descendant_ids,
            "matches": [
                {
                    "opponent": m.opponent_model_id,
                    "result": m.result,
                    "num_games": m.num_games,
                    "win_rate": m.win_rate,
                }
                for m in node.matches
            ],
            "promotions": [
                {
                    "from_rating": p.from_rating,
                    "to_rating": p.to_rating,
                    "reason": p.reason,
                }
                for p in node.promotions
            ],
        }

    # ---------------------------------------------------------------
    # Snapshot for WebUI
    # ---------------------------------------------------------------

    def to_snapshot(self) -> Dict[str, Any]:
        """Serialize the full graph for WebUI consumption.

        Returns a dict with ``nodes`` (list of node dicts) and ``edges``
        (list of parent→child pairs).
        """
        node_list = []
        edges = []

        for model_id, node in self.nodes.items():
            node_list.append(node.to_dict())
            if node.parent_model_id is not None:
                edges.append({
                    "from": node.parent_model_id,
                    "to": model_id,
                    "type": "parent",
                })

        return {
            "node_count": len(node_list),
            "edge_count": len(edges),
            "nodes": node_list,
            "edges": edges,
        }
