"""DemonstratorRunner — threaded inference-only exhibition matches."""

from __future__ import annotations

import logging
import random
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from keisei.training.match_utils import release_models
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role

logger = logging.getLogger(__name__)


def _get_policy_flat(model_output: Any, batch_size: int) -> torch.Tensor:
    """Extract flat policy logits from either BaseModel or KataGoBaseModel output.

    BaseModel.forward() returns (policy_logits, value) — policy already flat.
    KataGoBaseModel.forward() returns KataGoOutput — policy is (B, 9, 9, 139).
    """
    if isinstance(model_output, tuple):
        result: torch.Tensor = model_output[0]
        return result
    else:
        result = model_output.policy_logits.reshape(batch_size, -1)
        return result


@dataclass
class DemoMatchup:
    """A demonstrator game matchup."""

    slot: int
    entry_a: OpponentEntry
    entry_b: OpponentEntry


class DemonstratorRunner(threading.Thread):
    """Runs inference-only exhibition matches in a background thread.

    Error policy: crashes are non-fatal. The run() method wraps the loop in
    try/except, logs the exception, and stops. The training loop can check
    is_alive() at epoch boundaries.
    """

    def __init__(
        self,
        store: OpponentStore,
        db_path: str,
        num_slots: int = 3,
        moves_per_minute: int = 60,
        device: str = "cpu",
    ) -> None:
        super().__init__(daemon=True, name="DemonstratorRunner")
        self.store = store
        self.db_path = db_path  # reserved for future game-state persistence (not yet wired)
        self.num_slots = num_slots
        self.move_delay = 60.0 / max(moves_per_minute, 1)
        self.device = device
        self._stop_event = threading.Event()

        self._stream = None
        if device.startswith("cuda") and torch.cuda.is_available():
            self._stream = torch.cuda.Stream()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        """Thread main loop — wrapped in try/except for non-fatal crash policy."""
        try:
            self._run_loop()
        except Exception:
            logger.exception("DemonstratorRunner crashed — thread stopping")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            matchups = self._select_matchups()
            if not matchups:
                self._stop_event.wait(timeout=5.0)
                continue

            for matchup in matchups:
                if self._stop_event.is_set():
                    return
                try:
                    self._play_game(matchup)
                except Exception:
                    logger.exception("Demo slot %d game failed — skipping", matchup.slot)

    def _select_matchups(self) -> list[DemoMatchup]:
        """Select role-aware matchups for active demo slots (§11, §12.6).

        Slot 1: Cross-tier championship — top Dynamic vs top Frontier Static.
        Slot 2: Intra-tier — two Dynamic entries (or two from the largest tier).
        Slot 3: Random cross-tier pairing.

        Falls back gracefully when tiers are empty or too small.
        """
        entries = self.store.list_entries()
        if len(entries) < 2:
            return []

        by_role: dict[Role, list[OpponentEntry]] = {}
        for e in entries:
            by_role.setdefault(e.role, []).append(e)

        # Sort each tier by Elo descending
        for role in by_role:
            by_role[role].sort(key=lambda e: e.elo_rating, reverse=True)

        dynamic = by_role.get(Role.DYNAMIC, [])
        frontier = by_role.get(Role.FRONTIER_STATIC, [])
        recent = by_role.get(Role.RECENT_FIXED, [])
        matchups: list[DemoMatchup] = []

        # Slot 1: cross-tier championship (Dynamic #1 vs Frontier #1)
        if self.num_slots >= 1:
            if dynamic and frontier:
                matchups.append(DemoMatchup(slot=1, entry_a=dynamic[0], entry_b=frontier[0]))
            elif len(entries) >= 2:
                # Fallback: top-2 by Elo regardless of tier
                top = sorted(entries, key=lambda e: e.elo_rating, reverse=True)
                matchups.append(DemoMatchup(slot=1, entry_a=top[0], entry_b=top[1]))

        # Slot 2: intra-tier (prefer Dynamic vs Dynamic)
        if self.num_slots >= 2:
            if len(dynamic) >= 2:
                pair = random.sample(dynamic, 2)
                matchups.append(DemoMatchup(slot=2, entry_a=pair[0], entry_b=pair[1]))
            elif len(recent) >= 2:
                pair = random.sample(recent, 2)
                matchups.append(DemoMatchup(slot=2, entry_a=pair[0], entry_b=pair[1]))
            elif len(entries) >= 2:
                pair = random.sample(entries, 2)
                matchups.append(DemoMatchup(slot=2, entry_a=pair[0], entry_b=pair[1]))

        # Slot 3: random cross-tier pairing
        if self.num_slots >= 3 and len(entries) >= 2:
            populated_roles = [r for r in by_role if len(by_role[r]) > 0]
            if len(populated_roles) >= 2:
                r1, r2 = random.sample(populated_roles, 2)
                a = random.choice(by_role[r1])
                b = random.choice(by_role[r2])
                matchups.append(DemoMatchup(slot=3, entry_a=a, entry_b=b))
            else:
                pair = random.sample(entries, 2)
                matchups.append(DemoMatchup(slot=3, entry_a=pair[0], entry_b=pair[1]))

        return matchups

    def _play_game(self, matchup: DemoMatchup) -> None:
        """Play a single demonstrator game to completion."""
        # Pin entries for the ENTIRE duration of model usage (load + inference),
        # not just during loading.  Unpinning early would allow eviction to
        # delete the checkpoint while the model is still in use.
        self.store.pin(matchup.entry_a.id)
        self.store.pin(matchup.entry_b.id)
        model_a = None
        model_b = None
        try:
            try:
                model_a = self.store.load_opponent(matchup.entry_a, device=self.device)
                model_b = self.store.load_opponent(matchup.entry_b, device=self.device)
            except FileNotFoundError:
                logger.warning("Checkpoint missing for demo slot %d — skipping", matchup.slot)
                return
            except Exception:
                logger.exception("Failed to load models for demo slot %d — skipping", matchup.slot)
                return

            logger.info(
                "Demo slot %d: %s (elo=%.0f) vs %s (elo=%.0f)",
                matchup.slot,
                matchup.entry_a.architecture, matchup.entry_a.elo_rating,
                matchup.entry_b.architecture, matchup.entry_b.elo_rating,
            )

            try:
                from shogi_gym import VecEnv
                env = VecEnv(
                    num_envs=1, max_ply=512,
                    observation_mode="katago",  # type: ignore[call-arg]
                    action_mode="spatial",
                )
            except ImportError:
                logger.warning("shogi_gym not available — demo slot %d inactive", matchup.slot)
                return

            reset_result = env.reset()
            obs = torch.from_numpy(np.asarray(reset_result.observations)).to(self.device)
            legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(self.device)
            current_player = 0
            models = [model_a, model_b]
            done = False

            while not done and not self._stop_event.is_set():
                model = models[current_player]
                ctx = torch.cuda.stream(self._stream) if self._stream else nullcontext()
                with ctx:
                    with torch.no_grad():
                        # Guard: zero legal actions → all-inf softmax → NaN crash
                        legal_counts = legal_masks.sum(dim=-1)
                        if (legal_counts == 0).any():
                            logger.warning(
                                "Demo slot %d: zero legal actions detected — ending game",
                                matchup.slot,
                            )
                            break

                        output = model(obs)
                        flat = _get_policy_flat(output, obs.shape[0])
                        masked = flat.masked_fill(~legal_masks, float("-inf"))
                        probs = F.softmax(masked, dim=-1)
                        action = torch.distributions.Categorical(probs).sample()

                step_result = env.step(action.tolist())
                done = bool(step_result.terminated[0] or step_result.truncated[0])
                obs = torch.from_numpy(np.asarray(step_result.observations)).to(self.device)
                legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(self.device)
                current_player = int(step_result.current_players[0])
                time.sleep(self.move_delay)

            logger.info("Demo slot %d game completed", matchup.slot)
        finally:
            # Release GPU memory for loaded models to prevent OOM in daemon loop
            if model_a is not None or model_b is not None:
                models_to_release = [m for m in (model_a, model_b) if m is not None]
                release_models(*models_to_release, device_type=self.device.split(":")[0])
                del model_a, model_b
            self.store.unpin(matchup.entry_a.id)
            self.store.unpin(matchup.entry_b.id)
