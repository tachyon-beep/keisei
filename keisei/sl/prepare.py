"""SL data preparation: parse game records, encode positions, write shards."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from keisei.sl.dataset import OBS_SIZE, SCORE_NORMALIZATION, write_shard
from keisei.sl.parsers import (
    CSAParser,
    GameFilter,
    GameOutcome,
    GameParser,
    SFENParser,
)

logger = logging.getLogger(__name__)

def _build_parser_registry() -> dict[str, GameParser]:
    """Build a parser registry mapping file extensions to parser instances."""
    registry: dict[str, GameParser] = {}
    for parser_cls in [SFENParser, CSAParser]:
        parser = parser_cls()
        for ext in parser.supported_extensions():
            if ext in registry:
                raise ValueError(f"Duplicate parser for extension '{ext}'")
            registry[ext] = parser
    return registry


def _iter_records_safe(parser: GameParser, game_file: Path):
    """Yield records from parser, logging per-record errors without discarding prior results."""
    try:
        it = parser.parse(game_file)
        while True:
            try:
                yield next(it)
            except StopIteration:
                return
            except Exception:
                logger.exception("Failed to parse a record in %s — skipping record", game_file)
                yield None
    except Exception:
        logger.exception("Failed to open/parse %s — skipping file", game_file)
        yield None


def prepare_sl_data(
    game_sources: list[str],
    output_dir: str,
    min_ply: int = 40,
    min_rating: int | None = None,
    shard_size: int = 100_000,
) -> None:
    """Parse game records, encode positions, and write shards.

    For initial implementation, positions are encoded as flat observation
    tensors using the Rust VecEnv. This requires the shogi-gym native module.

    NOTE: For production scale, parallelize via multiprocessing or Rust rayon.
    This implementation is single-threaded for correctness validation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parsers = _build_parser_registry()
    game_filter = GameFilter(min_ply=min_ply, min_rating=min_rating)

    # Collect all game files
    game_files: list[Path] = []
    for source in game_sources:
        source_path = Path(source)
        if source_path.is_file():
            game_files.append(source_path)
        elif source_path.is_dir():
            for ext in parsers:
                game_files.extend(source_path.glob(f"*{ext}"))

    logger.info("Found %d game files across %d sources", len(game_files), len(game_sources))

    # Accumulate positions into shard buffers
    observations: list[np.ndarray] = []
    policy_targets: list[int] = []
    value_targets: list[int] = []
    score_targets: list[float] = []
    shard_idx = 0
    games_parsed = 0
    games_skipped = 0

    # Warn loudly about placeholder data
    logger.warning(
        "*** PLACEHOLDER MODE: observations are all-zeros, policy targets are all-zeros. ***\n"
        "    Shards produced are structurally valid but semantically useless for training.\n"
        "    Full observation/policy encoding requires the Rust engine (shogi-gym).\n"
        "    Use these shards ONLY for pipeline testing, NOT for model training."
    )

    parse_errors = 0
    for game_file in game_files:
        ext = game_file.suffix.lower()
        parser = parsers.get(ext)
        if parser is None:
            logger.warning("No parser for extension '%s', skipping %s", ext, game_file)
            continue

        for record in _iter_records_safe(parser, game_file):
            if record is None:
                parse_errors += 1
                continue
            if not game_filter.accepts(record):
                games_skipped += 1
                continue

            games_parsed += 1

            # Determine W/D/L target and score for each position
            for i, move in enumerate(record.moves):
                # Side-to-move perspective: even moves = Black, odd = White
                is_black_to_move = i % 2 == 0

                if record.outcome == GameOutcome.WIN_BLACK:
                    value_cat = 0 if is_black_to_move else 2  # W or L
                    raw_score = 1.0 if is_black_to_move else -1.0
                elif record.outcome == GameOutcome.WIN_WHITE:
                    value_cat = 2 if is_black_to_move else 0
                    raw_score = -1.0 if is_black_to_move else 1.0
                else:
                    value_cat = 1  # Draw
                    raw_score = 0.0

                # NOTE: observation encoding and policy target encoding
                # require the Rust engine to replay the position from SFEN.
                # For this initial implementation, we store placeholder
                # observations. A full implementation would:
                #   1. Create a GameState from the starting SFEN
                #   2. Replay moves to reach this position
                #   3. Generate observation via KataGoObservationGenerator
                #   4. Encode the played move via SpatialActionMapper
                #
                # This placeholder allows testing the shard write/read pipeline
                # without requiring the Rust engine.
                obs = np.zeros(OBS_SIZE, dtype=np.float32)  # placeholder
                policy_target = 0  # placeholder

                observations.append(obs)
                policy_targets.append(policy_target)
                value_targets.append(value_cat)
                # FIXME(keisei-8ad9dd8509): score_targets use game outcome (±1/76 ≈ ±0.013),
                # not material difference. The score head will learn near-zero targets from
                # this data. Real material scoring requires Rust replay of positions to compute
                # material_balance() at each move. This placeholder is structurally correct
                # (valid shard format) but semantically wrong for score head training.
                score_targets.append(raw_score / SCORE_NORMALIZATION)

                # Flush inside the per-move loop so shard_size is a true cap,
                # not a post-game threshold.
                if len(observations) >= shard_size:
                    _flush_shard(
                        output_path,
                        shard_idx,
                        observations,
                        policy_targets,
                        value_targets,
                        score_targets,
                    )
                    shard_idx += 1
                    observations.clear()
                    policy_targets.clear()
                    value_targets.clear()
                    score_targets.clear()

    # Flush remaining
    if observations:
        _flush_shard(
            output_path,
            shard_idx,
            observations,
            policy_targets,
            value_targets,
            score_targets,
        )
        shard_idx += 1

    # Write shard metadata so downstream consumers can detect placeholder data.
    meta_path = output_path / "shard_meta.json"
    meta = {
        "placeholder": True,
        "num_shards": shard_idx,
        "num_games": games_parsed,
    }
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
    tmp_meta.rename(meta_path)
    logger.info("Wrote %s (placeholder=%s)", meta_path, meta["placeholder"])

    logger.info(
        "Prepared %d shards from %d games (%d skipped by filter, %d parse errors)",
        shard_idx,
        games_parsed,
        games_skipped,
        parse_errors,
    )


def _flush_shard(
    output_path: Path,
    shard_idx: int,
    observations: list[np.ndarray],
    policy_targets: list[int],
    value_targets: list[int],
    score_targets: list[float],
) -> None:
    n = len(observations)
    shard_path = output_path / f"shard_{shard_idx:03d}.bin"
    write_shard(
        shard_path,
        np.array(observations, dtype=np.float32),
        np.array(policy_targets, dtype=np.int64),
        np.array(value_targets, dtype=np.int64),
        np.array(score_targets, dtype=np.float32),
    )
    logger.info("Wrote shard %s with %d positions", shard_path.name, n)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Prepare SL training data")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Directories or files containing game records",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed shards",
    )
    parser.add_argument("--min-ply", type=int, default=40)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=100_000)
    args = parser.parse_args()

    prepare_sl_data(
        game_sources=args.sources,
        output_dir=args.output,
        min_ply=args.min_ply,
        min_rating=args.min_rating,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
