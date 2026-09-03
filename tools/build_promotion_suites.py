"""Build three sealed promotion suites from disjoint Lichess eval windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import cast

import chess

from tools.build_position_suite import (
    MAX_ABS_EVAL_CP,
    MIN_PAWNS,
    MIN_PIECES,
    POSITIONS_DIRECTORY,
    SOURCE_DUMP_DATE,
    SOURCE_URL,
    PositionCandidate,
    StreamSummary,
    SuiteBuildError,
    build_promotion_positions,
    iter_jsonl,
    phase_for,
    suite_content,
    suite_statistics,
)

WINDOW_SIZE = 250_000


@dataclass(frozen=True)
class PromotionSpec:
    name: str
    filename: str
    record_start: int
    record_end: int
    seed: int


PROMOTION_SPECS = (
    PromotionSpec("promotion-a-96", "promotion-a-96.fens", 250_001, 500_000, 20_260_902),
    PromotionSpec("promotion-b-96", "promotion-b-96.fens", 500_001, 750_000, 20_260_903),
    PromotionSpec("promotion-c-96", "promotion-c-96.fens", 750_001, 1_000_000, 20_260_904),
)
METADATA_FILENAME = "promotion-suite-metadata.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build three sealed 96-position promotion suites from fixed, disjoint "
            "Lichess evaluation-database windows. URL .zst sources require zstandard."
        )
    )
    parser.add_argument("--source", default=SOURCE_URL, help="JSONL or JSONL.zst path/URL.")
    parser.add_argument("--source-date", default=SOURCE_DUMP_DATE)
    parser.add_argument("--output-directory", type=Path, default=POSITIONS_DIRECTORY)
    return parser


def build_all(
    records: Iterator[tuple[int, bytes]],
) -> dict[PromotionSpec, tuple[list[PositionCandidate], StreamSummary]]:
    """Build all suites in one forward-only source pass."""
    skipped = sum(1 for _ in islice(records, PROMOTION_SPECS[0].record_start - 1))
    expected_skip = PROMOTION_SPECS[0].record_start - 1
    if skipped != expected_skip:
        raise SuiteBuildError(
            f"source ended while skipping records: saw {skipped:,}; expected {expected_skip:,}"
        )

    built: dict[PromotionSpec, tuple[list[PositionCandidate], StreamSummary]] = {}
    for spec in PROMOTION_SPECS:
        if spec.record_end - spec.record_start + 1 != WINDOW_SIZE:
            raise SuiteBuildError(f"invalid window size for {spec.name}")
        positions, stream = build_promotion_positions(
            islice(records, WINDOW_SIZE),
            record_start=spec.record_start,
            record_limit=WINDOW_SIZE,
            seed=spec.seed,
        )
        built[spec] = (positions, stream)
    return built


def validate_suite(
    spec: PromotionSpec,
    path: Path,
    positions: Sequence[PositionCandidate],
    expected_hash: str,
) -> set[str]:
    """Validate generated data using python-chess, without invoking an engine."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise SuiteBuildError(f"hash mismatch after writing {path}")
    fens = [line for line in content.decode().splitlines() if line]
    if len(fens) != 96 or len(fens) != len(positions):
        raise SuiteBuildError(f"{spec.name} does not contain exactly 96 positions")
    if len(fens) != len(set(fens)):
        raise SuiteBuildError(f"{spec.name} contains duplicate positions")
    if len({position.pawn_structure_sha256 for position in positions}) != 96:
        raise SuiteBuildError(f"{spec.name} contains duplicate pawn structures")
    if Counter(position.category for position in positions) != {
        "opening_high_material": 24,
        "middlegame": 48,
        "endgame": 24,
    }:
        raise SuiteBuildError(f"{spec.name} has incorrect phase distribution")
    if Counter(position.side_to_move for position in positions) != {"white": 48, "black": 48}:
        raise SuiteBuildError(f"{spec.name} has incorrect side-to-move distribution")
    if Counter(position.capture_available for position in positions) != {False: 48, True: 48}:
        raise SuiteBuildError(f"{spec.name} has incorrect capture distribution")

    for fen, position in zip(fens, positions, strict=True):
        board = chess.Board(fen)
        if not board.is_valid() or board.outcome(claim_draw=False) is not None:
            raise SuiteBuildError(f"invalid or terminal generated position: {fen}")
        if board.is_check() or not any(board.legal_moves):
            raise SuiteBuildError(f"checked or move-less generated position: {fen}")
        if board.halfmove_clock != 0 or board.fullmove_number != 1:
            raise SuiteBuildError(f"non-normalized generated FEN counters: {fen}")
        if not spec.record_start <= position.source_record <= spec.record_end:
            raise SuiteBuildError(f"source record outside {spec.name} window")
        if abs(position.eval_cp) > MAX_ABS_EVAL_CP or position.depth < 1:
            raise SuiteBuildError(f"invalid evaluation provenance in {spec.name}")
        if position.category != phase_for(board):
            raise SuiteBuildError(f"phase metadata mismatch in {spec.name}")
        if position.capture_available != any(board.is_capture(move) for move in board.legal_moves):
            raise SuiteBuildError(f"capture metadata mismatch in {spec.name}")
        has_castling_rights = board.has_castling_rights(chess.WHITE) or board.has_castling_rights(
            chess.BLACK
        )
        if position.castling_rights != has_castling_rights:
            raise SuiteBuildError(f"castling metadata mismatch in {spec.name}")
    return set(fens)


def write_outputs(
    built: Mapping[PromotionSpec, tuple[Sequence[PositionCandidate], StreamSummary]],
    *,
    output_directory: Path,
    source: str,
    source_date: str,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    suite_metadata: dict[str, object] = {}
    all_fens: set[str] = set()

    for spec in PROMOTION_SPECS:
        positions, stream = built[spec]
        path = output_directory / spec.filename
        content = suite_content(positions)
        path.write_bytes(content)
        suite_hash = hashlib.sha256(content).hexdigest()
        suite_fens = validate_suite(spec, path, positions, suite_hash)
        if all_fens.intersection(suite_fens):
            raise SuiteBuildError(f"{spec.name} overlaps an earlier promotion suite")
        all_fens.update(suite_fens)

        suite_metadata[spec.name] = {
            "path": spec.filename,
            "sha256": suite_hash,
            "seed": spec.seed,
            "source_window": {
                "record_start": spec.record_start,
                "record_end": spec.record_end,
                "records_examined": stream.records_examined,
                "scanned_jsonl_bytes": stream.scanned_jsonl_bytes,
                "scanned_jsonl_sha256": stream.scanned_jsonl_sha256,
            },
            "statistics": suite_statistics(positions),
            "eligible_unique_positions": stream.eligible_unique_positions,
            "rejection_counts": stream.rejection_counts,
            "eligible_by_bucket": stream.eligible_by_bucket,
            "positions": [asdict(position) for position in positions],
        }

    metadata: dict[str, object] = {
        "schema": "aichessathon.promotion-suite-provenance",
        "schema_version": 1,
        "source": {
            "url_or_path": source,
            "database_dump_date": source_date,
            "license": "CC0",
            "evaluation_source": "Stockfish evaluations contributed through Lichess analysis",
        },
        "selection": {
            "method": (
                "per-stratum reservoir sampling independently within each fixed source window; "
                "deterministic final shuffle"
            ),
            "window_size": WINDOW_SIZE,
            "criteria": {
                "variant": "valid standard chess",
                "evaluation": (
                    f"highest-depth evaluation, first PV; no mate; abs(cp) <= {MAX_ABS_EVAL_CP}"
                ),
                "terminal": "non-terminal; no pre-root history; sufficient material",
                "material": f"at least {MIN_PIECES} pieces and {MIN_PAWNS} pawns",
                "check": "side to move is not in check",
                "fen_counters": "history-free normalization to halfmove 0 and fullmove 1",
                "position_deduplication": "first four canonical FEN fields",
                "pawn_structure_deduplication": "unique within each suite",
                "strata": (
                    "phase, side to move, capture availability, and "
                    "(outside endgames) castling rights"
                ),
            },
            "phase_definition": {
                "units": "queen=4, rook=2, bishop=1, knight=1; both colors; maximum 24",
                "opening_high_material": ">=18",
                "middlegame": "8..17",
                "endgame": "<=7",
            },
        },
        "suites": suite_metadata,
        "sealing_policy": {
            "next_available": "promotion-a-96",
            "order": ["promotion-a-96", "promotion-b-96", "promotion-c-96"],
            "policy": (
                "Promotion A is the next available promotion validation set. Once used for a "
                "promotion decision it is consumed, and Promotion B becomes next. Promotion C "
                "is reserved after B. A consumed suite must never be reused to tune a candidate."
            ),
            "generation_validation": (
                "Structural and legality validation used python-chess only; no agent or chess "
                "engine was run against any promotion suite during generation."
            ),
        },
    }
    (output_directory / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def print_report(metadata: Mapping[str, object]) -> None:
    suites = cast(Mapping[str, Mapping[str, object]], metadata["suites"])
    for name, suite in suites.items():
        print(f"{name} sha256: {suite['sha256']}")
        print(f"{name}: {suite['statistics']}")


def main() -> None:
    arguments = build_parser().parse_args()
    source = cast(str, arguments.source)
    try:
        with ExitStack() as stack:
            built = build_all(iter_jsonl(source, stack))
        metadata = write_outputs(
            built,
            output_directory=cast(Path, arguments.output_directory),
            source=source,
            source_date=cast(str, arguments.source_date),
        )
    except (OSError, SuiteBuildError, urllib.error.URLError) as error:
        raise SystemExit(f"error: {error}") from error
    print_report(metadata)


if __name__ == "__main__":
    main()
