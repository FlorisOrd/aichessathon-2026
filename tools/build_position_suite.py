"""Build deterministic development and holdout FEN suites from Lichess evals."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from statistics import median
from typing import Any, BinaryIO, Literal, cast

import chess

SOURCE_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
SOURCE_DUMP_DATE = "2026-08-02"
SEED = 20_260_901
SOURCE_RECORD_LIMIT = 250_000
MAX_ABS_EVAL_CP = 50
MIN_PIECES = 8
MIN_PAWNS = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POSITIONS_DIRECTORY = REPOSITORY_ROOT / "benchmarks" / "positions"
DEVELOPMENT_PATH = POSITIONS_DIRECTORY / "development-64.fens"
HOLDOUT_PATH = POSITIONS_DIRECTORY / "holdout-32.fens"
FAST_PATH = POSITIONS_DIRECTORY / "development-fast-16.fens"
METADATA_PATH = POSITIONS_DIRECTORY / "development-suite-metadata.json"

Phase = Literal["opening_high_material", "middlegame", "endgame"]
Side = Literal["white", "black"]
Bucket = tuple[Phase, Side, bool, bool]


class SuiteBuildError(Exception):
    """The source or selected corpus cannot produce the requested suites."""


@dataclass(frozen=True)
class PositionCandidate:
    fen: str
    eval_cp: int
    depth: int
    knodes: int | None
    first_pv: str
    category: Phase
    side_to_move: Side
    capture_available: bool
    castling_rights: bool
    piece_count: int
    pawn_count: int
    pawn_structure_sha256: str
    source_record: int


@dataclass(frozen=True)
class StreamSummary:
    record_start: int
    record_end: int
    records_examined: int
    eligible_unique_positions: int
    scanned_jsonl_bytes: int
    scanned_jsonl_sha256: str
    rejection_counts: dict[str, int]
    eligible_by_bucket: dict[str, int]


@dataclass(frozen=True)
class BuiltSuites:
    development: list[PositionCandidate]
    holdout: list[PositionCandidate]
    fast: list[PositionCandidate]
    stream: StreamSummary


@dataclass(frozen=True)
class SampledWindow:
    reservoirs: dict[Bucket, list[PositionCandidate]]
    stream: StreamSummary


PHASE_TOTALS: Mapping[Phase, int] = {
    "opening_high_material": 24,
    "middlegame": 48,
    "endgame": 24,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed, balanced FEN suites from a deterministic prefix of the public "
            "Lichess evaluation database. URL sources ending in .zst require zstandard; "
            "run with `uv run --with zstandard python -m tools.build_position_suite`."
        )
    )
    parser.add_argument("--source", default=SOURCE_URL, help="JSONL or JSONL.zst path/URL.")
    parser.add_argument("--source-date", default=SOURCE_DUMP_DATE)
    parser.add_argument(
        "--start-record",
        type=int,
        default=1,
        help="One-based first non-empty JSONL record to include (default: 1).",
    )
    parser.add_argument("--records", type=int, default=SOURCE_RECORD_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-directory", type=Path, default=POSITIONS_DIRECTORY)
    return parser


def normalize_fen(raw_fen: str) -> str:
    """Normalize a four- or six-field FEN to history-free six-field form."""
    fields = raw_fen.split()
    if len(fields) not in {4, 6}:
        raise ValueError("FEN must have four or six fields")
    # The dump documents four-field FENs. Even if a local input has counters, reset them:
    # neither the dump nor get_move() provides pre-root move/repetition history.
    return " ".join([*fields[:4], "0", "1"])


def phase_for(board: chess.Board) -> Phase:
    """Classify by conventional non-pawn material phase units (maximum 24)."""
    phase_units = 0
    for color in chess.COLORS:
        phase_units += 4 * len(board.pieces(chess.QUEEN, color))
        phase_units += 2 * len(board.pieces(chess.ROOK, color))
        phase_units += len(board.pieces(chess.BISHOP, color))
        phase_units += len(board.pieces(chess.KNIGHT, color))
    if phase_units >= 18:
        return "opening_high_material"
    if phase_units >= 8:
        return "middlegame"
    return "endgame"


def select_highest_depth_first_pv(record: Mapping[str, Any]) -> tuple[int, int, int | None, str]:
    evals = record.get("evals")
    if not isinstance(evals, list) or not evals:
        raise ValueError("missing evaluations")
    valid_evals = [
        item for item in evals if isinstance(item, dict) and isinstance(item.get("depth"), int)
    ]
    if not valid_evals:
        raise ValueError("missing evaluation depth")
    evaluation = max(valid_evals, key=lambda item: cast(int, item["depth"]))
    pvs = evaluation.get("pvs")
    if not isinstance(pvs, list) or not pvs or not isinstance(pvs[0], dict):
        raise ValueError("missing first PV")
    pv = pvs[0]
    if "mate" in pv or not isinstance(pv.get("cp"), int):
        raise ValueError("mate or missing centipawn evaluation")
    line = pv.get("line")
    if not isinstance(line, str) or not line:
        raise ValueError("missing PV line")
    knodes_value = evaluation.get("knodes")
    knodes = knodes_value if isinstance(knodes_value, int) else None
    return cast(int, pv["cp"]), cast(int, evaluation["depth"]), knodes, line


def candidate_from_record(record: Mapping[str, Any], source_record: int) -> PositionCandidate:
    raw_fen = record.get("fen")
    if not isinstance(raw_fen, str):
        raise ValueError("missing FEN")
    fen = normalize_fen(raw_fen)
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("invalid standard-chess position")
    if board.outcome(claim_draw=False) is not None or not any(board.legal_moves):
        raise ValueError("terminal position")
    if board.is_insufficient_material():
        raise ValueError("insufficient material")
    if board.is_check():
        raise ValueError("side to move is in check")

    piece_count = len(board.piece_map())
    pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) + len(
        board.pieces(chess.PAWN, chess.BLACK)
    )
    if piece_count < MIN_PIECES or pawn_count < MIN_PAWNS:
        raise ValueError("trivial material")

    eval_cp, depth, knodes, first_pv = select_highest_depth_first_pv(record)
    if abs(eval_cp) > MAX_ABS_EVAL_CP:
        raise ValueError("evaluation outside neutral window")

    legal_moves = list(board.legal_moves)
    pawn_key = (
        f"{board.pieces(chess.PAWN, chess.WHITE).mask:016x}:"
        f"{board.pieces(chess.PAWN, chess.BLACK).mask:016x}"
    )
    return PositionCandidate(
        fen=fen,
        eval_cp=eval_cp,
        depth=depth,
        knodes=knodes,
        first_pv=first_pv,
        category=phase_for(board),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        capture_available=any(board.is_capture(move) for move in legal_moves),
        castling_rights=bool(board.castling_rights),
        piece_count=piece_count,
        pawn_count=pawn_count,
        pawn_structure_sha256=hashlib.sha256(pawn_key.encode()).hexdigest(),
        source_record=source_record,
    )


def bucket_for(candidate: PositionCandidate) -> Bucket:
    # Castling is stratified only while it remains meaningful; requiring it in endgames
    # would create a distorted or impossible sample.
    castling = candidate.castling_rights if candidate.category != "endgame" else False
    return (
        candidate.category,
        candidate.side_to_move,
        candidate.capture_available,
        castling,
    )


def bucket_quota(bucket: Bucket) -> int:
    phase = bucket[0]
    return 6 if phase == "endgame" else PHASE_TOTALS[phase] // 8


def all_buckets() -> list[Bucket]:
    buckets: list[Bucket] = []
    for phase in PHASE_TOTALS:
        castling_values = (False,) if phase == "endgame" else (False, True)
        for side in ("white", "black"):
            for capture in (False, True):
                for castling in castling_values:
                    buckets.append((phase, side, capture, castling))
    return buckets


def bucket_name(bucket: Bucket) -> str:
    phase, side, capture, castling = bucket
    return f"{phase}|{side}|capture={str(capture).lower()}|castling={str(castling).lower()}"


def sample_window(
    records: Iterator[tuple[int, bytes]],
    *,
    record_start: int,
    record_limit: int,
    seed: int,
    unique_pawn_structures: bool = False,
) -> SampledWindow:
    """Reservoir-sample one already-sliced source window into the fixed strata."""
    if record_start < 1:
        raise SuiteBuildError("record start must be positive")
    if record_limit < 1:
        raise SuiteBuildError("record limit must be positive")

    rng = random.Random(seed)
    reservoirs: dict[Bucket, list[PositionCandidate]] = {bucket: [] for bucket in all_buckets()}
    eligible_by_bucket: Counter[Bucket] = Counter()
    rejections: Counter[str] = Counter()
    seen_positions: set[str] = set()
    seen_pawn_structures: set[str] = set()
    scanned_hash = hashlib.sha256()
    scanned_bytes = 0
    records_examined = 0
    first_record_number: int | None = None
    last_record_number: int | None = None

    for record_number, raw_line in records:
        if records_examined >= record_limit:
            break
        records_examined += 1
        first_record_number = first_record_number or record_number
        last_record_number = record_number
        scanned_hash.update(raw_line)
        scanned_bytes += len(raw_line)
        try:
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError("record is not an object")
            candidate = candidate_from_record(value, record_number)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            rejections[str(error)] += 1
            continue

        position_key = " ".join(candidate.fen.split()[:4])
        if position_key in seen_positions:
            rejections["duplicate position"] += 1
            continue
        seen_positions.add(position_key)
        if unique_pawn_structures and candidate.pawn_structure_sha256 in seen_pawn_structures:
            rejections["duplicate pawn structure"] += 1
            continue
        seen_pawn_structures.add(candidate.pawn_structure_sha256)

        bucket = bucket_for(candidate)
        eligible_by_bucket[bucket] += 1
        seen_in_bucket = eligible_by_bucket[bucket]
        reservoir = reservoirs[bucket]
        quota = bucket_quota(bucket)
        if len(reservoir) < quota:
            reservoir.append(candidate)
        else:
            replacement = rng.randrange(seen_in_bucket)
            if replacement < quota:
                reservoir[replacement] = candidate

    if records_examined < record_limit:
        raise SuiteBuildError(
            f"source ended after {records_examined:,} records; expected {record_limit:,}"
        )
    expected_record_end = record_start + record_limit - 1
    if first_record_number != record_start or last_record_number != expected_record_end:
        raise SuiteBuildError(
            "source iterator does not match requested record window: "
            f"saw {first_record_number}..{last_record_number}; "
            f"expected {record_start}..{expected_record_end}"
        )

    underfilled = {
        bucket_name(bucket): (len(reservoirs[bucket]), bucket_quota(bucket))
        for bucket in all_buckets()
        if len(reservoirs[bucket]) < bucket_quota(bucket)
    }
    if underfilled:
        raise SuiteBuildError(f"source window underfilled selection buckets: {underfilled}")

    stream = StreamSummary(
        record_start=record_start,
        record_end=record_start + records_examined - 1,
        records_examined=records_examined,
        eligible_unique_positions=sum(eligible_by_bucket.values()),
        scanned_jsonl_bytes=scanned_bytes,
        scanned_jsonl_sha256=scanned_hash.hexdigest(),
        rejection_counts=dict(sorted(rejections.items())),
        eligible_by_bucket={
            bucket_name(bucket): eligible_by_bucket[bucket] for bucket in all_buckets()
        },
    )
    return SampledWindow(reservoirs=reservoirs, stream=stream)


def build_suites(
    records: Iterator[tuple[int, bytes]],
    *,
    record_limit: int,
    seed: int,
    record_start: int = 1,
) -> BuiltSuites:
    """Build the legacy 64/32 suites from an explicit one-based source window."""
    window_records = islice(records, record_start - 1, record_start - 1 + record_limit)
    sampled = sample_window(
        window_records,
        record_start=record_start,
        record_limit=record_limit,
        seed=seed,
    )

    development: list[PositionCandidate] = []
    holdout: list[PositionCandidate] = []
    split_rng = random.Random(seed ^ 0x5A17_2026)
    for bucket in all_buckets():
        selected = list(sampled.reservoirs[bucket])
        split_rng.shuffle(selected)
        development_count = bucket_quota(bucket) * 2 // 3
        development.extend(selected[:development_count])
        holdout.extend(selected[development_count:])
    split_rng.shuffle(development)
    split_rng.shuffle(holdout)

    fast = fast_subset(development, seed)
    return BuiltSuites(
        development=development,
        holdout=holdout,
        fast=fast,
        stream=sampled.stream,
    )


def build_promotion_positions(
    records: Iterator[tuple[int, bytes]],
    *,
    record_start: int,
    record_limit: int,
    seed: int,
) -> tuple[list[PositionCandidate], StreamSummary]:
    """Build one 96-position promotion suite with unique pawn structures."""
    sampled = sample_window(
        records,
        record_start=record_start,
        record_limit=record_limit,
        seed=seed,
        unique_pawn_structures=True,
    )
    positions = [position for bucket in all_buckets() for position in sampled.reservoirs[bucket]]
    random.Random(seed ^ 0xA11C_E096).shuffle(positions)
    return positions, sampled.stream


def fast_subset(development: Sequence[PositionCandidate], seed: int) -> list[PositionCandidate]:
    by_phase: dict[Phase, list[PositionCandidate]] = defaultdict(list)
    for candidate in development:
        by_phase[candidate.category].append(candidate)
    rng = random.Random(seed ^ 0xFA57_0016)
    targets: Mapping[Phase, int] = {
        "opening_high_material": 4,
        "middlegame": 8,
        "endgame": 4,
    }
    fast: list[PositionCandidate] = []
    for phase, count in targets.items():
        choices = list(by_phase[phase])
        rng.shuffle(choices)
        fast.extend(choices[:count])
    rng.shuffle(fast)
    return fast


def iter_jsonl(source: str, stack: ExitStack) -> Iterator[tuple[int, bytes]]:
    if source.startswith(("https://", "http://")):
        request = urllib.request.Request(
            source, headers={"User-Agent": "aichessathon-suite-builder/1"}
        )
        raw = cast(BinaryIO, stack.enter_context(closing(urllib.request.urlopen(request))))
    else:
        raw = stack.enter_context(Path(source).open("rb"))  # noqa: SIM115 - owned by ExitStack

    stream: BinaryIO
    if source.lower().endswith(".zst"):
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError as error:
            raise SuiteBuildError(
                "zstandard is required for .zst input; run with `uv run --with zstandard`"
            ) from error
        stream = cast(
            BinaryIO, stack.enter_context(zstandard.ZstdDecompressor().stream_reader(raw))
        )
    else:
        stream = raw

    buffered = stack.enter_context(io.BufferedReader(cast(io.RawIOBase, stream)))
    number = 0
    for raw_line in buffered:
        if not raw_line.strip():
            continue
        number += 1
        yield number, raw_line


def suite_content(positions: Sequence[PositionCandidate]) -> bytes:
    return ("".join(f"{position.fen}\n" for position in positions)).encode()


def suite_statistics(positions: Sequence[PositionCandidate]) -> dict[str, object]:
    abs_evals = [abs(position.eval_cp) for position in positions]
    return {
        "positions": len(positions),
        "category_counts": dict(sorted(Counter(p.category for p in positions).items())),
        "side_to_move_counts": dict(sorted(Counter(p.side_to_move for p in positions).items())),
        "capture_available_counts": {
            str(key).lower(): value
            for key, value in sorted(Counter(p.capture_available for p in positions).items())
        },
        "castling_rights_counts": {
            str(key).lower(): value
            for key, value in sorted(Counter(p.castling_rights for p in positions).items())
        },
        "unique_pawn_structures": len({p.pawn_structure_sha256 for p in positions}),
        "absolute_eval_cp": {
            "min": min(abs_evals),
            "max": max(abs_evals),
            "median": median(abs_evals),
        },
        "piece_count": {
            "min": min(p.piece_count for p in positions),
            "max": max(p.piece_count for p in positions),
            "median": median(p.piece_count for p in positions),
        },
    }


def write_outputs(
    built: BuiltSuites,
    *,
    output_directory: Path,
    source: str,
    source_date: str,
    record_limit: int,
    seed: int,
    record_start: int = 1,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "development-64": output_directory / DEVELOPMENT_PATH.name,
        "holdout-32": output_directory / HOLDOUT_PATH.name,
        "development-fast-16": output_directory / FAST_PATH.name,
    }
    suites = {
        "development-64": built.development,
        "holdout-32": built.holdout,
        "development-fast-16": built.fast,
    }
    hashes: dict[str, str] = {}
    for name, positions in suites.items():
        content = suite_content(positions)
        paths[name].write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()

    metadata: dict[str, object] = {
        "schema": "aichessathon.position-suite-provenance",
        "schema_version": 1,
        "source": {
            "url_or_path": source,
            "database_dump_date": source_date,
            "license": "CC0",
            "evaluation_source": "Stockfish evaluations contributed through Lichess analysis",
            "selection_scope": (
                f"non-empty JSONL records {record_start} through "
                f"{record_start + record_limit - 1} inclusive"
            ),
            "scanned_jsonl_bytes": built.stream.scanned_jsonl_bytes,
            "scanned_jsonl_sha256": built.stream.scanned_jsonl_sha256,
        },
        "selection": {
            "seed": seed,
            "method": "per-stratum reservoir sampling followed by deterministic 2:1 splitting",
            "record_start": record_start,
            "record_end": record_start + record_limit - 1,
            "record_limit": record_limit,
            "criteria": {
                "variant": "valid standard chess",
                "evaluation": f"highest-depth eval first PV; no mate; abs(cp) <= {MAX_ABS_EVAL_CP}",
                "terminal": "non-terminal; no pre-root history; sufficient material",
                "material": f"at least {MIN_PIECES} pieces and {MIN_PAWNS} pawns",
                "check": "side to move is not in check",
                "fen_counters": "history-free normalization to halfmove 0 and fullmove 1",
                "deduplication": "first four canonical FEN fields",
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
            "records_examined": built.stream.records_examined,
            "eligible_unique_positions": built.stream.eligible_unique_positions,
            "rejection_counts": built.stream.rejection_counts,
            "eligible_by_bucket": built.stream.eligible_by_bucket,
        },
        "suites": {
            name: {
                "path": paths[name].name,
                "sha256": hashes[name],
                "statistics": suite_statistics(positions),
                "positions": [asdict(position) for position in positions],
            }
            for name, positions in suites.items()
        },
        "holdout_policy": (
            "The holdout is the deterministic one-third split within every sampling stratum. "
            "Do not inspect its game results or tune against it during routine Phase 2 work; "
            "reserve it for infrequent promotion decisions."
        ),
    }
    metadata_path = output_directory / METADATA_PATH.name
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_outputs(paths, suites, hashes)
    return metadata


def validate_outputs(
    paths: Mapping[str, Path],
    suites: Mapping[str, Sequence[PositionCandidate]],
    hashes: Mapping[str, str],
) -> None:
    all_primary_fens: list[str] = []
    for name, path in paths.items():
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != hashes[name]:
            raise SuiteBuildError(f"hash mismatch after writing {path}")
        fens = [line for line in content.decode().splitlines() if line]
        if len(fens) != len(suites[name]):
            raise SuiteBuildError(f"position count mismatch after writing {path}")
        for fen in fens:
            board = chess.Board(fen)
            if not board.is_valid() or board.outcome(claim_draw=False) is not None:
                raise SuiteBuildError(f"invalid or terminal generated position: {fen}")
            if not any(board.legal_moves):
                raise SuiteBuildError(f"generated position has no legal moves: {fen}")
        if name != "development-fast-16":
            all_primary_fens.extend(fens)
    if len(all_primary_fens) != len(set(all_primary_fens)):
        raise SuiteBuildError("development and holdout suites overlap or contain duplicates")
    if not set(position.fen for position in suites["development-fast-16"]).issubset(
        position.fen for position in suites["development-64"]
    ):
        raise SuiteBuildError("fast suite is not a subset of the development suite")


def print_report(metadata: Mapping[str, object]) -> None:
    selection = cast(Mapping[str, object], metadata["selection"])
    print(
        f"examined {cast(int, selection['records_examined']):,} records; "
        f"eligible unique positions {cast(int, selection['eligible_unique_positions']):,}"
    )
    suites = cast(Mapping[str, Mapping[str, object]], metadata["suites"])
    for name, suite in suites.items():
        stats = cast(Mapping[str, object], suite["statistics"])
        print(f"{name}: {stats}")
        print(f"{name} sha256: {suite['sha256']}")


def main() -> None:
    arguments = build_parser().parse_args()
    records = cast(int, arguments.records)
    record_start = cast(int, arguments.start_record)
    source = cast(str, arguments.source)
    seed = cast(int, arguments.seed)
    try:
        with ExitStack() as stack:
            built = build_suites(
                iter_jsonl(source, stack),
                record_limit=records,
                seed=seed,
                record_start=record_start,
            )
        metadata = write_outputs(
            built,
            output_directory=cast(Path, arguments.output_directory),
            source=source,
            source_date=cast(str, arguments.source_date),
            record_limit=records,
            seed=seed,
            record_start=record_start,
        )
    except (OSError, SuiteBuildError, urllib.error.URLError) as error:
        raise SystemExit(f"error: {error}") from error
    print_report(metadata)


if __name__ == "__main__":
    main()
