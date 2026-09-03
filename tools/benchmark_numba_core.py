"""Validate and benchmark the standalone Numba chess core against python-chess."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import chess
import numba
import numpy as np

from numba_core import (
    MAX_LEGAL_MOVES,
    UNDO_SIZE,
    generate_legal_moves_into,
    is_in_check,
    legal_moves,
    make_move_inplace,
    perft,
    state_tuple,
    unmake_move_inplace,
)
from tools.numba_core_validation import (
    deterministic_random_positions,
    differential_position,
    load_fens,
    python_perft,
    state_from_board,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "benchmarks" / "numba-core-v0-results.json"
RANDOM_SEED = 20_260_903

SPECIAL_CASES = {
    "white_kingside_and_queenside_castling": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "black_kingside_and_queenside_castling": "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "en_passant": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "promotions_and_underpromotions": "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "double_pawn_moves": chess.STARTING_FEN,
    "pinned_piece": "4k3/8/8/8/8/4r3/4R3/4K3 w - - 0 1",
    "discovered_check": "4k3/8/8/8/8/8/4B3/4R1K1 w - - 0 1",
    "double_check": "4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1",
    "checkmate": "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
    "stalemate": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
}

PERFT_CASES = (
    ("start", chess.STARTING_FEN, 4, 197_281),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        3,
        97_862,
    ),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2_812),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-positions", type=int, default=2_048)
    parser.add_argument("--movegen-repeats", type=int, default=25)
    parser.add_argument("--make-repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def compile_and_warm(state: np.ndarray) -> tuple[float, int]:
    output = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
    undo = np.empty(UNDO_SIZE, dtype=np.int16)
    started = perf_counter()
    count = int(generate_legal_moves_into(state, output))
    is_in_check(state)
    if count:
        move = int(output[0])
        make_move_inplace(state, move, undo)
        unmake_move_inplace(state, move, undo)
    perft(state, 2)
    return perf_counter() - started, count


def validate_corpus(boards: Sequence[chess.Board]) -> tuple[int, int, list[str]]:
    mismatch_count = 0
    transitions = 0
    examples: list[str] = []
    for board in boards:
        transitions += board.legal_moves.count()
        mismatches = differential_position(board)
        mismatch_count += len(mismatches)
        if mismatches and len(examples) < 10:
            examples.append(f"{board.fen(en_passant='fen')}: {mismatches}")
    return mismatch_count, transitions, examples


def benchmark_move_generation(boards: Sequence[chess.Board], repeats: int) -> dict[str, Any]:
    states = [state_from_board(board) for board in boards]
    output = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)

    internal_moves = 0
    started = perf_counter()
    for _ in range(repeats):
        for state in states:
            internal_moves += int(generate_legal_moves_into(state, output))
    internal_seconds = perf_counter() - started

    python_moves = 0
    started = perf_counter()
    for _ in range(repeats):
        for board in boards:
            python_moves += board.legal_moves.count()
    python_seconds = perf_counter() - started
    if internal_moves != python_moves:
        raise RuntimeError("move-generation benchmark counts diverged")

    position_calls = len(boards) * repeats
    return {
        "corpus_positions": len(boards),
        "repeats": repeats,
        "legal_moves_generated": internal_moves,
        "numba": {
            "elapsed_seconds": internal_seconds,
            "positions_per_second": position_calls / internal_seconds,
            "legal_moves_per_second": internal_moves / internal_seconds,
        },
        "python_chess": {
            "elapsed_seconds": python_seconds,
            "positions_per_second": position_calls / python_seconds,
            "legal_moves_per_second": python_moves / python_seconds,
        },
        "speedup": python_seconds / internal_seconds,
    }


def benchmark_make_unmake(boards: Sequence[chess.Board], repeats: int) -> dict[str, Any]:
    states = [state_from_board(board) for board in boards]
    internal_moves = [legal_moves(state) for state in states]
    python_moves = [list(board.legal_moves) for board in boards]
    round_trips = repeats * sum(len(moves) for moves in internal_moves)
    undo = np.empty(UNDO_SIZE, dtype=np.int16)
    originals = [state_tuple(state) for state in states]

    started = perf_counter()
    for _ in range(repeats):
        for state, encoded_moves in zip(states, internal_moves, strict=True):
            for encoded in encoded_moves:
                move = int(encoded)
                make_move_inplace(state, move, undo)
                unmake_move_inplace(state, move, undo)
    internal_seconds = perf_counter() - started
    if any(
        state_tuple(state) != original for state, original in zip(states, originals, strict=True)
    ):
        raise RuntimeError("internal state was not restored during benchmark")

    started = perf_counter()
    for _ in range(repeats):
        for board, oracle_moves in zip(boards, python_moves, strict=True):
            for oracle_move in oracle_moves:
                board.push(oracle_move)
                board.pop()
    python_seconds = perf_counter() - started

    return {
        "corpus_positions": len(boards),
        "repeats": repeats,
        "make_unmake_round_trips": round_trips,
        "numba": {
            "elapsed_seconds": internal_seconds,
            "round_trips_per_second": round_trips / internal_seconds,
        },
        "python_chess": {
            "elapsed_seconds": python_seconds,
            "round_trips_per_second": round_trips / python_seconds,
        },
        "speedup": python_seconds / internal_seconds,
    }


def corpus_sha256(boards: Sequence[chess.Board]) -> str:
    content = "".join(f"{board.fen(en_passant='fen')}\n" for board in boards).encode()
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    arguments = build_parser().parse_args()
    smoke_fens = load_fens(REPOSITORY_ROOT / "benchmarks" / "positions" / "smoke.fens")
    development_fens = load_fens(
        REPOSITORY_ROOT / "benchmarks" / "positions" / "development-64.fens"
    )
    random_boards = deterministic_random_positions(arguments.random_positions, RANDOM_SEED)
    corpus = [chess.Board(fen) for fen in smoke_fens + development_fens] + random_boards

    warm_state = state_from_board(chess.Board())
    warm_state_before = state_tuple(warm_state)
    compile_seconds, warm_moves = compile_and_warm(warm_state)
    if state_tuple(warm_state) != warm_state_before or warm_moves != 20:
        raise RuntimeError("warmup changed state or produced an incorrect start move count")

    mismatch_count, transitions, mismatch_examples = validate_corpus(corpus)
    special_results: dict[str, object] = {}
    for name, fen in SPECIAL_CASES.items():
        mismatches = differential_position(chess.Board(fen))
        mismatch_count += len(mismatches)
        special_results[name] = {"passed": not mismatches, "mismatches": mismatches}

    perft_results: dict[str, object] = {}
    for name, fen, depth, expected in PERFT_CASES:
        board = chess.Board(fen)
        state = state_from_board(board)
        original = state_tuple(state)
        python_nodes = python_perft(board, depth)
        internal_nodes = int(perft(state, depth))
        passed = (
            python_nodes == expected
            and internal_nodes == python_nodes
            and state_tuple(state) == original
        )
        if not passed:
            mismatch_count += 1
        perft_results[name] = {
            "depth": depth,
            "expected_nodes": expected,
            "python_chess_nodes": python_nodes,
            "numba_nodes": internal_nodes,
            "passed": passed,
        }

    result = {
        "schema": "aichessathon.numba-core-v0-results",
        "schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numba": numba.__version__,
            "numpy": np.__version__,
            "python_chess": chess.__version__,
        },
        "corpus": {
            "smoke_positions": len(smoke_fens),
            "development_positions": len(development_fens),
            "random_positions": len(random_boards),
            "random_seed": RANDOM_SEED,
            "total_positions": len(corpus),
            "legal_move_transitions_checked": transitions,
            "sha256": corpus_sha256(corpus),
        },
        "validation": {
            "mismatch_count": mismatch_count,
            "mismatch_examples": mismatch_examples,
            "special_cases": special_results,
            "perft": perft_results,
        },
        "compile_warmup_seconds": compile_seconds,
        "throughput": {
            "move_generation": benchmark_move_generation(corpus, arguments.movegen_repeats),
            "make_unmake": benchmark_make_unmake(corpus, arguments.make_repeats),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if mismatch_count:
        raise SystemExit(f"validation failed with {mismatch_count} mismatches")


if __name__ == "__main__":
    main()
