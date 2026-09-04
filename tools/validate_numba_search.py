"""Fixed-depth parity gate. Reads development data only, never promotion suites."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chess

import engine
import numba_engine
from tools.numba_core_validation import deterministic_random_positions, load_fens


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    boards = [
        chess.Board(fen)
        for name in ("smoke.fens", "development-64.fens")
        for fen in load_fens(root / "benchmarks" / "positions" / name)
    ]
    boards += deterministic_random_positions(500, 20260904)
    started = time.perf_counter()
    warmup = numba_engine.warmup()
    rows = []
    mismatches = []
    move_differences = []
    for index, board in enumerate(boards):
        for depth in (1, 2, 3) if index < 68 else (1, 2):
            before = board.fen(en_passant="fen")
            expected = engine.SearchEngine().search_fixed_depth(board, depth)
            actual = numba_engine.SearchEngine().search_fixed_depth(board, depth)
            row: dict[str, Any] = {
                "position": index,
                "fen": before,
                "depth": depth,
                "python": asdict(expected),
                "numba": asdict(actual),
            }
            row["python"]["move"] = expected.move.uci() if expected.move is not None else None
            row["numba"]["move"] = actual.move.uci() if actual.move is not None else None
            rows.append(row)
            if expected.score != actual.score or board.fen(en_passant="fen") != before:
                mismatches.append(row)
                print(
                    f"MISMATCH {index} depth {depth}: {expected.score} vs {actual.score}",
                    flush=True,
                )
            if expected.move != actual.move:
                move_differences.append(row)
        if index % 25 == 0:
            print(f"parity {index + 1}/{len(boards)}, mismatches={len(mismatches)}", flush=True)
    result = {
        "oracle_commit": "6686f7ca8412776247149512a28fe7ec18f034f9",
        "oracle_engine_sha256": hashlib.sha256((root / "engine.py").read_bytes()).hexdigest(),
        "positions": len(boards),
        "comparisons": len(rows),
        "mismatches": len(mismatches),
        "move_differences": len(move_differences),
        "warmup_seconds": warmup,
        "elapsed_seconds": time.perf_counter() - started,
        "results": rows,
    }
    output = root / "benchmarks" / "results" / "numba-search-v1-parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print({key: value for key, value in result.items() if key != "results"}, flush=True)
    if mismatches or move_differences:
        raise SystemExit("Investigate score/move differences before timed games.")


if __name__ == "__main__":
    main()
