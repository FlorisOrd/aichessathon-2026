"""Timed root diagnostics against the unchanged Python champion, development data only."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chess

import engine
import numba_engine
from tools.numba_core_validation import load_fens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("development-fast-16", "development-64"), required=True)
    parser.add_argument("--time-left-ms", type=int, default=5000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    gate = json.loads((root / "benchmarks/results/numba-search-v1-parity.json").read_text())
    if gate["mismatches"] or gate["move_differences"]:
        raise SystemExit("Fixed-depth parity gate has not passed.")
    warmup = numba_engine.warmup()
    rows = []
    for index, fen in enumerate(load_fens(root / "benchmarks/positions" / f"{args.suite}.fens")):
        row: dict[str, Any] = {"position": index, "fen": fen}
        implementations = [("python", engine.SearchEngine), ("numba", numba_engine.SearchEngine)]
        for name, implementation in implementations[:: (-1 if index % 2 else 1)]:
            board = chess.Board(fen)
            started = time.perf_counter()
            result = implementation().search(board, args.time_left_ms)
            elapsed = time.perf_counter() - started
            data = asdict(result)
            data["move"] = result.move.uci() if result.move else None
            data["wall_seconds"] = elapsed
            data["legal"] = result.move in board.legal_moves
            data["restored"] = board.fen() == chess.Board(fen).fen()
            row[name] = data
        rows.append(row)
    summary = {}
    for name in ("python", "numba"):
        elapsed = sum(row[name]["wall_seconds"] for row in rows)
        nodes = sum(row[name]["nodes"] for row in rows)
        qnodes = sum(row[name]["qnodes"] for row in rows)
        summary[name] = {
            "nodes": nodes,
            "qnodes": qnodes,
            "nodes_per_second": nodes / elapsed,
            "qnodes_per_second": qnodes / elapsed,
            "total_nodes_per_second": (nodes + qnodes) / elapsed,
            "mean_completed_depth": sum(row[name]["completed_depth"] for row in rows) / len(rows),
            "mean_elapsed_ms": elapsed * 1000 / len(rows),
            "failures": sum(not row[name]["legal"] or not row[name]["restored"] for row in rows),
        }
    output = root / "benchmarks/results" / f"numba-search-v1-{args.suite}-diagnostics.json"
    output.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "time_left_ms": args.time_left_ms,
                "warmup_seconds": warmup,
                "summary": summary,
                "positions": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if any(data["failures"] for data in summary.values()):
        raise SystemExit("diagnostic failure")


if __name__ == "__main__":
    main()
