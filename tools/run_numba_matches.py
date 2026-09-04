"""Paired development matches using unmodified referee/sandbox, with stderr telemetry."""

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.sandbox import local
from tools.numba_core_validation import load_fens

DIAGNOSTIC = re.compile(r"depth=(\d+).*?nodes=(\d+) qnodes=(\d+).*?elapsed_ms=([\d.]+)")


def telemetry(log: str) -> dict[str, Any]:
    moves = [
        {
            "depth": int(depth),
            "nodes": int(nodes),
            "qnodes": int(qnodes),
            "elapsed_ms": float(elapsed),
        }
        for depth, nodes, qnodes, elapsed in DIAGNOSTIC.findall(log)
    ]
    return {"moves": moves, "engine_errors": log.count("engine_error=")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("development-fast-16", "development-64"), required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=5000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--ply-cap", type=int, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidate_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    opponent_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.opponent, text=True
    ).strip()
    if opponent_commit != "6686f7ca8412776247149512a28fe7ec18f034f9":
        raise SystemExit("Opponent is not exact killer/history champion.")
    gate = json.loads((root / "benchmarks/results/numba-search-v1-parity.json").read_text())
    if gate["mismatches"] or gate["move_differences"]:
        raise SystemExit("Fixed-depth parity gate has not passed.")
    fens = load_fens(root / "benchmarks/positions" / f"{args.suite}.fens")
    started = time.perf_counter()

    def game(index: int, fen: str, candidate_white: bool) -> dict[str, Any]:
        candidate = local(root)
        opponent = local(args.opponent)
        before = time.perf_counter()
        outcome = play_match(
            candidate if candidate_white else opponent,
            opponent if candidate_white else candidate,
            args.base_ms,
            args.increment_ms,
            args.ply_cap,
            fen,
        )
        winner = "white" if candidate_white else "black"
        result = (
            "win" if outcome.result == winner else ("draw" if outcome.result == "draw" else "loss")
        )
        return {
            "position": index,
            "candidate_white": candidate_white,
            "result": result,
            "termination": outcome.termination,
            "elapsed_seconds": time.perf_counter() - before,
            "candidate": telemetry(candidate.stderr_tail),
            "opponent": telemetry(opponent.stderr_tail),
            "pgn": outcome.pgn,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(game, index, fen, white)
            for index, fen in enumerate(fens)
            for white in (True, False)
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{args.label}: {len(rows)}/{len(futures)} {row['result']} {row['termination']}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["position"], not row["candidate_white"]))
    summary: dict[str, Any] = {
        result: sum(row["result"] == result for row in rows) for result in ("win", "draw", "loss")
    }
    summary["score"] = (summary["win"] + summary["draw"] / 2) / len(rows)
    summary["referee_failures"] = sum(row["termination"] in FAILED_TERMINATIONS for row in rows)
    for name in ("candidate", "opponent"):
        moves = [move for row in rows for move in row[name]["moves"]]
        summary[name] = {
            "moves": len(moves),
            "engine_errors": sum(row[name]["engine_errors"] for row in rows),
            "mean_depth": sum(move["depth"] for move in moves) / max(1, len(moves)),
            "nodes": sum(move["nodes"] for move in moves),
            "qnodes": sum(move["qnodes"] for move in moves),
            "elapsed_ms": sum(move["elapsed_ms"] for move in moves),
        }
    summary["elapsed_seconds"] = time.perf_counter() - started
    output = root / "benchmarks/results" / f"numba-search-v1-{args.label}.json"
    output.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "base_ms": args.base_ms,
                "increment_ms": args.increment_ms,
                "ply_cap": args.ply_cap,
                "workers": args.workers,
                "opponent": str(args.opponent),
                "opponent_commit": opponent_commit,
                "candidate_commit": candidate_commit,
                "summary": summary,
                "games": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["referee_failures"] or summary["candidate"]["engine_errors"]:
        raise SystemExit("Match failure: do not advance benchmark gates.")


if __name__ == "__main__":
    main()
