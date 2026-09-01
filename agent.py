"""Competition entrypoint for the deterministic reference engine."""

import traceback

import chess

from engine import SearchEngine, ordered_moves

ENGINE = SearchEngine()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move for the side to move in ``fen``."""
    board = chess.Board(fen)
    legal_moves = ordered_moves(board)
    if not legal_moves:
        raise ValueError("get_move called for a position with no legal moves")
    fallback = legal_moves[0]

    try:
        result = ENGINE.search(board, time_left_ms)
    except Exception as error:
        print(f"engine_error={type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        return fallback.uci()

    move = result.move
    if move is None or move not in board.legal_moves:
        print("engine_error=search returned no legal move; using fallback", flush=True)
        move = fallback
    print(
        f"move={move.uci()} depth={result.completed_depth} score={result.score} "
        f"nodes={result.nodes} elapsed_ms={result.elapsed_ms:.1f} "
        f"timeout={'yes' if result.timed_out else 'no'} "
        f"tt_probes={result.tt_probes} tt_hits={result.tt_hits} "
        f"tt_usable={result.tt_usable_hits} tt_cutoffs={result.tt_cutoffs} "
        f"tt_stores={result.tt_stores} tt_hit_rate={result.tt_hit_rate:.1%}",
        flush=True,
    )
    return move.uci()
