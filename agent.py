"""Competition entrypoint for the parity-validated Numba killer/history speed port."""

import traceback

import chess

from engine import ordered_moves
from numba_engine import SearchEngine, warmup

WARMUP_SECONDS = warmup()
ENGINE = SearchEngine()
print(f"engine_init_warmup_seconds={WARMUP_SECONDS:.3f}", flush=True)


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
        f"nodes={result.nodes} qnodes={result.qnodes} cutoffs={result.beta_cutoffs} "
        f"killer_first={result.killer_first_searches} "
        f"history_moves={result.history_ordered_moves} elapsed_ms={result.elapsed_ms:.1f} "
        f"timeout={'yes' if result.timed_out else 'no'}",
        flush=True,
    )
    return move.uci()
