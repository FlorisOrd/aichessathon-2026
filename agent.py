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
    depth_nodes = ",".join(
        f"{item.depth}:{item.nodes}" for item in result.depth_diagnostics
    ) or "none"
    depth_cutoffs = ",".join(
        f"{item.depth}:{item.cutoffs}" for item in result.depth_diagnostics
    ) or "none"
    reusable_depths = max(0, len(result.depth_diagnostics) - 1)
    root_first = result.root_first_move.uci() if result.root_first_move is not None else "none"
    print(
        f"move={move.uci()} depth={result.completed_depth} score={result.score} "
        f"nodes={result.nodes} cutoffs={result.cutoffs} elapsed_ms={result.elapsed_ms:.1f} "
        f"root_first={root_first} "
        f"root_pv_reuse={result.root_pv_reuses}/{reusable_depths} "
        f"depth_nodes={depth_nodes} depth_cutoffs={depth_cutoffs} "
        f"timeout={'yes' if result.timed_out else 'no'}",
        flush=True,
    )
    return move.uci()
