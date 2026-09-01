"""Simple deterministic reference engine built on python-chess.

Evaluation scores are integer centipawns from the side-to-move perspective. Search uses
iterative-deepening negamax with alpha-beta pruning. FEN contains the halfmove clock but not the
game's earlier positions, so fifty-move claims can be represented while pre-root repetition
history cannot; this engine deliberately does not invent missing repetition state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import chess

MATE_SCORE = 1_000_000
MATE_THRESHOLD = MATE_SCORE - 10_000
INFINITY = MATE_SCORE + 1
MAX_PHASE = 24
MIN_SEARCH_TIME_MS = 30
MAX_SOFT_TIME_MS = 1_500
TIME_DIVISOR = 40
MAX_RESERVE_MS = 2_000

PIECE_TYPES: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
MIDDLEGAME_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
ENDGAME_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 120,
    chess.KNIGHT: 300,
    chess.BISHOP: 320,
    chess.ROOK: 510,
    chess.QUEEN: 900,
    chess.KING: 0,
}
PHASE_WEIGHTS: dict[chess.PieceType, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}


class SearchTimeout(Exception):
    """Abort the current iteration when its hard deadline is reached."""


@dataclass(frozen=True)
class TimeBudget:
    soft_ms: int
    hard_ms: int


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move | None
    score: int
    completed_depth: int
    nodes: int
    elapsed_ms: float
    timed_out: bool


def _centrality(square: chess.Square) -> int:
    file_distance = abs(2 * chess.square_file(square) - 7)
    rank_distance = abs(2 * chess.square_rank(square) - 7)
    return 14 - file_distance - rank_distance


def _piece_square_bonus(piece_type: chess.PieceType, square: chess.Square, endgame: bool) -> int:
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    center = _centrality(square)
    if piece_type == chess.PAWN:
        file_center = 7 - abs(2 * file - 7)
        return rank * (12 if endgame else 8) + (file_center if not endgame else 0)
    if piece_type == chess.KNIGHT:
        return (center - 6) * (6 if not endgame else 4)
    if piece_type == chess.BISHOP:
        return (center - 6) * 4
    if piece_type == chess.ROOK:
        return rank * 2 + (10 if rank == 6 else 0)
    if piece_type == chess.QUEEN:
        return (center - 6) * (1 if endgame else 2)
    if endgame:
        return (center - 6) * 5
    castled_bonus = 20 if rank == 0 and file in {1, 2, 6} else 0
    return castled_bonus - center * 3 - rank * 8


MIDDLEGAME_TABLES: dict[chess.PieceType, tuple[int, ...]] = {
    piece_type: tuple(
        _piece_square_bonus(piece_type, square, endgame=False) for square in chess.SQUARES
    )
    for piece_type in PIECE_TYPES
}
ENDGAME_TABLES: dict[chess.PieceType, tuple[int, ...]] = {
    piece_type: tuple(
        _piece_square_bonus(piece_type, square, endgame=True) for square in chess.SQUARES
    )
    for piece_type in PIECE_TYPES
}


def evaluate(board: chess.Board) -> int:
    """Return a tapered static score in centipawns for the side to move."""
    middlegame = 0
    endgame = 0
    phase = 0
    for piece_type in PIECE_TYPES:
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            squares = board.pieces(piece_type, color)
            phase += PHASE_WEIGHTS[piece_type] * len(squares)
            for square in squares:
                relative_square = square if color == chess.WHITE else chess.square_mirror(square)
                middlegame += sign * (
                    MIDDLEGAME_VALUES[piece_type] + MIDDLEGAME_TABLES[piece_type][relative_square]
                )
                endgame += sign * (
                    ENDGAME_VALUES[piece_type] + ENDGAME_TABLES[piece_type][relative_square]
                )

    phase = min(phase, MAX_PHASE)
    white_score = (middlegame * phase + endgame * (MAX_PHASE - phase)) // MAX_PHASE
    return white_score if board.turn == chess.WHITE else -white_score


def terminal_score(board: chess.Board, ply_from_root: int) -> int | None:
    """Return a side-to-move terminal score, or ``None`` for a live position."""
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        if outcome.winner is None:
            return 0
        return -MATE_SCORE + ply_from_root

    # The halfmove clock is present in FEN, unlike repetition history. The referee claims this
    # draw automatically, so a claimable fifty-move position is terminal for search purposes.
    if board.halfmove_clock >= 100:
        return 0
    return None


def move_order_score(board: chess.Board, move: chess.Move) -> int:
    """Rank promotions first, then MVV-LVA captures, then quiet moves."""
    score = 0
    if move.promotion is not None:
        score += 20_000 + MIDDLEGAME_VALUES[move.promotion]
    if board.is_capture(move):
        victim_type = (
            chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
        )
        attacker_type = board.piece_type_at(move.from_square)
        victim = MIDDLEGAME_VALUES[victim_type] if victim_type is not None else 0
        attacker = MIDDLEGAME_VALUES[attacker_type] if attacker_type is not None else 0
        score += 10_000 + 10 * victim - attacker
    return score


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    """Return legal moves in a deterministic, replaceable first-pass order."""
    return sorted(board.legal_moves, key=lambda move: (-move_order_score(board, move), move.uci()))


def allocate_time(time_left_ms: int) -> TimeBudget:
    """Choose conservative deadlines without assuming any increment.

    About 1/40 of the remaining clock is the normal target, capped at 1.5 seconds. The hard
    deadline is roughly 1.5 times that target, while a separate reserve is never knowingly
    touched. Very low time skips search and immediately uses the preselected legal fallback.
    """
    if time_left_ms <= MIN_SEARCH_TIME_MS:
        return TimeBudget(soft_ms=0, hard_ms=0)
    reserve_ms = max(15, min(MAX_RESERVE_MS, time_left_ms // 8))
    available_ms = max(0, time_left_ms - reserve_ms)
    soft_ms = min(MAX_SOFT_TIME_MS, max(2, time_left_ms // TIME_DIVISOR), available_ms)
    hard_ms = min(available_ms, max(soft_ms, soft_ms * 3 // 2 + 5))
    return TimeBudget(soft_ms=soft_ms, hard_ms=hard_ms)


class SearchEngine:
    """Stateful only for one search; the reference engine intentionally has no cache."""

    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock_ns = clock_ns
        self._deadline_ns: int | None = None
        self._nodes = 0

    def search(self, board: chess.Board, time_left_ms: int) -> SearchResult:
        """Search until the soft/hard budget and return the last completed iteration."""
        started_ns = self._clock_ns()
        moves = ordered_moves(board)
        root_terminal = terminal_score(board, 0)
        if not moves:
            return SearchResult(
                move=None,
                score=root_terminal if root_terminal is not None else 0,
                completed_depth=0,
                nodes=0,
                elapsed_ms=self._elapsed_ms(started_ns),
                timed_out=False,
            )

        best_move = moves[0]
        best_score = evaluate(board)
        completed_depth = 0
        timed_out = False
        self._nodes = 0
        budget = allocate_time(time_left_ms)
        if budget.hard_ms == 0:
            return SearchResult(
                move=best_move,
                score=best_score,
                completed_depth=0,
                nodes=0,
                elapsed_ms=self._elapsed_ms(started_ns),
                timed_out=False,
            )

        soft_deadline_ns = started_ns + budget.soft_ms * 1_000_000
        self._deadline_ns = started_ns + budget.hard_ms * 1_000_000
        depth = 1
        while True:
            try:
                iteration_move, iteration_score = self._search_depth(board, depth)
            except SearchTimeout:
                timed_out = True
                break
            if iteration_move is not None:
                best_move = iteration_move
                best_score = iteration_score
                completed_depth = depth
            if abs(best_score) >= MATE_THRESHOLD or self._clock_ns() >= soft_deadline_ns:
                break
            depth += 1

        self._deadline_ns = None
        return SearchResult(
            move=best_move,
            score=best_score,
            completed_depth=completed_depth,
            nodes=self._nodes,
            elapsed_ms=self._elapsed_ms(started_ns),
            timed_out=timed_out,
        )

    def search_fixed_depth(self, board: chess.Board, depth: int) -> SearchResult:
        """Search exactly one depth without a deadline; useful for reference tests."""
        if depth < 1:
            raise ValueError("depth must be at least 1")
        started_ns = self._clock_ns()
        self._nodes = 0
        self._deadline_ns = None
        move, score = self._search_depth(board, depth)
        return SearchResult(
            move=move,
            score=score,
            completed_depth=depth,
            nodes=self._nodes,
            elapsed_ms=self._elapsed_ms(started_ns),
            timed_out=False,
        )

    def _search_depth(self, board: chess.Board, depth: int) -> tuple[chess.Move | None, int]:
        self._visit_node()
        terminal = terminal_score(board, 0)
        if terminal is not None:
            return None, terminal

        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha = -INFINITY
        for move in ordered_moves(board):
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -INFINITY, -alpha, 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_move, best_score

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply_from_root: int,
    ) -> int:
        self._visit_node()
        terminal = terminal_score(board, ply_from_root)
        if terminal is not None:
            return terminal
        if depth == 0:
            return evaluate(board)

        best_score = -INFINITY
        for move in ordered_moves(board):
            board.push(move)
            try:
                score = -self._negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply_from_root + 1,
                )
            finally:
                board.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score

    def _visit_node(self) -> None:
        self._nodes += 1
        if self._deadline_ns is not None and self._clock_ns() >= self._deadline_ns:
            raise SearchTimeout

    def _elapsed_ms(self, started_ns: int) -> float:
        return max(0.0, (self._clock_ns() - started_ns) / 1_000_000)
