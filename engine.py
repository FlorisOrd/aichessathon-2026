"""Simple deterministic reference engine built on python-chess.

Evaluation scores are integer centipawns from the side-to-move perspective. Search uses
iterative-deepening negamax with alpha-beta pruning. FEN contains the halfmove clock but not the
game's earlier positions, so fifty-move claims can be represented while pre-root repetition
history cannot; this engine deliberately does not invent missing repetition state.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import chess
import chess.polyglot

MATE_SCORE = 1_000_000
MATE_THRESHOLD = MATE_SCORE - 10_000
INFINITY = MATE_SCORE + 1
MAX_PHASE = 24
MIN_SEARCH_TIME_MS = 30
MAX_SOFT_TIME_MS = 1_500
TIME_DIVISOR = 40
MAX_RESERVE_MS = 2_000
MIN_THREEFOLD_CLAIM_PLIES = 7
DEFAULT_TT_CAPACITY = 1 << 16
_TT_INDEX_MIX = 0x9E3779B97F4A7C15

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
    tt_probes: int
    tt_hits: int
    tt_usable_hits: int
    tt_cutoffs: int
    tt_stores: int

    @property
    def tt_hit_rate(self) -> float:
        """Return raw TT hits per probe; zero when no probes were made."""
        return self.tt_hits / self.tt_probes if self.tt_probes else 0.0


class BoundType(IntEnum):
    """How a stored fail-soft score relates to its original alpha-beta window."""

    EXACT = 0
    LOWER = 1
    UPPER = 2


@dataclass(frozen=True, slots=True)
class TTEntry:
    """One direct-mapped transposition-table entry."""

    position_key: int
    halfmove_clock: int
    depth: int
    score: int
    bound: BoundType
    best_move: chess.Move | None
    generation: int


@dataclass
class TTStats:
    probes: int = 0
    hits: int = 0
    usable_hits: int = 0
    cutoffs: int = 0
    stores: int = 0


def position_key(board: chess.Board) -> int:
    """Return python-chess's documented Polyglot Zobrist key for ``board``."""
    return chess.polyglot.zobrist_hash(board)


def score_to_tt(score: int, ply_from_root: int) -> int:
    """Remove root-relative mate distance before storing a score."""
    if score >= MATE_THRESHOLD:
        return score + ply_from_root
    if score <= -MATE_THRESHOLD:
        return score - ply_from_root
    return score


def score_from_tt(score: int, ply_from_root: int) -> int:
    """Restore root-relative mate distance after loading a score."""
    if score >= MATE_THRESHOLD:
        return score - ply_from_root
    if score <= -MATE_THRESHOLD:
        return score + ply_from_root
    return score


class TranspositionTable:
    """Fixed-capacity direct-mapped table with depth-preferred replacement."""

    def __init__(self, capacity: int = DEFAULT_TT_CAPACITY) -> None:
        if capacity < 1 or capacity & (capacity - 1):
            raise ValueError("TT capacity must be a positive power of two")
        self.capacity = capacity
        self._mask = capacity - 1
        self._entries: list[TTEntry | None] = [None] * capacity
        self.generation = 0

    def new_generation(self) -> None:
        """Age collision entries without clearing the fixed-size allocation."""
        self.generation += 1

    def probe(self, board: chess.Board) -> TTEntry | None:
        """Return the matching entry, verifying key and halfmove clock after indexing."""
        key = position_key(board)
        entry = self._entries[self._index(key, board.halfmove_clock)]
        if (
            entry is not None
            and entry.position_key == key
            and entry.halfmove_clock == board.halfmove_clock
        ):
            return entry
        return None

    def store(
        self,
        board: chess.Board,
        depth: int,
        score: int,
        bound: BoundType,
        best_move: chess.Move | None,
        ply_from_root: int,
    ) -> bool:
        """Store an entry when the direct-mapped replacement policy accepts it."""
        key = position_key(board)
        index = self._index(key, board.halfmove_clock)
        current = self._entries[index]
        same_position = (
            current is not None
            and current.position_key == key
            and current.halfmove_clock == board.halfmove_clock
        )
        should_replace = (
            current is None
            or (same_position and depth >= current.depth)
            or (
                not same_position
                and (current.generation != self.generation or depth >= current.depth)
            )
        )
        if not should_replace:
            return False
        self._entries[index] = TTEntry(
            position_key=key,
            halfmove_clock=board.halfmove_clock,
            depth=depth,
            score=score_to_tt(score, ply_from_root),
            bound=bound,
            best_move=best_move,
            generation=self.generation,
        )
        return True

    def clear(self) -> None:
        """Clear entries while retaining the same bounded allocation."""
        self._entries[:] = [None] * self.capacity

    @property
    def entry_count(self) -> int:
        return sum(entry is not None for entry in self._entries)

    @property
    def approximate_max_bytes(self) -> int:
        """Conservatively estimate a fully populated table's Python heap footprint."""
        sample = TTEntry(0, 0, 0, 0, BoundType.EXACT, chess.Move.null(), 0)
        referenced_values = 5 * sys.getsizeof(0) + sys.getsizeof(chess.Move.null())
        return sys.getsizeof(self._entries) + self.capacity * (
            sys.getsizeof(sample) + referenced_values
        )

    def _index(self, key: int, halfmove_clock: int) -> int:
        mixed = key ^ ((halfmove_clock + 1) * _TT_INDEX_MIX)
        mixed ^= mixed >> 32
        return mixed & self._mask


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

    # Match the referee's remaining claim_draw=True checks exactly. The cheap guards avoid move
    # generation/replay when a claim cannot yet be possible. A root FEN supplies the halfmove
    # clock but no earlier move stack: starting from that root, the earliest threefold claim is
    # at ply 7, by announcing a move that will create the third occurrence. If a caller supplies
    # a Board with real history, len(move_stack) preserves and exposes that available history.
    if board.halfmove_clock >= 99 and board.can_claim_fifty_moves():
        return 0
    if (
        len(board.move_stack) >= MIN_THREEFOLD_CLAIM_PLIES
        and board.can_claim_threefold_repetition()
    ):
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


def ordered_moves(
    board: chess.Board, tt_move: chess.Move | None = None
) -> list[chess.Move]:
    """Return deterministic tactical ordering, with a legal TT move first."""
    moves = sorted(
        board.legal_moves,
        key=lambda move: (-move_order_score(board, move), move.uci()),
    )
    if tt_move is not None and tt_move in moves:
        moves.remove(tt_move)
        moves.insert(0, tt_move)
    return moves


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
    """Iterative-deepening alpha-beta search with a bounded transposition table."""

    def __init__(
        self,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        tt_capacity: int = DEFAULT_TT_CAPACITY,
    ) -> None:
        self._clock_ns = clock_ns
        self._deadline_ns: int | None = None
        self._nodes = 0
        self.table = TranspositionTable(tt_capacity)
        self._tt_stats = TTStats()

    def search(self, board: chess.Board, time_left_ms: int) -> SearchResult:
        """Search until the soft/hard budget and return the last completed iteration."""
        self._deadline_ns = None
        self._start_tt_generation()
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
                **self._tt_result_fields(),
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
                **self._tt_result_fields(),
            )

        soft_deadline_ns = started_ns + budget.soft_ms * 1_000_000
        self._deadline_ns = started_ns + budget.hard_ms * 1_000_000
        depth = 1
        try:
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
        finally:
            self._deadline_ns = None

        return SearchResult(
            move=best_move,
            score=best_score,
            completed_depth=completed_depth,
            nodes=self._nodes,
            elapsed_ms=self._elapsed_ms(started_ns),
            timed_out=timed_out,
            **self._tt_result_fields(),
        )

    def search_fixed_depth(self, board: chess.Board, depth: int) -> SearchResult:
        """Search exactly one depth without a deadline; useful for reference tests."""
        if depth < 1:
            raise ValueError("depth must be at least 1")
        started_ns = self._clock_ns()
        self._nodes = 0
        self._deadline_ns = None
        self._start_tt_generation()
        move, score = self._search_depth(board, depth)
        return SearchResult(
            move=move,
            score=score,
            completed_depth=depth,
            nodes=self._nodes,
            elapsed_ms=self._elapsed_ms(started_ns),
            timed_out=False,
            **self._tt_result_fields(),
        )

    def _search_depth(self, board: chess.Board, depth: int) -> tuple[chess.Move | None, int]:
        self._visit_node()
        terminal = terminal_score(board, 0)
        if terminal is not None:
            return None, terminal

        original_alpha = -INFINITY
        original_beta = INFINITY
        entry, alpha, beta, cached_score = self._probe_tt(
            board,
            depth,
            original_alpha,
            original_beta,
            0,
        )
        tt_move = entry.best_move if entry is not None else None
        if cached_score is not None and tt_move is not None and tt_move in board.legal_moves:
            return tt_move, cached_score

        best_move: chess.Move | None = None
        best_score = -INFINITY
        for move in ordered_moves(board, tt_move):
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        self._store_tt(
            board,
            depth,
            best_score,
            self._bound_type(best_score, original_alpha, original_beta),
            best_move,
            0,
        )
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

        original_alpha = alpha
        original_beta = beta
        entry, alpha, beta, cached_score = self._probe_tt(
            board,
            depth,
            alpha,
            beta,
            ply_from_root,
        )
        if cached_score is not None:
            return cached_score

        tt_move = entry.best_move if entry is not None else None
        best_score = -INFINITY
        best_move: chess.Move | None = None
        for move in ordered_moves(board, tt_move):
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
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        self._store_tt(
            board,
            depth,
            best_score,
            self._bound_type(best_score, original_alpha, original_beta),
            best_move,
            ply_from_root,
        )
        return best_score

    def _start_tt_generation(self) -> None:
        self.table.new_generation()
        self._tt_stats = TTStats()

    def _probe_tt(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply_from_root: int,
    ) -> tuple[TTEntry | None, int, int, int | None]:
        """Probe and apply a draw-safe entry to the current alpha-beta window."""
        self._tt_stats.probes += 1
        entry = self.table.probe(board)
        if entry is None:
            return None, alpha, beta, None
        self._tt_stats.hits += 1

        # A zero halfmove clock follows a capture or pawn move. Both are irreversible, so no
        # position before that reset can recur and contribute to a future repetition claim.
        # At every history-sensitive node we use only the legal move hint, never the value.
        if board.halfmove_clock != 0 or entry.depth < depth:
            return entry, alpha, beta, None

        self._tt_stats.usable_hits += 1
        score = score_from_tt(entry.score, ply_from_root)
        if entry.bound == BoundType.EXACT:
            self._tt_stats.cutoffs += 1
            return entry, alpha, beta, score
        if entry.bound == BoundType.LOWER:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)
        if alpha >= beta:
            self._tt_stats.cutoffs += 1
            return entry, alpha, beta, score
        return entry, alpha, beta, None

    def _store_tt(
        self,
        board: chess.Board,
        depth: int,
        score: int,
        bound: BoundType,
        best_move: chess.Move | None,
        ply_from_root: int,
    ) -> None:
        if self.table.store(board, depth, score, bound, best_move, ply_from_root):
            self._tt_stats.stores += 1

    @staticmethod
    def _bound_type(score: int, alpha: int, beta: int) -> BoundType:
        if score <= alpha:
            return BoundType.UPPER
        if score >= beta:
            return BoundType.LOWER
        return BoundType.EXACT

    def _tt_result_fields(self) -> dict[str, int]:
        return {
            "tt_probes": self._tt_stats.probes,
            "tt_hits": self._tt_stats.hits,
            "tt_usable_hits": self._tt_stats.usable_hits,
            "tt_cutoffs": self._tt_stats.cutoffs,
            "tt_stores": self._tt_stats.stores,
        }

    def _visit_node(self) -> None:
        self._nodes += 1
        if self._deadline_ns is not None and self._clock_ns() >= self._deadline_ns:
            raise SearchTimeout

    def _elapsed_ms(self, started_ns: int) -> float:
        return max(0.0, (self._clock_ns() - started_ns) / 1_000_000)
