"""JIT speed-port of exact killer/history engine 6686f7c; no new chess heuristics."""

from __future__ import annotations

import time

import chess
import numpy as np
from numba import objmode

from engine import (
    ENDGAME_TABLES,
    ENDGAME_VALUES,
    HISTORY_MAX,
    INFINITY,
    MATE_SCORE,
    MATE_THRESHOLD,
    MIDDLEGAME_TABLES,
    MIDDLEGAME_VALUES,
    PHASE_WEIGHTS,
    SearchResult,
    allocate_time,
)
from numba_core import (
    CASTLING_RIGHTS,
    EP_SQUARE,
    FULLMOVE_NUMBER,
    HALFMOVE_CLOCK,
    MAX_LEGAL_MOVES,
    MOVE_EN_PASSANT,
    SIDE_TO_MOVE,
    UNDO_SIZE,
    _numba_njit,
    empty_state,
    generate_legal_moves_into,
    is_in_check,
    make_move_inplace,
    move_to_uci,
    unmake_move_inplace,
)

MAX_PLY = 512
LIVE = INFINITY + 1
MG = np.zeros((7, 64), dtype=np.int64)
EG = np.zeros((7, 64), dtype=np.int64)
VALUES = np.zeros(7, dtype=np.int64)
PHASE = np.zeros(7, dtype=np.int64)
for _piece in range(1, 7):
    MG[_piece] = np.asarray(MIDDLEGAME_TABLES[_piece]) + MIDDLEGAME_VALUES[_piece]
    EG[_piece] = np.asarray(ENDGAME_TABLES[_piece]) + ENDGAME_VALUES[_piece]
    VALUES[_piece] = MIDDLEGAME_VALUES[_piece]
    PHASE[_piece] = PHASE_WEIGHTS[_piece]


@_numba_njit
def evaluate(state: np.ndarray) -> int:
    mg = 0
    eg = 0
    phase = 0
    for square in range(64):
        piece = int(state[square])
        if piece:
            kind = abs(piece)
            sign = 1 if piece > 0 else -1
            relative = square if sign == 1 else square ^ 56
            mg += sign * MG[kind, relative]
            eg += sign * EG[kind, relative]
            phase += PHASE[kind]
    phase = min(24, phase)
    # Python floor division, including negative totals, is part of the oracle semantics.
    return ((mg * phase + eg * (24 - phase)) // 24) * int(state[SIDE_TO_MOVE])


@_numba_njit
def capture(state: np.ndarray, move: int) -> bool:
    return bool(state[(move >> 6) & 63] != 0 or move & MOVE_EN_PASSANT)


@_numba_njit
def tactical_score(state: np.ndarray, move: int) -> int:
    promotion = (move >> 12) & 7
    score = 20_000 + VALUES[promotion] if promotion else 0
    if capture(state, move):
        victim = 1 if move & MOVE_EN_PASSANT else abs(int(state[(move >> 6) & 63]))
        score += 10_000 + 10 * VALUES[victim] - VALUES[abs(int(state[move & 63]))]
    return score


@_numba_njit
def order_key(
    state: np.ndarray, move: int, ply: int, killers: np.ndarray, history: np.ndarray, qmode: bool
) -> tuple[int, int, int, int]:
    source = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    suffix = 0
    if promotion == 3:
        suffix = 1
    elif promotion == 2:
        suffix = 2
    elif promotion == 5:
        suffix = 3
    elif promotion == 4:
        suffix = 4
    uci = ((source % 8 * 8 + source // 8) * 64 + target % 8 * 8 + target // 8) * 5 + suffix
    tactical = tactical_score(state, move)
    if tactical > 0:
        return (0, -tactical, 0, uci)
    if not qmode:
        if killers[ply, 0] == move:
            return (1, 0, 0, uci)
        if killers[ply, 1] == move:
            return (1, 1, 0, uci)
        color = 0 if state[SIDE_TO_MOVE] == 1 else 1
        return (2, 0, -int(history[color, source, target]), uci)
    return (2, 0, 0, uci)


@_numba_njit
def order_moves(
    state: np.ndarray,
    moves: np.ndarray,
    count: int,
    ply: int,
    killers: np.ndarray,
    history: np.ndarray,
    qmode: bool,
) -> None:
    for index in range(1, count):
        move = int(moves[index])
        key = order_key(state, move, ply, killers, history, qmode)
        cursor = index - 1
        while (
            cursor >= 0 and order_key(state, int(moves[cursor]), ply, killers, history, qmode) > key
        ):
            moves[cursor + 1] = moves[cursor]
            cursor -= 1
        moves[cursor + 1] = move


@_numba_njit
def insufficient_material(state: np.ndarray) -> bool:
    knights = 0
    bishops = 0
    bishop_colors = 0
    for square in range(64):
        kind = abs(int(state[square]))
        if kind == 1 or kind == 4 or kind == 5:
            return False
        if kind == 2:
            knights += 1
        if kind == 3:
            bishops += 1
            bishop_colors |= 1 << ((square // 8 + square % 8) % 2)
    if knights:
        return knights == 1 and bishops == 0
    return bishop_colors != 3


@_numba_njit
def write_key(state: np.ndarray, moves: np.ndarray, count: int, key: np.ndarray) -> None:
    for index in range(66):
        key[index] = state[index]
    key[EP_SQUARE] = -1
    if state[EP_SQUARE] >= 0:
        for index in range(count):
            if moves[index] & MOVE_EN_PASSANT:
                key[EP_SQUARE] = state[EP_SQUARE]
                break


@_numba_njit
def same_key(first: np.ndarray, second: np.ndarray) -> bool:
    for index in range(67):  # noqa: SIM110 - Numba does not support generator-based all().
        if first[index] != second[index]:
            return False
    return True


@_numba_njit
def terminal(
    state: np.ndarray, moves: np.ndarray, count: int, ply: int, keys: np.ndarray, history_index: int
) -> int:
    write_key(state, moves, count, keys[history_index])
    if count == 0:
        return -MATE_SCORE + ply if is_in_check(state) else 0
    if insufficient_material(state):
        return 0
    if state[HALFMOVE_CLOCK] >= 100:
        return 0
    undo = np.empty(UNDO_SIZE, dtype=np.int16)
    child_moves = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
    if state[HALFMOVE_CLOCK] == 99:
        for index in range(count):
            move = int(moves[index])
            if abs(int(state[move & 63])) != 1 and not capture(state, move):
                make_move_inplace(state, move, undo)
                child_count = generate_legal_moves_into(state, child_moves)
                unmake_move_inplace(state, move, undo)
                # A mating/stalemating next move does not establish a fifty-move claim.
                if child_count > 0:
                    return 0
    if history_index < 7:
        return LIVE
    occurrences = 0
    for index in range(history_index, -1, -2):
        if same_key(keys[index], keys[history_index]):
            occurrences += 1
    if occurrences >= 3:
        return 0
    # Only construct successors when some prior opponent-to-move key occurred twice.
    possible = False
    for index in range(history_index - 1, -1, -2):
        for earlier in range(index - 2, -1, -2):
            if same_key(keys[index], keys[earlier]):
                possible = True
                break
        if possible:
            break
    if possible:
        child_key = np.empty(67, dtype=np.int16)
        for index in range(count):
            move = int(moves[index])
            make_move_inplace(state, move, undo)
            child_count = 0
            if state[EP_SQUARE] >= 0:
                child_count = generate_legal_moves_into(state, child_moves)
            write_key(state, child_moves, child_count, child_key)
            unmake_move_inplace(state, move, undo)
            occurrences = 0
            for previous in range(history_index - 1, -1, -2):
                if same_key(keys[previous], child_key):
                    occurrences += 1
            if occurrences >= 2:
                return 0
    return LIVE


@_numba_njit
def expired(deadline_ns: int) -> bool:
    # Portable monotonic wall clock; the chess recursion never enters Python.
    with objmode(now="int64"):
        now = time.perf_counter_ns()
    return bool(now >= deadline_ns)


@_numba_njit
def search_node(
    state: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    keys: np.ndarray,
    history_root: int,
    killers: np.ndarray,
    history: np.ndarray,
    stats: np.ndarray,
    deadline_ns: int,
    qmode: bool = False,
) -> tuple[int, int]:
    qmode = qmode or (depth == 0 and ply != 0)
    stats[1 if qmode else 0] += 1
    if ply + history_root >= MAX_PLY - 1:
        stats[5] = 2  # Abort, never silently substitute a static score at a storage limit.
        return (0, -1)
    if (
        deadline_ns >= 0
        and (stats[0] + stats[1] == 1 or (stats[0] + stats[1]) % 64 == 0)
        and expired(deadline_ns)
    ):
        stats[5] = 1
        return (0, -1)
    moves = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
    count = generate_legal_moves_into(state, moves)
    finish = terminal(state, moves, count, ply, keys, history_root + ply)
    if finish != LIVE:
        return (finish, -1)
    in_check = is_in_check(state)
    best = -INFINITY
    best_move = -1
    if qmode and not in_check:
        best = evaluate(state)
        if best >= beta:
            return (best, -1)
        alpha = max(alpha, best)
    order_moves(state, moves, count, ply, killers, history, qmode)
    undo = np.empty(UNDO_SIZE, dtype=np.int16)
    for index in range(count):
        move = int(moves[index])
        quiet = not capture(state, move) and ((move >> 12) & 7) == 0
        if qmode and not in_check and quiet:
            continue
        color = 0 if state[SIDE_TO_MOVE] == 1 else 1
        source = move & 63
        target = (move >> 6) & 63
        if not qmode and ply > 0 and quiet:
            if killers[ply, 0] == move or killers[ply, 1] == move:
                stats[3] += 1
            elif history[color, source, target] > 0:
                stats[4] += 1
        make_move_inplace(state, move, undo)
        child_score, _ = search_node(
            state,
            max(0, depth - 1),
            -beta,
            -alpha,
            ply + 1,
            keys,
            history_root,
            killers,
            history,
            stats,
            deadline_ns,
            qmode,
        )
        unmake_move_inplace(state, move, undo)
        if stats[5]:
            return (0, -1)
        score = -child_score
        if score > best:
            best = score
            best_move = move
        alpha = max(alpha, score)
        if alpha >= beta:
            if not qmode:
                stats[2] += 1
                if quiet:
                    if killers[ply, 0] != move:
                        killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = move
                    history[color, source, target] = min(
                        HISTORY_MAX, history[color, source, target] + depth * depth
                    )
            break
    return (best, best_move)


def state_from_board(board: chess.Board) -> np.ndarray:
    state = empty_state()
    for square, piece in board.piece_map().items():
        state[square] = piece.piece_type * (1 if piece.color else -1)
    state[SIDE_TO_MOVE] = 1 if board.turn else -1
    rights = 0
    for color, shift in ((chess.WHITE, 0), (chess.BLACK, 2)):
        if board.has_kingside_castling_rights(color):
            rights |= 1 << shift
        if board.has_queenside_castling_rights(color):
            rights |= 2 << shift
    state[CASTLING_RIGHTS] = rights
    state[EP_SQUARE] = board.ep_square if board.ep_square is not None else -1
    state[HALFMOVE_CLOCK] = board.halfmove_clock
    state[FULLMOVE_NUMBER] = board.fullmove_number
    return state


def root_context(board: chess.Board) -> tuple[np.ndarray, np.ndarray, int]:
    keys = np.zeros((MAX_PLY, 67), dtype=np.int16)
    root_index = len(board.move_stack)
    if root_index >= MAX_PLY - 1:
        raise ValueError("available move stack exceeds internal storage")
    if root_index:
        replay = board.root()
        moves = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
        for index, move in enumerate(board.move_stack):
            state = state_from_board(replay)
            count = generate_legal_moves_into(state, moves)
            write_key(state, moves, count, keys[index])
            replay.push(move)
    return state_from_board(board), keys, root_index


class SearchEngine:
    """Python outer budget/interface; all chess work below a root iteration is compiled."""

    def _run(self, board: chess.Board, depth: int, time_left_ms: int | None) -> SearchResult:
        started = time.perf_counter_ns()
        state, keys, root_index = root_context(board)
        original = state.copy()
        killers = np.full((MAX_PLY, 2), -1, dtype=np.int32)
        history = np.zeros((2, 64, 64), dtype=np.int64)
        stats = np.zeros(6, dtype=np.int64)
        moves = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
        count = generate_legal_moves_into(state, moves)
        order_moves(state, moves, count, 0, killers, history, True)
        best_move = int(moves[0]) if count else -1
        best_score = (
            evaluate(state) if count else terminal(state, moves, count, 0, keys, root_index)
        )
        completed = 0
        deadline = -1
        soft = -1
        if time_left_ms is not None:
            budget = allocate_time(time_left_ms)
            if budget.hard_ms == 0 or not count:
                depth = 0
            else:
                soft = started + budget.soft_ms * 1_000_000
                deadline = started + max(0, budget.hard_ms - 2) * 1_000_000
        current = depth if time_left_ms is None else 1
        while depth > 0:
            score, move = search_node(
                state,
                current,
                -INFINITY,
                INFINITY,
                0,
                keys,
                root_index,
                killers,
                history,
                stats,
                deadline,
                False,
            )
            if not np.array_equal(state, original):
                raise RuntimeError("search did not restore its root")
            if stats[5]:
                if time_left_ms is None:
                    raise RuntimeError("fixed-depth search exceeded internal storage")
                break
            if time_left_ms is None:
                best_move, best_score, completed = move, score, current
                break
            if move >= 0:
                best_move, best_score, completed = move, score, current
            if move < 0 or abs(best_score) >= MATE_THRESHOLD or time.perf_counter_ns() >= soft:
                break
            current += 1
        return SearchResult(
            move=chess.Move.from_uci(move_to_uci(best_move)) if best_move >= 0 else None,
            score=int(best_score),
            completed_depth=completed,
            nodes=int(stats[0]),
            qnodes=int(stats[1]),
            beta_cutoffs=int(stats[2]),
            killer_first_searches=int(stats[3]),
            history_ordered_moves=int(stats[4]),
            elapsed_ms=(time.perf_counter_ns() - started) / 1_000_000,
            timed_out=bool(stats[5]),
        )

    def search(self, board: chess.Board, time_left_ms: int) -> SearchResult:
        return self._run(board, 1, time_left_ms)

    def search_fixed_depth(self, board: chess.Board, depth: int) -> SearchResult:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        return self._run(board, depth, None)


def warmup() -> float:
    started = time.perf_counter()
    engine = SearchEngine()
    engine.search_fixed_depth(chess.Board(), 2)
    engine.search(chess.Board(), 100)
    # All calls use the same concrete array/scalar signatures regardless of board contents.
    return time.perf_counter() - started
