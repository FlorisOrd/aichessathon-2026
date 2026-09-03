"""python-chess oracle helpers for validating :mod:`numba_core`."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import chess
import numpy as np

from numba_core import (
    BLACK,
    BLACK_KINGSIDE,
    BLACK_QUEENSIDE,
    CASTLING_RIGHTS,
    EP_SQUARE,
    FULLMOVE_NUMBER,
    HALFMOVE_CLOCK,
    SIDE_TO_MOVE,
    UNDO_SIZE,
    WHITE,
    WHITE_KINGSIDE,
    WHITE_QUEENSIDE,
    empty_state,
    is_in_check,
    legal_moves,
    make_move_inplace,
    move_to_uci,
    moves_to_uci,
    state_tuple,
    unmake_move_inplace,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def state_from_board(board: chess.Board) -> np.ndarray:
    """Convert public python-chess state into the internal numeric representation."""
    state = empty_state()
    for square, piece in board.piece_map().items():
        state[square] = piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
    state[SIDE_TO_MOVE] = WHITE if board.turn == chess.WHITE else BLACK
    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= WHITE_KINGSIDE
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= WHITE_QUEENSIDE
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= BLACK_KINGSIDE
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= BLACK_QUEENSIDE
    state[CASTLING_RIGHTS] = rights
    state[EP_SQUARE] = board.ep_square if board.ep_square is not None else -1
    state[HALFMOVE_CLOCK] = board.halfmove_clock
    state[FULLMOVE_NUMBER] = board.fullmove_number
    return state


def state_mismatches(state: np.ndarray, board: chess.Board) -> list[str]:
    expected = state_from_board(board)
    mismatches: list[str] = []
    if not np.array_equal(state[:64], expected[:64]):
        mismatches.append("piece placement")
    labels = {
        SIDE_TO_MOVE: "side to move",
        CASTLING_RIGHTS: "castling rights",
        EP_SQUARE: "en-passant square",
        HALFMOVE_CLOCK: "halfmove clock",
        FULLMOVE_NUMBER: "fullmove number",
    }
    for index, label in labels.items():
        if state[index] != expected[index]:
            mismatches.append(f"{label}: {int(state[index])} != {int(expected[index])}")
    return mismatches


def differential_position(board: chess.Board) -> list[str]:
    """Compare a position and every legal one-ply transition against python-chess."""
    state = state_from_board(board)
    original = state_tuple(state)
    internal_moves = legal_moves(state)
    by_uci = {move_to_uci(int(move)): int(move) for move in internal_moves}
    oracle_uci = {move.uci() for move in board.legal_moves}
    mismatches: list[str] = []

    if set(by_uci) != oracle_uci:
        missing = sorted(oracle_uci - set(by_uci))
        extra = sorted(set(by_uci) - oracle_uci)
        mismatches.append(f"legal moves missing={missing} extra={extra}")
    if bool(is_in_check(state)) != board.is_check():
        mismatches.append(f"check status {bool(is_in_check(state))} != {board.is_check()}")
    mismatches.extend(state_mismatches(state, board))

    undo = np.empty(UNDO_SIZE, dtype=np.int16)
    for uci in sorted(oracle_uci & set(by_uci)):
        move = by_uci[uci]
        oracle_move = chess.Move.from_uci(uci)
        make_move_inplace(state, move, undo)
        board.push(oracle_move)
        for mismatch in state_mismatches(state, board):
            mismatches.append(f"after {uci}: {mismatch}")
        if bool(is_in_check(state)) != board.is_check():
            mismatches.append(
                f"after {uci}: check status {bool(is_in_check(state))} != {board.is_check()}"
            )
        board.pop()
        unmake_move_inplace(state, move, undo)
        if state_tuple(state) != original:
            mismatches.append(f"after undo {uci}: exact state not restored")
            state[:] = np.asarray(original, dtype=np.int16)
    return mismatches


def load_fens(path: Path) -> list[str]:
    return [
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


def deterministic_random_positions(count: int, seed: int) -> list[chess.Board]:
    """Create deterministic reachable positions across many independent playouts."""
    rng = random.Random(seed)
    positions: list[chess.Board] = []
    seen: set[str] = set()
    while len(positions) < count:
        board = chess.Board()
        target_plies = rng.randint(12, 100)
        for _ in range(target_plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[rng.randrange(len(moves))])
            key = board.fen(en_passant="fen")
            if key not in seen:
                seen.add(key)
                positions.append(board.copy(stack=False))
                if len(positions) == count:
                    break
    return positions


def python_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += python_perft(board, depth - 1)
        board.pop()
    return total


def boards_from_fens(fens: Iterable[str]) -> list[chess.Board]:
    return [chess.Board(fen) for fen in fens]


def legal_uci(state: np.ndarray) -> set[str]:
    return moves_to_uci(legal_moves(state))
