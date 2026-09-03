"""Numba-compatible chess state and legal move generation.

This module is deliberately independent from the active search implementation.  The compact
numeric representation and hot functions can be validated now and integrated into search later.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from numba import njit


def _numba_njit[FunctionT: Callable[..., Any]](function: FunctionT) -> FunctionT:
    """Keep precise static types while applying Numba's dynamically typed decorator."""
    return cast(FunctionT, njit(function))


PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

WHITE = 1
BLACK = -1

WHITE_KINGSIDE = 1
WHITE_QUEENSIDE = 2
BLACK_KINGSIDE = 4
BLACK_QUEENSIDE = 8

SIDE_TO_MOVE = 64
CASTLING_RIGHTS = 65
EP_SQUARE = 66
HALFMOVE_CLOCK = 67
FULLMOVE_NUMBER = 68
STATE_SIZE = 69

MOVE_FROM_MASK = 0x3F
MOVE_TO_SHIFT = 6
MOVE_PROMOTION_SHIFT = 12
MOVE_PROMOTION_MASK = 0x7
MOVE_EN_PASSANT = 1 << 15
MOVE_CASTLING = 1 << 16
MAX_LEGAL_MOVES = 256

UNDO_FROM = 0
UNDO_TO = 1
UNDO_MOVED_PIECE = 2
UNDO_CAPTURED_PIECE = 3
UNDO_CAPTURED_SQUARE = 4
UNDO_CASTLING_RIGHTS = 5
UNDO_EP_SQUARE = 6
UNDO_HALFMOVE_CLOCK = 7
UNDO_FULLMOVE_NUMBER = 8
UNDO_SIZE = 9

PROMOTION_PIECES = (QUEEN, ROOK, BISHOP, KNIGHT)
PROMOTION_NAMES = {QUEEN: "q", ROOK: "r", BISHOP: "b", KNIGHT: "n"}
PROMOTION_CODES = {value: key for key, value in PROMOTION_NAMES.items()}


def empty_state() -> np.ndarray:
    """Return an empty state array with neutral scalar defaults."""
    state = np.zeros(STATE_SIZE, dtype=np.int16)
    state[SIDE_TO_MOVE] = WHITE
    state[EP_SQUARE] = -1
    state[FULLMOVE_NUMBER] = 1
    return state


def encode_move(
    from_square: int,
    to_square: int,
    promotion: int = 0,
    *,
    en_passant: bool = False,
    castling: bool = False,
) -> int:
    """Encode a move into a stable integer representation."""
    move = from_square | (to_square << MOVE_TO_SHIFT) | (promotion << MOVE_PROMOTION_SHIFT)
    if en_passant:
        move |= MOVE_EN_PASSANT
    if castling:
        move |= MOVE_CASTLING
    return move


def move_from_square(move: int) -> int:
    return move & MOVE_FROM_MASK


def move_to_square(move: int) -> int:
    return (move >> MOVE_TO_SHIFT) & MOVE_FROM_MASK


def move_promotion(move: int) -> int:
    return (move >> MOVE_PROMOTION_SHIFT) & MOVE_PROMOTION_MASK


def square_name(square: int) -> str:
    if not 0 <= square < 64:
        raise ValueError(f"invalid square: {square}")
    return f"{chr(ord('a') + square % 8)}{square // 8 + 1}"


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise ValueError(f"invalid square: {name}")
    return (ord(name[0]) - ord("a")) + 8 * (int(name[1]) - 1)


def move_to_uci(move: int) -> str:
    uci = square_name(move_from_square(move)) + square_name(move_to_square(move))
    promotion = move_promotion(move)
    if promotion:
        uci += PROMOTION_NAMES[promotion]
    return uci


def uci_to_move(state: np.ndarray, uci: str) -> int:
    """Resolve UCI text to the matching legal internal move."""
    if len(uci) not in {4, 5}:
        raise ValueError(f"invalid UCI move: {uci}")
    promotion = PROMOTION_CODES.get(uci[4], 0) if len(uci) == 5 else 0
    from_square = parse_square(uci[:2])
    to_square = parse_square(uci[2:4])
    moves = legal_moves(state)
    for move in moves:
        value = int(move)
        if (
            move_from_square(value) == from_square
            and move_to_square(value) == to_square
            and move_promotion(value) == promotion
        ):
            return value
    raise ValueError(f"illegal UCI move: {uci}")


@_numba_njit
def _encode_move(
    from_square: int,
    to_square: int,
    promotion: int,
    en_passant: bool,
    castling: bool,
) -> int:
    move = from_square | (to_square << MOVE_TO_SHIFT) | (promotion << MOVE_PROMOTION_SHIFT)
    if en_passant:
        move |= MOVE_EN_PASSANT
    if castling:
        move |= MOVE_CASTLING
    return move


@_numba_njit
def _append_move(
    output: np.ndarray,
    count: int,
    from_square: int,
    to_square: int,
    promotion: int = 0,
    en_passant: bool = False,
    castling: bool = False,
) -> int:
    output[count] = _encode_move(from_square, to_square, promotion, en_passant, castling)
    return count + 1


@_numba_njit
def _append_pawn_move(
    output: np.ndarray,
    count: int,
    from_square: int,
    to_square: int,
    promotion_rank: int,
    en_passant: bool = False,
) -> int:
    if to_square // 8 == promotion_rank:
        count = _append_move(output, count, from_square, to_square, QUEEN)
        count = _append_move(output, count, from_square, to_square, ROOK)
        count = _append_move(output, count, from_square, to_square, BISHOP)
        count = _append_move(output, count, from_square, to_square, KNIGHT)
        return count
    return _append_move(output, count, from_square, to_square, 0, en_passant)


@_numba_njit
def is_square_attacked(state: np.ndarray, square: int, by_side: int) -> bool:
    """Return whether ``square`` is attacked by ``by_side``."""
    board = state[:64]
    rank = square // 8
    file = square % 8

    pawn_rank = rank - by_side
    if 0 <= pawn_rank < 8:
        if file > 0 and board[pawn_rank * 8 + file - 1] == by_side * PAWN:
            return True
        if file < 7 and board[pawn_rank * 8 + file + 1] == by_side * PAWN:
            return True

    knight_offsets = (
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    )
    for delta_rank, delta_file in knight_offsets:
        target_rank = rank + delta_rank
        target_file = file + delta_file
        if (
            0 <= target_rank < 8
            and 0 <= target_file < 8
            and board[target_rank * 8 + target_file] == by_side * KNIGHT
        ):
            return True

    for delta_rank in (-1, 0, 1):
        for delta_file in (-1, 0, 1):
            if delta_rank == 0 and delta_file == 0:
                continue
            target_rank = rank + delta_rank
            target_file = file + delta_file
            if (
                0 <= target_rank < 8
                and 0 <= target_file < 8
                and board[target_rank * 8 + target_file] == by_side * KING
            ):
                return True

    orthogonal = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for delta_rank, delta_file in orthogonal:
        target_rank = rank + delta_rank
        target_file = file + delta_file
        while 0 <= target_rank < 8 and 0 <= target_file < 8:
            piece = board[target_rank * 8 + target_file]
            if piece:
                if piece == by_side * ROOK or piece == by_side * QUEEN:
                    return True
                break
            target_rank += delta_rank
            target_file += delta_file

    diagonal = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for delta_rank, delta_file in diagonal:
        target_rank = rank + delta_rank
        target_file = file + delta_file
        while 0 <= target_rank < 8 and 0 <= target_file < 8:
            piece = board[target_rank * 8 + target_file]
            if piece:
                if piece == by_side * BISHOP or piece == by_side * QUEEN:
                    return True
                break
            target_rank += delta_rank
            target_file += delta_file
    return False


@_numba_njit
def is_in_check_for_side(state: np.ndarray, side: int) -> bool:
    king_square = -1
    for square in range(64):
        if state[square] == side * KING:
            king_square = square
            break
    if king_square < 0:
        return True
    return is_square_attacked(state, king_square, -side)


@_numba_njit
def is_in_check(state: np.ndarray) -> bool:
    return is_in_check_for_side(state, int(state[SIDE_TO_MOVE]))


@_numba_njit
def make_move_inplace(state: np.ndarray, move: int, undo: np.ndarray) -> None:
    """Make an encoded move and fill a fixed-size undo record."""
    from_square = move & MOVE_FROM_MASK
    to_square = (move >> MOVE_TO_SHIFT) & MOVE_FROM_MASK
    promotion = (move >> MOVE_PROMOTION_SHIFT) & MOVE_PROMOTION_MASK
    moved_piece = int(state[from_square])
    side = int(state[SIDE_TO_MOVE])
    captured_square = to_square
    if move & MOVE_EN_PASSANT:
        captured_square = to_square - side * 8
    captured_piece = int(state[captured_square])

    undo[UNDO_FROM] = from_square
    undo[UNDO_TO] = to_square
    undo[UNDO_MOVED_PIECE] = moved_piece
    undo[UNDO_CAPTURED_PIECE] = captured_piece
    undo[UNDO_CAPTURED_SQUARE] = captured_square
    undo[UNDO_CASTLING_RIGHTS] = state[CASTLING_RIGHTS]
    undo[UNDO_EP_SQUARE] = state[EP_SQUARE]
    undo[UNDO_HALFMOVE_CLOCK] = state[HALFMOVE_CLOCK]
    undo[UNDO_FULLMOVE_NUMBER] = state[FULLMOVE_NUMBER]

    state[from_square] = 0
    if captured_square != to_square:
        state[captured_square] = 0
    state[to_square] = side * promotion if promotion else moved_piece

    if move & MOVE_CASTLING:
        if to_square == 6:
            state[5] = state[7]
            state[7] = 0
        elif to_square == 2:
            state[3] = state[0]
            state[0] = 0
        elif to_square == 62:
            state[61] = state[63]
            state[63] = 0
        else:
            state[59] = state[56]
            state[56] = 0

    rights = int(state[CASTLING_RIGHTS])
    if moved_piece == WHITE * KING:
        rights &= ~(WHITE_KINGSIDE | WHITE_QUEENSIDE)
    elif moved_piece == BLACK * KING:
        rights &= ~(BLACK_KINGSIDE | BLACK_QUEENSIDE)
    if from_square == 0 or captured_square == 0:
        rights &= ~WHITE_QUEENSIDE
    if from_square == 7 or captured_square == 7:
        rights &= ~WHITE_KINGSIDE
    if from_square == 56 or captured_square == 56:
        rights &= ~BLACK_QUEENSIDE
    if from_square == 63 or captured_square == 63:
        rights &= ~BLACK_KINGSIDE
    state[CASTLING_RIGHTS] = rights

    state[EP_SQUARE] = -1
    if abs(moved_piece) == PAWN and abs(to_square - from_square) == 16:
        state[EP_SQUARE] = (from_square + to_square) // 2
    if abs(moved_piece) == PAWN or captured_piece:
        state[HALFMOVE_CLOCK] = 0
    else:
        state[HALFMOVE_CLOCK] += 1
    if side == BLACK:
        state[FULLMOVE_NUMBER] += 1
    state[SIDE_TO_MOVE] = -side


@_numba_njit
def unmake_move_inplace(state: np.ndarray, move: int, undo: np.ndarray) -> None:
    """Restore the exact state recorded by ``make_move_inplace``."""
    from_square = int(undo[UNDO_FROM])
    to_square = int(undo[UNDO_TO])
    captured_square = int(undo[UNDO_CAPTURED_SQUARE])

    state[SIDE_TO_MOVE] = -state[SIDE_TO_MOVE]
    state[CASTLING_RIGHTS] = undo[UNDO_CASTLING_RIGHTS]
    state[EP_SQUARE] = undo[UNDO_EP_SQUARE]
    state[HALFMOVE_CLOCK] = undo[UNDO_HALFMOVE_CLOCK]
    state[FULLMOVE_NUMBER] = undo[UNDO_FULLMOVE_NUMBER]

    if move & MOVE_CASTLING:
        if to_square == 6:
            state[7] = state[5]
            state[5] = 0
        elif to_square == 2:
            state[0] = state[3]
            state[3] = 0
        elif to_square == 62:
            state[63] = state[61]
            state[61] = 0
        else:
            state[56] = state[59]
            state[59] = 0

    state[from_square] = undo[UNDO_MOVED_PIECE]
    state[to_square] = 0
    state[captured_square] = undo[UNDO_CAPTURED_PIECE]


@_numba_njit
def _generate_pseudo_legal_moves(state: np.ndarray, output: np.ndarray) -> int:
    side = int(state[SIDE_TO_MOVE])
    count = 0

    for from_square in range(64):
        piece = int(state[from_square])
        if piece * side <= 0:
            continue
        piece_type = abs(piece)
        rank = from_square // 8
        file = from_square % 8

        if piece_type == PAWN:
            direction = 8 * side
            start_rank = 1 if side == WHITE else 6
            promotion_rank = 7 if side == WHITE else 0
            to_square = from_square + direction
            if 0 <= to_square < 64 and state[to_square] == 0:
                count = _append_pawn_move(output, count, from_square, to_square, promotion_rank)
                double_square = from_square + 2 * direction
                if rank == start_rank and state[double_square] == 0:
                    count = _append_move(output, count, from_square, double_square)

            capture_rank = rank + side
            if 0 <= capture_rank < 8:
                for target_file in (file - 1, file + 1):
                    if not 0 <= target_file < 8:
                        continue
                    target = capture_rank * 8 + target_file
                    target_piece = int(state[target])
                    if target_piece * side < 0 and abs(target_piece) != KING:
                        count = _append_pawn_move(
                            output, count, from_square, target, promotion_rank
                        )
                    elif target == state[EP_SQUARE]:
                        captured_square = target - side * 8
                        if state[captured_square] == -side * PAWN:
                            count = _append_pawn_move(
                                output,
                                count,
                                from_square,
                                target,
                                promotion_rank,
                                True,
                            )
            continue

        if piece_type == KNIGHT:
            for delta_rank, delta_file in (
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
                (2, -1),
                (2, 1),
            ):
                target_rank = rank + delta_rank
                target_file = file + delta_file
                if 0 <= target_rank < 8 and 0 <= target_file < 8:
                    target = target_rank * 8 + target_file
                    target_piece = int(state[target])
                    if target_piece * side <= 0 and abs(target_piece) != KING:
                        count = _append_move(output, count, from_square, target)
            continue

        if piece_type == KING:
            for delta_rank in (-1, 0, 1):
                for delta_file in (-1, 0, 1):
                    if delta_rank == 0 and delta_file == 0:
                        continue
                    target_rank = rank + delta_rank
                    target_file = file + delta_file
                    if 0 <= target_rank < 8 and 0 <= target_file < 8:
                        target = target_rank * 8 + target_file
                        target_piece = int(state[target])
                        if target_piece * side <= 0 and abs(target_piece) != KING:
                            count = _append_move(output, count, from_square, target)

            rights = int(state[CASTLING_RIGHTS])
            if side == WHITE and from_square == 4:
                if (
                    rights & WHITE_KINGSIDE
                    and state[7] == WHITE * ROOK
                    and state[5] == 0
                    and state[6] == 0
                    and not is_square_attacked(state, 4, BLACK)
                    and not is_square_attacked(state, 5, BLACK)
                    and not is_square_attacked(state, 6, BLACK)
                ):
                    count = _append_move(output, count, 4, 6, 0, False, True)
                if (
                    rights & WHITE_QUEENSIDE
                    and state[0] == WHITE * ROOK
                    and state[1] == 0
                    and state[2] == 0
                    and state[3] == 0
                    and not is_square_attacked(state, 4, BLACK)
                    and not is_square_attacked(state, 3, BLACK)
                    and not is_square_attacked(state, 2, BLACK)
                ):
                    count = _append_move(output, count, 4, 2, 0, False, True)
            elif side == BLACK and from_square == 60:
                if (
                    rights & BLACK_KINGSIDE
                    and state[63] == BLACK * ROOK
                    and state[61] == 0
                    and state[62] == 0
                    and not is_square_attacked(state, 60, WHITE)
                    and not is_square_attacked(state, 61, WHITE)
                    and not is_square_attacked(state, 62, WHITE)
                ):
                    count = _append_move(output, count, 60, 62, 0, False, True)
                if (
                    rights & BLACK_QUEENSIDE
                    and state[56] == BLACK * ROOK
                    and state[57] == 0
                    and state[58] == 0
                    and state[59] == 0
                    and not is_square_attacked(state, 60, WHITE)
                    and not is_square_attacked(state, 59, WHITE)
                    and not is_square_attacked(state, 58, WHITE)
                ):
                    count = _append_move(output, count, 60, 58, 0, False, True)
            continue

        directions = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
        for direction_index in range(8):
            if piece_type == BISHOP and direction_index < 4:
                continue
            if piece_type == ROOK and direction_index >= 4:
                continue
            delta_rank, delta_file = directions[direction_index]
            target_rank = rank + delta_rank
            target_file = file + delta_file
            while 0 <= target_rank < 8 and 0 <= target_file < 8:
                target = target_rank * 8 + target_file
                target_piece = int(state[target])
                if target_piece * side > 0:
                    break
                if abs(target_piece) != KING:
                    count = _append_move(output, count, from_square, target)
                if target_piece:
                    break
                target_rank += delta_rank
                target_file += delta_file
    return count


@_numba_njit
def generate_legal_moves_into(state: np.ndarray, output: np.ndarray) -> int:
    """Fill ``output`` with complete legal moves and return the count."""
    pseudo = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
    pseudo_count = _generate_pseudo_legal_moves(state, pseudo)
    undo = np.empty(UNDO_SIZE, dtype=np.int16)
    side = int(state[SIDE_TO_MOVE])
    legal_count = 0
    for index in range(pseudo_count):
        move = int(pseudo[index])
        make_move_inplace(state, move, undo)
        legal = not is_in_check_for_side(state, side)
        unmake_move_inplace(state, move, undo)
        if legal:
            output[legal_count] = move
            legal_count += 1
    return legal_count


def legal_moves(state: np.ndarray) -> np.ndarray:
    """Return a right-sized array of complete legal moves."""
    output = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
    count = generate_legal_moves_into(state, output)
    return output[:count].copy()


@_numba_njit
def perft(state: np.ndarray, depth: int) -> int:
    """Count legal leaf nodes; used only for differential validation."""
    if depth == 0:
        return 1
    moves = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
    count = generate_legal_moves_into(state, moves)
    if depth == 1:
        return count
    total = 0
    undo = np.empty(UNDO_SIZE, dtype=np.int16)
    for index in range(count):
        move = int(moves[index])
        make_move_inplace(state, move, undo)
        total += perft(state, depth - 1)
        unmake_move_inplace(state, move, undo)
    return total


def state_tuple(state: np.ndarray) -> tuple[int, ...]:
    """Return an immutable exact snapshot useful for restoration assertions."""
    return tuple(int(value) for value in state)


def moves_to_uci(moves: Sequence[int] | np.ndarray) -> set[str]:
    return {move_to_uci(int(move)) for move in moves}
