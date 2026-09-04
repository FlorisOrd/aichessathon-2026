import time
import unittest

import chess
import numpy as np

import engine as oracle
import numba_engine as native
from numba_core import MAX_LEGAL_MOVES, generate_legal_moves_into
from tools.numba_core_validation import deterministic_random_positions


def native_terminal(board: chess.Board, ply: int = 0) -> int | None:
    state, keys, index = native.root_context(board)
    original = state.copy()
    moves = np.empty(MAX_LEGAL_MOVES, dtype=np.int32)
    count = generate_legal_moves_into(state, moves)
    result = native.terminal(state, moves, count, ply, keys, index)
    np.testing.assert_array_equal(state, original)
    return None if result == native.LIVE else int(result)


class DrawParityTests(unittest.TestCase):
    def test_fifty_claims_and_terminal_precedence(self) -> None:
        for clock in (0, 98, 99, 100, 149, 150):
            for placement in (
                "4k3/8/8/8/8/8/P7/R3K3 w - -",
                "7k/6Q1/6K1/8/8/8/8/8 b - -",
                "7k/5Q2/6K1/8/8/8/8/8 b - -",
                "8/8/8/8/8/pk6/8/K7 w - -",
            ):
                board = chess.Board(f"{placement} {clock} 1")
                self.assertEqual(native_terminal(board, 5), oracle.terminal_score(board, 5))

    def test_repetition_immediate_next_move_and_no_invented_history(self) -> None:
        board = chess.Board()
        sequence = ["g1f3", "g8f6", "f3g1", "f6g8"] * 3
        for uci in sequence:
            board.push_uci(uci)
            self.assertEqual(native_terminal(board), oracle.terminal_score(board, 0))
            fresh = chess.Board(board.fen())
            self.assertEqual(native_terminal(fresh), oracle.terminal_score(fresh, 0))
        self.assertEqual(native_terminal(board), 0)
        self.assertIsNone(native_terminal(chess.Board(board.fen())))

    def test_terminal_random_history_and_insufficient_material(self) -> None:
        for fen in (
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2NNK3 w - - 0 1",
            "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",
            "2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
        ):
            board = chess.Board(fen)
            self.assertEqual(native_terminal(board), oracle.terminal_score(board, 0))
        board = chess.Board()
        import random

        rng = random.Random(20260904)
        for _ in range(120):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            self.assertEqual(native_terminal(board), oracle.terminal_score(board, 0))

    def test_ep_key_uses_legal_not_pseudo_capture(self) -> None:
        for fen, expected_ep in (
            ("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1", -1),
            ("4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1", chess.F6),
        ):
            state = native.state_from_board(chess.Board(fen))
            moves = np.empty(256, dtype=np.int32)
            count = generate_legal_moves_into(state, moves)
            key = np.empty(67, dtype=np.int16)
            native.write_key(state, moves, count, key)
            self.assertEqual(key[66], expected_ep)


class SearchParityTests(unittest.TestCase):
    def test_static_evaluation(self) -> None:
        for board in deterministic_random_positions(500, 20260904):
            self.assertEqual(
                native.evaluate(native.state_from_board(board)), oracle.evaluate(board)
            )

    def test_tactics_qsearch_and_fixed_depth(self) -> None:
        for fen in (
            chess.STARTING_FEN,
            "3r2k1/4q3/8/8/8/8/8/3Q2K1 w - - 0 1",
            "k3r3/8/8/8/8/8/8/4K3 w - - 0 1",
            "8/P6k/8/8/8/8/8/4K3 w - - 0 1",
            "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        ):
            board = chess.Board(fen)
            before = board.fen()
            for depth in (1, 2, 3):
                first = oracle.SearchEngine().search_fixed_depth(board, depth)
                second = native.SearchEngine().search_fixed_depth(board, depth)
                self.assertEqual((first.score, first.move), (second.score, second.move))
                self.assertEqual((first.nodes, first.qnodes), (second.nodes, second.qnodes))
                self.assertEqual(board.fen(), before)
            state, keys, root = native.root_context(board)
            score, _ = native.search_node(
                state,
                0,
                -oracle.INFINITY,
                oracle.INFINITY,
                0,
                keys,
                root,
                np.full((native.MAX_PLY, 2), -1, dtype=np.int32),
                np.zeros((2, 64, 64), dtype=np.int64),
                np.zeros(6, dtype=np.int64),
                -1,
                True,
            )
            expected = oracle.SearchEngine()._qsearch(board, -oracle.INFINITY, oracle.INFINITY, 0)
            self.assertEqual(score, expected)

    def test_deadline_unwind_and_fallback(self) -> None:
        board = chess.Board()
        native.warmup()
        before = board.fen()
        for remaining in (0, 1, 30, 100, 500):
            result = native.SearchEngine().search(board, remaining)
            self.assertIn(result.move, board.legal_moves)
            self.assertEqual(board.fen(), before)
        state, keys, root = native.root_context(board)
        original = state.copy()
        stats = np.zeros(6, dtype=np.int64)
        native.search_node(
            state,
            30,
            -oracle.INFINITY,
            oracle.INFINITY,
            0,
            keys,
            root,
            np.full((native.MAX_PLY, 2), -1, dtype=np.int32),
            np.zeros((2, 64, 64), dtype=np.int64),
            stats,
            time.perf_counter_ns() + 10_000_000,
            False,
        )
        self.assertEqual(stats[5], 1)
        np.testing.assert_array_equal(state, original)


if __name__ == "__main__":
    unittest.main()
