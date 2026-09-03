import unittest
from pathlib import Path

import chess
import numpy as np

from numba_core import (
    UNDO_SIZE,
    is_in_check,
    legal_moves,
    make_move_inplace,
    moves_to_uci,
    perft,
    state_tuple,
    uci_to_move,
    unmake_move_inplace,
)
from tools.numba_core_validation import (
    deterministic_random_positions,
    differential_position,
    load_fens,
    python_perft,
    state_from_board,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class NumbaCoreDifferentialTests(unittest.TestCase):
    @staticmethod
    def assert_matches_oracle(board: chess.Board) -> None:
        mismatches = differential_position(board)
        if mismatches:
            raise AssertionError(f"{board.fen(en_passant='fen')}: {mismatches}")

    def test_smoke_and_development_64(self) -> None:
        paths = (
            REPOSITORY_ROOT / "benchmarks" / "positions" / "smoke.fens",
            REPOSITORY_ROOT / "benchmarks" / "positions" / "development-64.fens",
        )
        for path in paths:
            for fen in load_fens(path):
                with self.subTest(path=path.name, fen=fen):
                    self.assert_matches_oracle(chess.Board(fen))

    def test_2048_deterministic_random_playout_positions(self) -> None:
        mismatch_count = 0
        details: list[str] = []
        for board in deterministic_random_positions(2_048, seed=20_260_903):
            mismatches = differential_position(board)
            mismatch_count += len(mismatches)
            if mismatches and len(details) < 10:
                details.append(f"{board.fen(en_passant='fen')}: {mismatches}")
        self.assertEqual(mismatch_count, 0, "\n".join(details))


class NumbaCoreSpecialCaseTests(unittest.TestCase):
    def assert_case_matches(self, fen: str) -> tuple[np.ndarray, set[str]]:
        board = chess.Board(fen)
        mismatches = differential_position(board)
        self.assertEqual(mismatches, [], f"{fen}: {mismatches}")
        state = state_from_board(board)
        return state, moves_to_uci(legal_moves(state))

    def test_kingside_and_queenside_castling(self) -> None:
        state, moves = self.assert_case_matches("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.assertIn("e1g1", moves)
        self.assertIn("e1c1", moves)
        self.assertFalse(is_in_check(state))

        _, black_moves = self.assert_case_matches("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
        self.assertIn("e8g8", black_moves)
        self.assertIn("e8c8", black_moves)

    def test_en_passant(self) -> None:
        _, moves = self.assert_case_matches(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        )
        self.assertIn("e5f6", moves)

    def test_promotions_and_all_underpromotions(self) -> None:
        _, moves = self.assert_case_matches("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1")
        self.assertTrue({"a7a8q", "a7a8r", "a7a8b", "a7a8n"}.issubset(moves))

    def test_double_pawn_moves(self) -> None:
        _, moves = self.assert_case_matches(chess.STARTING_FEN)
        self.assertIn("e2e4", moves)

    def test_pinned_piece(self) -> None:
        _, moves = self.assert_case_matches("4k3/8/8/8/8/4r3/4R3/4K3 w - - 0 1")
        self.assertNotIn("e2d2", moves)
        self.assertIn("e2e3", moves)

    def test_discovered_check(self) -> None:
        fen = "4k3/8/8/8/8/8/4B3/4R1K1 w - - 0 1"
        state, _ = self.assert_case_matches(fen)
        move = uci_to_move(state, "e2b5")
        undo = np.empty(UNDO_SIZE, dtype=np.int16)
        original = state_tuple(state)
        make_move_inplace(state, move, undo)
        self.assertTrue(is_in_check(state))
        unmake_move_inplace(state, move, undo)
        self.assertEqual(state_tuple(state), original)

    def test_double_check(self) -> None:
        state, moves = self.assert_case_matches("4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1")
        self.assertTrue(is_in_check(state))
        self.assertTrue(all(move.startswith("e1") for move in moves))

    def test_checkmate(self) -> None:
        state, moves = self.assert_case_matches("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(is_in_check(state))
        self.assertEqual(moves, set())

    def test_stalemate(self) -> None:
        state, moves = self.assert_case_matches("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertFalse(is_in_check(state))
        self.assertEqual(moves, set())


class NumbaCorePerftTests(unittest.TestCase):
    def test_differential_perft_and_restoration(self) -> None:
        cases = (
            (chess.STARTING_FEN, 4, 197_281),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                3,
                97_862,
            ),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2_812),
        )
        for fen, depth, known_nodes in cases:
            with self.subTest(fen=fen, depth=depth):
                board = chess.Board(fen)
                state = state_from_board(board)
                original = state_tuple(state)
                oracle_nodes = python_perft(board, depth)
                internal_nodes = int(perft(state, depth))
                self.assertEqual(oracle_nodes, known_nodes)
                self.assertEqual(internal_nodes, oracle_nodes)
                self.assertEqual(state_tuple(state), original)


if __name__ == "__main__":
    unittest.main()
