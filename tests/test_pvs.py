import unittest
from unittest.mock import patch

import chess

from engine import INFINITY, SearchEngine, SearchTimeout


class PvsTests(unittest.TestCase):
    def test_first_child_uses_full_negated_window(self) -> None:
        engine = SearchEngine()
        board = chess.Board()
        with patch.object(engine, "_negamax", return_value=-25) as search:
            score = engine._search_child(board, 2, 10, 100, 3, first=True)
        self.assertEqual(score, 25)
        search.assert_called_once_with(board, 2, -100, -10, 3)
        self.assertEqual(engine._pvs_null_window_searches, 0)
        self.assertEqual(engine._pvs_researches, 0)

    def test_alpha_improvement_is_researched_with_full_window(self) -> None:
        engine = SearchEngine()
        board = chess.Board()
        with patch.object(engine, "_negamax", side_effect=[-11, -35]) as search:
            score = engine._search_child(board, 2, 10, 100, 3, first=False)
        self.assertEqual(score, 35)
        self.assertEqual(
            [call.args for call in search.call_args_list],
            [(board, 2, -11, -10, 3), (board, 2, -100, -10, 3)],
        )
        self.assertEqual(engine._pvs_null_window_searches, 1)
        self.assertEqual(engine._pvs_researches, 1)

    def test_fail_low_and_beta_fail_high_do_not_research(self) -> None:
        for score in (9, 10, 100, 125):
            with self.subTest(score=score):
                engine = SearchEngine()
                board = chess.Board()
                with patch.object(engine, "_negamax", return_value=-score) as search:
                    result = engine._search_child(board, 2, 10, 100, 3, first=False)
                self.assertEqual(result, score)
                search.assert_called_once_with(board, 2, -11, -10, 3)
                self.assertEqual(engine._pvs_null_window_searches, 1)
                self.assertEqual(engine._pvs_researches, 0)

    def test_null_parent_window_cannot_trigger_research(self) -> None:
        engine = SearchEngine()
        with patch.object(engine, "_negamax", return_value=-11):
            self.assertEqual(
                engine._search_child(chess.Board(), 2, 10, 11, 3, first=False), 11
            )
        self.assertEqual(engine._pvs_researches, 0)

    def test_normal_node_fail_high_cuts_off_and_retains_killer_history(self) -> None:
        engine = SearchEngine()
        board = chess.Board()
        moves = engine._ordered_search_moves(board, 0)
        original = board.fen()
        with patch.object(engine, "_search_child", side_effect=[5, 100]) as search:
            score = engine._negamax(board, 2, 0, 100, 0)
        self.assertEqual(score, 100)
        self.assertEqual(search.call_count, 2)
        self.assertEqual([call.kwargs["first"] for call in search.call_args_list], [True, False])
        self.assertEqual(engine._beta_cutoffs, 1)
        self.assertEqual(engine._killers[0], [moves[1]])
        self.assertEqual(engine._history_score(board.turn, moves[1]), 4)
        self.assertEqual(board.fen(), original)

    def test_fixed_depth_scores_match_full_window_baseline(self) -> None:
        positions = (
            chess.STARTING_FEN,
            "3r2k1/4q3/8/8/8/8/8/3Q2K1 w - - 0 1",
            "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            "k3r3/8/8/8/8/8/8/4K3 w - - 0 1",
            "8/P6k/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/R7/4K3 w - - 99 1",
        )
        for fen in positions:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                baseline = SearchEngine(use_pvs=False).search_fixed_depth(board, 3)
                candidate = SearchEngine().search_fixed_depth(board, 3)
                repeated = SearchEngine().search_fixed_depth(board, 3)
                self.assertEqual(candidate.score, baseline.score)
                self.assertEqual(candidate.move, repeated.move)
                self.assertEqual(candidate.score, repeated.score)
                self.assertEqual(candidate.nodes, repeated.nodes)
                self.assertEqual(candidate.qnodes, repeated.qnodes)
                self.assertEqual(candidate.pvs_researches, repeated.pvs_researches)
                self.assertEqual(board.fen(), fen)
                self.assertEqual(board.move_stack, [])

    def test_timeouts_in_scout_and_research_restore_board_and_deadline(self) -> None:
        for results, researches in (([0, SearchTimeout()], 0), ([0, -10, SearchTimeout()], 1)):
            with self.subTest(researches=researches):
                engine = SearchEngine(clock_ns=lambda: 0)
                board = chess.Board()
                board.push_uci("g1f3")
                original_fen = board.fen()
                original_stack = list(board.move_stack)
                with patch.object(engine, "_negamax", side_effect=results):
                    result = engine.search(board, 1000)
                self.assertTrue(result.timed_out)
                self.assertEqual(result.completed_depth, 0)
                self.assertIn(result.move, board.legal_moves)
                self.assertEqual(result.pvs_null_window_searches, 1)
                self.assertEqual(result.pvs_researches, researches)
                self.assertEqual(board.fen(), original_fen)
                self.assertEqual(board.move_stack, original_stack)
                self.assertIsNone(engine._deadline_ns)

    def test_disabled_pvs_and_direct_qsearch_have_no_pvs_counts(self) -> None:
        disabled = SearchEngine(use_pvs=False).search_fixed_depth(chess.Board(), 2)
        self.assertEqual(disabled.pvs_null_window_searches, 0)
        self.assertEqual(disabled.pvs_researches, 0)
        engine = SearchEngine()
        engine._qsearch(chess.Board("k3r3/8/8/8/8/8/8/4K3 w - - 0 1"), -INFINITY, INFINITY, 0)
        self.assertEqual(engine._pvs_null_window_searches, 0)
        self.assertEqual(engine._pvs_researches, 0)

    def test_pvs_counters_reset_between_root_searches(self) -> None:
        engine = SearchEngine()
        result = engine.search_fixed_depth(chess.Board(), 2)
        self.assertGreater(result.pvs_null_window_searches, 0)
        reset = engine.search(chess.Board(), 0)
        self.assertEqual(reset.pvs_null_window_searches, 0)
        self.assertEqual(reset.pvs_researches, 0)


if __name__ == "__main__":
    unittest.main()
