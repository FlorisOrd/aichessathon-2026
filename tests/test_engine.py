import unittest

import chess

import agent
from engine import (
    DELTA_MARGIN,
    INFINITY,
    MATE_SCORE,
    MATE_THRESHOLD,
    SearchEngine,
    evaluate,
    move_order_score,
    ordered_moves,
    terminal_score,
)


class TickingClock:
    def __init__(self, step_ns: int) -> None:
        self.value = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        self.value += self.step_ns
        return self.value


class ExplodingQSearchEngine(SearchEngine):
    def _qsearch(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
        ply_from_root: int,
    ) -> int:
        raise RuntimeError("deliberate test failure")

    def deadline_is_clear(self) -> bool:
        return self._deadline_ns is None


class EvaluationTests(unittest.TestCase):
    def test_evaluation_uses_side_to_move_perspective(self) -> None:
        board = chess.Board("k7/8/8/8/8/8/4Q3/4K3 w - - 0 1")
        white_score = evaluate(board)
        board.turn = chess.BLACK
        self.assertGreater(white_score, 0)
        self.assertEqual(evaluate(board), -white_score)

    def test_starting_position_is_balanced(self) -> None:
        self.assertEqual(evaluate(chess.Board()), 0)


class SearchTests(unittest.TestCase):
    def test_fixed_depth_choice_is_deterministic(self) -> None:
        first = SearchEngine().search_fixed_depth(chess.Board(), 2)
        second = SearchEngine().search_fixed_depth(chess.Board(), 2)
        self.assertEqual(first.move, second.move)
        self.assertEqual(first.score, second.score)

    def test_mate_in_one_is_selected(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        result = SearchEngine().search_fixed_depth(board, 1)
        self.assertIsNotNone(result.move)
        assert result.move is not None
        board.push(result.move)
        self.assertTrue(board.is_checkmate())

    def test_stalemate_is_a_terminal_draw(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        result = SearchEngine().search_fixed_depth(board, 1)
        self.assertIsNone(result.move)
        self.assertEqual(result.score, 0)

    def test_insufficient_material_is_a_terminal_draw(self) -> None:
        board = chess.Board("8/8/8/8/8/4k3/8/4K3 w - - 0 1")
        self.assertEqual(terminal_score(board, 0), 0)

    def test_fifty_move_claim_includes_claim_by_next_move(self) -> None:
        claimable = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 99 1")
        nearby = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 98 1")
        claim = claimable.outcome(claim_draw=True)
        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual(claim.termination, chess.Termination.FIFTY_MOVES)
        self.assertEqual(terminal_score(claimable, 0), 0)
        self.assertIsNone(nearby.outcome(claim_draw=True))
        self.assertIsNone(terminal_score(nearby, 0))

    def test_threefold_claim_uses_only_available_move_history(self) -> None:
        board = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
            board.push_uci(uci)

        claim = board.outcome(claim_draw=True)
        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual(claim.termination, chess.Termination.THREEFOLD_REPETITION)
        self.assertEqual(terminal_score(board, 0), 0)

        reconstructed = chess.Board(board.fen())
        self.assertEqual(len(reconstructed.move_stack), 0)
        self.assertIsNone(reconstructed.outcome(claim_draw=True))
        self.assertIsNone(terminal_score(reconstructed, 0))

    def test_timeout_returns_legal_fallback(self) -> None:
        board = chess.Board()
        original_fen = board.fen()
        result = SearchEngine(TickingClock(step_ns=5_000_000)).search(board, 100)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.completed_depth, 0)
        self.assertGreater(result.qnodes, 0)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(board.fen(), original_fen)

    def test_board_and_deadline_are_restored_after_qsearch_exception(self) -> None:
        board = chess.Board()
        original_fen = board.fen()
        engine = ExplodingQSearchEngine()
        with self.assertRaisesRegex(RuntimeError, "deliberate test failure"):
            engine.search(board, 1_000)
        self.assertEqual(board.fen(), original_fen)
        self.assertTrue(engine.deadline_is_clear())

    def test_qsearch_avoids_a_horizon_capture_blunder(self) -> None:
        board = chess.Board("3r2k1/4q3/8/8/8/8/8/3Q2K1 w - - 0 1")
        static_result = SearchEngine(use_quiescence=False).search_fixed_depth(board, 1)
        qsearch_result = SearchEngine().search_fixed_depth(board, 1)
        self.assertEqual(static_result.move, chess.Move.from_uci("d1d8"))
        self.assertNotEqual(qsearch_result.move, chess.Move.from_uci("d1d8"))
        self.assertEqual(static_result.qnodes, 0)
        self.assertGreater(qsearch_result.qnodes, 0)

    def test_in_check_qsearch_searches_all_quiet_evasions(self) -> None:
        board = chess.Board("k3r3/8/8/8/8/8/8/4K3 w - - 0 1")
        evasions = ordered_moves(board)
        self.assertTrue(board.is_check())
        self.assertTrue(
            all(not board.is_capture(move) and move.promotion is None for move in evasions)
        )

        engine = SearchEngine()
        original_fen = board.fen()
        engine._qsearch(board, -INFINITY, INFINITY, 0)
        self.assertEqual(engine._qnodes, 1 + len(evasions))
        self.assertEqual(engine._delta_pruned, 0)
        self.assertEqual(board.fen(), original_fen)

    def test_qsearch_includes_promotions(self) -> None:
        board = chess.Board("8/P6k/8/8/8/8/8/4K3 w - - 0 1")
        stand_pat = evaluate(board)
        engine = SearchEngine()
        score = engine._qsearch(board, -INFINITY, INFINITY, 0)
        self.assertGreater(score, stand_pat)
        self.assertEqual(engine._qnodes, 5)
        self.assertEqual(engine._delta_pruned, 0)

    def test_clearly_irrelevant_low_value_capture_is_delta_pruned(self) -> None:
        board = chess.Board("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1")
        stand_pat = evaluate(board)
        engine = SearchEngine()

        score = engine._qsearch(
            board,
            stand_pat + 120 + DELTA_MARGIN,
            INFINITY,
            0,
        )

        self.assertEqual(score, stand_pat)
        self.assertEqual(engine._qnodes, 1)
        self.assertEqual(engine._delta_pruned, 1)

    def test_tactical_capture_near_alpha_is_not_delta_pruned(self) -> None:
        board = chess.Board("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1")
        stand_pat = evaluate(board)
        engine = SearchEngine()

        engine._qsearch(
            board,
            stand_pat + 120 + DELTA_MARGIN - 1,
            INFINITY,
            0,
        )

        self.assertEqual(engine._qnodes, 2)
        self.assertEqual(engine._delta_pruned, 0)

    def test_capture_promotions_are_never_delta_pruned(self) -> None:
        board = chess.Board("r7/1P5k/8/8/8/8/8/2K5 w - - 0 1")
        promotions = [move for move in board.legal_moves if move.promotion is not None]
        self.assertEqual(len(promotions), 8)
        engine = SearchEngine()

        engine._qsearch(board, evaluate(board) + 2_000, INFINITY, 0)

        self.assertEqual(engine._qnodes, 1 + len(promotions))
        self.assertEqual(engine._delta_pruned, 0)

    def test_check_evasions_are_never_delta_pruned_even_with_high_alpha(self) -> None:
        board = chess.Board("k3r3/8/8/8/8/8/8/4K3 w - - 0 1")
        evasions = list(board.legal_moves)
        engine = SearchEngine()

        engine._qsearch(board, 100_000, INFINITY, 0)

        self.assertEqual(engine._qnodes, 1 + len(evasions))
        self.assertEqual(engine._delta_pruned, 0)

    def test_mate_score_window_disables_delta_pruning(self) -> None:
        board = chess.Board("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1")
        engine = SearchEngine()

        engine._qsearch(board, MATE_THRESHOLD, INFINITY, 0)

        self.assertEqual(engine._qnodes, 2)
        self.assertEqual(engine._delta_pruned, 0)

    def test_en_passant_uses_conservative_pawn_value(self) -> None:
        board = chess.Board("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1")
        stand_pat = evaluate(board)
        near_engine = SearchEngine()
        prune_engine = SearchEngine()

        near_engine._qsearch(
            board,
            stand_pat + 100 + DELTA_MARGIN + 1,
            INFINITY,
            0,
        )
        prune_engine._qsearch(
            board,
            stand_pat + 120 + DELTA_MARGIN,
            INFINITY,
            0,
        )

        self.assertEqual(near_engine._qnodes, 2)
        self.assertEqual(near_engine._delta_pruned, 0)
        self.assertEqual(prune_engine._qnodes, 1)
        self.assertEqual(prune_engine._delta_pruned, 1)

    def test_delta_pruning_restores_the_board(self) -> None:
        board = chess.Board("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1")
        original_fen = board.fen()
        engine = SearchEngine()

        engine._qsearch(board, evaluate(board) + 120 + DELTA_MARGIN, INFINITY, 0)

        self.assertEqual(engine._delta_pruned, 1)
        self.assertEqual(board.fen(), original_fen)

    def test_terminal_capture_is_not_delta_pruned(self) -> None:
        board = chess.Board("7k/6r1/5KQ1/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci("g6g7")
        self.assertIn(move, board.legal_moves)
        board.push(move)
        self.assertTrue(board.is_checkmate())
        board.pop()
        engine = SearchEngine()

        score = engine._qsearch(board, evaluate(board) + 1_000, INFINITY, 0)

        self.assertGreaterEqual(score, MATE_THRESHOLD)
        self.assertEqual(engine._delta_pruned, 0)

    def test_terminal_draw_capture_is_not_delta_pruned(self) -> None:
        board = chess.Board("8/7k/8/8/3n4/2B5/8/K7 w - - 0 1")
        move = chess.Move.from_uci("c3d4")
        self.assertIn(move, board.legal_moves)
        self.assertFalse(board.gives_check(move))
        board.push(move)
        self.assertTrue(board.is_insufficient_material())
        board.pop()
        engine = SearchEngine()

        engine._qsearch(board, evaluate(board) + 1_000, INFINITY, 0)

        self.assertEqual(engine._qnodes, 2)
        self.assertEqual(engine._delta_pruned, 0)

    def test_qsearch_preserves_terminal_and_mate_distance_scores(self) -> None:
        checkmate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        engine = SearchEngine()
        self.assertEqual(engine._qsearch(checkmate, -INFINITY, INFINITY, 4), -MATE_SCORE + 4)
        self.assertEqual(engine._qsearch(stalemate, -INFINITY, INFINITY, 4), 0)

    def test_zero_time_agent_move_is_legal(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci(agent.get_move(board.fen(), 0))
        self.assertIn(move, board.legal_moves)

    def test_move_order_is_deterministic_and_score_sorted(self) -> None:
        board = chess.Board("k7/4P3/8/8/8/8/8/4K3 w - - 0 1")
        first = ordered_moves(board)
        second = ordered_moves(board)
        self.assertEqual(first, second)
        scores = [move_order_score(board, move) for move in first]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(first[0].promotion, chess.QUEEN)

    def test_mate_distance_prefers_faster_wins_and_slower_losses(self) -> None:
        checkmated = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        fast_loss = terminal_score(checkmated, 1)
        slow_loss = terminal_score(checkmated, 5)
        self.assertEqual(fast_loss, -MATE_SCORE + 1)
        self.assertEqual(slow_loss, -MATE_SCORE + 5)
        assert fast_loss is not None and slow_loss is not None
        self.assertGreater(slow_loss, fast_loss)
        self.assertGreater(-fast_loss, -slow_loss)


if __name__ == "__main__":
    unittest.main()
