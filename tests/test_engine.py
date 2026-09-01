import unittest

import chess

import agent
from engine import (
    MATE_SCORE,
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


class ExplodingSearchEngine(SearchEngine):
    def _search_depth(self, board: chess.Board, depth: int) -> tuple[chess.Move | None, int]:
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
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(board.fen(), original_fen)

    def test_deadline_is_cleared_after_unexpected_exception(self) -> None:
        engine = ExplodingSearchEngine()
        with self.assertRaisesRegex(RuntimeError, "deliberate test failure"):
            engine.search(chess.Board(), 1_000)
        self.assertTrue(engine.deadline_is_clear())

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

    def test_preferred_move_is_first_without_reordering_the_rest(self) -> None:
        board = chess.Board()
        default = ordered_moves(board)
        preferred = chess.Move.from_uci("g1f3")
        reordered = ordered_moves(board, preferred)
        self.assertEqual(reordered[0], preferred)
        self.assertEqual(reordered[1:], [move for move in default if move != preferred])

    def test_iterative_deepening_reuses_the_completed_principal_variation(self) -> None:
        board = chess.Board()
        result = SearchEngine(TickingClock(step_ns=1_000)).search(board, 1_000)
        repeated = SearchEngine(TickingClock(step_ns=1_000)).search(board, 1_000)
        self.assertGreaterEqual(result.completed_depth, 2)
        self.assertEqual(result.move, repeated.move)
        self.assertEqual(result.score, repeated.score)
        self.assertEqual(result.completed_depth, repeated.completed_depth)
        self.assertEqual(result.nodes, repeated.nodes)
        self.assertEqual(result.cutoffs, repeated.cutoffs)
        self.assertEqual(result.depth_diagnostics, repeated.depth_diagnostics)
        self.assertEqual(
            result.root_pv_reuses,
            len(result.depth_diagnostics) - 1,
        )
        self.assertFalse(result.depth_diagnostics[0].root_pv_reused)
        self.assertTrue(all(item.nodes > 0 for item in result.depth_diagnostics))
        for item in result.depth_diagnostics[1:]:
            self.assertTrue(item.root_pv_reused)
            self.assertIsNotNone(item.first_move)
        self.assertGreater(result.cutoffs, 0)

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
