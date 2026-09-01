import unittest

import chess

import agent
from engine import (
    DEFAULT_TT_CAPACITY,
    INFINITY,
    MATE_SCORE,
    BoundType,
    SearchEngine,
    TranspositionTable,
    evaluate,
    move_order_score,
    ordered_moves,
    score_from_tt,
    score_to_tt,
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

    def test_exact_tt_hit_returns_without_research(self) -> None:
        board = chess.Board()
        engine = SearchEngine(tt_capacity=16)
        move = chess.Move.from_uci("e2e4")
        engine.table.store(board, 3, 123, BoundType.EXACT, move, 0)

        result = engine.search_fixed_depth(board, 3)

        self.assertEqual(result.move, move)
        self.assertEqual(result.score, 123)
        self.assertEqual(result.nodes, 1)
        self.assertEqual(result.tt_usable_hits, 1)
        self.assertEqual(result.tt_cutoffs, 1)

    def test_lower_and_upper_bounds_cut_off_only_the_matching_window(self) -> None:
        board = chess.Board()
        lower_engine = SearchEngine(tt_capacity=16)
        lower_engine.table.store(board, 3, 50, BoundType.LOWER, None, 0)
        lower_engine._start_tt_generation()
        _, alpha, beta, score = lower_engine._probe_tt(board, 3, 0, 40, 0)
        self.assertEqual((alpha, beta, score), (50, 40, 50))

        upper_engine = SearchEngine(tt_capacity=16)
        upper_engine.table.store(board, 3, -50, BoundType.UPPER, None, 0)
        upper_engine._start_tt_generation()
        _, alpha, beta, score = upper_engine._probe_tt(board, 3, -40, 0, 0)
        self.assertEqual((alpha, beta, score), (-40, -50, -50))

        non_cutoff = SearchEngine(tt_capacity=16)
        non_cutoff.table.store(board, 3, 10, BoundType.LOWER, None, 0)
        non_cutoff._start_tt_generation()
        _, alpha, beta, score = non_cutoff._probe_tt(board, 3, 0, 40, 0)
        self.assertEqual((alpha, beta, score), (10, 40, None))

    def test_insufficient_depth_entry_only_supplies_move_ordering(self) -> None:
        board = chess.Board()
        engine = SearchEngine(tt_capacity=16)
        engine.table.store(
            board,
            1,
            987_654,
            BoundType.EXACT,
            chess.Move.from_uci("e2e4"),
            0,
        )

        result = engine.search_fixed_depth(board, 2)

        self.assertGreater(result.nodes, 1)
        self.assertNotEqual(result.score, 987_654)
        self.assertGreaterEqual(result.tt_hits, 1)
        self.assertEqual(result.tt_usable_hits, 0)

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

    def test_legal_tt_move_is_first_and_illegal_hint_is_ignored(self) -> None:
        board = chess.Board()
        tt_move = chess.Move.from_uci("h2h3")
        self.assertEqual(ordered_moves(board, tt_move)[0], tt_move)
        self.assertEqual(
            ordered_moves(board, chess.Move.from_uci("e7e5")),
            ordered_moves(board),
        )

    def test_mate_distance_prefers_faster_wins_and_slower_losses(self) -> None:
        checkmated = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        fast_loss = terminal_score(checkmated, 1)
        slow_loss = terminal_score(checkmated, 5)
        self.assertEqual(fast_loss, -MATE_SCORE + 1)
        self.assertEqual(slow_loss, -MATE_SCORE + 5)
        assert fast_loss is not None and slow_loss is not None
        self.assertGreater(slow_loss, fast_loss)
        self.assertGreater(-fast_loss, -slow_loss)

    def test_tt_mate_scores_are_normalized_across_root_distances(self) -> None:
        winning = MATE_SCORE - 7
        losing = -MATE_SCORE + 7
        self.assertEqual(score_from_tt(score_to_tt(winning, 3), 5), MATE_SCORE - 9)
        self.assertEqual(score_from_tt(score_to_tt(losing, 3), 5), -MATE_SCORE + 9)
        self.assertEqual(score_from_tt(score_to_tt(42, 3), 5), 42)


class TranspositionTableTests(unittest.TestCase):
    def test_collision_replacement_is_depth_preferred_then_generation_aged(self) -> None:
        table = TranspositionTable(capacity=1)
        first = chess.Board()
        second = chess.Board()
        second.push_uci("e2e4")
        self.assertTrue(table.store(first, 4, 10, BoundType.EXACT, None, 0))
        self.assertFalse(table.store(second, 2, 20, BoundType.EXACT, None, 0))
        self.assertIsNotNone(table.probe(first))
        self.assertIsNone(table.probe(second))

        table.new_generation()
        self.assertTrue(table.store(second, 2, 20, BoundType.EXACT, None, 0))
        self.assertIsNone(table.probe(first))
        self.assertIsNotNone(table.probe(second))

    def test_halfmove_clock_is_keyed_and_history_sensitive_scores_are_refused(self) -> None:
        safe = chess.Board()
        sensitive = chess.Board(safe.fen())
        sensitive.halfmove_clock = 8
        table = TranspositionTable(capacity=16)
        table.store(safe, 2, 99, BoundType.EXACT, None, 0)
        self.assertIsNotNone(table.probe(safe))
        self.assertIsNone(table.probe(sensitive))

        engine = SearchEngine(tt_capacity=16)
        engine.table.store(sensitive, 2, 456_789, BoundType.EXACT, None, 0)
        engine._start_tt_generation()
        _, alpha, beta, score = engine._probe_tt(sensitive, 2, -INFINITY, INFINITY, 0)
        self.assertEqual((alpha, beta, score), (-INFINITY, INFINITY, None))
        self.assertEqual(engine._tt_stats.hits, 1)
        self.assertEqual(engine._tt_stats.usable_hits, 0)

    def test_terminal_repetition_is_checked_before_tt_probe(self) -> None:
        repeated = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
            repeated.push_uci(uci)
        reconstructed = chess.Board(repeated.fen())
        engine = SearchEngine(tt_capacity=16)
        engine.table.store(
            reconstructed,
            4,
            456_789,
            BoundType.EXACT,
            next(iter(reconstructed.legal_moves)),
            0,
        )

        result = engine.search_fixed_depth(repeated, 4)

        self.assertEqual(result.score, 0)
        self.assertIsNone(result.move)
        self.assertEqual(result.tt_probes, 0)

    def test_table_capacity_and_memory_estimate_are_bounded(self) -> None:
        table = TranspositionTable(capacity=4)
        board = chess.Board()
        for move in list(board.legal_moves)[:8]:
            board.push(move)
            table.store(board, 1, 0, BoundType.EXACT, None, 0)
            board.pop()
        self.assertLessEqual(table.entry_count, table.capacity)
        self.assertEqual(table.capacity, 4)

        default = TranspositionTable()
        self.assertEqual(default.capacity, DEFAULT_TT_CAPACITY)
        self.assertLess(default.approximate_max_bytes, 64 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
