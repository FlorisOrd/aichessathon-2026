import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import chess

from tools.build_position_suite import (
    PHASE_TOTALS,
    PositionCandidate,
    all_buckets,
    bucket_quota,
    build_promotion_positions,
    build_suites,
    candidate_from_record,
    iter_jsonl,
    normalize_fen,
    phase_for,
    select_highest_depth_first_pv,
)


def make_candidate(index: int, bucket: tuple[str, str, bool, bool]) -> PositionCandidate:
    phase, side, capture, castling = bucket
    return PositionCandidate(
        fen=f"position-{index} {side} {castling} {capture} 0 1",
        eval_cp=index % 51,
        depth=20,
        knodes=100,
        first_pv="e2e4",
        category=phase,  # type: ignore[arg-type]
        side_to_move=side,  # type: ignore[arg-type]
        capture_available=capture,
        castling_rights=castling,
        piece_count=16,
        pawn_count=8,
        pawn_structure_sha256=f"pawn-{index}",
        source_record=index,
    )


class RecordParsingTests(unittest.TestCase):
    def test_normalizes_four_or_six_field_fen_without_history(self) -> None:
        four_fields = "8/8/8/8/8/8/4K3/7k w - -"
        self.assertEqual(normalize_fen(four_fields), f"{four_fields} 0 1")
        self.assertEqual(normalize_fen(f"{four_fields} 87 42"), f"{four_fields} 0 1")

    def test_selects_first_pv_from_highest_depth_evaluation(self) -> None:
        record = {
            "evals": [
                {"depth": 18, "knodes": 5, "pvs": [{"cp": 1, "line": "a2a3"}]},
                {
                    "depth": 24,
                    "knodes": 9,
                    "pvs": [
                        {"cp": -7, "line": "e2e4 e7e5"},
                        {"cp": 3, "line": "d2d4 d7d5"},
                    ],
                },
            ]
        }
        self.assertEqual(select_highest_depth_first_pv(record), (-7, 24, 9, "e2e4 e7e5"))

    def test_rejects_mate_in_highest_depth_first_pv(self) -> None:
        record = {"evals": [{"depth": 30, "pvs": [{"mate": 2, "line": "f7f8"}]}]}
        with self.assertRaisesRegex(ValueError, "mate"):
            select_highest_depth_first_pv(record)

    def test_candidate_uses_valid_neutral_nonterminal_position(self) -> None:
        record = {
            "fen": chess.STARTING_FEN.rsplit(" ", 2)[0],
            "evals": [{"depth": 22, "knodes": 1000, "pvs": [{"cp": 12, "line": "e2e4"}]}],
        }
        candidate = candidate_from_record(record, 17)
        self.assertEqual(candidate.fen, chess.STARTING_FEN)
        self.assertEqual(candidate.category, "opening_high_material")
        self.assertEqual(candidate.source_record, 17)
        self.assertFalse(candidate.capture_available)
        self.assertTrue(candidate.castling_rights)

    def test_phase_boundaries_cover_all_three_categories(self) -> None:
        self.assertEqual(phase_for(chess.Board()), "opening_high_material")
        self.assertEqual(
            phase_for(chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")),
            "middlegame",
        )
        self.assertEqual(
            phase_for(chess.Board("4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1")),
            "endgame",
        )


class ReservoirTests(unittest.TestCase):
    def test_sampling_and_split_are_deterministic_balanced_and_disjoint(self) -> None:
        candidates: list[PositionCandidate] = []
        for bucket in all_buckets():
            for _ in range(bucket_quota(bucket) + 4):
                candidates.append(make_candidate(len(candidates) + 1, bucket))
        raw_records = [
            (index, (json.dumps({"index": index}) + "\n").encode())
            for index in range(1, len(candidates) + 1)
        ]

        def parsed(record: object, source_record: int) -> PositionCandidate:
            del record
            return candidates[source_record - 1]

        with patch("tools.build_position_suite.candidate_from_record", side_effect=parsed):
            first = build_suites(iter(raw_records), record_limit=len(raw_records), seed=2026)
            second = build_suites(iter(raw_records), record_limit=len(raw_records), seed=2026)

        self.assertEqual(first, second)
        self.assertEqual(len(first.development), 64)
        self.assertEqual(len(first.holdout), 32)
        self.assertEqual(len(first.fast), 16)
        self.assertEqual(
            {
                phase: sum(item.category == phase for item in first.development)
                for phase in PHASE_TOTALS
            },
            {"opening_high_material": 16, "middlegame": 32, "endgame": 16},
        )
        development_fens = {item.fen for item in first.development}
        holdout_fens = {item.fen for item in first.holdout}
        self.assertTrue(development_fens.isdisjoint(holdout_fens))
        self.assertTrue({item.fen for item in first.fast}.issubset(development_fens))
        self.assertEqual(
            sum(item.side_to_move == "white" for item in first.development),
            32,
        )
        self.assertEqual(
            sum(item.capture_available for item in first.development),
            32,
        )

    def test_explicit_start_record_selects_exact_window(self) -> None:
        candidates: dict[int, PositionCandidate] = {}
        for bucket in all_buckets():
            for _ in range(bucket_quota(bucket)):
                source_record = len(candidates) + 6
                candidates[source_record] = make_candidate(source_record, bucket)
        raw_records = [
            (index, (json.dumps({"index": index}) + "\n").encode())
            for index in range(1, len(candidates) + 6)
        ]

        def parsed(record: object, source_record: int) -> PositionCandidate:
            del record
            return candidates[source_record]

        with patch("tools.build_position_suite.candidate_from_record", side_effect=parsed):
            built = build_suites(
                iter(raw_records),
                record_start=6,
                record_limit=len(candidates),
                seed=2026,
            )

        self.assertEqual(built.stream.record_start, 6)
        self.assertEqual(built.stream.record_end, len(candidates) + 5)
        self.assertEqual(built.stream.records_examined, len(candidates))
        self.assertTrue(
            all(position.source_record >= 6 for position in built.development + built.holdout)
        )

    def test_promotion_sampling_is_deterministic_and_balanced(self) -> None:
        candidates: list[PositionCandidate] = []
        for bucket in all_buckets():
            for _ in range(bucket_quota(bucket) + 2):
                candidates.append(make_candidate(len(candidates) + 1, bucket))
        raw_records = [
            (index, (json.dumps({"index": index}) + "\n").encode())
            for index in range(1, len(candidates) + 1)
        ]

        def parsed(record: object, source_record: int) -> PositionCandidate:
            del record
            return candidates[source_record - 1]

        with patch("tools.build_position_suite.candidate_from_record", side_effect=parsed):
            first, first_stream = build_promotion_positions(
                iter(raw_records), record_start=1, record_limit=len(raw_records), seed=2026
            )
            second, second_stream = build_promotion_positions(
                iter(raw_records), record_start=1, record_limit=len(raw_records), seed=2026
            )

        self.assertEqual(first, second)
        self.assertEqual(first_stream, second_stream)
        self.assertEqual(len(first), 96)
        self.assertEqual(len({item.pawn_structure_sha256 for item in first}), 96)
        self.assertEqual(sum(item.side_to_move == "white" for item in first), 48)
        self.assertEqual(sum(item.capture_available for item in first), 48)
        self.assertEqual(
            {phase: sum(item.category == phase for item in first) for phase in PHASE_TOTALS},
            dict(PHASE_TOTALS),
        )

    def test_plain_jsonl_stream_can_be_kept_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "evals.jsonl"
            source.write_bytes(b"{}\n")
            with ExitStack() as stack:
                self.assertEqual(list(iter_jsonl(str(source), stack)), [(1, b"{}\n")])


if __name__ == "__main__":
    unittest.main()
