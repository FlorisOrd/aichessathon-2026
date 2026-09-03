import hashlib
import json
import unittest

from tools.build_position_suite import POSITIONS_DIRECTORY, PositionCandidate
from tools.build_promotion_suites import PROMOTION_SPECS, validate_suite


class CommittedPromotionSuiteTests(unittest.TestCase):
    def test_committed_suites_match_metadata_and_are_disjoint(self) -> None:
        metadata_path = POSITIONS_DIRECTORY / "promotion-suite-metadata.json"
        metadata = json.loads(metadata_path.read_bytes())
        all_fens: set[str] = set()

        for spec in PROMOTION_SPECS:
            suite = metadata["suites"][spec.name]
            positions = [PositionCandidate(**position) for position in suite["positions"]]
            path = POSITIONS_DIRECTORY / spec.filename
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), suite["sha256"])
            suite_fens = validate_suite(spec, path, positions, suite["sha256"])
            self.assertTrue(all_fens.isdisjoint(suite_fens))
            all_fens.update(suite_fens)

            source_window = suite["source_window"]
            self.assertEqual(source_window["record_start"], spec.record_start)
            self.assertEqual(source_window["record_end"], spec.record_end)
            self.assertEqual(source_window["records_examined"], 250_000)

        self.assertEqual(len(all_fens), 288)


if __name__ == "__main__":
    unittest.main()
