import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models import load_split_ids


class TestSplitArtifactValidation(unittest.TestCase):
    def _write_splits(self, path, train, validation, test):
        path.mkdir()
        for name, ids in (("train", train), ("validation", validation), ("test", test)):
            pd.DataFrame({"SK_ID_CURR": ids}).to_csv(path / f"{name}_ids.csv", index=False)

    def test_split_artifacts_require_disjoint_complete_population(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "splits"
            self._write_splits(folder, [1, 2], [3], [4, 5])
            splits = load_split_ids([1, 2, 3, 4, 5], folder)
            self.assertEqual([len(splits[k]) for k in ("train", "validation", "test")], [2, 1, 2])

    def test_overlap_and_population_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "splits"
            self._write_splits(folder, [1, 2], [2], [3, 4])
            with self.assertRaises(ValueError):
                load_split_ids([1, 2, 3, 4], folder)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "splits"
            self._write_splits(folder, [1], [2], [3])
            with self.assertRaises(ValueError):
                load_split_ids([1, 2, 3, 4], folder)

    def test_committed_phase8_artifacts_match_full_population(self):
        data_path = Path("data/processed/feature_master.csv")
        split_path = Path("data/processed/splits")
        if not data_path.exists() or not split_path.exists():
            self.skipTest("Local Home Credit feature data and frozen split artifacts are required")
        ids = pd.read_csv(data_path, usecols=["SK_ID_CURR"]).SK_ID_CURR
        splits = load_split_ids(ids, split_path)
        self.assertEqual([len(splits[k]) for k in ("train", "validation", "test")],
                         [215257, 46127, 46127])
        self.assertEqual(len(set.union(*(set(v) for v in splits.values()))), 307511)


if __name__ == "__main__":
    unittest.main()
