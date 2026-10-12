"""Tests for FolderState.promote_champion."""

import tempfile
import unittest
from pathlib import Path

from system.helper import FolderState


class ChampionPromotion(unittest.TestCase):
    def test_writes_code_and_record(self):
        with tempfile.TemporaryDirectory() as directory:
            state = FolderState(Path(directory))
            record = {"name": "candidate", "module": "champion/algorithm.py",
                      "objective": "measured objective", "derivation": {}}
            source = "UPDATE = None\nOBJECTIVE = None\nPARAMS = {}\n"
            state.promote_champion(record, source)
            self.assertEqual(state.read_champion(), record)
            self.assertEqual((Path(directory) / "champion/algorithm.py").read_text(), source)

    def test_invalid_code_leaves_champion_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            state = FolderState(Path(directory))
            record = {"name": "good", "module": "champion/algorithm.py"}
            source = "UPDATE = None\n"
            state.promote_champion(record, source)
            with self.assertRaises(SyntaxError):
                state.promote_champion({"name": "bad", "module": "champion/algorithm.py"},
                                       "def broken(\n")
            self.assertEqual(state.read_champion(), record)
            self.assertEqual((Path(directory) / "champion/algorithm.py").read_text(), source)

    def test_rejects_accuracy_in_record(self):
        with tempfile.TemporaryDirectory() as directory:
            state = FolderState(Path(directory))
            with self.assertRaises(ValueError):
                state.promote_champion(
                    {"module": "champion/algorithm.py", "accuracy": {}},
                    "UPDATE = None\n")


if __name__ == "__main__":
    unittest.main()
