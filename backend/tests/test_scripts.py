"""Dataset split integration checks using temporary generated images."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts import split_dataset


class DatasetScriptTests(unittest.TestCase):
    def test_split_cli_writes_training_layout_reproducibly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "faces"
            first_output = root / "first"
            second_output = root / "second"
            rng = np.random.default_rng(12)
            for label in ("real", "fake"):
                folder = source / label
                folder.mkdir(parents=True)
                for index in range(8):
                    image = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
                    self.assertTrue(cv2.imwrite(str(folder / f"{index}.jpg"), image))

            for destination in (first_output, second_output):
                with patch.object(sys, "argv", ["split_dataset", "--input-dir", str(source), "--output-dir", str(destination), "--seed", "17"]):
                    split_dataset.main()

            for split in ("train", "val", "test"):
                for label in ("real", "fake"):
                    first = sorted(path.name for path in (first_output / split / label).glob("*.jpg"))
                    second = sorted(path.name for path in (second_output / split / label).glob("*.jpg"))
                    self.assertEqual(first, second)
            self.assertTrue((first_output / "train" / "real").is_dir())
            self.assertTrue((first_output / "val" / "fake").is_dir())
            self.assertTrue((first_output / "test" / "real").is_dir())


if __name__ == "__main__":
    unittest.main()
