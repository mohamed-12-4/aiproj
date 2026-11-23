import unittest
import tempfile
import io
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from diagnose import class_counts, source_confusion, dead_row, coverage


class DiagnosticTests(unittest.TestCase):
    def test_original_categories_and_missed_denominators(self):
        original = np.array([[3, 20, 23, 32, 37], [0, 21, 6, 4, 5]], dtype=np.uint8)
        prediction = np.array([[0, 1, 2, 3, 3], [3, 3, 1, 2, 0]])
        matrix = source_confusion(original, prediction)
        row = dead_row(matrix)
        self.assertEqual(row["dead_pixels"], 5)
        self.assertEqual(row["dead_correct"], 2)
        self.assertEqual(row["dead_recall"], .4)
        self.assertEqual(row["dead_to_Healthy_fraction_all_dead"], .2)
        self.assertEqual(row["dead_to_Healthy_fraction_missed_dead"], 1 / 3)
        self.assertEqual(matrix[32, 3], 1)
        self.assertEqual(class_counts(original).tolist(), [1, 1, 1, 5, 2])

    def test_absent_dead_is_undefined(self):
        row = dead_row(source_confusion(np.array([[6, 4, 5]], dtype=np.uint8), np.array([[1, 2, 0]])))
        self.assertIsNone(row["dead_recall"])
        self.assertIsNone(row["dead_to_Background_fraction_missed_dead"])

    def test_coverage_reads_nested_parquet_and_counts_images(self):
        masks = [np.array([[3, 6, 4, 5], [0, 21, 27, 37]], dtype=np.uint8),
                 np.array([[20, 20, 6, 6], [23, 23, 32, 32]], dtype=np.uint8)]
        samples = []
        for i, mask in enumerate(masks):
            stream = io.BytesIO()
            Image.fromarray(mask).save(stream, format="PNG")
            samples.append({"image": {"path": f"image_{i}.png", "bytes": b"unused"},
                            "label": {"path": f"mask_{i}.png", "bytes": stream.getvalue()}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            pq.write_table(pa.Table.from_pylist(samples), root / "data/train-000.parquet")
            pq.write_table(pa.Table.from_pylist(samples[:1]), root / "data/validation-000.parquet")
            summary = coverage(root, 2, root / "report")
            classes = {r["class"]: r for r in summary["original"]["classes"]}
            self.assertEqual(summary["images"], 2)
            self.assertEqual(summary["original"]["valid_pixels"], 13)
            self.assertEqual(summary["original"]["ignored_pixels"], 3)
            self.assertEqual(classes["Dead"]["pixels"], 8)
            self.assertEqual(classes["Healthy"]["images_present"], 2)
            self.assertEqual(classes["Bleached"]["images_present"], 1)
            sources = json.loads((root / "report/validation_sources.json").read_text())
            self.assertEqual(sources[0]["image"], "image_0.png")


if __name__ == "__main__":
    unittest.main()
