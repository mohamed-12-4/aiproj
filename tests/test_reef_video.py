"""Small fixtures for moving-video geometry and consistent category colors."""
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from reef_video_demo import display_metadata, pad_frame, render


class ReefVideoTests(unittest.TestCase):
    def test_padding_preserves_original_frame_and_repeats_edge(self):
        array = np.random.default_rng(12).integers(0, 255, (540, 960, 3), dtype=np.uint8)
        actual = np.asarray(pad_frame(Image.fromarray(array)))
        self.assertEqual(actual.shape, (544, 960, 3))
        np.testing.assert_array_equal(actual[:540], array)
        np.testing.assert_array_equal(actual[540:], np.repeat(array[-1:], 4, axis=0))

    def test_shared_palette_and_synchronized_panel_geometry(self):
        lookup, palette, labels = display_metadata(Path(__file__).resolve().parents[1])
        self.assertEqual(lookup.shape, (96,))
        self.assertEqual(palette.shape, (40, 3))
        image = Image.new('RGB', (960, 540))
        pred = np.full((540, 960), 25, dtype=np.uint8)
        hero, comparison = render(image, [pred] * 5, lookup, palette, labels, [9, 13, 14], 1.)
        self.assertEqual(hero.size, (1920, 720))
        self.assertEqual(comparison.size, (1920, 1080))


if __name__ == '__main__':
    unittest.main()
