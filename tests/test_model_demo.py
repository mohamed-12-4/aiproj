"""Check deterministic demonstration selection and render without model execution."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from model_demo import comparison_frame, select_examples


class DemoTests(unittest.TestCase):
    def test_midpoint_selection_is_independent_of_model_scores(self):
        dataset = SimpleNamespace(pairs=[(Path(f'test/{site}/{i}.png'), Path('mask.png'))
            for site, size in [('site1', 3), ('site22', 4)] for i in range(size)])
        self.assertEqual(select_examples(dataset), [1, 5])

    def test_comparison_has_video_geometry_and_real_prediction_panels(self):
        image = Image.new('RGB', (40, 20), '#203040')
        mask = np.ones((20, 40), dtype=np.int64)
        groups = np.array([4, 1])
        names = ['Public', 'LoRA', 'Rein++', '50/50']
        metrics = {name: dict(native_miou=.4, dead_iou=None) for name in names}
        frame = comparison_frame(image, mask, {name: mask for name in names}, groups,
                                 'site1', 'test.png', metrics)
        self.assertEqual(frame.size, (1920, 1080))
        self.assertEqual(frame.mode, 'RGB')


if __name__ == '__main__':
    unittest.main()
