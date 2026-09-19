"""Direct decoded-frame quality metrics using an in-memory video decoder."""

import ast
import contextlib
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class VideoMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        source = ROOT / "scripts/run/common.py"
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("measure_video_quality", "_video_frame_count")]
        cls.functions = {"Path": Path}
        exec(compile(ast.Module(nodes, []), str(source), "exec"), cls.functions)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def compare(self, left, right, *, same_path=False):
        import numpy as np
        import torch

        def frame(value):
            array = np.full((16, 16, 3), value, dtype=np.uint8)
            return types.SimpleNamespace(to_ndarray=lambda **_: array)

        def open_video(name):
            values = left if name == "actual.mp4" else right
            video = types.SimpleNamespace(decode=lambda **_: iter(frame(v) for v in values))
            return contextlib.nullcontext(video)

        decoder = types.SimpleNamespace(open=open_video)
        with patch.dict(sys.modules, {"av": decoder}), patch.object(torch.cuda, "is_available", return_value=False):
            return self.functions["measure_video_quality"](
                Path("actual.mp4"), Path("actual.mp4" if same_path else "reference.mp4")
            )

    def test_distinct_identical_videos(self):
        result = self.compare([80, 120], [80, 120])
        self.assertEqual(set(result), {"video_psnr_db", "video_ssim", "video_quality_frames"})
        self.assertEqual(result["video_psnr_db"], "Infinity")
        self.assertAlmostEqual(result["video_ssim"], 1.0, places=6)
        self.assertEqual(result["video_quality_frames"], 2)

    def test_self_reference(self):
        result = self.compare([80, 120], [], same_path=True)
        self.assertEqual(result, {"video_psnr_db": "Infinity", "video_ssim": 1.0, "video_quality_frames": 2})

    def test_psnr_uses_aggregate_error(self):
        result = self.compare([0, 128], [255, 128])
        self.assertAlmostEqual(result["video_psnr_db"], -10 * math.log10(0.5), places=5)
        self.assertAlmostEqual(result["video_ssim"], (1 + 0.01**2 / (1 + 0.01**2)) / 2, places=5)

    def test_different_frame_counts_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "frame counts differ"):
            self.compare([80, 120], [80])


if __name__ == "__main__":
    unittest.main()
