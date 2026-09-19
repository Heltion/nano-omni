"""Tests for compact benchmark tables using in-memory measurement fixtures."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("table_readme", ROOT / "hooks/readme.py")
assert SPEC is not None and SPEC.loader is not None
readme = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readme)


def report(seconds):
    return {
        "status": "complete",
        "mean_seconds": seconds,
        "latency_seconds": [seconds],
        "repetitions": 1,
        "workload": {
            "pipeline": "h3_fl2va", "mode": "fl2va", "width": 1344,
            "height": 768, "requested_frames": 192, "frames": 192,
            "video_latent_frames": 57, "audio_latent_frames": 320,
            "steps": 4, "seed": 0, "lora": "turbo_4step",
            "lora_strength": 1.0, "video_shift": 12.0, "audio_shift": 3.0,
        },
    }


class ReadmeTableTests(unittest.TestCase):
    def test_single_run_comparison(self):
        self.assertEqual(readme.speedup(report(12), report(10)), "1.20x")

    def test_single_run_slowdown(self):
        self.assertEqual(readme.speedup(report(10), report(12)), "0.83x")

    def test_incomplete_comparison(self):
        actual = report(10)
        actual["status"] = "incomplete"
        self.assertEqual(readme.speedup(report(12), actual), readme.MISSING)

    def test_invalid_latency(self):
        for value in (0, -1, True, float("nan"), float("inf"), None):
            with self.subTest(value=value):
                self.assertEqual(readme.speedup(report(12), report(value)), readme.MISSING)

    def test_mismatched_workload(self):
        actual = report(10)
        actual["workload"]["seed"] = 1
        self.assertEqual(readme.speedup(report(12), actual), readme.MISSING)

    def test_missing_workload_field(self):
        actual = report(10)
        del actual["workload"]["seed"]
        self.assertEqual(readme.speedup(report(12), actual), readme.MISSING)

    def test_compact_tables_include_core_profile_metrics(self):
        paths = []

        def load(path, **kwargs):
            paths.append(path.name)
            if path.name == "error.json":
                return {"video_psnr_db": 20.5, "video_ssim": 0.75}
            data = report(12 if path.name == "comfy.json" else 10)
            data.update(tensor_percent=60.0, kernel_percent=80.0, gpu_percent=90.0,
                        profile={"capture": {"gpu_complete": True}})
            return data

        with patch.object(readme, "load", side_effect=load):
            text = readme.render()
        self.assertIn("nano-nsys.json", paths)
        self.assertIn("| Target | Workload | Nano (s) | Tensor% | Kernel% | GPU% | ComfyUI (s) | Speedup |", text)
        self.assertIn("| Routing | Precision | Total (s) | Tensor% | Kernel% | GPU% | PSNR (dB) | SSIM |", text)
        self.assertIn("| 10.000 | 60.00 | 80.00 | 90.00 | 12.000 | 1.20x |", text)
        self.assertIn("| 20.500 | 0.750000 |", text)
        self.assertNotIn("| N |", text)
        self.assertNotIn("| Status |", text)
        self.assertEqual(text.count("| Tensor% | Kernel% | GPU% |"), 2)
        self.assertLess(len(text.splitlines()), 55)

    def test_identity_quality_format(self):
        identity = {"video_psnr_db": "Infinity", "video_ssim": 1.0}
        self.assertEqual(readme.quality_number(identity, "video_psnr_db", 3), "∞")
        self.assertEqual(readme.quality_number(identity, "video_ssim", 6), "1.000000")

    def test_inputs_are_not_modified(self):
        left, right = report(12), report(10)
        before = copy.deepcopy((left, right))
        readme.speedup(left, right)
        self.assertEqual((left, right), before)

    def test_generation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(readme, "ROOT", Path(directory)):
                readme.main()
                path = Path(directory) / "README.md"
                stamp = path.stat().st_mtime_ns
                readme.main()
                self.assertEqual(path.stat().st_mtime_ns, stamp)
                self.assertEqual(path.read_text(encoding="utf-8"), readme.render())


if __name__ == "__main__":
    unittest.main()
