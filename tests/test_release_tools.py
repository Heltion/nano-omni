"""Dependency-free tests for release reporting; these do not execute GPU kernels."""

import ast
import copy
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_readme", ROOT / "hooks/readme.py")
assert SPEC is not None and SPEC.loader is not None
readme = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readme)

COMFY_SPEC = importlib.util.spec_from_file_location(
    "release_comfy_runner", ROOT / "scripts/run/comfy-ui.py"
)
assert COMFY_SPEC is not None and COMFY_SPEC.loader is not None
comfy_runner = importlib.util.module_from_spec(COMFY_SPEC)
sys.modules[COMFY_SPEC.name] = comfy_runner
COMFY_SPEC.loader.exec_module(comfy_runner)


def measurement(seconds=10.0):
    return {
        "status": "complete",
        "mean_seconds": seconds,
        "latency_seconds": [seconds],
        "repetitions": 1,
        "workload": {
            "pipeline": "h3_fl2va",
            "mode": "fl2va",
            "width": 1344,
            "height": 768,
            "requested_frames": 192,
            "frames": 197,
            "video_latent_frames": 49,
            "audio_latent_frames": 192,
            "steps": 4,
            "seed": 0,
            "lora": "turbo_4step",
            "lora_strength": 1.0,
            "video_shift": 12.0,
            "audio_shift": 3.0,
        },
    }


def runner_functions(root):
    """Load only host reporting functions, without importing CUDA or TileLang."""
    source = ROOT / "scripts/run/nano-omni.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"same_run_metadata", "read_metadata", "command", "run_parent"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    namespace = {
        "Path": Path, "json": json, "statistics": statistics, "os": os, "sys": sys,
        "ROOT": root, "CHILD": "NANO_OMNI_RUN_CHILD",
        "uuid": uuid,
        "timing_statistics": lambda samples: {
            "mean_seconds": statistics.mean(samples),
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
            "range_seconds": max(samples) - min(samples),
            "sample_stdev_seconds": (
                statistics.stdev(samples) if len(samples) > 1 else None
            ),
        },
        "check_gpu_temperature": lambda _: None,
        "merge_measurements": merge_measurements,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


def merge_measurements(previous, current):
    source = ROOT / "scripts/run/common.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("merge_measurements", "timing_statistics")
    ]
    namespace = {"statistics": statistics}
    exec(compile(ast.Module(nodes, []), str(source), "exec"), namespace)
    return namespace["merge_measurements"](previous, current)


class ReadmeTests(unittest.TestCase):
    def test_finite_numbers_exclude_booleans(self):
        for value in (True, False, float("nan"), float("inf"), -float("inf"), "12"):
            with self.subTest(value=value):
                self.assertFalse(readme.is_number(value))
        self.assertTrue(readme.is_number(12))
        self.assertTrue(readme.is_number(0.5))

    def test_successful_comparison(self):
        baseline, result = measurement(12), measurement(10)
        baseline["latency_seconds"] = [12, 12]
        result["latency_seconds"] = [10, 10]
        self.assertEqual(readme.speedup(baseline, result), "1.20x")
        self.assertEqual(readme.saved_seconds(baseline, result), "2.000")

    def test_single_pair_reports_the_observed_speedup(self):
        self.assertEqual(readme.speedup(measurement(12), measurement(10)), "1.20x")
        self.assertEqual(readme.saved_seconds(measurement(12), measurement(10)), "2.000")

    def test_multi_sample_summary_and_paired_ratio(self):
        baseline, result = measurement(12), measurement(10)
        baseline.update(latency_seconds=[12.0, 14.0], repetitions=2)
        result.update(latency_seconds=[10.0, 10.0], repetitions=2)
        self.assertEqual(readme.timing_summary(baseline), "13.000 [12.000, 14.000]")
        self.assertEqual(readme.saved_seconds(baseline, result), "3.000")
        self.assertEqual(readme.paired_ratio(baseline, result), "1.30x [1.20x, 1.40x]")

    def test_slowdown_is_not_hidden(self):
        baseline, result = measurement(10), measurement(12)
        baseline["latency_seconds"] = [10, 10]
        result["latency_seconds"] = [12, 12]
        self.assertEqual(readme.speedup(baseline, result), "0.83x")
        self.assertEqual(readme.saved_seconds(baseline, result), "-2.000")

    def test_invalid_times_do_not_produce_a_ratio(self):
        for value in (0, -1, True, float("nan"), float("inf"), None):
            with self.subTest(value=value):
                self.assertEqual(readme.speedup(measurement(), measurement(value)), readme.MISSING)
                self.assertEqual(readme.saved_seconds(measurement(value), measurement()), readme.MISSING)

    def test_failed_and_incomplete_reports_are_excluded(self):
        for status in ("failed", "incomplete", "running", None):
            report = measurement()
            report["status"] = status
            self.assertEqual(readme.speedup(report, measurement()), readme.MISSING)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                self.assertIsNone(readme.load(path, require_complete=True))

    def test_geometry_mismatch_is_excluded(self):
        for field in readme.WORKLOAD_FIELDS:
            changed = measurement()
            changed["workload"][field] += 1
            self.assertFalse(readme.comparable(measurement(), changed))
        changed = measurement()
        changed["workload"]["width"] = True
        self.assertEqual(readme.workload(changed), readme.MISSING)

    def test_common_optional_workload_fields_must_match(self):
        left, right = measurement(), measurement()
        left["workload"]["mode"], right["workload"]["mode"] = "t2va", "fl2va"
        self.assertFalse(readme.comparable(left, right))

    def test_missing_semantic_workload_field_is_rejected(self):
        baseline, result = measurement(), measurement()
        del result["workload"]["seed"]
        self.assertFalse(readme.comparable(baseline, result))

    def test_missing_files_and_invalid_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            self.assertIsNone(readme.load(path))
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(TypeError):
                readme.load(path)

    def test_quality_infinity_is_psnr_only(self):
        report = {"video_psnr_db": "Infinity", "video_ssim": "Infinity"}
        self.assertEqual(readme.quality_number(report, "video_psnr_db", 3), "∞")
        self.assertEqual(readme.quality_number(report, "video_ssim", 3), readme.MISSING)

    def test_gpu_coverage_is_visible(self):
        report = {"gpu_percent": 50.0, "profile": {"capture": {"gpu_complete": False}}}
        self.assertEqual(readme.gpu_percent(report), "≥50.00")
        report["profile"]["capture"]["gpu_complete"] = True
        self.assertEqual(readme.gpu_percent(report), "50.00")
        report.pop("profile")
        self.assertEqual(readme.gpu_percent(report), "50.00?")

    def test_sample_count_requires_positive_integer(self):
        self.assertEqual(readme.samples(measurement()), "1")
        for value in (0, -1, True, 1.5, None):
            self.assertEqual(readme.samples({"repetitions": value}), readme.MISSING)

    def test_render_uses_corrected_sol_measurements_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(readme, "ROOT", Path(directory)):
            text = readme.render()
            self.assertIn("# Nano-Omni", text)
            self.assertNotIn("Release status", text)
            self.assertNotIn("| N |", text)
            self.assertNotIn("| Status |", text)
            self.assertNotIn("known bug", text)
            readme.main()
            path = Path(directory) / "README.md"
            before = path.stat().st_mtime_ns
            readme.main()
            self.assertEqual(path.read_text(encoding="utf-8"), text)
            self.assertEqual(path.stat().st_mtime_ns, before)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.functions = runner_functions(self.root)
        self.workload = measurement()["workload"]
        self.metadata = {
            "workload": self.workload,
            "tops": {"bf16": 123},
            "memory": {"peak_rss_bytes": 100, "trims": 0},
            "environment": {"gpu": "fixture", "python": "fixture"},
        }
        self.output = self.root / "nano.mp4"
        self.config = types.SimpleNamespace(
            output=self.output, pipeline="fixture", maximum_gpu_temperature_celsius=None,
            quality_reference=None, model_dump=lambda **_: {"pipeline": "fixture"},
        )
        self.functions["pipeline_for"] = lambda _: types.SimpleNamespace(workload=lambda: self.workload)
        self.functions["write_json"] = lambda path, value: path.write_text(json.dumps(value), encoding="utf-8")

    def run_parent(self, repetitions=1, warmup=0):
        self.functions["run_parent"](
            self.root / "config.yaml",
            [],
            self.config,
            warmup,
            repetitions,
            False,
            None,
            False,
        )

    def test_memory_changes_are_not_workload_changes(self):
        other = copy.deepcopy(self.metadata)
        other["memory"] = {"peak_rss_bytes": 999, "trims": 7}
        self.assertTrue(self.functions["same_run_metadata"](self.metadata, other))
        for field in ("workload", "tops", "environment"):
            changed = copy.deepcopy(other)
            changed[field] = {"changed": True}
            self.assertFalse(self.functions["same_run_metadata"](self.metadata, changed))

    def test_repetitions_keep_each_memory_sample(self):
        calls = []

        def invoke(command, log, **kwargs):
            current = copy.deepcopy(self.metadata)
            current["memory"]["peak_rss_bytes"] += len(calls)
            calls.append(current)
            log.write_text("\n".join(name + "=" + json.dumps(value) for name, value in current.items()), encoding="utf-8")
            return 0, float(len(calls))

        self.functions["invoke"] = invoke
        self.run_parent(repetitions=2)
        result = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["latency_seconds"], [1.0, 2.0])
        self.assertEqual(result["mean_seconds"], 1.5)
        self.assertEqual([item["peak_rss_bytes"] for item in result["memory_samples"]], [100, 101])

    def test_launch_exception_invalidates_previous_success(self):
        self.output.with_suffix(".json").write_text(json.dumps(measurement()))

        def invoke(*args, **kwargs):
            raise OSError("fixture launch failure")

        self.functions["invoke"] = invoke
        with self.assertRaises(OSError):
            self.run_parent()
        self.assertEqual(json.loads(self.output.with_suffix(".json").read_text())["status"], "incomplete")

    def test_failed_compile_warmup_invalidates_previous_success(self):
        self.output.with_suffix(".json").write_text(json.dumps(measurement()))
        self.functions["invoke"] = lambda *args, **kwargs: (1, 0.5)
        with self.assertRaises(RuntimeError):
            self.run_parent(warmup=1)
        self.assertNotEqual(json.loads(self.output.with_suffix(".json").read_text())["status"], "complete")

    def test_workload_change_still_fails(self):
        count = 0

        def invoke(command, log, **kwargs):
            nonlocal count
            current = copy.deepcopy(self.metadata)
            current["workload"]["steps"] += count
            count += 1
            log.write_text(
                "\n".join(name + "=" + json.dumps(value) for name, value in current.items()),
                encoding="utf-8",
            )
            return 0, 1.0

        self.functions["invoke"] = invoke
        with self.assertRaisesRegex(ValueError, "different workload metadata"):
            self.run_parent(repetitions=2)
        self.assertNotEqual(
            json.loads(self.output.with_suffix(".json").read_text())["status"],
            "complete",
        )

    def test_independent_samples_merge_only_with_matching_identity(self):
        previous = {
            **copy.deepcopy(self.metadata),
            "status": "complete",
            "run_id": "first",
            "resolved_config": {"pipeline": "fixture"},
            "latency_seconds": [2.0],
        }
        current = {
            **copy.deepcopy(self.metadata),
            "status": "complete",
            "run_id": "second",
            "resolved_config": {"pipeline": "fixture"},
            "latency_seconds": [4.0],
        }
        result = merge_measurements(previous, current)
        self.assertEqual(result["run_ids"], ["first", "second"])
        self.assertEqual(result["latency_seconds"], [2.0, 4.0])
        self.assertEqual(result["mean_seconds"], 3.0)
        changed = copy.deepcopy(current)
        changed["resolved_config"]["pipeline"] = "different"
        with self.assertRaisesRegex(AssertionError, "changed resolved_config"):
            merge_measurements(previous, changed)

    def test_append_temperature_rejection_preserves_previous_report(self):
        previous = measurement(12)
        self.output.with_suffix(".json").write_text(json.dumps(previous))

        def reject(_):
            raise RuntimeError("too hot")

        self.functions["check_gpu_temperature"] = reject
        with self.assertRaisesRegex(RuntimeError, "too hot"):
            self.functions["run_parent"](
                self.root / "config.yaml",
                [],
                self.config,
                0,
                1,
                False,
                None,
                True,
            )
        self.assertEqual(
            json.loads(self.output.with_suffix(".json").read_text()), previous
        )


class ComfyParameterTests(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = {
            "pipeline": "h3_fl2va",
            "attention": "dense",
            "attention_precision": "bf16",
            "lora": "turbo_4step",
            "steps": 4,
            "video_shift": None,
            "audio_shift": None,
            "seed": 0,
            "lora_strength": 1.0,
            "seconds": 4.0,
            "width": 1344,
            "height": 768,
            "first_frame": None,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_shift_overrides_reach_workload(self):
        config = self.config(video_shift=7.5, audio_shift=2.5, seed=9)
        parameters = comfy_runner.comfy_parameters(config)
        self.assertEqual((parameters.video_shift, parameters.audio_shift), (7.5, 2.5))
        workload = comfy_runner.workload(config)
        self.assertEqual(workload["seed"], 9)
        self.assertEqual(workload["video_shift"], 7.5)
        self.assertEqual(workload["audio_shift"], 2.5)

    def test_unsupported_attention_is_rejected(self):
        for attention, precision in (("sol", "bf16"), ("dense", "int8_fp8")):
            with (
                self.subTest(attention=attention, precision=precision),
                self.assertRaisesRegex(ValueError, "only dense BF16"),
            ):
                comfy_runner.comfy_parameters(
                    self.config(attention=attention, attention_precision=precision)
                )

    def test_custom_lora_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "custom LoRA"):
            comfy_runner.comfy_parameters(self.config(lora="models/custom.safetensors"))

    def test_ref2va_requires_its_fixed_lora_and_step_count(self):
        with self.assertRaisesRegex(ValueError, "fixed turbo_4step"):
            comfy_runner.comfy_parameters(
                self.config(pipeline="h3_ref2va", lora="turbo_8step")
            )
        with self.assertRaisesRegex(ValueError, "four steps"):
            comfy_runner.comfy_parameters(
                self.config(pipeline="h3_ref2va", lora="turbo_4step", steps=8)
            )

    def test_only_current_run_outputs_are_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comfy.mp4"
            stale = output.with_name("comfy_00001.mp4")
            changed = output.with_name("comfy_00002.mp4")
            stale.write_bytes(b"old")
            changed.write_bytes(b"before")
            previous = {
                stale: (stale.stat().st_mtime_ns, stale.stat().st_size),
                changed: (changed.stat().st_mtime_ns, changed.stat().st_size),
            }
            changed.write_bytes(b"after-current-run")
            created = output.with_name("comfy_00003.mp4")
            created.write_bytes(b"new")
            self.assertEqual(
                set(comfy_runner.changed_outputs(output, previous)),
                {changed, created},
            )

    def test_comfy_child_metadata_is_read_from_its_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "comfy.log"
            log.write_text(
                'noise\nbackend={"implementation":"fixture"}\n'
                'environment={"gpu":"fixture"}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                comfy_runner.read_child_metadata(log),
                {
                    "backend": {"implementation": "fixture"},
                    "environment": {"gpu": "fixture"},
                },
            )

    def test_fl2va_workflow_is_one_connected_prompt(self):
        graph = comfy_runner.fl2va_workflow(
            prompt="fixture",
            output_prefix="comfy",
            width=1344,
            height=768,
            seconds=8.0,
            checkpoint="model.safetensors",
            lora_name="lora.safetensors",
            lora_strength=1.0,
            steps=4,
            seed=0,
            video_shift=12.0,
            audio_shift=3.0,
            first_frame_name="first.png",
            last_frame_name=None,
        )
        self.assertEqual(graph["sample"]["inputs"]["latent_image"], ["conditioning", 1])
        self.assertEqual(graph["separate"]["inputs"]["av_latent"], ["sample", 0])
        self.assertEqual(graph["save"]["inputs"]["video"], ["video", 0])
        self.assertEqual(graph["save"]["inputs"]["format"], "mp4")
        self.assertEqual(graph["save"]["inputs"]["format.codec"], "h264")
        self.assertIn("lora", graph)
        self.assertIn("first_frame", graph)
        self.assertNotIn("last_frame", graph)

    def test_ref2va_workflow_routes_both_reference_audio_sources(self):
        graph = comfy_runner.ref2va_workflow(
            prompt="fixture",
            output_prefix="comfy",
            width=864,
            height=480,
            seconds=5.0,
            checkpoint="model.safetensors",
            lora_name="lora.safetensors",
            lora_strength=1.0,
            steps=4,
            seed=0,
            video_shift=12.0,
            audio_shift=3.0,
            reference_video_name="reference.mp4",
            reference_audio_name="reference.mp3",
        )
        inputs = graph["conditioning"]["inputs"]
        self.assertEqual(inputs["ref_videos.ref_video_0"], ["reference_components", 0])
        self.assertEqual(
            inputs["ref_video_audios.ref_video_audio_0"],
            ["reference_components", 1],
        )
        self.assertEqual(inputs["ref_audios.ref_audio_0"], ["reference_audio", 0])

    def test_monitor_thread_errors_are_raised_by_context_exit(self):
        monitor = comfy_runner.Monitor(1 << 60)
        monitor.thread = types.SimpleNamespace(join=lambda: None)
        monitor.error = RuntimeError("monitor failed")
        with self.assertRaisesRegex(RuntimeError, "monitor failed"):
            monitor.__exit__()

class DocumentationTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("release_docs", ROOT / "hooks/docs.py")
        assert spec is not None and spec.loader is not None
        self.docs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.docs)

    def test_new_performance_source_pair_is_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instance = root / "perf/new-instance"
            instance.mkdir(parents=True)
            english, chinese = instance / "SOURCES.md", instance / "SOURCES.zh.md"
            english.write_text("[中文](SOURCES.zh.md)", encoding="utf-8")
            chinese.write_text("[English](SOURCES.md)", encoding="utf-8")
            with patch.object(self.docs, "ROOT", root):
                self.assertEqual(self.docs.documentation_pairs(), [(english, chinese)])

    def test_one_sided_source_note_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instance = root / "perf/new-instance"
            instance.mkdir(parents=True)
            chinese = instance / "SOURCES.zh.md"
            chinese.write_text("fixture", encoding="utf-8")
            with patch.object(self.docs, "ROOT", root):
                self.assertEqual(
                    self.docs.documentation_pairs(),
                    [(instance / "SOURCES.md", chinese)],
                )

    def test_scripts_inventory_preserves_both_language_contracts(self):
        expected = {
            "scripts/run/config.py", "scripts/run/nano-omni.py",
            "scripts/run/comfy-ui.py", "scripts/run/common.py",
            "scripts/tune.py", "scripts/update.py",
        }
        self.docs.verify_inventory(
            ((ROOT / "docs/en/scripts.md",), (ROOT / "docs/zh/scripts.md",)),
            expected,
        )

    def test_new_guides_have_relative_counterpart_links(self):
        for name in ("getting-started.md", "benchmarking.md", "scripts.md"):
            english = ROOT / "docs/en" / name
            chinese = ROOT / "docs/zh" / name
            self.assertIn(f"](../zh/{name})", english.read_text(encoding="utf-8"))
            self.assertIn(f"](../en/{name})", chinese.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
