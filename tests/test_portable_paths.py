"""Portable configuration paths and the published benchmark file set."""

import ast
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/run"))

from config import PATH_FIELDS, RunConfig, load, relative_path, save

SPEC = importlib.util.spec_from_file_location("portable_readme", ROOT / "hooks/readme.py")
assert SPEC is not None and SPEC.loader is not None
readme = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readme)


class PortablePathTests(unittest.TestCase):
    def config(self, **overrides):
        return RunConfig.model_validate({
            "pipeline": "h3_fl2va", "prompt": "fixture", "width": 256,
            "height": 256, "steps": 1, **overrides,
        })

    def test_both_separator_styles_are_relative(self):
        for separator in ("/", chr(92)):
            self.assertEqual(relative_path(separator.join(("inputs", "frame.png"))), Path("inputs/frame.png"))

    def test_rejects_native_and_foreign_roots(self):
        absolute = PurePosixPath("/") / "fixture" / "data"
        drive = PureWindowsPath("Z:" + chr(92)) / "fixture" / "data"
        unc = chr(92) * 2 + chr(92).join(("fixture", "share", "data"))
        for value in (str(absolute), str(drive), drive.as_posix(), unc, "Z:data", chr(92) + "data"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                relative_path(value)

    def test_rejects_home_and_parent_paths(self):
        for value in ("", ".", "..", "~/data", "~fixture/data", "inputs/../data", "../data", "data:stream", "data" + chr(0), 42):
            with self.subTest(value=value), self.assertRaises(ValueError):
                relative_path(value)

    def test_every_config_path_rejects_an_absolute_value(self):
        with tempfile.TemporaryDirectory() as directory:
            absolute = Path(directory) / "asset"
            for name in PATH_FIELDS:
                with self.subTest(field=name), self.assertRaises(ValueError) as raised:
                    self.config(**{name: absolute})
                self.assertNotIn(str(absolute), str(raised.exception))

    def test_custom_lora_follows_path_rules(self):
        self.assertEqual(self.config(lora="loras" + chr(92) + "custom.safetensors").lora, "loras/custom.safetensors")
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            self.config(lora=str(Path(directory) / "custom.safetensors"))
        for alias in (None, "none", "turbo_4step", "turbo_8step"):
            steps = 8 if alias == "turbo_8step" else 4
            self.assertEqual(self.config(lora=alias, width=1344, height=768, steps=steps).lora, alias)

    def test_json_and_yaml_use_forward_slashes(self):
        config = self.config(output=chr(92).join(("outputs", "movie.mp4")))
        self.assertEqual(config.model_dump(mode="json")["output"], "outputs/movie.mp4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save(config, path)
            self.assertEqual(load(path), config)
            self.assertIn("outputs/movie.mp4", path.read_text())

    def test_same_config_works_under_two_roots(self):
        config = self.config(prompt=None, prompt_file="inputs/prompt.txt")
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "inputs").mkdir()
                (root / "inputs/prompt.txt").write_text("portable fixture")
                self.assertEqual(config.read_prompt(root), "portable fixture")

    def test_child_output_argument_is_relative(self):
        source = ROOT / "scripts/run/nano-omni.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "command")
        import os
        namespace = {"Path": Path, "os": os, "sys": sys, "ROOT": ROOT}
        exec(compile(ast.Module([function], []), str(source), "exec"), namespace)
        command = namespace["command"](Path("config.yaml"), [], ROOT / "outputs/result.mp4")
        self.assertEqual(command[-2:], ["--output", "outputs/result.mp4"])

    def test_published_inputs_exist_and_configs_roundtrip(self):
        instances = {i for _, i in readme.INSTANCES} | {i for _, _, i in readme.ATTENTION_STUDY}
        for instance in instances:
            path = ROOT / "perf" / instance / "config.yaml"
            config = load(path)
            self.assertEqual(RunConfig.model_validate_json(config.model_dump_json()), config)
            for name in PATH_FIELDS:
                value = getattr(config, name)
                if value is not None and name not in ("output", "latent_output"):
                    with self.subTest(instance=instance, field=name):
                        self.assertTrue((ROOT / value).is_file())

    def test_published_reports_have_no_absolute_paths(self):
        path_fields = set(PATH_FIELDS) | {"actual", "reference", "sqlite", "lora"}
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in path_fields and isinstance(item, str):
                        self.assertFalse(relative_path(item).is_absolute())
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        for path in (ROOT / "perf").rglob("*.json"):
            with self.subTest(path=path.relative_to(ROOT)):
                check(json.loads(path.read_text()))

    def test_perf_tracks_only_table_inputs_and_outputs(self):
        main = {i for _, i in readme.INSTANCES}
        attention = {i for _, _, i in readme.ATTENTION_STUDY}
        allowed = {Path("perf/.gitignore"), Path("perf/hardware.json")}
        for instance in main | attention:
            directory = Path("perf") / instance
            allowed.update(directory / name for name in ("config.yaml", "SOURCES.md", "SOURCES.zh.md", "nano.json", "nano-nsys.json", "nano.mp4"))
            if instance in main:
                allowed.update(directory / name for name in ("comfy.json", "comfy.mp4"))
            if instance in attention:
                allowed.add(directory / "error.json")
            config = load(ROOT / directory / "config.yaml")
            allowed.update(value for name in PATH_FIELDS if name not in ("output", "latent_output") and (value := getattr(config, name)) is not None)
        tracked = subprocess.check_output(["git", "ls-files", "perf"], cwd=ROOT, text=True).splitlines()
        actual = {Path(p) for p in tracked if (ROOT / p).is_file()}
        self.assertEqual(actual, allowed)
        for instance in main | attention:
            for name in ("kernels.csv", "tuning.csv", "scratch.log"):
                result = subprocess.run(["git", "check-ignore", "--no-index", "-q", f"perf/{instance}/{name}"], cwd=ROOT)
                self.assertEqual(result.returncode, 0)

    def test_tracked_text_has_no_machine_paths(self):
        # Check stored text, not absolute paths resolved at runtime for local I/O.
        patterns = (
            re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\[A-Za-z0-9_.-]+[\\])"),
            re.compile(r"(?:/(?:Users|home)/|[\\/](?:Users|Documents and Settings)[\\/]|/mnt/[a-z]/)", re.I),
        )
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        for name in tracked:
            path = ROOT / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                continue
            for pattern in patterns:
                self.assertIsNone(pattern.search(text), name)


if __name__ == "__main__":
    unittest.main()
