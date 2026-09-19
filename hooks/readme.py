"""Generate the README from measurement JSON."""

import json
import math
import statistics
from pathlib import Path
from typing import TypeGuard, cast

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type Measurement = dict[str, JsonValue]

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = (
    ("T2VA", "h3-t2va-official-dense-bf16-4step"),
    ("FL2VA", "h3-fl2va-official-dense-bf16-4step"),
    ("REF2VA", "h3-ref2va-0.4mp-dense-bf16-4step"),
)
ATTENTION_STUDY = (
    ("dense", "bf16", "h3-fl2va-official-dense-bf16-4step"),
    ("sol", "bf16", "h3-fl2va-official-sol-bf16-4step"),
    ("dense", "int8-fp8", "h3-fl2va-official-dense-int8-fp8-4step"),
    ("sol", "int8-fp8", "h3-fl2va-official-sol-int8-fp8-4step"),
    ("dense", "nvfp4", "h3-fl2va-official-dense-nvfp4-4step"),
    ("sol", "nvfp4", "h3-fl2va-official-sol-nvfp4-4step"),
)
MISSING = "Missing"
WORKLOAD_FIELDS = ("width", "height", "requested_frames", "steps")
COMPARISON_FIELDS = (
    *WORKLOAD_FIELDS,
    "pipeline",
    "mode",
    "frames",
    "video_latent_frames",
    "audio_latent_frames",
    "seed",
    "lora",
    "lora_strength",
    "video_shift",
    "audio_shift",
)


def is_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def load(path: Path, *, require_complete: bool = False) -> Measurement | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Measurement must be a JSON object: {path}")
    if require_complete and value.get("status") != "complete":
        return None
    return cast(Measurement, value)


def number(data: Measurement | None, name: str, digits: int = 2) -> str:
    value = None if data is None else data.get(name)
    if not is_number(value) or (name == "mean_seconds" and value <= 0):
        return MISSING
    return f"{value:.{digits}f}"


def quality_number(data: Measurement | None, name: str, digits: int) -> str:
    value = None if data is None else data.get(name)
    if name == "video_psnr_db" and value == "Infinity":
        return "∞"
    if not is_number(value):
        return MISSING
    return f"{value:.{digits}f}"


def workload_fields(data: Measurement | None) -> Measurement | None:
    value = None if data is None else data.get("workload")
    if not isinstance(value, dict):
        return None
    for name in WORKLOAD_FIELDS:
        item = value.get(name)
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            return None
    return value


def workload(data: Measurement | None) -> str:
    value = workload_fields(data)
    if value is None:
        return MISSING
    return (
        f"{value['width']}x{value['height']}, {value['requested_frames']} frames, "
        f"{value['steps']} steps"
    )


def comparable(baseline: Measurement | None, result: Measurement | None) -> bool:
    """Require successful reports with matching workload fields."""
    if baseline is None or result is None:
        return False
    if baseline.get("status") != "complete" or result.get("status") != "complete":
        return False
    left, right = workload_fields(baseline), workload_fields(result)
    if left is None or right is None:
        return False
    return all(
        name in left and name in right and left[name] == right[name]
        for name in COMPARISON_FIELDS
    )


def profile_comparable(normal: Measurement | None, profiled: Measurement | None) -> bool:
    """Match an ordinary run to its profiler run."""
    if normal is None or profiled is None:
        return False
    if normal.get("status") != "complete" or profiled.get("status") != "complete":
        return False
    left, right = workload_fields(normal), workload_fields(profiled)
    if left is None or right is None:
        return False
    return all(left.get(name) == right.get(name) for name in WORKLOAD_FIELDS)


def comparison_times(
    baseline: Measurement | None, result: Measurement | None
) -> tuple[int | float, int | float] | None:
    if not comparable(baseline, result):
        return None
    assert baseline is not None and result is not None
    left_samples, right_samples = latencies(baseline), latencies(result)
    if (
        left_samples is None
        or right_samples is None
        or len(left_samples) != len(right_samples)
    ):
        return None
    return statistics.median(left_samples), statistics.median(right_samples)


def speedup(baseline: Measurement | None, result: Measurement | None) -> str:
    times = comparison_times(baseline, result)
    return MISSING if times is None else f"{times[0] / times[1]:.2f}x"


def saved_seconds(baseline: Measurement | None, result: Measurement | None) -> str:
    times = comparison_times(baseline, result)
    return MISSING if times is None else f"{times[0] - times[1]:.3f}"


def samples(data: Measurement | None) -> str:
    value = None if data is None else data.get("repetitions")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return MISSING
    return str(value)


def latencies(data: Measurement | None) -> list[int | float] | None:
    values = None if data is None else data.get("latency_seconds")
    if not isinstance(values, list) or not values or not all(
        is_number(value) and value > 0 for value in values
    ):
        return None
    return cast("list[int | float]", values)


def timing_summary(data: Measurement | None) -> str:
    values = latencies(data)
    if values is None:
        return MISSING
    median = statistics.median(values)
    return f"{median:.3f} [{min(values):.3f}, {max(values):.3f}]"


def paired_ratio(baseline: Measurement | None, result: Measurement | None) -> str:
    if not comparable(baseline, result):
        return MISSING
    left, right = latencies(baseline), latencies(result)
    if left is None or right is None or len(left) < 2 or len(left) != len(right):
        return MISSING
    ratios = [
        baseline_time / result_time
        for baseline_time, result_time in zip(left, right, strict=True)
    ]
    return (
        f"{statistics.median(ratios):.2f}x "
        f"[{min(ratios):.2f}x, {max(ratios):.2f}x]"
    )


def gpu_percent(data: Measurement | None) -> str:
    value = number(data, "gpu_percent")
    if data is None or value == MISSING:
        return value
    profile = data.get("profile")
    capture = profile.get("capture") if isinstance(profile, dict) else None
    complete = capture.get("gpu_complete") if isinstance(capture, dict) else None
    if complete is False:
        return f"≥{value}"
    return value if complete is True else f"{value}?"


def render() -> str:
    lines = [
        "<!-- Generated by hooks/readme.py. -->",
        "<!-- cspell:words PSNR SSIM -->",
        "# Nano-Omni",
        "",
        "TileLang inference for MiniMax H3 on memory-constrained NVIDIA GPUs, supporting text-to-audio/video (T2VA), first/last-frame conditioning (FL2VA), and reference-audio/video conditioning (REF2VA).",
        "",
        "[Getting started](docs/en/getting-started.md) · [中文入门](docs/zh/getting-started.md) · [Documentation](docs/README.md)",
        "",
        "## Run",
        "",
        "```powershell",
        "git clone https://github.com/Heltion/nano-omni.git",
        "cd nano-omni",
        "uv sync --locked",
        "uv run scripts/run/nano-omni.py --config perf/h3-fl2va-official-dense-bf16-4step/config.yaml --output outputs/example.mp4",
        "```",
        "",
        "Models download automatically into `models/`. Run from the repository root; see the setup guide for requirements and configuration.",
        "",
        "## Performance",
        "",
        "RTX 5060 Ti on Windows. Dense BF16 DiT attention; one complete process run per configuration. Speedup = ComfyUI / Nano-Omni. [Measurement details](docs/en/benchmarking.md).",
        "",
        "| Target | Workload | Nano (s) | Tensor% | Kernel% | GPU% | ComfyUI (s) | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, instance in INSTANCES:
        directory = ROOT / "perf" / instance
        normal = load(directory / "nano.json", require_complete=True)
        baseline = load(directory / "comfy.json", require_complete=True)
        profiled = load(directory / "nano-nsys.json", require_complete=True)
        if not profile_comparable(normal, profiled):
            profiled = None
        lines.append(
            f"| [{label}](perf/{instance}/config.yaml) | {workload(normal)} | "
            f"{number(normal, 'mean_seconds', 3)} | {number(normal, 'tensor_percent')} | "
            f"{number(profiled, 'kernel_percent')} | {gpu_percent(profiled)} | "
            f"{number(baseline, 'mean_seconds', 3)} | {speedup(baseline, normal)} |"
        )
    lines.extend(
        [
            "",
            "## FL2VA attention",
            "",
            "Same seed-0 input. PSNR and SSIM compare decoded video with Nano-Omni dense BF16; they measure agreement, not absolute quality or audio quality.",
            "",
            "| Routing | Precision | Total (s) | Tensor% | Kernel% | GPU% | PSNR (dB) | SSIM |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for routing, precision, instance in ATTENTION_STUDY:
        directory = ROOT / "perf" / instance
        normal = load(directory / "nano.json", require_complete=True)
        profiled = load(directory / "nano-nsys.json", require_complete=True)
        if not profile_comparable(normal, profiled):
            profiled = None
        error = load(directory / "error.json") if normal is not None else None
        lines.append(
            f"| {routing} | {precision} | {number(normal, 'mean_seconds', 3)} | "
            f"{number(normal, 'tensor_percent')} | "
            f"{number(profiled, 'kernel_percent')} | {gpu_percent(profiled)} | "
            f"{quality_number(error, 'video_psnr_db', 3)} | "
            f"{quality_number(error, 'video_ssim', 6)} |"
        )
    lines.extend(
        [
            "",
            "Run `scripts/measure.ps1` to measure the suite and regenerate this page. Detailed timing and profiling data live under `perf/`; see the [scripts guide](docs/en/scripts.md).",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    content = render()
    path = ROOT / "README.md"
    if not path.is_file() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
