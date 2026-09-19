import contextlib
import itertools
import json
import sqlite3
import statistics

# cspell:ignore noheader nounits
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import config
from config import RunConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nano_omni.core.kernel import MmaType
from nano_omni.core.runtime.synchronization import Synchronization

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Interval:
    start: int
    end: int


class _CopyBreakdown(TypedDict):
    kind: str
    source: str
    destination: str
    count: int
    bytes: int
    seconds: float
    without_kernel_seconds: float
    gigabytes_per_second: float | None


class CaptureMetrics(TypedDict):
    seconds: float
    kernel_union_seconds: float
    gpu_union_seconds: float
    video_union_seconds: float
    gpu_complete: bool
    video_api_calls: int
    ignored_video_ranges: int
    missing_video_ranges: int
    gpu_coverage_reason: str
    bubble_seconds: float
    bubble_percent: float


class TraceAnalysis(TypedDict):
    sqlite: str
    definition: str
    transfers: dict[str, object]
    capture: CaptureMetrics
    kernel_span: dict[str, float]
    outside_kernel_span_seconds: float
    largest_gaps: list[dict[str, object]]


def union(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for interval in sorted(intervals, key=lambda item: item.start):
        if merged and interval.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = Interval(previous.start, max(previous.end, interval.end))
        else:
            merged.append(interval)
    return merged


def duration(intervals: list[Interval]) -> int:
    return sum(interval.end - interval.start for interval in intervals)


def clipped_duration(intervals: list[Interval], start: int, end: int) -> int:
    return duration(
        union(
            [
                Interval(max(start, item.start), min(end, item.end))
                for item in intervals
                if item.end > start and item.start < end
            ]
        )
    )


def extract_process_streams(path: Path, directory: Path) -> list[dict[str, object]]:
    """Recover target logs captured inside the trace, separate from profiler output."""
    with contextlib.closing(sqlite3.connect(path)) as connection:
        connection.text_factory = lambda value: value.decode("utf-8", errors="replace")
        tables: set[str] = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "ProcessStreams" not in tables:
            return []
        rows: list[tuple[int, str, str]] = connection.execute(
            "SELECT p.globalPid, f.value, c.value FROM ProcessStreams p "
            "JOIN StringIds f ON f.id=p.filenameId "
            "JOIN StringIds c ON c.id=p.contentId ORDER BY p.globalPid, p.filenameId"
        ).fetchall()
    records: list[dict[str, object]] = []
    directory.mkdir(parents=True, exist_ok=True)
    for pid, filename, content in rows:
        basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        stream = basename.split("_", 1)[0].split(".", 1)[0]
        if stream not in ("stdout", "stderr"):
            continue
        artifact = f"profile-process-{pid}-{len(records)}-{stream}.log"
        (directory / artifact).write_text(content, encoding="utf-8")
        records.append({"global_pid": pid, "stream": stream, "artifact": artifact})
    return records


def analyze(
    path: Path, largest: int = 20, *, expects_video: bool = True
) -> TraceAnalysis:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        connection.text_factory = lambda value: value.decode("utf-8", errors="replace")
        tables: set[str] = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        strings: dict[int, str] = dict(
            connection.execute("SELECT id, value FROM StringIds")
        )
        kernel_rows: list[tuple[int, int, int, int]] = connection.execute(
            "SELECT start, end, shortName, globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchall()
        kernels = [Interval(start, end) for start, end, _, _ in kernel_rows]
        copy_rows: list[tuple[int, int, int, int, int, int, int]] = (
            connection.execute(
                "SELECT start, end, bytes, copyKind, srcKind, dstKind, streamId "
                "FROM CUPTI_ACTIVITY_KIND_MEMCPY"
            ).fetchall()
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables
            else []
        )
        copies = [Interval(start, end) for start, end, *_ in copy_rows]
        memsets: list[Interval] = (
            [
                Interval(start, end)
                for start, end in connection.execute(
                    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET"
                )
            ]
            if "CUPTI_ACTIVITY_KIND_MEMSET" in tables
            else []
        )
        runtime: list[tuple[Interval, str]] = [
            (Interval(start, end), strings.get(name_id, str(name_id)))
            for start, end, name_id in connection.execute(
                "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            )
        ]
        names = [
            (start, end, strings.get(name_id, str(name_id)))
            for start, end, name_id, _ in kernel_rows
        ]
        metadata: dict[str, str] = dict(
            connection.execute("SELECT name, value FROM META_DATA_CAPTURE")
        )
        video: list[Interval] = []
        ignored_video_ranges = 0
        if "GPU_VIDEO_ENGINE_WORKLOAD" in tables:
            target_pids = {pid for _, _, _, pid in kernel_rows}
            workloads: list[tuple[int, int, int]] = connection.execute(
                "SELECT start, end, globalPid FROM GPU_VIDEO_ENGINE_WORKLOAD"
            ).fetchall()
            video = [
                Interval(start, end)
                for start, end, pid in workloads
                if pid in target_pids
            ]
            ignored_video_ranges = len(workloads) - len(video)
        video_calls = 0
        for table in ("NVVIDEO_ENCODER_API", "NVVIDEO_DECODER_API"):
            if table in tables:
                video_calls += sum(
                    1
                    for (name,) in connection.execute(f"SELECT nameId FROM {table}")
                    if any(
                        operation in strings.get(name, "").lower()
                        for operation in ("encodepicture", "decodepicture")
                    )
                )
        video_traced = "NVVIDEO_ENCODER_API" in tables
        gpu_complete = bool(video) if expects_video else video_calls == 0 or bool(video)
        if video_traced and video_calls and not video:
            gpu_complete = False
        missing_video_ranges = 0
        if "GPU_VIDEO_ENGINE_MISSING" in tables:
            missing_video_ranges = connection.execute(
                "SELECT COALESCE(SUM(rangeCount), 0) FROM GPU_VIDEO_ENGINE_MISSING"
            ).fetchone()[0]
            if missing_video_ranges and (expects_video or video_calls):
                gpu_complete = False

    merged = union(kernels)
    assert merged
    kernel_ns = duration(merged)
    copy_ns = duration(union(copies))
    kernel_or_copy_ns = duration(union([*merged, *copies]))
    exposed_copy_ns = kernel_or_copy_ns - kernel_ns
    capture_ns = int(metadata["RUN_DURATION_MS"]) * 1_000_000
    span_ns = merged[-1].end - merged[0].start

    before_names = {end: name for _, end, name in names}
    after_names = {start: name for start, _, name in names}
    gap_intervals = [
        Interval(previous.end, following.start)
        for previous, following in itertools.pairwise(merged)
    ]
    gap_intervals.sort(key=lambda item: item.end - item.start, reverse=True)

    copy_kind_names = {1: "H2D", 2: "D2H", 8: "D2D"}
    memory_kind_names = {0: "unknown", 1: "pinned", 2: "device"}
    grouped_copies: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    for (
        start,
        end,
        nbytes,
        copy_kind,
        source_kind,
        destination_kind,
        stream_id,
    ) in copy_rows:
        grouped_copies.setdefault(
            (copy_kind, source_kind, destination_kind), []
        ).append((start, end, nbytes, stream_id))
    copy_breakdown: list[_CopyBreakdown] = []
    for (copy_kind, source_kind, destination_kind), rows in grouped_copies.items():
        intervals = [Interval(start, end) for start, end, _, _ in rows]
        elapsed_ns = duration(union(intervals))
        total_bytes = sum(row[2] for row in rows)
        combined_ns = duration(union([*merged, *intervals]))
        exposed_ns = combined_ns - kernel_ns
        copy_breakdown.append(
            {
                "kind": copy_kind_names.get(copy_kind, str(copy_kind)),
                "source": memory_kind_names.get(source_kind, str(source_kind)),
                "destination": memory_kind_names.get(
                    destination_kind, str(destination_kind)
                ),
                "count": len(rows),
                "bytes": total_bytes,
                "seconds": elapsed_ns / 1e9,
                "without_kernel_seconds": exposed_ns / 1e9,
                "gigabytes_per_second": (
                    total_bytes / elapsed_ns if elapsed_ns else None
                ),
            }
        )
    copy_breakdown.sort(key=lambda item: item["seconds"], reverse=True)
    longest_copies: list[dict[str, object]] = []
    for (
        start,
        end,
        nbytes,
        copy_kind,
        source_kind,
        destination_kind,
        stream_id,
    ) in sorted(copy_rows, key=lambda row: row[1] - row[0], reverse=True)[:20]:
        elapsed_ns = end - start
        longest_copies.append(
            {
                "kind": copy_kind_names.get(copy_kind, str(copy_kind)),
                "source": memory_kind_names.get(source_kind, str(source_kind)),
                "destination": memory_kind_names.get(
                    destination_kind, str(destination_kind)
                ),
                "stream": stream_id,
                "start_seconds": start / 1e9,
                "seconds": elapsed_ns / 1e9,
                "bytes": nbytes,
                "gigabytes_per_second": nbytes / elapsed_ns if elapsed_ns else None,
            }
        )

    gaps: list[dict[str, object]] = []
    for gap in gap_intervals[:largest]:
        start, end = gap.start, gap.end
        calls: dict[str, int] = {}
        for interval, name in runtime:
            overlap = max(0, min(end, interval.end) - max(start, interval.start))
            if overlap:
                calls[name] = calls.get(name, 0) + overlap
        gaps.append(
            {
                "seconds": (end - start) / 1e9,
                "start_seconds": start / 1e9,
                "end_seconds": end / 1e9,
                "before_kernel": before_names.get(start, "overlapping kernels"),
                "after_kernel": after_names.get(end, "overlapping kernels"),
                "memcpy_seconds": clipped_duration(copies, start, end) / 1e9,
                "memset_seconds": clipped_duration(memsets, start, end) / 1e9,
                "host_cuda_calls": [
                    {"name": name, "seconds": elapsed / 1e9}
                    for name, elapsed in sorted(
                        calls.items(), key=lambda item: item[1], reverse=True
                    )[:5]
                ],
            }
        )
    return {
        "sqlite": path.name,
        "definition": "only CUDA kernel interval union is non-bubble",
        "transfers": {
            "union_seconds": copy_ns / 1e9,
            "overlapping_kernel_seconds": (copy_ns - exposed_copy_ns) / 1e9,
            "without_kernel_seconds": exposed_copy_ns / 1e9,
            "bubble_without_transfer_seconds": (capture_ns - kernel_or_copy_ns) / 1e9,
            "breakdown": copy_breakdown,
            "longest": longest_copies,
        },
        "capture": {
            "seconds": capture_ns / 1e9,
            "kernel_union_seconds": kernel_ns / 1e9,
            "gpu_union_seconds": duration(union([*kernels, *copies, *memsets, *video]))
            / 1e9,
            "video_union_seconds": duration(union(video)) / 1e9,
            "gpu_complete": gpu_complete,
            "video_api_calls": video_calls,
            "ignored_video_ranges": ignored_video_ranges,
            "missing_video_ranges": missing_video_ranges,
            "gpu_coverage_reason": "complete for captured CUDA/video workload"
            if gpu_complete
            else "video engine device intervals unavailable; CUDA/video union is only a lower bound",
            "bubble_seconds": (capture_ns - kernel_ns) / 1e9,
            "bubble_percent": 100.0 * (capture_ns - kernel_ns) / capture_ns,
        },
        "kernel_span": {
            "seconds": span_ns / 1e9,
            "kernel_union_seconds": kernel_ns / 1e9,
            "bubble_seconds": (span_ns - kernel_ns) / 1e9,
            "bubble_percent": 100.0 * (span_ns - kernel_ns) / span_ns,
        },
        "outside_kernel_span_seconds": (capture_ns - span_ns) / 1e9,
        "largest_gaps": gaps,
    }


class Hardware(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gpu: str
    dense_tops: dict[str, float]
    assumptions: str
    sources: list[str] = Field(default_factory=list)

    @field_validator("dense_tops")
    @classmethod
    def valid_rates(cls, rates: dict[str, float]) -> dict[str, float]:
        import math

        for instruction, rate in rates.items():
            MmaType(instruction)
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("dense TOPS must be finite and positive")
        return rates


def resolve(path: Path, overrides: dict[str, object]) -> RunConfig:
    synchronization = {
        name: overrides.pop(name)
        for name in Synchronization.model_fields
        if name in overrides
    }
    resolved = config.load(path, overrides)
    return resolved.model_copy(
        update={
            "synchronization": Synchronization.model_validate(
                {**resolved.synchronization.model_dump(), **synchronization}
            )
        }
    )


def overrides(arguments: list[str]) -> dict[str, object]:
    allowed = set(RunConfig.model_fields) | set(Synchronization.model_fields)
    result: dict[str, object] = {}
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if not option.startswith("--"):
            raise ValueError(f"Expected an option, received {option!r}")
        option, separator, inline = option[2:].partition("=")
        negative = option.startswith("no-")
        name = (option[3:] if negative else option).replace("-", "_")
        if name not in allowed:
            raise ValueError(f"Unknown configuration field: {name}")
        if negative:
            if name == "lora":
                result[name] = None
            else:
                raise ValueError(f"--no-{option[3:]} is not a boolean option")
        elif separator:
            result[name] = inline
        else:
            index += 1
            if index == len(arguments):
                raise ValueError(f"--{option} requires a value")
            result[name] = arguments[index]
        index += 1
    return result


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge_measurements(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Append independent child samples after verifying their recorded identity."""
    assert previous.get("status") == current.get("status") == "complete"
    for name in ("resolved_config", "workload", "environment", "backend"):
        if name in previous or name in current:
            assert previous.get(name) == current.get(name), f"changed {name}"
    left = previous.get("latency_seconds")
    right = current.get("latency_seconds")
    assert isinstance(left, list) and isinstance(right, list)
    samples = [float(value) for value in (*left, *right)]
    timing = timing_statistics(samples)
    previous_ids = previous.get("run_ids")
    result = dict(current)
    result.update(
        repetitions=len(samples),
        latency_seconds=samples,
        **timing,
        run_ids=[
            *(previous_ids if isinstance(previous_ids, list) else [previous.get("run_id")]),
            current.get("run_id"),
        ],
    )
    left_memory = previous.get("memory_samples")
    right_memory = current.get("memory_samples")
    if isinstance(left_memory, list) and isinstance(right_memory, list):
        result["memory_samples"] = [*left_memory, *right_memory]
    tensor_seconds = result.get("tensor_seconds")
    if isinstance(tensor_seconds, int | float):
        mean_seconds = timing["mean_seconds"]
        assert mean_seconds is not None
        result["tensor_percent"] = 100 * tensor_seconds / mean_seconds
    return result


def timing_statistics(samples: list[float]) -> dict[str, float | None]:
    """Summarize preserved wall-time samples without hiding their spread."""
    assert samples
    return {
        "mean_seconds": statistics.mean(samples),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "range_seconds": max(samples) - min(samples),
        "sample_stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else None,
    }


def measure_video_quality(actual: Path, reference: Path) -> dict[str, object]:
    """Compare decoded RGB frames with PSNR and Gaussian-window SSIM."""
    import math

    if actual.resolve() == reference.resolve():
        return {
            "video_psnr_db": "Infinity",
            "video_ssim": 1.0,
            "video_quality_frames": _video_frame_count(actual),
        }

    import av
    import torch
    from torch.nn import functional

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coordinates = torch.arange(11, device=device, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coordinates * coordinates) / (2 * 1.5 * 1.5))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, 11, 11)
    squared_error = 0.0
    element_count = 0
    ssim_sum = 0.0
    frame_count = 0

    with (
        av.open(str(actual)) as actual_container,
        av.open(str(reference)) as reference_container,
    ):
        actual_frames = actual_container.decode(video=0)
        reference_frames = reference_container.decode(video=0)
        while True:
            left = next(actual_frames, None)
            right = next(reference_frames, None)
            assert (left is None) == (right is None), "video frame counts differ"
            if left is None:
                break
            assert right is not None
            left_tensor = torch.from_numpy(left.to_ndarray(format="rgb24")).to(
                device=device, dtype=torch.float32
            )
            right_tensor = torch.from_numpy(right.to_ndarray(format="rgb24")).to(
                device=device, dtype=torch.float32
            )
            assert left_tensor.shape == right_tensor.shape, "video frame sizes differ"
            left_tensor = left_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
            right_tensor = right_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
            difference = left_tensor - right_tensor
            squared_error += float(torch.sum(difference * difference))
            element_count += difference.numel()

            left_mean = functional.conv2d(left_tensor, window, groups=3)
            right_mean = functional.conv2d(right_tensor, window, groups=3)
            left_variance = (
                functional.conv2d(left_tensor * left_tensor, window, groups=3)
                - left_mean * left_mean
            )
            right_variance = (
                functional.conv2d(right_tensor * right_tensor, window, groups=3)
                - right_mean * right_mean
            )
            covariance = (
                functional.conv2d(left_tensor * right_tensor, window, groups=3)
                - left_mean * right_mean
            )
            ssim = (
                (2 * left_mean * right_mean + 0.01**2) * (2 * covariance + 0.03**2)
            ) / (
                (left_mean * left_mean + right_mean * right_mean + 0.01**2)
                * (left_variance + right_variance + 0.03**2)
            )
            ssim_sum += float(ssim.mean())
            frame_count += 1

    assert frame_count and element_count
    mse = squared_error / element_count
    return {
        "video_psnr_db": -10 * math.log10(mse) if mse else "Infinity",
        "video_ssim": ssim_sum / frame_count,
        "video_quality_frames": frame_count,
    }


def _video_frame_count(path: Path) -> int:
    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def write_video_quality(actual: Path, reference: Path, run_id: str) -> None:
    """Merge decoded-video quality metrics into the run's error report."""
    path = actual.with_name("error.json")
    result: dict[str, object] = {
        "run_id": run_id,
        "actual": actual.name,
        "reference": reference.name,
        **measure_video_quality(actual, reference),
    }
    write_json(path, result)


def check_gpu_temperature(maximum_celsius: int | None) -> None:
    """Reject a measurement when GPU 0 is not below its configured limit."""
    if maximum_celsius is None:
        return
    import subprocess

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        text=True,
    )
    temperature = int(output.strip())
    if temperature >= maximum_celsius:
        raise RuntimeError(
            f"GPU temperature is {temperature}C; require <{maximum_celsius}C"
        )
    print(f"gpu_temperature={temperature}C", flush=True)


def invoke(
    command_line: list[str],
    log: Path,
    *,
    environment: Mapping[str, str] | None = None,
    maximum_gpu_temperature_celsius: int | None = None,
) -> tuple[int, float]:
    import subprocess
    import time

    check_gpu_temperature(maximum_gpu_temperature_celsius)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command_line,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode, time.perf_counter() - started


def tensor_metrics(
    operations: dict[str, int], hardware_path: Path, latency: float
) -> tuple[dict[str, object], float, float]:
    hardware = Hardware.model_validate_json(hardware_path.read_text(encoding="utf-8"))
    missing = sorted(set(operations) - set(hardware.dense_tops))
    if missing:
        raise ValueError(f"hardware profile missing MMA input pairs: {missing}")
    seconds = sum(
        count / (hardware.dense_tops[pair] * 1e12) for pair, count in operations.items()
    )
    return hardware.model_dump(), seconds, 100 * seconds / latency