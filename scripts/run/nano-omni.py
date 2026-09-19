"""Run one pipeline and write a measurement beside its output."""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypedDict

import typer

ROOT = Path(__file__).resolve().parents[2]
CHILD = "NANO_OMNI_RUN_CHILD"

from common import (
    analyze,
    check_gpu_temperature,
    invoke,
    merge_measurements,
    overrides,
    resolve,
    tensor_metrics,
    timing_statistics,
    write_json,
    write_video_quality,
)

if TYPE_CHECKING:
    from config import RunConfig

    from nano_omni.core.kernel import MmaType
    from nano_omni.core.model import PlannedModel
    from nano_omni.core.pipeline import Pipeline
    from nano_omni.core.runtime.execution import CudaRuntime
    from nano_omni.core.tensor import TensorDesc

app = typer.Typer(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)


class RunMetadata(TypedDict, total=False):
    workload: dict[str, object]
    tops: dict[str, int]
    memory: dict[str, object]
    environment: dict[str, object]


def measurement_environment() -> dict[str, object]:
    """Capture the runtime versions and hardware used for one measurement."""
    import tilelang
    import torch

    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tilelang": tilelang.__version__,
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "execution_backend": "nvrtc",
    }


def pipeline_for(config: RunConfig) -> Pipeline:
    if config.pipeline == "h3_fl2va":
        from nano_omni.pipelines.h3_fl2va import H3Fl2vaPipeline

        return H3Fl2vaPipeline(config, ROOT)
    if config.pipeline == "h3_ref2va":
        from nano_omni.pipelines.h3_ref2va import H3Ref2vaPipeline

        return H3Ref2vaPipeline(config, ROOT)
    raise ValueError(f"Unsupported pipeline: {config.pipeline}")


def child(config: RunConfig, mode: str) -> None:
    from nano_omni.core.model import PlannedModel
    from nano_omni.core.platform.memory import WorkingSetLimit

    if mode == "warmup":
        with WorkingSetLimit(config.working_set_gib << 30):
            pipeline_for(config).warmup()
        return

    original = PlannedModel.execute
    totals: collections.Counter[MmaType] = collections.Counter()

    def measured[Spec, Args, Config](
        self: PlannedModel[Spec, Args, Config],
        runtime: CudaRuntime,
        inputs: tuple[TensorDesc, ...],
        outputs: tuple[TensorDesc, ...],
    ) -> None:
        original(self, runtime, inputs, outputs)
        totals.update(self.tops)

    PlannedModel.execute = measured
    try:
        with WorkingSetLimit(config.working_set_gib << 30) as guard:
            pipeline = pipeline_for(config)
            output = pipeline.run()
            typer.echo("workload=" + json.dumps(pipeline.workload(), sort_keys=True))
            typer.echo("tops=" + json.dumps(dict(totals), sort_keys=True))
            typer.echo("output=" + Path(os.path.relpath(output, ROOT)).as_posix())
            typer.echo("memory=" + json.dumps(dataclasses.asdict(guard.usage)))
            typer.echo("environment=" + json.dumps(measurement_environment()))
    finally:
        PlannedModel.execute = original


def read_metadata(log: Path) -> RunMetadata:
    result: RunMetadata = {}
    for line in log.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        for name in ("workload", "tops", "memory", "environment"):
            if line.startswith(name + "="):
                result[name] = json.loads(line.removeprefix(name + "="))
    return result


def same_run_metadata(previous: RunMetadata, current: RunMetadata) -> bool:
    """Compare run identity, not process-dependent memory peaks or trim counts."""
    return all(
        previous.get(name) == current.get(name)
        for name in ("workload", "tops", "environment")
    )


def command(config: Path, arguments: list[str], output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run/nano-omni.py"),
        "--config",
        str(config),
        *arguments,
        "--output",
        Path(os.path.relpath(output, ROOT)).as_posix(),
    ]


def run_parent(
    config_path: Path,
    arguments: list[str],
    resolved: RunConfig,
    warmup: int,
    repetitions: int,
    nsys: bool,
    hardware: Path | None,
    append: bool,
) -> None:
    output = Path(resolved.output or ROOT / "outputs" / f"{resolved.pipeline}.mp4")
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    report = output.with_name(output.stem + "-nsys") if nsys else output
    report_path = report.with_suffix(".json")
    previous: dict[str, object] | None = None
    if append:
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict) and loaded.get("status") == "complete"
        previous = loaded
        check_gpu_temperature(resolved.maximum_gpu_temperature_celsius)
    # Invalidate an older success before warmup, launch, export, or parsing can fail.
    # An interrupted parent leaves an incomplete report, never a stale success.
    write_json(
        report_path,
        {"status": "incomplete", "output": output.name, "run_id": run_id},
    )
    workload = pipeline_for(resolved).workload()
    environment = {**os.environ, CHILD: "run", "NANO_OMNI_RUN_ID": run_id}
    if warmup:
        warmup_log = output.with_name(f".{output.stem}-warmup.log")
        code, _ = invoke(
            command(config_path, arguments, output),
            warmup_log,
            environment={**os.environ, CHILD: "warmup", "NANO_OMNI_RUN_ID": run_id},
            maximum_gpu_temperature_celsius=(
                None if append else resolved.maximum_gpu_temperature_celsius
            ),
        )
        if code:
            raise RuntimeError("compile warmup failed")
        warmup_log.unlink(missing_ok=True)

    if nsys:
        prefix = output.with_name(output.stem + "-nsys")
        log = prefix.with_suffix(".log")
        profiled = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,nvvideo",
            "--gpu-video-devices=0",
            "--sample=none",
            "--force-overwrite=true",
            "--output",
            str(prefix),
            *command(config_path, arguments, output),
        ]
        code, latency = invoke(
            profiled,
            log,
            environment=environment,
            maximum_gpu_temperature_celsius=resolved.maximum_gpu_temperature_celsius,
        )
        result: dict[str, object] = {
            "status": "complete" if code == 0 else "failed",
            "output": output.name,
            "run_id": run_id,
            "resolved_config": resolved.model_dump(mode="json"),
            "measurement": {
                "timing_boundary": "NSys capture around the complete child process",
                "process": "fresh profiled child",
                "cache_state": "uncontrolled existing OS, CUDA, and TileLang caches",
                "missing_assets": "downloaded inside the timed child",
            },
            "warmup": warmup,
            "repetitions": 1,
            "latency_seconds": latency,
            "returncode": code,
            "artifacts": {
                "log": log.name,
                "report": prefix.with_suffix(".nsys-rep").name,
                "sqlite": prefix.with_suffix(".sqlite").name,
            },
        }
        if code == 0:
            result["environment"] = read_metadata(log).get("environment")
            sqlite = prefix.with_suffix(".sqlite")
            with log.open("a", encoding="utf-8") as stream:
                exported = subprocess.run(
                    [
                        "nsys",
                        "export",
                        "--type=sqlite",
                        "--force-overwrite=true",
                        "--output",
                        str(sqlite),
                        str(prefix.with_suffix(".nsys-rep")),
                    ],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if exported.returncode:
                result.update(status="failed", export_returncode=exported.returncode)
            else:
                profile = analyze(sqlite)
                profile["sqlite"] = sqlite.name
                capture = profile["capture"]
                elapsed = capture["seconds"]
                result.update(
                    workload=workload,
                    kernel_seconds=capture["kernel_union_seconds"],
                    kernel_percent=100 * capture["kernel_union_seconds"] / elapsed,
                    gpu_seconds=capture["gpu_union_seconds"],
                    gpu_percent=100 * capture["gpu_union_seconds"] / elapsed,
                    profile=profile,
                )
        write_json(prefix.with_suffix(".json"), result)
        if result["status"] != "complete":
            raise RuntimeError(f"NSys run failed; see {log}")
        return

    samples: list[float] = []
    metadata: RunMetadata = {}
    memory_samples: list[dict[str, object] | None] = []
    temporary_log = output.with_name(f".{output.stem}-run.log")
    for index in range(repetitions):
        code, latency = invoke(
            command(config_path, arguments, output),
            temporary_log,
            environment=environment,
            maximum_gpu_temperature_celsius=(
                None if append else resolved.maximum_gpu_temperature_celsius
            ),
        )
        if code:
            write_json(
                output.with_suffix(".json"),
                {
                    "status": "failed",
                    "run_id": run_id,
                    "resolved_config": resolved.model_dump(mode="json"),
                    "returncode": code,
                    "completed_samples": samples,
                },
            )
            raise RuntimeError(f"run {index} failed; see {temporary_log}")
        samples.append(latency)
        current = read_metadata(temporary_log)
        if index and not same_run_metadata(metadata, current):
            raise ValueError("repetitions produced different workload metadata")
        metadata = current
        memory_samples.append(current.get("memory"))
    temporary_log.unlink(missing_ok=True)
    timing = timing_statistics(samples)
    mean_seconds = timing["mean_seconds"]
    assert mean_seconds is not None
    result = {
        "status": "complete",
        "output": output.name,
        "run_id": run_id,
        "resolved_config": resolved.model_dump(mode="json"),
        "workload": workload,
        "warmup": warmup,
        "repetitions": repetitions,
        "latency_seconds": samples,
        **timing,
        "tops": metadata["tops"],
        "memory": metadata.get("memory"),
        "memory_samples": memory_samples,
        "environment": metadata.get("environment"),
        "measurement": {
            "timing_boundary": "parent wall time from child launch through child exit",
            "process": "fresh child per sample",
            "cache_state": "uncontrolled existing OS, CUDA, and TileLang caches",
            "missing_assets": "downloaded inside the timed child",
        },
    }
    if hardware is not None:
        path = hardware if hardware.is_absolute() else ROOT / hardware
        profile, seconds, percent = tensor_metrics(metadata["tops"], path, mean_seconds)
        result.update(hardware=profile, tensor_seconds=seconds, tensor_percent=percent)
    if previous is not None:
        result = merge_measurements(previous, result)
    write_json(output.with_suffix(".json"), result)
    if resolved.quality_reference is not None:
        reference = resolved.quality_reference
        if not reference.is_absolute():
            reference = ROOT / reference
        write_video_quality(output, reference, run_id)


@app.command()
def main(
    context: typer.Context,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    warmup: Annotated[int, typer.Option(min=0, max=1)] = 0,
    repetitions: Annotated[int, typer.Option(min=1)] = 1,
    nsys: Annotated[bool, typer.Option()] = False,
    hardware: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    append: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run a pipeline in a child process and write a JSON measurement."""
    try:
        values = overrides(context.args)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    resolved = resolve(config, dict(values))
    if mode := os.environ.get(CHILD):
        child(resolved, mode)
        return
    if nsys and repetitions != 1:
        raise typer.BadParameter("--nsys requires --repetitions 1")
    if nsys and hardware is not None:
        raise typer.BadParameter("--hardware is unavailable with --nsys")
    if nsys and append:
        raise typer.BadParameter("--append is unavailable with --nsys")
    if append and (warmup or repetitions != 1):
        raise typer.BadParameter("--append requires --warmup 0 and --repetitions 1")
    run_parent(
        config,
        context.args,
        resolved,
        warmup,
        repetitions,
        nsys,
        hardware or resolved.hardware,
        append,
    )


if __name__ == "__main__":
    app()
