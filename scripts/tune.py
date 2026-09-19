"""Measure one explicit Kernel configuration through the common Kernel contract."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.platform.memory import WorkingSetLimit
from nano_omni.core.tensor import DType, TensorDesc, TensorKind
from nano_omni.kernels import classes as kernel_classes
from nano_omni.kernels import contract

app = typer.Typer()

TUNE_FIELDS = (
    "kernel",
    "workload",
    "config",
    "dynamic",
    "latency",
    "correctness",
    "warmup",
    "repetitions",
    "trials",
)


def compiler_environment() -> dict[str, str]:
    """Return an environment containing the MSVC tools required by NVRTC."""
    environment = dict(os.environ)
    if os.name != "nt" or shutil.which("cl"):
        return environment
    locator = (
        Path(os.environ["ProgramFiles(x86)"])
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    installation = subprocess.check_output(
        [
            str(locator),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    script = Path(installation) / "Common7/Tools/VsDevCmd.bat"
    assert installation and script.is_file(), (
        "Visual Studio x64 C++ tools are required for CUDA workloads"
    )
    output = subprocess.check_output(
        [
            "cmd",
            "/d",
            "/s",
            "/c",
            f'chcp 65001 >nul && call "{script}" -no_logo -arch=x64 -host_arch=x64 >nul && set',
        ],
        encoding="utf-8",
    )
    for line in output.splitlines():
        name, separator, value = line.partition("=")
        if separator and name:
            environment[name] = value
    return environment


def limit_vram() -> None:
    """Restrict Torch allocations to 85 percent of physical VRAM."""
    import torch

    free, total = torch.cuda.mem_get_info()
    budget = int(total * 0.85) - (total - free) + torch.cuda.memory_reserved()
    assert budget > 0, "current GPU use already reaches the VRAM budget"
    torch.cuda.set_per_process_memory_fraction(budget / total)


def json_object(value: str) -> dict[str, object]:
    """Parse one command-line JSON object."""
    data = json.loads(value)
    if not isinstance(data, dict):
        raise typer.BadParameter("JSON object required")
    return data


def _random_tensor(desc: TensorDesc, generator: Any) -> Any:
    """Allocate one CUDA tensor using the common value rule for its dtype."""
    import torch

    torch_dtype = {
        DType.F32: torch.float32,
        DType.F16: torch.float16,
        DType.I32: torch.int32,
        DType.U32: torch.uint32,
        DType.I8: torch.int8,
        DType.BF16: torch.bfloat16,
        DType.FP8_E4M3: torch.float8_e4m3fn,
        DType.FP8_UE4M3: torch.float8_e4m3fn,
        DType.U8: torch.uint8,
    }[desc.dtype]
    if desc.dtype in (DType.I32, DType.U32, DType.I8, DType.U8):
        return torch.randint(
            0, 3, desc.shape, generator=generator, device="cuda", dtype=torch.int64
        ).to(torch_dtype)
    if desc.dtype == DType.FP8_UE4M3:
        return (torch.rand(desc.shape, generator=generator, device="cuda") + 0.125).to(
            torch_dtype
        )
    return (torch.randn(desc.shape, generator=generator, device="cuda") * 0.1).to(
        torch_dtype
    )


def _address(
    template: Arguments, *, reference: bool = True
) -> tuple[Arguments, Arguments | None, list[Any]]:
    """Address one invocation and optionally clone it for the reference."""
    import torch

    generator = torch.Generator(device="cuda").manual_seed(0)
    actual_owners: list[Any] = []
    expected_owners: list[Any] = []
    actual_by_id: dict[int, TensorDesc] = {}
    expected_by_id: dict[int, TensorDesc] = {}

    def actual(desc: TensorDesc) -> TensorDesc:
        cached = actual_by_id.get(id(desc))
        if cached is not None:
            return cached
        tensor = _random_tensor(desc, generator)
        actual_owners.append(tensor)
        addressed = TensorDesc(
            desc.dtype, desc.shape, TensorKind.DEVICE, (tensor.data_ptr(), 0)
        )
        actual_by_id[id(desc)] = addressed
        return addressed

    actual_arguments = template.map(actual)
    if not reference:
        return actual_arguments, None, actual_owners

    def expected(desc: TensorDesc) -> TensorDesc:
        cached = expected_by_id.get(id(desc))
        if cached is not None:
            return cached
        source = actual_by_id[id(desc)].as_torch()
        tensor = source.clone()
        expected_owners.append(tensor)
        addressed = TensorDesc(
            desc.dtype, desc.shape, TensorKind.DEVICE, (tensor.data_ptr(), 0)
        )
        expected_by_id[id(desc)] = addressed
        return addressed

    expected_arguments = template.map(expected)
    return actual_arguments, expected_arguments, [*actual_owners, *expected_owners]


def with_dynamic_parameters(
    arguments: Arguments, overrides: dict[str, int | float]
) -> Arguments:
    """Return arguments with validated runtime scalar overrides."""
    available = dict(arguments.dynamic_parameters())
    assert overrides.keys() <= available.keys(), (
        f"unknown dynamic parameters: {sorted(overrides.keys() - available.keys())}"
    )
    for parameter_name, value in overrides.items():
        assert type(value) is type(available[parameter_name]), (
            f"invalid dynamic parameter: {parameter_name}"
        )
    return dataclasses.replace(cast(Any, arguments), **overrides) if overrides else arguments


def _abi_values(invocation: Kernel[Any, Any, Any]) -> list[Any]:
    """Expand runtime scalars and tensors in the generated TileLang ABI order."""
    values = invocation.arguments.values(include_absent=True)
    placeholder = next(value for value in values if value is not None)
    return [
        *(value for _, value in invocation.arguments.dynamic_parameters()),
        *((placeholder if value is None else value).as_torch() for value in values),
    ]


def _assert_close(
    actual: Arguments, expected: Arguments, *, rtol: float, atol: float
) -> None:
    """Compare every tensor leaf in bounded-memory chunks with dtype rules."""
    import torch

    actual_values = actual.values()
    expected_values = expected.values()
    assert len(actual_values) == len(expected_values)
    for observed, reference in zip(actual_values, expected_values, strict=True):
        left = observed.as_torch().view(-1)
        right = reference.as_torch().view(-1)
        assert left.numel() == right.numel()
        elements = 16 << 20
        for start in range(0, left.numel(), elements):
            left_chunk = left[start : start + elements]
            right_chunk = right[start : start + elements]
            if left.dtype.is_floating_point:
                torch.testing.assert_close(
                    left_chunk.float(),
                    right_chunk.float(),
                    rtol=rtol,
                    atol=atol,
                )
            elif observed.dtype in (DType.I8, DType.U8):
                torch.testing.assert_close(left_chunk, right_chunk, rtol=0, atol=1)
            else:
                torch.testing.assert_close(left_chunk, right_chunk, rtol=0, atol=0)


def _measure(
    name: str,
    kernel_type: type[Kernel[Any, Any, Any]],
    workload: Workload,
    config: Config,
    warmup: int,
    repetitions: int,
    verify: bool = True,
    dynamic: dict[str, int | float] | None = None,
) -> dict[str, object]:
    """Compile, verify and time one explicit configuration."""
    import torch

    template = kernel_type.make_arguments(workload)
    template = with_dynamic_parameters(template, dynamic or {})
    actual, expected, owners = _address(template, reference=verify)
    invocation = kernel_type(actual, config)
    invocation.compile()
    assert invocation.compiled is not None
    if verify:
        assert expected is not None
        kernel_type.ref_program(expected)
        invocation.compiled(*_abi_values(invocation))
        _assert_close(
            actual,
            expected,
            rtol=kernel_type.reference_rtol,
            atol=kernel_type.reference_atol,
        )
    if os.environ.get("NANO_OMNI_NVTX"):
        with torch.cuda.nvtx.range("measure"):
            invocation.compiled(*_abi_values(invocation))
        torch.cuda.synchronize()
    latency = invocation.compiled.get_profiler().do_bench(
        n_warmup=warmup,
        n_repeat=repetitions,
        input_tensors=_abi_values(invocation),
        backend="event",
    )
    assert math.isfinite(latency) and latency > 0
    del owners
    torch.cuda.synchronize()
    return {
        "kernel": name,
        "workload": workload.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "dynamic": dict(template.dynamic_parameters()),
        "latency_ms": latency,
        "correctness": "passed" if verify else "not_run",
        "warmup": warmup,
        "repetitions": repetitions,
        "method": "explicit_config",
    }


def record_candidate(path: Path, result: dict[str, object]) -> None:
    """Insert or replace one explicit candidate measurement in a CSV file."""
    row = {
        "kernel": str(result["kernel"]),
        "workload": json.dumps(result["workload"], sort_keys=True, separators=(",", ":")),
        "config": json.dumps(result["config"], sort_keys=True, separators=(",", ":")),
        "dynamic": json.dumps(result["dynamic"], sort_keys=True, separators=(",", ":")),
        "latency": str(result["latency_ms"]),
        "correctness": str(result["correctness"]),
        "warmup": str(result["warmup"]),
        "repetitions": str(result["repetitions"]),
        "trials": str(result["trials"]),
    }
    rows: list[dict[str, str]] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert tuple(reader.fieldnames or ()) == TUNE_FIELDS, (
                f"unexpected candidate CSV fields: {path}"
            )
            rows = list(reader)
    identity = (row["kernel"], row["workload"], row["config"], row["dynamic"])
    rows = [
        existing
        for existing in rows
        if (
            existing["kernel"],
            existing["workload"],
            existing["config"],
            existing["dynamic"],
        )
        != identity
    ]
    rows.append(row)
    rows.sort(
        key=lambda item: (
            item["kernel"],
            item["workload"],
            item["dynamic"],
            float(item["latency"]),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TUNE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@app.command()
def main(
    kernel: Annotated[str, typer.Option()],
    workload: Annotated[dict[str, object], typer.Option(parser=json_object)],
    config: Annotated[dict[str, object], typer.Option(parser=json_object)],
    dynamic: Annotated[dict[str, object] | None, typer.Option(parser=json_object)] = None,
    warmup: Annotated[int, typer.Option(min=0)] = 2,
    repetitions: Annotated[int, typer.Option(min=1)] = 10,
    trials: Annotated[int, typer.Option(min=1)] = 1,
    working_set_gib: Annotated[int, typer.Option(min=8, max=24)] = 24,
    record: Annotated[Path | None, typer.Option()] = None,
    verify: Annotated[bool, typer.Option(hidden=True)] = True,
) -> None:
    """Check and measure one explicit Kernel configuration."""
    classes = kernel_classes()
    if kernel not in classes:
        raise typer.BadParameter("unknown Kernel", param_hint="--kernel")
    kernel_type = classes[kernel]
    workload_type, config_type = contract(kernel_type)
    typed_workload = workload_type.model_validate(workload)
    typed_config = config_type.model_validate(config)
    if typed_config.model_fields_set != set(config_type.model_fields):
        raise typer.BadParameter(
            "every Config field must be explicit", param_hint="--config"
        )
    os.environ.update(compiler_environment())
    with WorkingSetLimit(working_set_gib << 30):
        limit_vram()
        dynamic_values = dynamic or {}
        assert all(type(value) in (int, float) for value in dynamic_values.values())
        measurements = [
            _measure(
                kernel,
                kernel_type,
                typed_workload,
                typed_config,
                warmup,
                repetitions,
                verify,
                dynamic=cast(dict[str, int | float], dynamic_values),
            )
            for _ in range(trials)
        ]
        result = measurements[-1] | {
            "latency_ms": statistics.median(
                cast(float, measurement["latency_ms"])
                for measurement in measurements
            ),
            "trials": trials,
        }
    if record is not None:
        record_candidate(record, result)
    typer.echo(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    app()
