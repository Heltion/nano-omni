"""Update one model inventory or measured kernel row."""

import csv
import dataclasses
import io
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypedDict, cast

import tune
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent / "run"))
from config import RunConfig, load

from nano_omni.core.kernel import Config, Kernel
from nano_omni.core.model import ModelRequest
from nano_omni.core.platform.memory import WorkingSetLimit
from nano_omni.kernels import classes, contract

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


# CSV input starts as text; measured columns are replaced with numeric values.
CsvRow = TypedDict(
    "CsvRow",
    {
        "model": str,
        "kernel": str,
        "percent": str | float,
        "latency": str | float,
        "launch": str | int,
        "mem%": str | float,
        "ncu%": str | float,
        "tops": str,
        "workload": str,
        "config": str,
    },
)

ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer(no_args_is_help=True)
CSV_FIELDS: tuple[str, ...] = (
    "model",
    "kernel",
    "percent",
    "latency",
    "launch",
    "mem%",
    "ncu%",
    "tops",
    "workload",
    "config",
)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def spec_json(value: object) -> dict[str, object]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(cast("DataclassInstance", value))
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Unsupported model workload: {type(value).__name__}")


def instance_path(instance: str) -> Path:
    """Resolve one performance instance that owns a config and kernel table."""
    directory = ROOT / "perf" / instance
    if not (directory / "config.yaml").is_file():
        raise typer.BadParameter(
            "Unknown performance instance", param_hint="--instance"
        )
    return directory


def table(instance: str) -> Path:
    return instance_path(instance) / "kernels.csv"


def read_rows(path: Path) -> list[CsvRow]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and tuple(rows[0]) != CSV_FIELDS:
        raise ValueError(f"Unexpected CSV schema: {path}")
    return cast(list[CsvRow], rows)


def row_key(model: str, kernel: str, workload: object) -> tuple[str, str, str]:
    return model, kernel, canonical(workload)


def model_name(request: ModelRequest) -> str:
    """Return the stable CSV model name for one concrete model request."""
    module = request.model.__module__.rsplit(".", 1)[-1]
    if module.endswith("_diffusion"):
        return "diffusion"
    if module == "audio_vae_decoder":
        return "audio_vae"
    return module


def normalize(rows: list[CsvRow]) -> list[CsvRow]:
    elapsed = [float(row["latency"]) * int(row["launch"]) for row in rows]
    total = sum(elapsed)
    for row, duration in zip(rows, elapsed, strict=True):
        row["percent"] = duration / total * 100 if total else 0
    return sorted(rows, key=lambda row: float(row["percent"]), reverse=True)


def write_rows(path: Path, rows: list[CsvRow]) -> list[CsvRow]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        ordered = normalize(rows)
        writer.writerows(ordered)
    temporary.replace(path)
    return ordered


def model_requests(instance: str) -> Iterator[ModelRequest]:
    yield from requests(load(instance_path(instance) / "config.yaml"), ROOT)


def request_for(
    instance: str,
    model: str,
    workload: dict[str, object],
    candidates: Iterable[ModelRequest] | None = None,
) -> ModelRequest:
    if candidates is None:
        candidates = model_requests(instance)
    matches = list(
        dict.fromkeys(
            request
            for request in candidates
            if model_name(request) == model
            and canonical(spec_json(request.spec)) == canonical(workload)
        )
    )
    if len(matches) != 1:
        available = [
            canonical(spec_json(request.spec))
            for request in candidates
            if model_name(request) == model
        ]
        raise ValueError(
            f"Expected one {model} model workload, found {len(matches)}; "
            f"requested={canonical(workload)}, available={available}"
        )
    return matches[0]


def current_inventory(
    instance: str, model: str, workload: dict[str, object]
) -> list[tuple[Kernel, int]]:
    import torch

    config = load(instance_path(instance) / "config.yaml")
    candidates = tuple(model_requests(instance))
    selected = request_for(instance, model, workload, candidates)
    kernels: list[Kernel] = []
    for request in candidates:
        if request.model is selected.model:
            kernels.extend(
                request.kernel_inventory(
                    torch.cuda.get_device_properties(0).total_memory,
                    config.workspace_nbytes,
                )
            )
    return aggregate(kernels)


@app.command()
def model(
    instance: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    workload: Annotated[dict[str, object], typer.Option(parser=tune.json_object)],
    working_set_gib: Annotated[int, typer.Option(min=8, max=24)] = 24,
) -> None:
    """Replace one model inventory while retaining every other model."""
    path = table(instance)
    old = {
        row_key(row["model"], row["kernel"], json.loads(row["workload"])): row
        for row in read_rows(path)
    }
    os.environ.update(tune.compiler_environment())
    with WorkingSetLimit(working_set_gib << 30):
        inventory = current_inventory(instance, model, workload)
    retained = [row for row in read_rows(path) if row["model"] != model]
    rows: list[CsvRow] = []
    for invocation, count in inventory:
        key = row_key(
            model,
            invocation.name,
            invocation.workload.model_dump(mode="json"),
        )
        tops = canonical(
            {
                mma.value: operations * count
                for mma, operations in invocation.tops(invocation.arguments).items()
            }
        )
        if key in old:
            row: CsvRow = old[key].copy()
            row["launch"] = count
            row["tops"] = tops
        else:
            row = {
                "model": model,
                "kernel": key[1],
                "percent": 0,
                "latency": 0,
                "launch": count,
                "mem%": 0,
                "ncu%": 0,
                "tops": tops,
                "workload": key[2],
                "config": "{}",
            }
        rows.append(row)
    write_rows(path, [*retained, *rows])
    typer.echo(str(path.relative_to(ROOT)))


def update_kernels(
    path: Path,
    selected: tuple[str, str, dict[str, object]] | None,
    warmup: int,
    repetitions: int,
    working_set_gib: int,
    dynamic: dict[str, int | float] | None = None,
) -> None:
    """Update selected rows, retaining an atomic checkpoint after each success."""
    rows = read_rows(path)
    rows_by_key: dict[tuple[str, str, str], list[CsvRow]] = {}
    pending = [] if selected is None else [selected]
    for row in rows:
        workload = json.loads(row["workload"])
        key = row_key(row["model"], row["kernel"], workload)
        rows_by_key.setdefault(key, []).append(row)
        if selected is None and (
            float(row["latency"]) == 0
            or float(row["mem%"]) == 0
            or float(row["ncu%"]) == 0
            or row["config"] in ("", "{}")
        ):
            pending.append((row["model"], row["kernel"], workload))
    inventory_by_key: dict[tuple[str, str, str], list[tuple[Kernel, int]]] = {}
    if selected is None:
        import torch

        run_config = load(path.parent / "config.yaml")
        inventories = model_inventories(
            run_config,
            ROOT,
            torch.cuda.get_device_properties(0).total_memory,
        )
        for inventory_model, inventory in inventories.items():
            for invocation, count in inventory:
                identity = row_key(
                    inventory_model,
                    invocation.name,
                    invocation.workload.model_dump(mode="json"),
                )
                inventory_by_key.setdefault(identity, []).append((invocation, count))

    for model, kernel_name, workload in pending:
        key = row_key(model, kernel_name, workload)
        matches = rows_by_key.get(key, [])
        if len(matches) != 1:
            raise ValueError(f"Expected one kernel workload row, found {len(matches)}")
        row = matches[0]
        if selected is not None:
            kernel_type = classes().get(kernel_name)
            if kernel_type is None:
                raise ValueError(f"Unknown kernel: {kernel_name}")
            workload_type, _ = contract(kernel_type)
            arguments = kernel_type.make_arguments(workload_type.model_validate(workload))
            invocation = kernel_type(
                tune.with_dynamic_parameters(arguments, dynamic or {})
            )
            count = int(row["launch"])
        else:
            matching_inventory = inventory_by_key.get(key, [])
            if len(matching_inventory) != 1:
                raise ValueError(
                    f"Expected one live kernel workload, found {len(matching_inventory)}"
                )
            invocation, count = matching_inventory[0]
        full_measurement = (
            selected is not None
            or float(row["latency"]) == 0
            or row["config"] in ("", "{}")
        )
        if full_measurement:
            assert isinstance(invocation.config, Config)
            selected_config = invocation.config
        else:
            _, config_type = contract(type(invocation))
            selected_config = config_type.model_validate_json(row["config"])
        config_data = selected_config.model_dump(mode="json")
        # Profile before the ordinary measurement retains large argument tensors
        # in the parent process and leaves too little VRAM for the NCU child.
        memory, compute = profile_config(
            kernel_name,
            workload,
            config_data,
            working_set_gib,
            dynamic or {},
        )
        latency = float(row["latency"])
        if full_measurement:
            measured = tune._measure(
                kernel_name,
                type(invocation),
                invocation.workload,
                selected_config,
                warmup=warmup,
                repetitions=repetitions,
                verify=False,
                dynamic=dynamic or {},
            )
            latency = cast(float, measured["latency_ms"])
        row.update(
            latency=latency,
            **{"mem%": memory, "ncu%": compute},
            tops=canonical(
                {
                    mma.value: operations * count
                    for mma, operations in invocation.tops(invocation.arguments).items()
                }
            ),
            config=canonical(config_data),
        )
        # The next checkpoint uses the previous persisted order for stable ties.
        rows = write_rows(path, rows)


@app.command()
def kernel(
    instance: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    kernel: Annotated[str, typer.Option()],
    workload: Annotated[dict[str, object], typer.Option(parser=tune.json_object)],
    dynamic: Annotated[
        dict[str, object] | None, typer.Option(parser=tune.json_object)
    ] = None,
    warmup: Annotated[int, typer.Option(min=0)] = 2,
    repetitions: Annotated[int, typer.Option(min=1)] = 10,
    working_set_gib: Annotated[int, typer.Option(min=8, max=24)] = 24,
) -> None:
    """Measure and replace one kernel workload row."""
    os.environ.update(tune.compiler_environment())
    with WorkingSetLimit(working_set_gib << 30):
        tune.limit_vram()
        update_kernels(
            table(instance),
            (model, kernel, workload),
            warmup,
            repetitions,
            working_set_gib,
            cast(dict[str, int | float], dynamic or {}),
        )


@app.command("all-kernels")
def all_kernels(
    instance: Annotated[str, typer.Option()],
    warmup: Annotated[int, typer.Option(min=0)] = 2,
    repetitions: Annotated[int, typer.Option(min=1)] = 10,
    working_set_gib: Annotated[int, typer.Option(min=8, max=24)] = 24,
) -> None:
    """Fill every row with missing latency, config, or NCU measurements."""
    path = table(instance)
    os.environ.update(tune.compiler_environment())
    with WorkingSetLimit(working_set_gib << 30):
        tune.limit_vram()
        update_kernels(path, None, warmup, repetitions, working_set_gib)


def profile_config(
    kernel: str,
    workload: dict[str, object],
    config: dict[str, object],
    working_set_gib: int,
    dynamic: dict[str, int | float],
) -> tuple[float, float]:
    executable = shutil.which("ncu")
    if executable is not None and Path(executable).suffix.lower() == ".bat":
        native = (
            Path(executable).parent
            / "target"
            / "windows-desktop-win7-x64"
            / "ncu.exe"
        )
        if native.is_file():
            executable = str(native)
    if executable is None:
        root = Path(os.environ["ProgramFiles"]) / "NVIDIA Corporation"
        executable = str(
            next(root.glob("Nsight Compute */target/windows-desktop-win7-x64/ncu.exe"))
        )
    command = [
        executable,
        "--csv",
        "--page",
        "raw",
        "--nvtx",
        "--nvtx-include",
        "measure/",
        "--metrics",
        "gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed",
        sys.executable,
        str(ROOT / "scripts/tune.py"),
        "--kernel",
        kernel,
        "--workload",
        canonical(workload),
        "--config",
        canonical(config),
        "--dynamic",
        canonical(dynamic),
        "--warmup",
        "0",
        "--repetitions",
        "1",
        "--working-set-gib",
        str(working_set_gib),
        "--no-verify",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ | {"NANO_OMNI_NVTX": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )
    lines = list(csv.reader(io.StringIO(result.stdout)))
    header = next(index for index, line in enumerate(lines) if line and line[0] == "ID")
    names = lines[header]
    duration_column = names.index("gpu__time_duration.sum")
    memory_column = names.index("dram__throughput.avg.pct_of_peak_sustained_elapsed")
    compute_column = names.index("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    memory, compute, duration = 0.0, 0.0, 0.0
    for line in lines[header + 2 :]:
        if line and line[0].isdigit():
            elapsed = float(line[duration_column].replace(",", ""))
            memory += elapsed * float(line[memory_column].replace(",", ""))
            compute += elapsed * float(line[compute_column].replace(",", ""))
            duration += elapsed
    if duration == 0:
        raise RuntimeError("NCU returned no kernels inside the measurement range")
    return memory / duration, compute / duration


def fl2va_requests(config: RunConfig, root: Path) -> Iterator[ModelRequest]:
    from nano_omni.pipelines.h3_fl2va import H3Fl2vaPipeline

    pipeline = H3Fl2vaPipeline(config, root)
    pipeline.prepare()
    yield from pipeline.model_requests()


def ref2va_requests(config: RunConfig, root: Path) -> Iterator[ModelRequest]:
    from nano_omni.pipelines.h3_ref2va import H3Ref2vaPipeline

    pipeline = H3Ref2vaPipeline(config, root)
    pipeline.prepare()
    yield from pipeline.model_requests()


def requests(config: RunConfig, root: Path) -> Iterator[ModelRequest]:
    factory = fl2va_requests if config.pipeline == "h3_fl2va" else ref2va_requests
    yield from factory(config, root)


def aggregate(kernels: Iterable[Kernel]) -> list[tuple[Kernel, int]]:
    """Count equivalent invocations without wrapping Kernel in another entity."""
    rows: dict[tuple[type[Kernel], object], tuple[Kernel, int]] = {}
    for invocation in kernels:
        key = (type(invocation), invocation.workload)
        existing = rows.get(key)
        rows[key] = (
            invocation,
            1 if existing is None else existing[1] + 1,
        )
    return list(rows.values())


def model_inventories(
    config: RunConfig,
    root: Path,
    total_memory: int,
    *,
    model: str | None = None,
) -> dict[str, list[tuple[Kernel, int]]]:
    """Build inventories for every model, or only the requested CSV model."""
    models: dict[str, list[Kernel]] = {}
    inventories: dict[ModelRequest, list[Kernel]] = {}
    for request in requests(config, root):
        name = model_name(request)
        if model is not None and name != model:
            continue
        if request not in inventories:
            inventories[request] = request.kernel_inventory(
                total_memory, config.workspace_nbytes
            )
        models.setdefault(name, []).extend(inventories[request])
    return {name: aggregate(kernels) for name, kernels in models.items()}


def inventory(
    config: RunConfig, root: Path, total_memory: int
) -> list[tuple[Kernel, int]]:
    kernels: list[Kernel] = []
    for model_rows in model_inventories(config, root, total_memory).values():
        for invocation, count in model_rows:
            kernels.extend([invocation] * count)
    return aggregate(kernels)


if __name__ == "__main__":
    app()
