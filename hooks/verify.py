"""Verify tuning CSV schema, arithmetic, ordering, and kernel contracts."""

import csv
import inspect
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from nano_omni.kernels import classes as kernel_classes
from nano_omni.kernels import contract

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
TUNE_FIELDS: tuple[str, ...] = (
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


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_kernel_source_ownership(classes: Mapping[str, type[object]]) -> None:
    kernels_by_source: defaultdict[Path, list[str]] = defaultdict(list)
    for name, kernel_type in classes.items():
        source = inspect.getsourcefile(kernel_type)
        if source is not None and source.endswith(".py"):
            kernels_by_source[Path(source).resolve()].append(name)

    violations = {
        source: sorted(names)
        for source, names in kernels_by_source.items()
        if len(names) > 1
    }
    if violations:
        details = "; ".join(
            f"{source.relative_to(ROOT)}: {', '.join(names)}"
            for source, names in sorted(violations.items())
        )
        raise ValueError(
            "Each Python source file must define at most one registered concrete "
            f"Kernel: {details}"
        )


def main() -> None:
    classes = kernel_classes()
    verify_kernel_source_ownership(classes)
    for path in sorted((ROOT / "perf").glob("*/tuning.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != TUNE_FIELDS:
                raise ValueError(f"Invalid tuning candidate fields: {path}")
            rows = list(reader)
        seen_candidates: set[tuple[str, str, str, str]] = set()
        for row in rows:
            kernel = row["kernel"]
            if kernel not in classes:
                raise ValueError(f"Unknown tuning candidate Kernel: {path}: {kernel}")
            workload_type, config_type = contract(classes[kernel])
            workload = workload_type.model_validate_json(row["workload"])
            config = config_type.model_validate_json(row["config"])
            dynamic = json.loads(row["dynamic"])
            if not isinstance(dynamic, dict):
                raise TypeError(f"Invalid tuning candidate dynamic values: {path}")
            expected_dynamic = dict(classes[kernel].make_arguments(workload).dynamic_parameters())
            if dynamic.keys() != expected_dynamic.keys() or any(
                type(value) is not type(expected_dynamic[name])
                for name, value in dynamic.items()
            ):
                raise ValueError(f"Invalid tuning candidate dynamic values: {path}: {kernel}")
            key = (
                kernel,
                canonical(workload.model_dump()),
                canonical(config.model_dump()),
                canonical(dynamic),
            )
            if key in seen_candidates:
                raise ValueError(f"Repeated tuning candidate: {path}: {key}")
            seen_candidates.add(key)
            if (
                float(row["latency"]) <= 0
                or int(row["repetitions"]) < 1
                or int(row["trials"]) < 1
            ):
                raise ValueError(f"Invalid tuning candidate measurement: {path}: {key}")
            if row["correctness"] not in ("passed", "not_run"):
                raise ValueError(f"Invalid tuning candidate correctness: {path}: {key}")
    for path in sorted((ROOT / "perf").glob("*/*.csv")):
        if path.name == "tuning.csv":
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValueError(f"Invalid tuning CSV fields: {path}")
            rows = list(reader)
        totals = [float(row["latency"]) * int(row["launch"]) for row in rows]
        summed = sum(totals)
        previous = math.inf
        seen: set[tuple[str, str, str]] = set()
        for row, total in zip(rows, totals, strict=True):
            key = (
                row["model"],
                row["kernel"],
                canonical(json.loads(row["workload"])),
            )
            if key in seen:
                raise ValueError(
                    f"Repeated model kernel workload: {path}: {key[0]}:{key[1]}"
                )
            seen.add(key)
            percent = float(row["percent"])
            expected = total / summed * 100 if summed else 0
            if not math.isclose(percent, expected, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Incorrect percentage: {path}: {key[0]}:{key[1]}")
            if percent > previous:
                raise ValueError(f"Rows are not sorted by percentage: {path}")
            previous = percent
            kernel_type = classes[row["kernel"]]
            workload_type, config_type = contract(kernel_type)
            workload = workload_type.model_validate_json(row["workload"])
            reconstructed = kernel_type.make_workload(
                kernel_type.make_arguments(workload)
            )
            if reconstructed != workload:
                raise ValueError(
                    f"Kernel argument roundtrip changed workload: "
                    f"{path}: {key[0]}:{key[1]}"
                )
            if float(row["latency"]):
                config = config_type.model_validate_json(row["config"])
                assert kernel_type.make_config(workload) == config, (
                    f"Config does not match production dispatch: "
                    f"{path}: {key[0]}:{key[1]}"
                )
            elif any(float(row[name]) for name in ("percent", "mem%", "ncu%")):
                raise ValueError(
                    f"Unmeasured row contains measurements: {path}: {key[0]}:{key[1]}"
                )


if __name__ == "__main__":
    main()
