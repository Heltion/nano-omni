"""Compile unique Kernel specializations and bind their CUDA launch metadata."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import TYPE_CHECKING

from cuda.bindings import driver  # pyrefly: ignore[missing-module-attribute]

from nano_omni.core.kernel import Config, Kernel
from nano_omni.core.planning.scheduling import Action
from nano_omni.core.runtime import execution, observation
from nano_omni.core.tensor import DType, TensorDesc, TensorKind

if TYPE_CHECKING:
    from nano_omni.core.model import PlannedModel


type Specialization = tuple[type[Kernel], tuple[tuple[str, object], ...], Config]


def _zero_arguments(
    kernel: Kernel,
    values: dict[tuple[DType, tuple[int, ...]], TensorDesc],
) -> Kernel:
    """Address equal tensor shapes with one reusable null descriptor."""

    def tensor(value: TensorDesc) -> TensorDesc:
        key = value.dtype, value.shape
        if key not in values:
            values[key] = TensorDesc(
                value.dtype,
                value.shape,
                TensorKind.DEVICE,
                (0, 0),
            )
        return values[key]

    return kernel.map_arguments(tensor)


def _key(kernel: Kernel) -> Specialization:
    dynamic = {name for name, _ in kernel.arguments.dynamic_parameters()}
    workload = tuple(
        sorted(
            (name, value)
            for name, value in kernel.workload.model_dump().items()
            if name not in dynamic
        )
    )
    return type(kernel), workload, kernel.config


def prepare(commands: Iterable[Action], *, label: str = "") -> None:
    """Populate TileLang caches once per specialization without submitting kernels."""
    started = perf_counter()
    arguments: dict[tuple[DType, tuple[int, ...]], TensorDesc] = {}
    unique: dict[Specialization, Kernel] = {}
    for item in commands:
        if isinstance(item, Kernel):
            unique.setdefault(_key(item), _zero_arguments(item, arguments))
    if label:
        observation.record_timing(label + "_kernel_requests", started, perf_counter())
    compiled = perf_counter()
    for kernel in unique.values():
        kernel.compile()
        # The planning process only needs TileLang's disk-cache side effect.
        kernel.compiled = None
    if label:
        observation.record_timing(
            label + "_compile_kernels",
            compiled,
            perf_counter(),
            kernels=len(unique),
        )


def bind_kernel_modules[Spec, Args, ModelConfig](
    model: PlannedModel[Spec, Args, ModelConfig],
    cuda: execution.CudaRuntime,
    *,
    label: str = "",
) -> None:
    """Compile once, share programs, and bind CUDA attributes in the parent process."""
    execution.check_cuda(driver.cuCtxSetCurrent(cuda.context))
    started = perf_counter()
    arguments: dict[tuple[DType, tuple[int, ...]], TensorDesc] = {}
    compiled = cuda.kernel_modules
    kernels = [item for item in model.commands if isinstance(item, Kernel)]
    representatives: list[Kernel] = []
    for kernel in kernels:
        key = _key(kernel)
        if key not in compiled:
            prepared = _zero_arguments(kernel, arguments)
            prepared.compile()
            compiled[key] = prepared.compiled, prepared.config
            representatives.append(prepared)
        kernel.compiled, kernel.config = compiled[key]
    if label:
        observation.record_timing(label + "_bind_arguments", started, perf_counter())

    bound = perf_counter()
    token = execution._compiling.set("bind")
    try:
        for kernel in representatives:
            kernel.submit()
    finally:
        execution._compiling.reset(token)
    if label:
        observation.record_timing(label + "_bind_attributes", bound, perf_counter())
