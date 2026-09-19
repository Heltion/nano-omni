"""Own CUDA resources, bind addressed plans, and launch prepared kernels."""

from __future__ import annotations

import contextlib
import dataclasses
import types
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, overload

from cuda.bindings import (
    driver,  # pyrefly: ignore[missing-module-attribute]  # Cython extension has no Python stub.
)

from nano_omni.core.planning.submission import StreamPlan
from nano_omni.core.runtime.host import HostTransfers
from nano_omni.core.runtime.synchronization import Synchronization
from nano_omni.core.tensor import TensorDesc, TensorKind

if TYPE_CHECKING:

    class _TilelangModule(Protocol):
        cuKernelSetAttribute: Callable[
            [driver.CUfunction_attribute, int, driver.CUkernel, driver.CUdevice],
            tuple[driver.CUresult],
        ]
        _nano_omni_kernel_attribute_results: dict[
            tuple[tuple[type[object], int], ...], tuple[driver.CUresult]
        ]


class Stream(StrEnum):
    MEMMOVE = "memmove"
    COPY = "copy"
    COMPUTE = "compute"


@overload
def check_cuda[T](result: tuple[driver.CUresult, T, *tuple[object, ...]]) -> T: ...


@overload
def check_cuda(result: tuple[driver.CUresult]) -> None: ...


def check_cuda[T](result: tuple[driver.CUresult, *tuple[T, ...]]) -> T | None:
    error, *values = result
    if error != driver.CUresult.CUDA_SUCCESS:
        _, name = driver.cuGetErrorName(error)
        _, message = driver.cuGetErrorString(error)
        raise RuntimeError(f"{name}: {message}")
    return values[0] if values else None


@dataclasses.dataclass(slots=True)
class CudaRuntime:
    """One CUDA context, four streams, and the pipeline's shared host/device arenas."""

    host_transfers: HostTransfers = dataclasses.field(
        default_factory=HostTransfers, init=False
    )
    synchronization: Synchronization = dataclasses.field(
        default_factory=Synchronization
    )
    ordinal: int = 0
    workspace: TensorDesc | None = None
    kernel_modules: dict[object, Any] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )
    device: driver.CUdevice = dataclasses.field(init=False)
    context: driver.CUcontext = dataclasses.field(init=False)
    streams: dict[Stream, driver.CUstream] = dataclasses.field(init=False)
    pinned_nbytes: int = 0
    pinned_address: int = dataclasses.field(init=False, default=0)
    cleanup: contextlib.ExitStack = dataclasses.field(
        default_factory=contextlib.ExitStack, init=False, repr=False
    )

    def __post_init__(self) -> None:
        with contextlib.ExitStack() as pending:
            pending.callback(self.host_transfers.close)
            check_cuda(driver.cuInit(0))
            self.device = check_cuda(driver.cuDeviceGet(self.ordinal))
            self.context = check_cuda(driver.cuDevicePrimaryCtxRetain(self.device))
            pending.callback(
                self.release_cuda, driver.cuDevicePrimaryCtxRelease, self.device
            )
            check_cuda(driver.cuCtxSetCurrent(self.context))
            self.streams = {}
            for kind in Stream:
                stream = check_cuda(
                    driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING)
                )
                self.streams[kind] = stream
                pending.callback(self.release_cuda, driver.cuStreamDestroy, stream)
            self.cleanup = pending.pop_all()

    @staticmethod
    def release_cuda[*Args](
        function: Callable[[*Args], tuple[driver.CUresult]], *arguments: *Args
    ) -> None:
        check_cuda(function(*arguments))

    def __enter__(self) -> Self:
        try:
            check_cuda(driver.cuCtxSetCurrent(self.context))
            if self.pinned_nbytes:
                self.pinned_address = int(
                    check_cuda(driver.cuMemHostAlloc(self.pinned_nbytes, 0))
                )
                self.cleanup.callback(
                    self.release_cuda, driver.cuMemFreeHost, self.pinned_address
                )
        except BaseException:
            self._close()
            raise
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        exception: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        try:
            # Synchronize every stream even when one reports an asynchronous error.
            with contextlib.ExitStack() as synchronizing:
                for stream in self.streams.values():
                    synchronizing.callback(
                        self.release_cuda, driver.cuStreamSynchronize, stream
                    )
        finally:
            self._close()

    def _close(self) -> None:
        """Release registered resources and clear handles even when release fails."""
        try:
            self.cleanup.close()
        finally:
            self.kernel_modules.clear()
            self.streams.clear()
            self.pinned_address = 0
            self.workspace = None

    @property
    def total_memory(self) -> int:
        return check_cuda(driver.cuDeviceTotalMem(self.device))

    def synchronize(self, stream: Stream) -> None:
        check_cuda(driver.cuStreamSynchronize(self.streams[stream]))


@dataclasses.dataclass(slots=True)
class Runtime:
    """Keep one invocation's addressed resources alive."""

    cuda: CudaRuntime
    inputs: Sequence[TensorDesc]
    outputs: Sequence[TensorDesc]
    paths: Sequence[Path]
    workspace_nbytes: int
    pointer: TensorDesc = dataclasses.field(init=False)
    events: dict[int, driver.CUevent] = dataclasses.field(
        default_factory=dict, init=False
    )
    mapped_hosts: list[int] = dataclasses.field(default_factory=list, init=False)

    @property
    def pinned_address(self) -> int:
        return self.cuda.pinned_address

    def __enter__(self) -> Self:
        assert _runtime.get() is None, "nested Runtime contexts are forbidden"
        shared = self.cuda.workspace
        assert shared is not None, "pipeline must allocate the shared workspace"
        assert self.workspace_nbytes <= shared.num_bytes, (
            f"runtime needs {self.workspace_nbytes} workspace bytes, "
            f"pipeline arena has {shared.num_bytes}"
        )
        self.pointer = shared
        _runtime.set(self)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        exception: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        try:
            with contextlib.ExitStack() as cleanup:
                for address in reversed(self.mapped_hosts):
                    cleanup.callback(
                        CudaRuntime.release_cuda, driver.cuMemFreeHost, address
                    )
                for event in reversed(self.events.values()):
                    cleanup.callback(
                        CudaRuntime.release_cuda, driver.cuEventDestroy, event
                    )
                for stream_kind in reversed(Stream):
                    cleanup.callback(self.cuda.synchronize, stream_kind)
        finally:
            self.events.clear()
            self.mapped_hosts.clear()
            _runtime.set(None)

_runtime: ContextVar[Runtime | None] = ContextVar("runtime", default=None)
_active_stream: ContextVar[Stream] = ContextVar("active_stream", default=Stream.COMPUTE)
# Preparation warms factories; binding prepares attributes with launch calls disabled.
# Context-local modes allow CPU preparation alongside another thread's execution.
_compiling: ContextVar[Literal["prepare", "bind"] | None] = ContextVar(
    "compiling", default=None
)


def current() -> Runtime:
    state = _runtime.get()
    assert state is not None, "runtime access requires `with Runtime(...)`"
    return state


def stream() -> driver.CUstream:
    return current().cuda.streams[_active_stream.get()]


# TileLang's backend-specific adapter and generated Python launcher are dynamic.
def tilelang_kernel_handles(compiled: Any) -> dict[str, driver.CUkernel]:
    """Bind handles per adapter; NVRTC's class-level dictionary aliases variants."""
    adapter = compiled.adapter
    kernels: dict[str, driver.CUkernel] | None = getattr(
        adapter, "_nano_omni_kernels", None
    )
    if kernels is None:
        library = adapter.lib_generator.culib
        kernels = {
            name: check_cuda(driver.cuLibraryGetKernel(library, bytes(name, "utf-8")))
            for name in adapter.function_names
        }
        adapter._nano_omni_kernels = kernels
    return kernels


def launch_tilelang(compiled: Any, *arguments: object) -> None:
    if _compiling.get() == "prepare":
        # Factory compilation already ran before this launch boundary.
        return
    adapter = compiled.adapter
    cache_tilelang_kernel_attributes(adapter.pymodule)
    kernels = tilelang_kernel_handles(compiled)
    compiling = _compiling.get() == "bind"
    device_arguments = tuple(
        dataclasses.replace(
            argument,
            kind=TensorKind.DEVICE,
            info=(0, 0),
        )
        if compiling and isinstance(argument, TensorDesc)
        else argument
        for argument in arguments
    )
    if compiling:
        # NVRTC emits launch/TMA calls inline in this function. Keep its binding
        # stubs private: cached specializations also serve concurrent execution.
        call: types.FunctionType = adapter.pymodule.call
        namespace = call.__globals__.copy()
        for name in (
            "cuLaunchKernelEx",
            "cuTensorMapEncodeTiled",
            "cuTensorMapEncodeIm2col",
        ):
            if namespace.get(name) is not None:
                result = (
                    (driver.CUresult.CUDA_SUCCESS, driver.CUtensorMap())
                    if name != "cuLaunchKernelEx"
                    else (driver.CUresult.CUDA_SUCCESS,)
                )
                namespace[name] = lambda *_args, _result=result, **_kwargs: _result
        bind_call = types.FunctionType(
            call.__code__, namespace, call.__name__, call.__defaults__, call.__closure__
        )
        bind_call.__kwdefaults__ = call.__kwdefaults__
        bind_call(kernels, *device_arguments, stream=0)
        return
    adapter.pymodule.call(kernels, *device_arguments, stream=int(stream()))


def cache_tilelang_kernel_attributes(pymodule: _TilelangModule) -> None:
    if "_nano_omni_kernel_attribute_results" in vars(pymodule):
        return
    original = pymodule.cuKernelSetAttribute
    results: dict[tuple[tuple[type[object], int], ...], tuple[driver.CUresult]] = {}

    def cached(
        *arguments: *tuple[
            driver.CUfunction_attribute, int, driver.CUkernel, driver.CUdevice
        ],
    ) -> tuple[driver.CUresult]:
        key = tuple((type(argument), int(argument)) for argument in arguments)
        result = results.get(key)
        if result is None:
            attribute, requested, kernel, device = arguments
            if (
                attribute
                == driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                and requested
                <= check_cuda(driver.cuKernelGetAttribute(attribute, kernel, device))
            ):
                # The launcher needs capacity, not an exact upper bound. Avoid
                # changing a device-wide kernel attribute when it already fits.
                result = (driver.CUresult.CUDA_SUCCESS,)
            else:
                result = original(*arguments)
            results[key] = result
        return result

    pymodule.cuKernelSetAttribute = cached
    pymodule._nano_omni_kernel_attribute_results = results


def zero_u32(value: TensorDesc) -> None:
    if _compiling.get() is not None:
        return
    check_cuda(driver.cuMemsetD32Async(value.data_ptr(), 0, 1, stream()))


def run(commands: StreamPlan) -> None:
    from nano_omni.core.runtime import submission

    submission.run(commands)
