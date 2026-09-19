"""Shared CUDA buffer ownership for complete generation pipelines."""

import contextlib
import dataclasses
from types import TracebackType
from typing import Self

import numpy

# cuda-bindings ships this Cython extension without a Python stub.
from cuda.bindings import driver  # pyrefly: ignore[missing-module-attribute]
from numpy.typing import DTypeLike, NDArray

from nano_omni.core.runtime.execution import CudaRuntime, check_cuda
from nano_omni.core.runtime.synchronization import Synchronization
from nano_omni.core.tensor import DType, TensorDesc, TensorKind

NUMPY_TO_DTYPE = {
    numpy.dtype(numpy.float32): DType.F32,
    numpy.dtype(numpy.float16): DType.F16,
    numpy.dtype(numpy.uint16): DType.BF16,
    numpy.dtype(numpy.int32): DType.I32,
    numpy.dtype(numpy.uint32): DType.U32,
    numpy.dtype(numpy.int8): DType.I8,
    numpy.dtype(numpy.uint8): DType.U8,
}
DTYPE_TO_NUMPY = {dtype: numpy_dtype for numpy_dtype, dtype in NUMPY_TO_DTYPE.items()}


@dataclasses.dataclass(slots=True)
class PipelineMemory:
    pinned_nbytes: int
    workspace_nbytes: int = 0
    synchronization: Synchronization = dataclasses.field(
        default_factory=Synchronization
    )
    vram_fraction: float | None = None
    runtime: CudaRuntime = dataclasses.field(init=False)
    capacities: dict[int, int] = dataclasses.field(init=False, default_factory=dict)
    inactive: set[int] = dataclasses.field(init=False, default_factory=set)

    def __enter__(self) -> Self:
        self.runtime = CudaRuntime(
            pinned_nbytes=self.pinned_nbytes,
            synchronization=self.synchronization,
        )
        try:
            self.runtime.__enter__()
            if self.workspace_nbytes:
                self.runtime.workspace = self.empty(
                    (self.workspace_nbytes,), numpy.uint8
                )
        except BaseException:
            for pointer in tuple(self.capacities):
                check_cuda(driver.cuMemFree(driver.CUdeviceptr(pointer)))
            self.capacities.clear()
            self.runtime.cleanup.close()
            raise
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            with contextlib.ExitStack() as cleanup:
                cleanup.callback(
                    self.runtime.__exit__, exception_type, exception, traceback
                )
                for pointer in reversed(self.capacities):
                    cleanup.callback(
                        CudaRuntime.release_cuda,
                        driver.cuMemFree,
                        driver.CUdeviceptr(pointer),
                    )
                for kind in reversed(self.runtime.streams):
                    cleanup.callback(self.runtime.synchronize, kind)
        finally:
            self.capacities.clear()
            self.inactive.clear()

    def check_capacity(self, nbytes: int) -> None:
        if self.vram_fraction is None:
            return
        result = driver.cuMemGetInfo()
        check_cuda(result)
        _, free, total = result
        limit = int(total * self.vram_fraction)
        if total - free + nbytes > limit:
            raise MemoryError(
                f"VRAM budget exceeded: used={total - free}, request={nbytes}, limit={limit}"
            )

    def empty(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> TensorDesc:
        resolved = numpy.dtype(dtype)
        assert resolved in NUMPY_TO_DTYPE, f"unsupported pipeline dtype: {resolved}"
        nbytes = int(numpy.prod(shape)) * resolved.itemsize
        for pointer in tuple(self.inactive):
            if self.capacities[pointer] >= nbytes:
                self.inactive.remove(pointer)
                return TensorDesc(
                    NUMPY_TO_DTYPE[resolved], shape, TensorKind.DEVICE, (pointer, 0)
                )
        self.check_capacity(nbytes)
        pointer = int(check_cuda(driver.cuMemAlloc(nbytes)))
        self.capacities[pointer] = nbytes
        return TensorDesc(
            NUMPY_TO_DTYPE[resolved], shape, TensorKind.DEVICE, (pointer, 0)
        )

    def upload(self, value: NDArray[numpy.generic]) -> TensorDesc:
        host = numpy.ascontiguousarray(value)
        buffer = self.empty(host.shape, host.dtype)
        check_cuda(
            driver.cuMemcpyHtoD(
                driver.CUdeviceptr(buffer.data_ptr()), host.ctypes.data, host.nbytes
            )
        )
        return buffer

    def download(self, buffer: TensorDesc) -> NDArray[numpy.generic]:
        output = numpy.empty(buffer.shape, dtype=DTYPE_TO_NUMPY[buffer.dtype])
        check_cuda(
            driver.cuMemcpyDtoH(
                output.ctypes.data,
                driver.CUdeviceptr(buffer.data_ptr()),
                buffer.num_bytes,
            )
        )
        return output

    def release(self, *buffers: TensorDesc | None) -> None:
        for buffer in buffers:
            if buffer is not None:
                pointer = buffer.data_ptr()
                assert pointer in self.capacities, "pipeline does not own tensor"
                assert pointer not in self.inactive, "pipeline tensor already released"
                self.inactive.add(pointer)

    def trim(self) -> None:
        """Return inactive stage buffers after all stream users have finished."""
        for kind in self.runtime.streams:
            self.runtime.synchronize(kind)
        for pointer in tuple(self.inactive):
            check_cuda(driver.cuMemFree(driver.CUdeviceptr(pointer)))
            del self.capacities[pointer]
            self.inactive.remove(pointer)

    def concatenate_rows(self, groups: tuple[TensorDesc, ...]) -> TensorDesc:
        assert groups
        tail = groups[0].shape[1:]
        dtype = groups[0].dtype
        assert all(item.shape[1:] == tail and item.dtype == dtype for item in groups)
        output = self.empty(
            (sum(item.shape[0] for item in groups), *tail), DTYPE_TO_NUMPY[dtype]
        )
        offset = 0
        for item in groups:
            check_cuda(
                driver.cuMemcpyDtoD(
                    driver.CUdeviceptr(output.data_ptr() + offset),
                    driver.CUdeviceptr(item.data_ptr()),
                    item.num_bytes,
                )
            )
            offset += item.num_bytes
        return output

    def memory_info(self) -> tuple[int, int]:
        result = driver.cuMemGetInfo()
        check_cuda((result[0],))
        return int(result[1]), int(result[2])
