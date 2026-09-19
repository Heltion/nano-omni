"""A tensor description shared by model construction, planning, and execution."""

# cspell:ignore typestr

from __future__ import annotations

import dataclasses
import math
from enum import StrEnum
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DType(StrEnum):
    F32 = "F32"
    F16 = "F16"
    I32 = "I32"
    U32 = "U32"
    I8 = "I8"
    BF16 = "BF16"
    FP8_E4M3 = "FP8_E4M3"
    FP8_UE4M3 = "FP8_UE4M3"
    U8 = "U8"
    FP4 = "U8"

    @classmethod
    def from_safetensors(cls, value: str) -> DType:
        """Translate a safetensors dtype tag into the project dtype vocabulary."""
        if value == "F8_E4M3":
            return cls.FP8_E4M3
        return cls(value)

    @property
    def itemsize(self) -> int:
        if self in (DType.F32, DType.I32, DType.U32):
            return 4
        return 2 if self in (DType.F16, DType.BF16) else 1


class TensorKind(StrEnum):
    EMPTY = "empty"
    FILE = "file"
    INPUT = "input"
    OUTPUT = "output"
    ACTIVATION = "activation"
    SCRATCH = "scratch"
    SCALAR = "scalar"
    WEIGHT = "weight"
    WORKSPACE = "workspace"
    PINNED = "pinned"
    HOST = "host"
    DEVICE = "device"


@dataclasses.dataclass(frozen=True, slots=True)
class Device:
    index: int = 0


DEVICE = Device()


@dataclasses.dataclass(frozen=True, slots=True)
class TensorDesc:
    """Tensor shape and dtype paired with one logical or addressed position."""

    dtype: DType
    shape: tuple[int, ...]
    kind: TensorKind
    info: tuple[int, ...]

    @classmethod
    def activation(
        cls,
        info: int | tuple[int, int],
        dtype: DType,
        shape: tuple[int, ...],
    ) -> TensorDesc:
        return cls(dtype, shape, TensorKind.ACTIVATION, _info(info))

    @classmethod
    def empty(cls, dtype: DType, shape: tuple[int, ...]) -> TensorDesc:
        """Describe a tensor shape without assigning storage."""
        return cls(dtype, shape, TensorKind.EMPTY, ())

    @classmethod
    def scratch(cls, offset: int, dtype: DType, shape: tuple[int, ...]) -> TensorDesc:
        return cls(dtype, shape, TensorKind.SCRATCH, (0, offset))

    @classmethod
    def bytes(
        cls,
        kind: TensorKind,
        index: int,
        offset: int,
        num_bytes: int,
    ) -> TensorDesc:
        """Describe an untyped byte range used by a copy command."""
        return cls(DType.U8, (num_bytes,), kind, (index, offset))

    @classmethod
    def from_pointer(cls, pointer: int, num_bytes: int) -> TensorDesc:
        """Describe an addressed device byte range owned by its caller."""
        return cls.bytes(TensorKind.DEVICE, pointer, 0, num_bytes)

    @property
    def num_bytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    def data_ptr(self) -> int:
        assert self.kind == TensorKind.DEVICE, "tensor has not been addressed"
        assert len(self.info) == 2, "addressed tensor requires base and offset"
        base, byte_offset = self.info
        return base + byte_offset

    def as_torch(self) -> torch.Tensor:
        """Create a zero-copy contiguous Torch view of an addressed CUDA tensor."""
        import torch

        carriers = {
            DType.F32: ("<f4", torch.float32),
            DType.F16: ("<f2", torch.float16),
            DType.I32: ("<i4", torch.int32),
            DType.U32: ("<u4", torch.uint32),
            DType.I8: ("|i1", torch.int8),
            DType.BF16: ("<u2", torch.bfloat16),
            DType.FP8_E4M3: ("|u1", torch.float8_e4m3fn),
            # Torch has no UE4M3 dtype. Positive E4M3 values have the same byte
            # encoding and provide a zero-copy carrier for NVFP4 scale factors.
            DType.FP8_UE4M3: ("|u1", torch.float8_e4m3fn),
            DType.U8: ("|u1", torch.uint8),
        }
        typestr, dtype = carriers[self.dtype]
        interface = SimpleNamespace(
            __cuda_array_interface__={
                "shape": self.shape,
                "strides": None,
                "typestr": typestr,
                "data": (self.data_ptr(), False),
                "version": 3,
            }
        )
        tensor = torch.as_tensor(interface, device=f"cuda:{self.device.index}")
        return tensor if tensor.dtype == dtype else tensor.view(dtype)

    def view(self, shape: tuple[int, ...], byte_offset: int = 0) -> TensorDesc:
        """Describe a contiguous subregion without changing its storage identity."""
        assert byte_offset >= 0, "tensor view offset must be nonnegative"
        assert byte_offset % self.dtype.itemsize == 0, (
            "tensor view offset must satisfy dtype alignment"
        )
        view = dataclasses.replace(
            self,
            shape=shape,
            info=(self.info[0], self.info[1] + byte_offset),
        )
        assert byte_offset + view.num_bytes <= self.num_bytes, (
            "tensor view exceeds source region"
        )
        return view

    @property
    def device(self) -> Device:
        return DEVICE


def _info(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, 0) if isinstance(value, int) else value
