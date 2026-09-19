"""Swap the two FP4 nibbles stored in each byte."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class SwapFp4NibblesWorkload(Workload):
    """Physical dimensions of the packed byte matrix."""

    rows: int
    packed_columns: int


@dataclasses.dataclass(frozen=True, slots=True)
class SwapFp4NibblesArguments(Arguments):
    """Packed input and output byte matrices."""

    input: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def swap_fp4_nibbles(rows, packed_columns, tile_elements=4096, threads=256):
    """Build the elementwise nibble permutation."""
    import tilelang.language as T

    elements = rows * packed_columns

    @T.prim_func
    def main(
        input: T.Tensor([rows, packed_columns], T.uint8),
        output: T.Tensor([rows, packed_columns], T.uint8),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    row = index // packed_columns
                    column = index % packed_columns
                    packed = input[row, column]
                    output[row, column] = ((packed & 15) << 4) | ((packed & 240) >> 4)

    return main


class SwapFp4NibblesKernel(
    ElementwiseKernel[
        SwapFp4NibblesArguments,
        SwapFp4NibblesWorkload,
    ]
):
    """Exchange the low and high nibble of every packed FP4 byte."""

    name = "swap_fp4_nibbles"
    program = swap_fp4_nibbles
    launch = (4096, 256)

    @classmethod
    def make_arguments(
        cls, workload: SwapFp4NibblesWorkload
    ) -> SwapFp4NibblesArguments:
        """Describe distinct packed input and output tensors."""
        shape = (workload.rows, workload.packed_columns)
        return SwapFp4NibblesArguments(
            input=TensorDesc.empty(DType.U8, shape),
            output=TensorDesc.empty(DType.U8, shape),
        )

    @classmethod
    def make_workload(
        cls, arguments: SwapFp4NibblesArguments
    ) -> SwapFp4NibblesWorkload:
        """Validate matching packed byte matrices and recover their dimensions."""
        assert arguments.input.dtype == arguments.output.dtype == DType.U8
        assert len(arguments.input.shape) == 2
        assert arguments.output.shape == arguments.input.shape
        return SwapFp4NibblesWorkload(
            rows=arguments.input.shape[0],
            packed_columns=arguments.input.shape[1],
        )

    @classmethod
    def ref_program(cls, arguments: SwapFp4NibblesArguments) -> None:
        """Exchange both nibbles directly with Torch bit operations."""
        packed = arguments.input.as_torch()
        arguments.output.as_torch().copy_(((packed & 15) << 4) | ((packed & 240) >> 4))
