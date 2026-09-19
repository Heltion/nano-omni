"""Add weighted position-table rows to vision patch tokens."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class VisionPositionWorkload(Workload):
    """Patch rows, feature width, and position-table row count."""

    tokens: int
    columns: int
    table_rows: int


@dataclasses.dataclass(frozen=True, slots=True)
class VisionPositionArguments(Arguments):
    """Patch tokens, position table, four interpolation rows, and output."""

    input: TensorDesc
    table: TensorDesc
    indices: TensorDesc
    weights: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def vision_position(tokens, columns, table_rows, tile_elements=128, threads=64):
    """Build four-row weighted position interpolation."""
    import tilelang.language as T

    elements = tokens * columns

    @T.prim_func
    def main(
        input: T.Tensor([tokens, columns], T.bfloat16),
        table: T.Tensor([table_rows, columns], T.bfloat16),
        indices: T.Tensor([4, tokens], T.uint32),
        weights: T.Tensor([4, tokens], T.float32),
        output: T.Tensor([tokens, columns], T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                offset = block * tile_elements + local
                if offset < elements:
                    token = offset // columns
                    column = offset % columns
                    output[token, column] = (
                        T.cast(input[token, column], T.float32)
                        + T.cast(table[indices[0, token], column], T.float32)
                        * weights[0, token]
                        + T.cast(table[indices[1, token], column], T.float32)
                        * weights[1, token]
                        + T.cast(table[indices[2, token], column], T.float32)
                        * weights[2, token]
                        + T.cast(table[indices[3, token], column], T.float32)
                        * weights[3, token]
                    )

    return main


class VisionPositionKernel(
    ElementwiseKernel[VisionPositionArguments, VisionPositionWorkload]
):
    """Interpolate position embeddings and add them to vision tokens."""

    name = "vision_position"
    program = vision_position
    launch = (128, 64)

    @classmethod
    def make_arguments(
        cls, workload: VisionPositionWorkload
    ) -> VisionPositionArguments:
        """Describe token, lookup, interpolation, and output tensors."""
        shape = (workload.tokens, workload.columns)
        return VisionPositionArguments(
            input=TensorDesc.empty(DType.BF16, shape),
            table=TensorDesc.empty(DType.BF16, (workload.table_rows, workload.columns)),
            indices=TensorDesc.empty(DType.U32, (4, workload.tokens)),
            weights=TensorDesc.empty(DType.F32, (4, workload.tokens)),
            output=TensorDesc.empty(DType.BF16, shape),
        )

    @classmethod
    def make_workload(
        cls, arguments: VisionPositionArguments
    ) -> VisionPositionWorkload:
        """Validate tensor shapes and recover the interpolation workload."""
        assert len(arguments.input.shape) == 2
        tokens, columns = arguments.input.shape
        assert arguments.input.dtype == arguments.table.dtype == DType.BF16
        assert arguments.output.dtype == DType.BF16
        assert arguments.output.shape == arguments.input.shape
        assert arguments.indices.dtype == DType.U32
        assert arguments.weights.dtype == DType.F32
        assert arguments.indices.shape == arguments.weights.shape == (4, tokens)
        assert arguments.table.shape[1] == columns
        return VisionPositionWorkload(
            tokens=tokens,
            columns=columns,
            table_rows=arguments.table.shape[0],
        )

    @classmethod
    def ref_program(cls, arguments: VisionPositionArguments) -> None:
        """Evaluate all four weighted gathers in one Torch expression."""
        indices = arguments.indices.as_torch().long()
        additions = arguments.table.as_torch()[indices].float()
        additions *= arguments.weights.as_torch()[:, :, None]
        arguments.output.as_torch().copy_(
            arguments.input.as_torch().float() + additions.sum(dim=0)
        )
