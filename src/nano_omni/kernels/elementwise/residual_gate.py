"""Add a per-column gated update to a residual tensor."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class ResidualGateWorkload(Workload):
    num_tokens: int
    columns: int


@dataclasses.dataclass(frozen=True, slots=True)
class ResidualGateArguments(Arguments):
    residual: TensorDesc
    update: TensorDesc
    gate: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.residual.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def residual_gate(
    num_tokens: int,
    columns: int,
    tile_elements: int = 1024,
    threads: int = 128,
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    num_elements = dynamic_num_tokens * columns

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        residual: T.Tensor((dynamic_num_tokens, columns), T.bfloat16),
        update: T.Tensor((dynamic_num_tokens, columns), T.bfloat16),
        gate: T.Tensor((columns,), T.bfloat16),
        output: T.Tensor((dynamic_num_tokens, columns), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(num_elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < num_elements:
                    row, column = index // columns, index % columns
                    output[row, column] = (
                        residual[row, column] + update[row, column] * gate[column]
                    )

    return main.with_attr(
        "global_symbol", f"residual_gate_{columns}_{tile_elements}_{threads}"
    )


class ResidualGateKernel(
    ElementwiseKernel[ResidualGateArguments, ResidualGateWorkload]
):
    name = "residual_gate"
    program = residual_gate
    launch = (512, 128)

    @classmethod
    def make_arguments(cls, workload: ResidualGateWorkload) -> ResidualGateArguments:
        """Describe BF16 residual, update, gate, and output tensors."""
        shape = (workload.num_tokens, workload.columns)
        return ResidualGateArguments(
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(DType.BF16, (workload.columns,)),
            TensorDesc.empty(DType.BF16, shape),
        )

    @classmethod
    def make_workload(cls, arguments: ResidualGateArguments) -> ResidualGateWorkload:
        assert all(tensor.dtype == DType.BF16 for tensor in arguments.values())
        num_tokens, columns = arguments.residual.shape
        assert arguments.update.shape == (num_tokens, columns)
        assert arguments.output.shape == (num_tokens, columns)
        assert arguments.gate.shape == (columns,)
        return ResidualGateWorkload(num_tokens=num_tokens, columns=columns)

    @classmethod
    def ref_program(cls, arguments: ResidualGateArguments) -> None:
        """Apply the per-column gated update into output."""
        value = arguments.residual.as_torch() + (
            arguments.update.as_torch() * arguments.gate.as_torch()
        )
        arguments.output.as_torch().copy_(value)
