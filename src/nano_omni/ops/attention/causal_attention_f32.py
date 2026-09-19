"""Bind equal-width F32 query, key, and value matrices for causal attention."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.causal_attention import (
    CausalAttentionArguments,
    CausalAttentionKernel,
)


class CausalAttentionF32Op(Op[None]):
    def __init__(
        self,
        query: int | tuple[int, int],
        key: int | tuple[int, int],
        value: int | tuple[int, int],
        *,
        rows: int,
        width: int,
        dim: int,
    ) -> None:
        assert width % dim == 0
        shape = (rows, width // dim, dim)
        inputs = tuple(
            TensorDesc.activation(position, DType.F32, shape)
            for position in (query, key, value)
        )
        super().__init__(*inputs, outputs=((DType.F32, shape),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        query, key, value = self.inputs
        return (
            CausalAttentionKernel(
                CausalAttentionArguments(
                    query,
                    key,
                    value,
                    output,
                    query.shape[0],
                )
            ),
        )
