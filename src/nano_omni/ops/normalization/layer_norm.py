"""Bind row-wise layer normalization with optional affine weights."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.normalization.layer_norm import (
    LayerNormArguments,
    LayerNormKernel,
)


class LayerNormOp(Op[tuple[TensorDesc | None, TensorDesc | None]]):
    def __init__(
        self,
        input: int | tuple[int, int],
        *,
        dtype: DType,
        shape: tuple[int, int],
        epsilon: float = 1e-6,
    ) -> None:
        self.epsilon = epsilon
        super().__init__(
            TensorDesc.activation(input, dtype, shape), outputs=((dtype, shape),)
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        weight, bias = self.weights if self.weights is not None else (None, None)
        return (
            LayerNormKernel(
                LayerNormArguments(
                    self.inputs[0],
                    weight,
                    bias,
                    output,
                    self.epsilon,
                )
            ),
        )
