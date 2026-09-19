"""Bind F32 transposed audio convolution without allocating scratch."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.convolution.conv_transpose1d import (
    ConvTranspose1dArguments,
    ConvTranspose1dKernel,
)


class ConvTranspose1dOp(Op[tuple[TensorDesc, TensorDesc | None]]):
    def __init__(
        self,
        input: int | tuple[int, int],
        *,
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        stereo: int,
        stride: int,
        padding: int = 0,
    ) -> None:
        self.stride, self.padding = stride, padding
        assert input_shape[0] % stereo == 0
        assert output_shape[0] % stereo == 0
        super().__init__(
            TensorDesc.activation(
                input, DType.F32, (stereo, input_shape[0] // stereo, input_shape[1])
            ),
            outputs=(
                (
                    DType.F32,
                    (stereo, output_shape[0] // stereo, output_shape[1]),
                ),
            ),
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        weight, bias = self.bound_weights
        return (
            ConvTranspose1dKernel(
                ConvTranspose1dArguments(
                    self.inputs[0],
                    weight,
                    bias,
                    output,
                    self.stride,
                    self.padding,
                )
            ),
        )
