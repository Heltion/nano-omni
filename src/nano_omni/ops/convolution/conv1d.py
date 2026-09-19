"""Bind F32 audio convolution in flattened [stereo * frames, channels] layout."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.convolution.conv1d import Conv1dArguments, Conv1dKernel


class Conv1dOp(Op[tuple[TensorDesc, TensorDesc | None]]):
    """Run Conv1d with explicit activation and weight bindings."""

    def __init__(
        self,
        input: int | tuple[int, int],
        residual: int | tuple[int, int] | None = None,
        *,
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        stereo: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        clamp: bool = False,
    ) -> None:
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.clamp = clamp
        assert input_shape[0] % stereo == 0 and output_shape[0] % stereo == 0
        input_tensor_shape = (stereo, input_shape[0] // stereo, input_shape[1])
        output_tensor_shape = (stereo, output_shape[0] // stereo, output_shape[1])
        inputs = [TensorDesc.activation(input, DType.F32, input_tensor_shape)]
        if residual is not None:
            inputs.append(
                TensorDesc.activation(residual, DType.F32, output_tensor_shape)
            )
        super().__init__(*inputs, outputs=((DType.F32, output_tensor_shape),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        weight, bias = self.bound_weights
        residual = self.inputs[1] if len(self.inputs) == 2 else None
        return (
            Conv1dKernel(
                Conv1dArguments(
                    self.inputs[0],
                    weight,
                    bias,
                    residual,
                    output,
                    self.stride,
                    self.dilation,
                    self.padding,
                    self.clamp,
                )
            ),
        )
