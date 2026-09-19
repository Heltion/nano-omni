"""Denormalize channel-first audio latents while transposing to token rows."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.channel_affine_transpose import (
    ChannelAffineTransposeArguments,
    ChannelAffineTransposeKernel,
)


class ChannelAffineTransposeOp(Op[tuple[TensorDesc, TensorDesc]]):
    def __init__(
        self, input: int | tuple[int, int], *, channels: int, stereo: int, frames: int
    ) -> None:
        input_shape = (channels, stereo, frames)
        output_shape = (stereo * frames, channels)
        super().__init__(
            TensorDesc.activation(input, DType.F32, input_shape),
            outputs=((DType.F32, output_shape),),
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        mean, std = self.bound_weights
        return (
            ChannelAffineTransposeKernel(
                ChannelAffineTransposeArguments(
                    self.inputs[0],
                    mean,
                    std,
                    output,
                )
            ),
        )
