"""Adaptive RMS normalization over model-selected contiguous rows."""

from collections.abc import Iterable

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import TensorDesc
from nano_omni.kernels.normalization.adaptive_rms_norm import (
    AdaptiveRmsNormArguments,
    AdaptiveRmsNormKernel,
)


class AdaptiveRmsNorm(Op[TensorDesc]):
    def __init__(self, input: TensorDesc, shift: TensorDesc, scale: TensorDesc) -> None:
        super().__init__(input, shift, scale, outputs=((input.dtype, input.shape),))

    def kernels(self, scratch: ScratchLayout) -> Iterable[Kernel]:
        del scratch
        input, shift, scale = self.inputs
        yield AdaptiveRmsNormKernel(
            AdaptiveRmsNormArguments(
                input, self.bound_weights, shift, scale, self.bound_outputs[0]
            )
        )
