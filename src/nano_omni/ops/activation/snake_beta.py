"""Bind resampled SnakeBeta with log alpha/beta and up/down filters."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.activation.snake_beta import SnakeBetaArguments, SnakeBetaKernel


class SnakeBetaOp(Op[tuple[TensorDesc, TensorDesc, TensorDesc, TensorDesc]]):
    def __init__(
        self,
        input: int | tuple[int, int],
        *,
        shape: tuple[int, int],
        stereo: int,
        up_ratio: int = 2,
        down_ratio: int = 2,
    ) -> None:
        self.up_ratio, self.down_ratio = up_ratio, down_ratio
        assert shape[0] % stereo == 0
        tensor_shape = (stereo, shape[0] // stereo, shape[1])
        super().__init__(
            TensorDesc.activation(input, DType.F32, tensor_shape),
            outputs=((DType.F32, tensor_shape),),
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        alpha, beta, up_filter, down_filter = self.bound_weights
        return (
            SnakeBetaKernel(
                SnakeBetaArguments(
                    self.inputs[0],
                    alpha,
                    beta,
                    up_filter,
                    down_filter,
                    output,
                    self.up_ratio,
                    self.down_ratio,
                )
            ),
        )
