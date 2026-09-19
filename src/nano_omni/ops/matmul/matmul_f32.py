"""Emit one F32 matrix projection with optional bias and BF16 output."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.matmul_f32 import MatmulF32Arguments, MatmulF32Kernel


class MatmulF32Op(Op[tuple[TensorDesc, TensorDesc | None]]):
    def __init__(
        self,
        input: int | tuple[int, int],
        *,
        rows: int,
        input_columns: int,
        output_columns: int,
        output_bf16: bool = False,
    ) -> None:
        self.output_bf16 = output_bf16
        output_dtype = DType.BF16 if output_bf16 else DType.F32
        super().__init__(
            TensorDesc.activation(input, DType.F32, (rows, input_columns)),
            outputs=((output_dtype, (rows, output_columns)),),
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        (output,) = self.bound_outputs
        matrix, bias = self.bound_weights
        return (
            MatmulF32Kernel(
                MatmulF32Arguments(
                    self.inputs[0],
                    matrix,
                    bias,
                    output,
                    self.output_bf16,
                )
            ),
        )
