"""Per-layer timestep modulation into a fixed model activation."""

from __future__ import annotations

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.matmul_f16 import MatmulF16Arguments, MatmulF16Kernel


class Modulation(Op[tuple[TensorDesc, TensorDesc]]):
    def __init__(
        self, curve: int | tuple[int, int], *, rows: int, columns: int = 18 * 5376
    ) -> None:
        super().__init__(
            TensorDesc.activation(curve, DType.F32, (rows, 8)),
            outputs=((DType.BF16, (rows, columns)),),
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel, ...]:
        outputs = self.bound_outputs
        weights = self.bound_weights
        assert len(outputs) == 1 and len(weights) == 2, "invalid modulation bindings"
        weight, bias = weights
        input, output = self.inputs[0], outputs[0]
        return (
            MatmulF16Kernel(
                MatmulF16Arguments(input, weight, bias, None, None, None, output, False)
            ),
        )
