"""Residual gate over model-selected contiguous rows."""

from collections.abc import Iterable

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import TensorDesc
from nano_omni.kernels.elementwise.residual_gate import (
    ResidualGateArguments,
    ResidualGateKernel,
)


class ResidualGate(Op[None]):
    def __init__(
        self, residual: TensorDesc, update: TensorDesc, gate: TensorDesc
    ) -> None:
        assert residual.shape == update.shape
        super().__init__(
            residual, update, gate, outputs=((residual.dtype, residual.shape),)
        )

    def kernels(self, scratch: ScratchLayout) -> Iterable[Kernel]:
        del scratch
        residual, update, gate = self.inputs
        yield ResidualGateKernel(
            ResidualGateArguments(residual, update, gate, self.bound_outputs[0])
        )
