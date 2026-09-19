"""Text refinement operators with fixed activation bindings."""

from __future__ import annotations

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.residual import ResidualArguments, ResidualKernel
from nano_omni.kernels.normalization.rms_norm import RmsNormArguments, RmsNormKernel
from nano_omni.ops.matmul.sequence import KernelSequence
from nano_omni.ops.minimax_h3.attention import AttentionWeights, project_attention
from nano_omni.ops.minimax_h3.mlp import MLPWeights


class ContextProjection(Op[tuple[TensorDesc, TensorDesc]]):
    def __init__(self, context: int | tuple[int, int], *, rows: int) -> None:
        self.rows = rows
        super().__init__(
            TensorDesc.activation(context, DType.BF16, (rows, 5120)),
            outputs=((DType.BF16, (rows, 5376)),),
        )

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        outputs = self.bound_outputs
        weights = self.bound_weights
        weight, bias = weights
        sequence = KernelSequence(scratch)
        sequence.linear(
            "context",
            self.inputs[0],
            weight,
            outputs[0],
            bias=bias,
        )
        return sequence.calls


class Refiner[WeightsT](Op[WeightsT]):
    def __init__(self, hidden: int | tuple[int, int], *, rows: int) -> None:
        self.rows = rows
        shape = (rows, 5376)
        super().__init__(
            TensorDesc.activation(hidden, DType.BF16, shape),
            outputs=((DType.BF16, shape),),
        )


class Normalize(Refiner[TensorDesc]):
    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        outputs = self.bound_outputs
        weights = self.bound_weights
        sequence = KernelSequence(scratch)
        sequence.emit(
            RmsNormKernel,
            RmsNormArguments(
                self.inputs[0],
                weights,
                outputs[0],
                1e-6,
            ),
        )
        return sequence.calls


class RefinerAttention(Refiner[AttentionWeights]):
    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        outputs = self.bound_outputs
        weights = self.bound_weights
        sequence = KernelSequence(scratch)
        hidden, output = self.inputs[0], outputs[0]
        normalized = sequence.temporary("normalized", DType.BF16, hidden.shape)
        sequence.emit(
            RmsNormKernel,
            RmsNormArguments(hidden, weights.norm, normalized, 1e-6),
        )
        attended = project_attention(sequence, weights, normalized)
        # Preserve the linear epilogue's residual rounding while writing the fixed state.
        weights.output.emit(sequence, "output", attended, output, residual=hidden)
        return sequence.calls


class RefinerMLP(Refiner[MLPWeights]):
    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        outputs = self.bound_outputs
        weights = self.bound_weights
        sequence = KernelSequence(scratch)
        hidden, output = self.inputs[0], outputs[0]
        normalized = sequence.temporary("normalized", DType.BF16, hidden.shape)
        sequence.emit(
            RmsNormKernel,
            RmsNormArguments(hidden, weights.norm, normalized, 1e-6),
        )
        shape = (self.rows, weights.gate.weight.shape[0])
        gate = sequence.temporary("gate", DType.BF16, shape)
        up = sequence.temporary("up", DType.BF16, shape)
        update = sequence.temporary("update", DType.BF16, hidden.shape)
        weights.gate.emit(sequence, "gate", normalized, gate, use_silu=True)
        weights.up.emit(sequence, "up", normalized, up, multiply=gate)
        weights.down.emit(sequence, "down", up, update)
        sequence.emit(ResidualKernel, ResidualArguments(update, hidden, output))
        return sequence.calls
