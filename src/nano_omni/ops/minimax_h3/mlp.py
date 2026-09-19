"""One complete H3 MLP slice, with model-selected activation bindings."""

from __future__ import annotations

import dataclasses
from typing import TypedDict, Unpack

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.residual_gate import (
    ResidualGateArguments,
    ResidualGateKernel,
)
from nano_omni.kernels.matmul.matmul_int8 import MatmulInt8Arguments
from nano_omni.kernels.matmul.matmul_int8_lora_residual_gate import (
    MatmulInt8LoraResidualGateArguments,
    MatmulInt8LoraResidualGateKernel,
)
from nano_omni.kernels.normalization.adaptive_rms_norm import (
    AdaptiveRmsNormArguments,
    AdaptiveRmsNormKernel,
)
from nano_omni.ops.matmul.sequence import KernelSequence, PreparedInput


class ProjectionOptions(TypedDict, total=False):
    bias: TensorDesc | None
    residual: TensorDesc | None
    multiply: TensorDesc | None
    use_silu: bool


@dataclasses.dataclass(frozen=True)
class Projection:
    weight: TensorDesc
    weight_scale: float | TensorDesc = 1.0
    input_scale: TensorDesc | None = None
    loras: tuple[tuple[TensorDesc, TensorDesc, float], ...] = ()
    rotation_group: int = 0

    def emit(
        self,
        sequence: KernelSequence,
        name: str,
        input: TensorDesc | PreparedInput,
        output: TensorDesc,
        **kwargs: Unpack[ProjectionOptions],
    ) -> None:
        sequence.linear(
            name,
            input,
            self.weight,
            output,
            weight_scale=self.weight_scale,
            input_scale=self.input_scale,
            loras=self.loras,
            rotation_group=self.rotation_group,
            **kwargs,
        )


@dataclasses.dataclass(frozen=True)
class MLPWeights:
    norm: TensorDesc
    gate: Projection
    up: Projection
    down: Projection


class MLP(Op[MLPWeights]):
    def __init__(
        self,
        hidden: TensorDesc,
        shift: TensorDesc,
        scale: TensorDesc,
        gate: TensorDesc,
    ) -> None:
        rows, width = hidden.shape
        assert shift.shape == scale.shape == gate.shape == (width,)
        self.rows = rows
        self.width = width
        super().__init__(
            hidden, shift, scale, gate, outputs=((DType.BF16, hidden.shape),)
        )

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        outputs = self.bound_outputs
        assert isinstance(self.weights, MLPWeights), "MLP weights are required"
        assert len(outputs) == 1, "MLP requires one output"
        weights = self.weights
        shape = (self.rows, self.width)
        hidden, shift, scale, gate_modulation = self.inputs
        output = outputs[0]
        sequence = KernelSequence(scratch)
        normalized = sequence.temporary("normalized", DType.BF16, shape)
        expanded = (self.rows, weights.gate.weight.shape[0])
        gate = sequence.temporary("gate", DType.BF16, expanded)
        up = sequence.temporary("up", DType.BF16, expanded)
        # The INT8 + LoRA epilogue retains BF16 rounding and skips update storage.
        # The measured no-LoRA path keeps its existing projection and gate calls.
        fused_down = weights.down.weight.dtype == DType.I8 and bool(weights.down.loras)
        update = (
            output if fused_down else sequence.temporary("update", DType.BF16, shape)
        )
        sequence.emit(
            AdaptiveRmsNormKernel,
            AdaptiveRmsNormArguments(
                hidden,
                weights.norm,
                shift,
                scale,
                normalized,
            ),
        )
        expanded_input = sequence.prepare(normalized)
        weights.gate.emit(sequence, "gate", expanded_input, gate, use_silu=True)
        weights.up.emit(sequence, "up", expanded_input, up, multiply=gate)
        weights.down.emit(sequence, "down", up, update)
        if fused_down:
            # Reuse all preparation and ordered LoRA residuals emitted by linear.
            projection = sequence.calls.pop().arguments
            assert isinstance(projection, MatmulInt8Arguments)
            assert projection.residual is not None
            sequence.emit(
                MatmulInt8LoraResidualGateKernel,
                MatmulInt8LoraResidualGateArguments(
                    projection.input,
                    projection.weight,
                    projection.residual,
                    output,
                    projection.weight_scale,
                    projection.input_scale,
                    hidden,
                    gate_modulation,
                ),
            )
            return sequence.calls
        sequence.emit(
            ResidualGateKernel,
            ResidualGateArguments(
                hidden,
                update,
                gate_modulation,
                output,
            ),
        )
        return sequence.calls
