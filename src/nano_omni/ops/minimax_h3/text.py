"""Qwen text operators emit kernels in execution order with explicit views."""

import dataclasses
import math

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.causal_attention import (
    CausalAttentionArguments,
    CausalAttentionKernel,
)
from nano_omni.kernels.attention.qk_norm_rope import (
    QkNormRopeArguments,
    QkNormRopeKernel,
)
from nano_omni.kernels.attention.qwen_mrope import QwenMropeArguments, QwenMropeKernel
from nano_omni.kernels.attention.qwen_rope import QwenRopeArguments, QwenRopeKernel
from nano_omni.kernels.layout.embedding_int8 import (
    EmbeddingInt8Arguments,
    EmbeddingInt8Kernel,
)
from nano_omni.kernels.matmul.matmul_fp4 import MatmulFp4Arguments, MatmulFp4Kernel
from nano_omni.kernels.normalization.rms_norm import RmsNormArguments, RmsNormKernel
from nano_omni.kernels.quantization.fp8_scale import Fp8ScaleArguments, Fp8ScaleKernel
from nano_omni.kernels.quantization.quantize_fp4 import (
    QuantizeFp4Arguments,
    QuantizeFp4Kernel,
)
from nano_omni.kernels.quantization.swap_fp4_nibbles import (
    SwapFp4NibblesArguments,
    SwapFp4NibblesKernel,
)
from nano_omni.ops.matmul.sequence import KernelSequence


@dataclasses.dataclass(frozen=True)
class Fp4Projection:
    weight: TensorDesc
    scales: TensorDesc
    weight_scale: TensorDesc
    input_scale: TensorDesc | None = None
    pre_scale: TensorDesc | None = None


@dataclasses.dataclass(frozen=True)
class TextLayerWeights:
    input_norm: TensorDesc
    query_norm: TensorDesc
    key_norm: TensorDesc
    query: Fp4Projection
    key: Fp4Projection
    value: Fp4Projection
    output: Fp4Projection
    post_norm: TensorDesc
    gate: Fp4Projection
    up: Fp4Projection
    down: Fp4Projection


class TextEmbedding(Op[tuple[TensorDesc, TensorDesc]]):
    def __init__(
        self, tokens: int | tuple[int, int], *, sequence: int, rows: int, width: int
    ) -> None:
        super().__init__(
            TensorDesc.activation(tokens, DType.U32, (sequence,)),
            outputs=((DType.BF16, (rows, width)),),
        )

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel, ...]:
        outputs, weights = self.bound_outputs, self.bound_weights
        assert len(outputs) == 1 and len(weights) == 2, "invalid embedding bindings"
        table, scales = weights
        return (
            EmbeddingInt8Kernel(
                EmbeddingInt8Arguments(
                    self.inputs[0],
                    table,
                    scales,
                    outputs[0],
                )
            ),
        )


class TextRope(Op[None]):
    def __init__(
        self,
        positions: int | tuple[int, int] | None = None,
        inverse_frequencies: int | tuple[int, int] | None = None,
        *,
        sequence: int,
        mrope: bool,
    ) -> None:
        inputs: tuple[TensorDesc, ...] = ()
        if positions is None:
            assert inverse_frequencies is None, "frequencies require positions"
        else:
            assert inverse_frequencies is not None, (
                "rotary positions require frequencies"
            )
            inputs = (
                TensorDesc.activation(positions, DType.F32, (sequence, 3)),
                TensorDesc.activation(inverse_frequencies, DType.F32, (64,)),
            )
        self.sequence, self.mrope = sequence, mrope
        super().__init__(*inputs, outputs=((DType.F32, (sequence, 64)),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel, ...]:
        output = self.bound_outputs[0]
        if not self.mrope:
            return (QwenRopeKernel(QwenRopeArguments(output)),)
        return (
            QwenMropeKernel(
                QwenMropeArguments(
                    self.inputs[0],
                    self.inputs[1],
                    output,
                )
            ),
        )


class TextLayer(Op[TextLayerWeights]):
    def __init__(
        self,
        hidden: int | tuple[int, int],
        angles: int | tuple[int, int],
        *,
        rows: int,
        sequence: int,
    ) -> None:
        self.rows, self.sequence = rows, sequence
        shape = (rows, 5120)
        inputs = (
            TensorDesc.activation(hidden, DType.BF16, shape),
            TensorDesc.activation(angles, DType.F32, (sequence, 64)),
        )
        super().__init__(*inputs, outputs=((DType.BF16, shape),))

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        outputs, weights = self.bound_outputs, self.bound_weights
        assert isinstance(weights, TextLayerWeights), "text layer weights are required"
        assert len(outputs) == 1, "text layer requires one output"
        sequence = KernelSequence(scratch)
        hidden, angles = self.inputs
        output = outputs[0]
        normalized = sequence.temporary("input_norm", DType.BF16, hidden.shape)
        sequence.emit(
            RmsNormKernel,
            RmsNormArguments(hidden, weights.input_norm, normalized, 1e-6),
        )
        projections = (
            weights.query,
            weights.key,
            weights.value,
            weights.output,
            weights.gate,
            weights.up,
            weights.down,
        )
        weight_buffer = scratch.reserve(
            "weight",
            DType.U8,
            (max(math.prod(projection.weight.shape) for projection in projections),),
        )
        padded_rows = -(-self.rows // 128) * 128
        maximum_columns = max(
            2 * projection.weight.shape[1] for projection in projections
        )
        packed_values = scratch.reserve(
            "packed_values", DType.FP4, (self.rows, maximum_columns // 2)
        )
        packed_scales = scratch.reserve(
            "packed_scales",
            DType.FP8_UE4M3,
            (padded_rows, maximum_columns // 16),
        )
        prepared_key: tuple[
            TensorDesc, TensorDesc | None, TensorDesc | None
        ] | None = None

        def project(
            name: str,
            source: TensorDesc,
            projection: Fp4Projection,
            *,
            residual: TensorDesc | None = None,
            multiply: TensorDesc | None = None,
            silu: bool = False,
            destination: TensorDesc | None = None,
        ) -> TensorDesc:
            nonlocal prepared_key
            result = (
                destination
                if destination is not None
                else sequence.temporary(
                    name, DType.BF16, (source.shape[0], projection.weight.shape[0])
                )
            )
            swapped = TensorDesc.scratch(
                weight_buffer.info[1], DType.FP4, projection.weight.shape
            )
            sequence.emit(
                SwapFp4NibblesKernel,
                SwapFp4NibblesArguments(projection.weight, swapped),
            )
            input_values = packed_values.view((source.shape[0], source.shape[1] // 2))
            input_scales = packed_scales.view(
                (-(-source.shape[0] // 128) * 128, source.shape[1] // 16)
            )
            key = source, projection.pre_scale, projection.input_scale
            if key != prepared_key:
                dynamic = None
                input_scale = projection.input_scale
                if input_scale is None:
                    dynamic = sequence.temporary(name + ".scale", DType.F32, (1,))
                    sequence.emit(
                        Fp8ScaleKernel,
                        Fp8ScaleArguments(source, projection.pre_scale, dynamic),
                    )
                    input_scale = dynamic
                sequence.emit(
                    QuantizeFp4Kernel,
                    QuantizeFp4Arguments(
                        source,
                        projection.pre_scale,
                        dynamic,
                        input_values,
                        input_scales,
                        input_scale,
                    ),
                )
                prepared_key = key
            sequence.emit(
                MatmulFp4Kernel,
                MatmulFp4Arguments(
                    input_values,
                    swapped,
                    input_scales,
                    projection.scales,
                    None,
                    residual,
                    multiply,
                    result,
                    projection.weight_scale,
                    projection.input_scale,
                    silu,
                ),
            )
            return result

        query = project("query", normalized, weights.query)
        key = project("key", normalized, weights.key)
        value = project("value", normalized, weights.value)
        query_rotated = sequence.temporary("query_rope", DType.BF16, query.shape)
        key_rotated = sequence.temporary("key_rope", DType.BF16, key.shape)
        query_heads = query.view((query.shape[0], 64, 128))
        key_heads = key.view((key.shape[0], 8, 128))
        query_rotated_heads = query_rotated.view(query_heads.shape)
        key_rotated_heads = key_rotated.view(key_heads.shape)
        sequence.emit(
            QkNormRopeKernel,
            QkNormRopeArguments(
                query_heads,
                weights.query_norm,
                angles,
                query_rotated_heads,
            ),
        )
        sequence.emit(
            QkNormRopeKernel,
            QkNormRopeArguments(
                key_heads,
                weights.key_norm,
                angles,
                key_rotated_heads,
            ),
        )
        context = sequence.temporary("context", DType.BF16, query.shape)
        context_heads = context.view(query_heads.shape)
        value_heads = value.view(key_heads.shape)
        sequence.emit(
            CausalAttentionKernel,
            CausalAttentionArguments(
                query_rotated_heads,
                key_rotated_heads,
                value_heads,
                context_heads,
                self.sequence,
            ),
        )
        attended = project("attention", context, weights.output, residual=hidden)
        post = sequence.temporary("post_norm", DType.BF16, hidden.shape)
        sequence.emit(
            RmsNormKernel,
            RmsNormArguments(attended, weights.post_norm, post, 1e-6),
        )
        gate = project("gate", post, weights.gate, silu=True)
        activated = project("up", post, weights.up, multiply=gate)
        # The last projection writes the bound output directly; no dead down temporary.
        project("down", activated, weights.down, residual=attended, destination=output)
        return sequence.calls
