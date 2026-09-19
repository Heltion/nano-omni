"""Vision operators keep attention and merger views explicit in kernel order."""

import dataclasses

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.activation.gelu import GeluArguments, GeluKernel
from nano_omni.kernels.attention.self_attention import (
    SelfAttentionArguments,
    SelfAttentionKernel,
)
from nano_omni.kernels.normalization.layer_norm import (
    LayerNormArguments,
    LayerNormKernel,
)
from nano_omni.kernels.specialized.h3_qkv_rope import (
    H3QkvRopeArguments,
    H3QkvRopeKernel,
)
from nano_omni.kernels.specialized.h3_rope import H3RopeArguments, H3RopeKernel
from nano_omni.kernels.specialized.vision_position import (
    VisionPositionArguments,
    VisionPositionKernel,
)
from nano_omni.ops.matmul.sequence import KernelSequence

WIDTH, HEADS, HEAD_DIM, MERGE_WIDTH = 1152, 16, 72, 4608


@dataclasses.dataclass(frozen=True)
class VisionLayerWeights:
    norm1_weight: TensorDesc
    norm1_bias: TensorDesc
    qkv_weight: TensorDesc
    qkv_bias: TensorDesc
    projection_weight: TensorDesc
    projection_bias: TensorDesc
    norm2_weight: TensorDesc
    norm2_bias: TensorDesc
    fc1_weight: TensorDesc
    fc1_bias: TensorDesc
    fc2_weight: TensorDesc
    fc2_bias: TensorDesc


@dataclasses.dataclass(frozen=True)
class VisionMergerWeights:
    norm_weight: TensorDesc
    norm_bias: TensorDesc
    fc1_weight: TensorDesc
    fc1_bias: TensorDesc
    fc2_weight: TensorDesc
    fc2_bias: TensorDesc


def _normalize(
    sequence: KernelSequence,
    name: str,
    source: TensorDesc,
    weight: TensorDesc,
    bias: TensorDesc,
) -> TensorDesc:
    output = sequence.temporary(name, DType.BF16, source.shape)
    sequence.emit(
        LayerNormKernel,
        LayerNormArguments(source, weight, bias, output, 1e-5),
    )
    return output


def _mlp(
    sequence: KernelSequence,
    source: TensorDesc,
    output: TensorDesc,
    weights: VisionLayerWeights | VisionMergerWeights,
    *,
    residual: TensorDesc | None = None,
) -> None:
    expanded_shape = (source.shape[0], weights.fc1_weight.shape[0])
    expanded = sequence.temporary("expanded", DType.BF16, expanded_shape)
    sequence.linear("fc1", source, weights.fc1_weight, expanded, bias=weights.fc1_bias)
    activated = sequence.temporary("activated", DType.BF16, expanded_shape)
    sequence.emit(GeluKernel, GeluArguments(expanded, activated))
    sequence.linear(
        "fc2",
        activated,
        weights.fc2_weight,
        output,
        bias=weights.fc2_bias,
        residual=residual,
    )


class VisionPatch(Op[tuple[TensorDesc, TensorDesc, TensorDesc]]):
    def __init__(
        self,
        patches: int | tuple[int, int],
        indices: int | tuple[int, int],
        position_weights: int | tuple[int, int],
        *,
        tokens: int,
    ) -> None:
        self.tokens = tokens
        inputs = (
            TensorDesc.activation(patches, DType.BF16, (tokens, 1536)),
            TensorDesc.activation(indices, DType.U32, (4, tokens)),
            TensorDesc.activation(position_weights, DType.F32, (4, tokens)),
        )
        super().__init__(*inputs, outputs=((DType.BF16, (tokens, WIDTH)),))

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        patch_weight, patch_bias, position_table = self.bound_weights
        sequence = KernelSequence(scratch)
        shape = (self.tokens, WIDTH)
        projected = sequence.temporary("projected", DType.BF16, shape)
        sequence.linear(
            "patch",
            self.inputs[0],
            patch_weight,
            projected,
            bias=patch_bias,
        )
        sequence.emit(
            VisionPositionKernel,
            VisionPositionArguments(
                projected,
                position_table,
                self.inputs[1],
                self.inputs[2],
                self.bound_outputs[0],
            ),
        )
        return sequence.calls


class VisionRope(Op[None]):
    def __init__(
        self,
        positions: int | tuple[int, int],
        frequencies: int | tuple[int, int],
        *,
        tokens: int,
    ) -> None:
        self.tokens = tokens
        inputs = (
            TensorDesc.activation(positions, DType.F32, (tokens, 2)),
            TensorDesc.activation(frequencies, DType.F32, (HEAD_DIM // 4,)),
        )
        super().__init__(*inputs, outputs=((DType.F32, (2, tokens, HEAD_DIM // 2)),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel, ...]:
        return (
            H3RopeKernel(
                H3RopeArguments(
                    self.inputs[0],
                    self.inputs[1],
                    self.bound_outputs[0],
                )
            ),
        )


class VisionLayer(Op[VisionLayerWeights]):
    def __init__(
        self, hidden: int | tuple[int, int], rope: int | tuple[int, int], *, tokens: int
    ) -> None:
        self.tokens = tokens
        rope_shape = (tokens, HEAD_DIM // 2)
        rope_bytes = tokens * (HEAD_DIM // 2) * DType.F32.itemsize
        rope_position = (rope, 0) if isinstance(rope, int) else rope
        rope_input = TensorDesc.activation(rope_position, DType.F32, (2, *rope_shape))
        inputs = (
            TensorDesc.activation(hidden, DType.BF16, (tokens, WIDTH)),
            rope_input.view(rope_shape),
            rope_input.view(rope_shape, rope_bytes),
        )
        super().__init__(*inputs, outputs=((DType.BF16, (tokens, WIDTH)),))

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        weights = self.bound_weights
        assert isinstance(weights, VisionLayerWeights), (
            "vision layer weights are required"
        )
        sequence = KernelSequence(scratch)
        hidden, cosines, sines = self.inputs
        output = self.bound_outputs[0]
        norm1 = _normalize(
            sequence, "norm1", hidden, weights.norm1_weight, weights.norm1_bias
        )
        qkv = sequence.temporary("qkv", DType.BF16, (self.tokens, WIDTH * 3))
        sequence.linear("qkv", norm1, weights.qkv_weight, qkv, bias=weights.qkv_bias)
        packed = sequence.temporary(
            "packed", DType.BF16, (3, self.tokens, HEADS, HEAD_DIM)
        )
        sequence.emit(
            H3QkvRopeKernel,
            H3QkvRopeArguments(
                qkv,
                None,
                None,
                cosines,
                sines,
                packed,
                HEADS,
                HEAD_DIM,
                HEAD_DIM,
                False,
                True,
                False,
            ),
        )
        # RoPE output stores consecutive Q, K, V planes in the same scratch region.
        plane_bytes = self.tokens * WIDTH * DType.BF16.itemsize
        query, key, value = (
            packed.view(hidden.shape, plane * plane_bytes) for plane in range(3)
        )
        context = sequence.temporary("context", DType.BF16, hidden.shape)
        sequence.emit(
            SelfAttentionKernel,
            SelfAttentionArguments(
                query,
                key,
                value,
                context,
                HEADS,
                HEAD_DIM,
                True,
                False,
            ),
        )
        attended = sequence.temporary("attended", DType.BF16, hidden.shape)
        sequence.linear(
            "projection",
            context,
            weights.projection_weight,
            attended,
            bias=weights.projection_bias,
            residual=hidden,
        )
        norm2 = _normalize(
            sequence, "norm2", attended, weights.norm2_weight, weights.norm2_bias
        )
        _mlp(sequence, norm2, output, weights, residual=attended)
        return sequence.calls


class VisionMerger(Op[VisionMergerWeights]):
    def __init__(
        self, hidden: int | tuple[int, int], *, tokens: int, pre_merge_norm: bool
    ) -> None:
        self.tokens, self.pre_merge_norm = tokens, pre_merge_norm
        input_shape = (tokens, WIDTH) if pre_merge_norm else (tokens // 4, MERGE_WIDTH)
        output_shape = (tokens // 4, 5120)
        super().__init__(
            TensorDesc.activation(hidden, DType.BF16, input_shape),
            outputs=((DType.BF16, output_shape),),
        )

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        weights = self.bound_weights
        assert isinstance(weights, VisionMergerWeights), (
            "vision merger weights are required"
        )
        sequence = KernelSequence(scratch)
        merged_shape = (self.tokens // 4, MERGE_WIDTH)
        # Main merger normalizes tokens first; deepstack normalizes merged groups.
        source = self.inputs[0]
        normalized = _normalize(
            sequence, "normalized", source, weights.norm_weight, weights.norm_bias
        )
        merged = normalized.view(merged_shape) if self.pre_merge_norm else normalized
        output = self.bound_outputs[0]
        _mlp(sequence, merged, output, weights)
        return sequence.calls
