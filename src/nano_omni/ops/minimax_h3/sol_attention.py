"""Head-sliced BF16 Sol attention for H3 diffusion blocks."""

from __future__ import annotations

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.sol.pool import (
    SolPoolArguments,
    SolPoolKernel,
)
from nano_omni.kernels.attention.sol.stats import (
    SolStatsArguments,
    SolStatsKernel,
)
from nano_omni.kernels.attention.sol_attention import (
    SolAttentionArguments,
    SolAttentionKernel,
)
from nano_omni.kernels.specialized.h3_head_norm_rope import (
    H3HeadNormRopeArguments,
    H3HeadNormRopeKernel,
)
from nano_omni.ops.matmul.sequence import KernelSequence
from nano_omni.ops.minimax_h3.attention import AttentionWeights
from nano_omni.ops.minimax_h3.mlp import Projection

TOTAL_HEADS = 56
SLICE_HEADS = 14
HEAD_DIM = 128


def _rows(tensor: TensorDesc, start: int, count: int) -> TensorDesc:
    """Take a contiguous first-dimension slice from a weight tensor."""
    assert tensor.shape and start >= 0 and count > 0
    row_bytes = tensor.num_bytes // tensor.shape[0]
    assert start + count <= tensor.shape[0]
    return tensor.view((count, *tensor.shape[1:]), start * row_bytes)


def _projection_heads(projection: Projection, start: int, heads: int) -> Projection:
    """Restrict a projection and every row-shaped scale/LoRA update to heads."""
    row_start = start * HEAD_DIM
    row_count = heads * HEAD_DIM
    weight_scale = projection.weight_scale
    if isinstance(weight_scale, TensorDesc):
        weight_scale = _rows(weight_scale, row_start, row_count)
    loras = tuple(
        (down, _rows(up, row_start, row_count), strength)
        for down, up, strength in projection.loras
    )
    return Projection(
        _rows(projection.weight, row_start, row_count),
        weight_scale,
        projection.input_scale,
        loras,
        projection.rotation_group,
    )


class SolAttention(Op[AttentionWeights]):
    """Compute four 14-head slices, then apply the complete output projection once."""

    def __init__(
        self,
        normalized: TensorDesc,
        cosines: TensorDesc,
        sines: TensorDesc,
        *,
        tau: float = 1.0,
        protected_tokens: int,
    ) -> None:
        assert normalized.dtype == DType.BF16 and normalized.shape[1] == 5376
        assert tau == 1.0, "H3 Sol attention currently fixes tau to one"
        assert 0 <= protected_tokens <= normalized.shape[0]
        self.tau = tau
        self.protected_blocks = -(-protected_tokens // 64)
        super().__init__(
            normalized,
            cosines,
            sines,
            outputs=((DType.BF16, normalized.shape),),
        )

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        weights = self.bound_weights
        normalized, cosines, sines = self.inputs
        rows = normalized.shape[0]
        blocks = -(-rows // 64)
        slice_shape = (rows, SLICE_HEADS * HEAD_DIM)
        head_shape = (rows, SLICE_HEADS, HEAD_DIM)
        sequence = KernelSequence(scratch)
        prepared = sequence.prepare(normalized)
        query = sequence.temporary("query", DType.BF16, slice_shape)
        key = sequence.temporary("key", DType.BF16, slice_shape)
        value = sequence.temporary("value", DType.BF16, slice_shape)
        pooled = sequence.temporary(
            "pooled", DType.BF16, (2, SLICE_HEADS, blocks, HEAD_DIM)
        )
        stats = sequence.temporary("stats", DType.F32, (SLICE_HEADS, 2, HEAD_DIM))
        attended = sequence.temporary(
            "attended", DType.BF16, (rows, TOTAL_HEADS, HEAD_DIM)
        )
        for head_start in range(0, TOTAL_HEADS, SLICE_HEADS):
            _projection_heads(weights.query, head_start, SLICE_HEADS).emit(
                sequence, f"query.{head_start}", prepared, query
            )
            sequence.emit(
                H3HeadNormRopeKernel,
                H3HeadNormRopeArguments(
                    query.view(head_shape),
                    weights.query_norm,
                    cosines,
                    sines,
                    query.view(head_shape),
                ),
            )
            _projection_heads(weights.key, head_start, SLICE_HEADS).emit(
                sequence, f"key.{head_start}", prepared, key
            )
            sequence.emit(
                H3HeadNormRopeKernel,
                H3HeadNormRopeArguments(
                    key.view(head_shape),
                    weights.key_norm,
                    cosines,
                    sines,
                    key.view(head_shape),
                ),
            )
            _projection_heads(weights.value, head_start, SLICE_HEADS).emit(
                sequence, f"value.{head_start}", prepared, value
            )
            sequence.emit(
                SolPoolKernel,
                SolPoolArguments(key.view(head_shape), value.view(head_shape), pooled),
            )
            sequence.emit(SolStatsKernel, SolStatsArguments(pooled, stats))
            sequence.emit(
                SolAttentionKernel,
                SolAttentionArguments(
                    query.view(head_shape),
                    key.view(head_shape),
                    value.view(head_shape),
                    pooled,
                    stats,
                    attended,
                    self.tau,
                    head_start,
                    self.protected_blocks,
                ),
            )
        weights.output.emit(
            sequence,
            "output",
            attended.view((rows, TOTAL_HEADS * HEAD_DIM)),
            self.bound_outputs[0],
        )
        return sequence.calls
