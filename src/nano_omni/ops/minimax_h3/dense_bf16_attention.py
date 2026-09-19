"""Head-sliced dense BF16 attention for H3 diffusion blocks."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.dense_bf16 import (
    DenseBf16AttentionArguments,
    DenseBf16AttentionKernel,
)
from nano_omni.kernels.specialized.h3_qk_norm_rope import (
    H3QkNormRopeArguments,
    H3QkNormRopeKernel,
)
from nano_omni.ops.matmul.sequence import KernelSequence
from nano_omni.ops.minimax_h3.attention import AttentionWeights
from nano_omni.ops.minimax_h3.sol_attention import (
    HEAD_DIM,
    TOTAL_HEADS,
    _projection_heads,
)


class DenseBf16Attention(Op[AttentionWeights]):
    """Compute dense attention in head slices and project the complete result once."""

    def __init__(
        self,
        normalized: TensorDesc,
        cosines: TensorDesc,
        sines: TensorDesc,
        *,
        slice_heads: int,
    ) -> None:
        assert normalized.dtype == DType.BF16 and normalized.shape[1] == 5376
        assert slice_heads > 0 and TOTAL_HEADS % slice_heads == 0
        self.slice_heads = slice_heads
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
        slice_heads = self.slice_heads
        slice_shape = (rows, slice_heads * HEAD_DIM)
        head_shape = (rows, slice_heads, HEAD_DIM)
        sequence = KernelSequence(scratch)
        prepared = sequence.prepare(normalized)
        query = sequence.temporary("query", DType.BF16, slice_shape)
        key = sequence.temporary("key", DType.BF16, slice_shape)
        value = sequence.temporary("value", DType.BF16, slice_shape)
        attended = sequence.temporary(
            "attended", DType.BF16, (rows, TOTAL_HEADS, HEAD_DIM)
        )
        for head_start in range(0, TOTAL_HEADS, slice_heads):
            for name, projection, target in (
                ("query", weights.query, query),
                ("key", weights.key, key),
            ):
                _projection_heads(projection, head_start, slice_heads).emit(
                    sequence, f"{name}.{head_start}", prepared, target
                )
            sequence.emit(
                H3QkNormRopeKernel,
                H3QkNormRopeArguments(
                    query.view(head_shape),
                    key.view(head_shape),
                    weights.query_norm,
                    weights.key_norm,
                    cosines,
                    sines,
                ),
            )
            _projection_heads(weights.value, head_start, slice_heads).emit(
                sequence, f"value.{head_start}", prepared, value
            )
            sequence.emit(
                DenseBf16AttentionKernel,
                DenseBf16AttentionArguments(
                    query.view(head_shape),
                    key.view(head_shape),
                    value.view(head_shape),
                    attended,
                    head_start,
                ),
            )
        weights.output.emit(
            sequence,
            "output",
            attended.view((rows, TOTAL_HEADS * HEAD_DIM)),
            self.bound_outputs[0],
        )
        return sequence.calls
