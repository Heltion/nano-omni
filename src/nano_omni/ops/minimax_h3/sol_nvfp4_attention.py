"""Head-sliced BF16-routed Sol attention with NVFP4 exact blocks."""

from nano_omni.kernels.attention.nvfp4.sol_finalize import (
    SolNvfp4FinalizeArguments,
    SolNvfp4FinalizeKernel,
)

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.nvfp4.sol_pool import (
    SolNvfp4PoolArguments,
    SolNvfp4PoolKernel,
)
from nano_omni.kernels.attention.nvfp4.sol_select import (
    SolNvfp4SelectArguments,
    SolNvfp4SelectKernel,
)
from nano_omni.kernels.attention.nvfp4_prepare import (
    Nvfp4PrepareArguments,
    Nvfp4PrepareKernel,
)
from nano_omni.kernels.attention.sol.stats import (
    SolStatsArguments,
    SolStatsKernel,
)
from nano_omni.kernels.attention.sol_nvfp4 import (
    SolNvfp4Arguments,
    SolNvfp4Kernel,
)
from nano_omni.kernels.normalization.token_group_mean import (
    TokenGroupMeanArguments,
    TokenGroupMeanKernel,
)
from nano_omni.kernels.specialized.h3_head_norm_rope import (
    H3HeadNormRopeArguments,
    H3HeadNormRopeKernel,
)
from nano_omni.ops.matmul.sequence import KernelSequence
from nano_omni.ops.minimax_h3.attention import AttentionWeights
from nano_omni.ops.minimax_h3.sol_attention import (
    HEAD_DIM,
    SLICE_HEADS,
    TOTAL_HEADS,
    _projection_heads,
)


class SolNvfp4Attention(Op[AttentionWeights]):
    """Route in BF16 and execute selected blocks with block-scaled FP4 MMA."""

    def __init__(
        self,
        normalized: TensorDesc,
        cosines: TensorDesc,
        sines: TensorDesc,
        *,
        protected_tokens: int,
        tau: float = 1.0,
    ) -> None:
        assert normalized.dtype == DType.BF16 and normalized.shape[1] == 5376
        assert 0 <= protected_tokens <= normalized.shape[0]
        assert tau == 1.0
        self.protected_blocks = -(-protected_tokens // 64)
        self.tau = tau
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
        padded = -(-rows // 128) * 128
        slice_shape = (rows, SLICE_HEADS * HEAD_DIM)
        head_shape = (rows, SLICE_HEADS, HEAD_DIM)
        sequence = KernelSequence(scratch)
        prepared = sequence.prepare(normalized)
        query = sequence.temporary("query.bf16", DType.BF16, slice_shape)
        key = sequence.temporary("key.bf16", DType.BF16, slice_shape)
        value = sequence.temporary("value.bf16", DType.BF16, slice_shape)
        mean = sequence.temporary("key.mean", DType.F32, (1, SLICE_HEADS * 128))
        query_fp4 = sequence.temporary("query.fp4", DType.U8, (SLICE_HEADS, padded, 64))
        key_fp4 = sequence.temporary("key.fp4", DType.U8, (SLICE_HEADS, padded, 64))
        value_fp4 = sequence.temporary(
            "value.fp4", DType.U8, (SLICE_HEADS, 128, padded // 2)
        )
        query_scale = sequence.temporary(
            "query.scale", DType.FP8_UE4M3, (SLICE_HEADS, padded, 8)
        )
        key_scale = sequence.temporary(
            "key.scale", DType.FP8_UE4M3, (SLICE_HEADS, padded, 8)
        )
        value_scale = sequence.temporary(
            "value.scale", DType.FP8_UE4M3, (SLICE_HEADS, 128, padded // 16)
        )
        pooled = sequence.temporary(
            "pooled", DType.BF16, (3, SLICE_HEADS, blocks, HEAD_DIM)
        )
        stats = sequence.temporary("stats", DType.F32, (SLICE_HEADS, 2, HEAD_DIM))
        selected = sequence.temporary(
            "selected", DType.U8, (blocks, SLICE_HEADS, blocks)
        )
        attended = sequence.temporary(
            "attended", DType.BF16, (rows, TOTAL_HEADS, HEAD_DIM)
        )
        exact = sequence.temporary("exact", DType.BF16, (rows, TOTAL_HEADS, HEAD_DIM))
        state = sequence.temporary("state", DType.F32, (rows, TOTAL_HEADS, 2))

        for head_start in range(0, TOTAL_HEADS, SLICE_HEADS):
            for name, projection, norm, target in (
                ("query", weights.query, weights.query_norm, query),
                ("key", weights.key, weights.key_norm, key),
            ):
                _projection_heads(projection, head_start, SLICE_HEADS).emit(
                    sequence, f"{name}.{head_start}", prepared, target
                )
                sequence.emit(
                    H3HeadNormRopeKernel,
                    H3HeadNormRopeArguments(
                        target.view(head_shape),
                        norm,
                        cosines,
                        sines,
                        target.view(head_shape),
                    ),
                )
            _projection_heads(weights.value, head_start, SLICE_HEADS).emit(
                sequence, f"value.{head_start}", prepared, value
            )
            sequence.emit(
                TokenGroupMeanKernel,
                TokenGroupMeanArguments(key, mean, rows, False),
            )
            sequence.emit(
                Nvfp4PrepareKernel,
                Nvfp4PrepareArguments(
                    query, None, query_fp4, query_scale, SLICE_HEADS, False
                ),
            )
            sequence.emit(
                Nvfp4PrepareKernel,
                Nvfp4PrepareArguments(
                    key,
                    mean.view((SLICE_HEADS, 128)),
                    key_fp4,
                    key_scale,
                    SLICE_HEADS,
                    False,
                ),
            )
            sequence.emit(
                Nvfp4PrepareKernel,
                Nvfp4PrepareArguments(
                    value, None, value_fp4, value_scale, SLICE_HEADS, True
                ),
            )
            sequence.emit(
                SolNvfp4PoolKernel,
                SolNvfp4PoolArguments(
                    key.view(head_shape),
                    value.view(head_shape),
                    mean.view((SLICE_HEADS, 128)),
                    pooled,
                ),
            )
            sequence.emit(
                SolStatsKernel,
                SolStatsArguments(
                    pooled.view((2, SLICE_HEADS, blocks, HEAD_DIM)), stats
                ),
            )
            sequence.emit(
                SolNvfp4SelectKernel,
                SolNvfp4SelectArguments(
                    query.view(head_shape),
                    pooled.view(
                        (SLICE_HEADS, blocks, HEAD_DIM),
                    ),
                    stats,
                    selected,
                    self.tau,
                    self.protected_blocks,
                ),
            )
            sequence.emit(
                SolNvfp4Kernel,
                SolNvfp4Arguments(
                    query_fp4,
                    key_fp4,
                    value_fp4,
                    query_scale,
                    key_scale,
                    value_scale,
                    selected,
                    exact,
                    state,
                    rows,
                    rows,
                    head_start,
                    self.tau,
                    self.protected_blocks,
                ),
            )
            sequence.emit(
                SolNvfp4FinalizeKernel,
                SolNvfp4FinalizeArguments(
                    query.view(head_shape),
                    pooled,
                    selected,
                    exact,
                    state,
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
