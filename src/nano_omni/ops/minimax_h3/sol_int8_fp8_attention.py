"""Head-sliced BF16-routed Sol attention with INT8 QK and FP8 PV."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.int8_fp8.quantize_qk import (
    Int8Fp8QuantizeQKArguments,
    Int8Fp8QuantizeQKKernel,
)
from nano_omni.kernels.attention.int8_fp8.quantize_value import (
    Int8Fp8QuantizeValueArguments,
    Int8Fp8QuantizeValueKernel,
)
from nano_omni.kernels.attention.int8_fp8.sol import (
    SolInt8Fp8AttentionArguments,
    SolInt8Fp8AttentionKernel,
)
from nano_omni.kernels.attention.int8_fp8.sol_pool import (
    SolInt8Fp8PoolArguments,
    SolInt8Fp8PoolKernel,
)
from nano_omni.kernels.attention.int8_fp8.value_scale import (
    Int8Fp8ValueScaleArguments,
    Int8Fp8ValueScaleKernel,
)
from nano_omni.kernels.attention.nvfp4.sol_finalize import (
    SolNvfp4FinalizeArguments,
    SolNvfp4FinalizeKernel,
)
from nano_omni.kernels.attention.nvfp4.sol_select import (
    SolNvfp4SelectArguments,
    SolNvfp4SelectKernel,
)
from nano_omni.kernels.attention.sol.stats import (
    SolStatsArguments,
    SolStatsKernel,
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


class SolInt8Fp8Attention(Op[AttentionWeights]):
    """Route in BF16 and compute selected blocks with INT8 QK and FP8 PV."""

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
        qrows = -(-rows // 32) * 32
        krows = blocks * 64
        slice_shape = (rows, SLICE_HEADS * HEAD_DIM)
        head_shape = (rows, SLICE_HEADS, HEAD_DIM)
        sequence = KernelSequence(scratch)
        prepared = sequence.prepare(normalized)
        query = sequence.temporary("query.bf16", DType.BF16, slice_shape)
        key = sequence.temporary("key.bf16", DType.BF16, slice_shape)
        value = sequence.temporary("value.bf16", DType.BF16, slice_shape)
        mean = sequence.temporary("key.mean", DType.F32, (1, SLICE_HEADS * 128))
        query_i8 = sequence.temporary("query.int8", DType.I8, (SLICE_HEADS, qrows, 128))
        key_i8 = sequence.temporary("key.int8", DType.I8, (SLICE_HEADS, krows, 128))
        query_scale = sequence.temporary(
            "query.scale", DType.F32, (SLICE_HEADS, qrows // 32)
        )
        key_scale = sequence.temporary("key.scale", DType.F32, (SLICE_HEADS, blocks))
        value_scale = sequence.temporary("value.scale", DType.F32, (SLICE_HEADS, 128))
        value_fp8 = sequence.temporary(
            "value.fp8", DType.FP8_E4M3, (SLICE_HEADS, 128, krows)
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
        # Finalize loads each disjoint query/head tile before writing it, so the
        # exact numerator and finalized attention may share one allocation.
        exact = attended
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
                Int8Fp8QuantizeQKKernel,
                Int8Fp8QuantizeQKArguments(
                    query, mean, query_i8, query_scale, SLICE_HEADS, 32, False
                ),
            )
            sequence.emit(
                Int8Fp8QuantizeQKKernel,
                Int8Fp8QuantizeQKArguments(
                    key, mean, key_i8, key_scale, SLICE_HEADS, 64, True
                ),
            )
            sequence.emit(
                Int8Fp8ValueScaleKernel,
                Int8Fp8ValueScaleArguments(value, value_scale, SLICE_HEADS),
            )
            sequence.emit(
                Int8Fp8QuantizeValueKernel,
                Int8Fp8QuantizeValueArguments(
                    value, value_scale, value_fp8, SLICE_HEADS
                ),
            )
            sequence.emit(
                SolInt8Fp8PoolKernel,
                SolInt8Fp8PoolArguments(
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
                    pooled.view((SLICE_HEADS, blocks, HEAD_DIM)),
                    stats,
                    selected,
                    self.tau,
                    self.protected_blocks,
                ),
            )
            sequence.emit(
                SolInt8Fp8AttentionKernel,
                SolInt8Fp8AttentionArguments(
                    query_i8,
                    key_i8,
                    value_fp8,
                    query_scale,
                    key_scale,
                    value_scale,
                    selected,
                    exact,
                    state,
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
