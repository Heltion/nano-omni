"""Head-sliced dense NVFP4 attention for H3 diffusion blocks."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.dense_nvfp4 import (
    DenseNvfp4Arguments,
    DenseNvfp4Kernel,
)
from nano_omni.kernels.attention.nvfp4_prepare import (
    Nvfp4PrepareArguments,
    Nvfp4PrepareKernel,
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


class DenseNvfp4Attention(Op[AttentionWeights]):
    """Quantize Q/K/V per head slice and execute dense block-scaled attention."""

    def __init__(
        self, normalized: TensorDesc, cosines: TensorDesc, sines: TensorDesc
    ) -> None:
        assert normalized.dtype == DType.BF16 and normalized.shape[1] == 5376
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
        padded = -(-rows // 128) * 128
        slice_shape = (rows, SLICE_HEADS * HEAD_DIM)
        head_shape = (rows, SLICE_HEADS, HEAD_DIM)
        sequence = KernelSequence(scratch)
        prepared = sequence.prepare(normalized)
        query = sequence.temporary("query.bf16", DType.BF16, slice_shape)
        key = sequence.temporary("key.bf16", DType.BF16, slice_shape)
        value = sequence.temporary("value.bf16", DType.BF16, slice_shape)
        mean = sequence.temporary("key.mean", DType.F32, (1, SLICE_HEADS * 128))
        query_fp4 = sequence.temporary(
            "query.fp4", DType.U8, (SLICE_HEADS, padded, 64)
        )
        key_fp4 = sequence.temporary(
            "key.fp4", DType.U8, (SLICE_HEADS, padded, 64)
        )
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
        attended = sequence.temporary(
            "attended", DType.BF16, (rows, TOTAL_HEADS, HEAD_DIM)
        )

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
                DenseNvfp4Kernel,
                DenseNvfp4Arguments(
                    query_fp4,
                    key_fp4,
                    value_fp4,
                    query_scale,
                    key_scale,
                    value_scale,
                    attended,
                    rows,
                    rows,
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
