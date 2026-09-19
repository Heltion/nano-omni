"""H3 attention with direct dense INT8/FP8 kernels and fixed local scratch regions."""

from __future__ import annotations

import dataclasses

from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.int8_fp8.dense import (
    DenseInt8Fp8AttentionArguments,
    DenseInt8Fp8AttentionKernel,
)
from nano_omni.kernels.attention.int8_fp8.quantize_qk import (
    Int8Fp8QuantizeQKArguments,
    Int8Fp8QuantizeQKKernel,
)
from nano_omni.kernels.attention.int8_fp8.quantize_value import (
    Int8Fp8QuantizeValueArguments,
    Int8Fp8QuantizeValueKernel,
)
from nano_omni.kernels.attention.int8_fp8.value_scale import (
    Int8Fp8ValueScaleArguments,
    Int8Fp8ValueScaleKernel,
)
from nano_omni.kernels.normalization.token_group_mean import (
    TokenGroupMeanArguments,
    TokenGroupMeanKernel,
)
from nano_omni.kernels.specialized.h3_head_norm_rope import (
    H3HeadNormRopeArguments,
    H3HeadNormRopeKernel,
)
from nano_omni.kernels.specialized.h3_query_prepare import (
    H3QueryPrepareArguments,
    H3QueryPrepareKernel,
)
from nano_omni.ops.matmul.sequence import KernelSequence
from nano_omni.ops.minimax_h3.mlp import Projection


@dataclasses.dataclass(frozen=True)
class AttentionWeights:
    norm: TensorDesc
    query_norm: TensorDesc
    key_norm: TensorDesc
    query: Projection
    key: Projection
    value: Projection
    output: Projection


def project_query(
    sequence: KernelSequence,
    projection: Projection,
    norm: TensorDesc,
    normalized: TensorDesc,
    cosines: TensorDesc | None = None,
    sines: TensorDesc | None = None,
) -> tuple[TensorDesc, TensorDesc]:
    rows, heads = normalized.shape[0], 56
    projected = sequence.temporary("query", DType.BF16, (rows, heads * 128))
    projection.emit(sequence, "query", normalized, projected)
    qrows = -(-rows // 32) * 32
    query = sequence.temporary("query.values", DType.I8, (heads, qrows, 128))
    query_scale = sequence.temporary("query.qk_scales", DType.F32, (heads, qrows // 32))
    sequence.emit(
        H3QueryPrepareKernel,
        H3QueryPrepareArguments(projected, norm, cosines, sines, query, query_scale),
    )
    return query, query_scale


def project_attention(
    sequence: KernelSequence,
    weights: AttentionWeights,
    normalized: TensorDesc,
    cosines: TensorDesc | None = None,
    sines: TensorDesc | None = None,
) -> TensorDesc:
    rows, heads = normalized.shape[0], 56
    shape = (rows, heads * 128)
    query, query_scale = project_query(
        sequence, weights.query, weights.query_norm, normalized, cosines, sines
    )
    projected_key = sequence.temporary("key", DType.BF16, shape)
    weights.key.emit(sequence, "key", normalized, projected_key)
    rotated_key = sequence.temporary("key.rotated", DType.BF16, shape)
    head_shape = (rows, heads, 128)
    sequence.emit(
        H3HeadNormRopeKernel,
        H3HeadNormRopeArguments(
            projected_key.view(head_shape),
            weights.key_norm,
            cosines,
            sines,
            rotated_key.view(head_shape),
        ),
    )
    mean = sequence.temporary("key.mean", DType.F32, (1, heads * 128))
    sequence.emit(
        TokenGroupMeanKernel, TokenGroupMeanArguments(rotated_key, mean, rows, False)
    )
    krows = -(-rows // 64) * 64
    key = sequence.temporary("key.values", DType.I8, (heads, krows, 128))
    key_scale = sequence.temporary("key.scales", DType.F32, (heads, krows // 64))
    sequence.emit(
        Int8Fp8QuantizeQKKernel,
        Int8Fp8QuantizeQKArguments(rotated_key, mean, key, key_scale, heads, 64, True),
    )
    projected_value = sequence.temporary("value", DType.BF16, shape)
    weights.value.emit(sequence, "value", normalized, projected_value)
    scales = sequence.temporary("value.fp8_scales", DType.F32, (heads, 128))
    sequence.emit(
        Int8Fp8ValueScaleKernel,
        Int8Fp8ValueScaleArguments(projected_value, scales, heads),
    )
    value = sequence.temporary(
        "value.packed", DType.FP8_E4M3, (heads, 128, -(-rows // 64) * 64)
    )
    sequence.emit(
        Int8Fp8QuantizeValueKernel,
        Int8Fp8QuantizeValueArguments(projected_value, scales, value, heads),
    )
    attended = sequence.temporary("attended", DType.BF16, (rows, heads * 128))
    sequence.emit(
        DenseInt8Fp8AttentionKernel,
        DenseInt8Fp8AttentionArguments(
            query,
            key,
            value,
            query_scale,
            key_scale,
            scales,
            attended,
            rows,
            heads,
            rows,
        ),
    )
    return attended
