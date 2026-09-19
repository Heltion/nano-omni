"""Persistent H3 K/V preparation and independently sliced queries."""

from __future__ import annotations

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
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
from nano_omni.ops.matmul.sequence import KernelSequence
from nano_omni.ops.minimax_h3.attention import AttentionWeights, project_query


def _rotate(
    sequence: KernelSequence,
    projected: TensorDesc,
    norm: TensorDesc,
    cosines: TensorDesc,
    sines: TensorDesc,
) -> TensorDesc:
    rows, width = projected.shape
    assert width % 128 == 0
    head_shape = (rows, width // 128, 128)
    sequence.emit(
        H3HeadNormRopeKernel,
        H3HeadNormRopeArguments(
            projected.view(head_shape),
            norm,
            cosines,
            sines,
            projected.view(head_shape),
        ),
    )
    return projected


class PrepareKeyValue(Op[AttentionWeights]):
    def __init__(
        self, normalized: TensorDesc, cosines: TensorDesc, sines: TensorDesc
    ) -> None:
        rows = normalized.shape[0]
        padded_rows = -(-rows // 64) * 64
        super().__init__(
            normalized,
            cosines,
            sines,
            outputs=(
                (DType.I8, (56, padded_rows, 128)),
                (DType.F32, (56, padded_rows // 64)),
                (DType.FP8_E4M3, (56, 128, padded_rows)),
                (DType.F32, (56, 128)),
            ),
        )

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        key, key_scale, value, value_scale = self.bound_outputs
        weights = self.bound_weights
        normalized, cosines, sines = self.inputs
        rows = normalized.shape[0]
        sequence = KernelSequence(scratch)
        projected = sequence.temporary("projected", DType.BF16, (rows, 7168))
        prepared = sequence.prepare(normalized)
        weights.key.emit(sequence, "key", prepared, projected)
        rotated = _rotate(sequence, projected, weights.key_norm, cosines, sines)
        mean = sequence.temporary("mean", DType.F32, (1, 7168))
        sequence.emit(
            TokenGroupMeanKernel,
            TokenGroupMeanArguments(rotated, mean, rows, False),
        )
        sequence.emit(
            Int8Fp8QuantizeQKKernel,
            Int8Fp8QuantizeQKArguments(
                rotated, mean, key, key_scale, 56, 64, True
            ),
        )
        weights.value.emit(sequence, "value", prepared, projected)
        sequence.emit(
            Int8Fp8ValueScaleKernel,
            Int8Fp8ValueScaleArguments(projected, value_scale, 56),
        )
        sequence.emit(
            Int8Fp8QuantizeValueKernel,
            Int8Fp8QuantizeValueArguments(projected, value_scale, value, 56),
        )
        return sequence.calls


class QuerySlice(Op[AttentionWeights]):
    def __init__(
        self,
        normalized: TensorDesc,
        cosines: TensorDesc,
        sines: TensorDesc,
        key: TensorDesc,
        key_scale: TensorDesc,
        value: TensorDesc,
        value_scale: TensorDesc,
        *,
        total_tokens: int,
    ) -> None:
        self.total_tokens = total_tokens
        super().__init__(
            normalized,
            cosines,
            sines,
            key,
            key_scale,
            value,
            value_scale,
            outputs=((DType.BF16, normalized.shape),),
        )

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        weights = self.bound_weights
        normalized, cosines, sines, key, key_scale, value, value_scale = self.inputs
        rows = normalized.shape[0]
        sequence = KernelSequence(scratch)
        query, query_scale = project_query(
            sequence, weights.query, weights.query_norm, normalized, cosines, sines
        )
        attended = sequence.temporary("attended", DType.BF16, (rows, 7168))
        sequence.emit(
            DenseInt8Fp8AttentionKernel,
            DenseInt8Fp8AttentionArguments(
                query,
                key,
                value,
                query_scale,
                key_scale,
                value_scale,
                attended,
                self.total_tokens,
                56,
                rows,
            ),
        )
        weights.output.emit(sequence, "output", attended, self.bound_outputs[0])
        return sequence.calls
