"""FP16/BF16 attention, optionally using fixed-scale INT8 Q/K dot products."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.base import AttentionConfig


class SelfAttentionWorkload(Workload):
    """One unmasked sequence with matching Q/K/V heads and head dimensions."""

    heads: int
    num_tokens: int
    dim: int
    probability_fp16: bool = False
    int8_fixed: bool = False
    dtype: DType = DType.BF16


@dataclasses.dataclass(frozen=True, slots=True)
class SelfAttentionArguments(Arguments):
    query: TensorDesc
    key: TensorDesc
    value: TensorDesc
    output: TensorDesc
    heads: int
    dim: int
    probability_fp16: bool
    int8_fixed: bool


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def self_attention(
    heads,
    num_tokens,
    dim,
    probability_fp16,
    int8_fixed,
    dtype,
    tile_query_tokens=64,
    tile_kv_tokens=64,
    stages=1,
    threads=128,
):
    import tilelang.language as T

    padded_dim = (dim + 15) // 16 * 16
    probability_dtype = T.float16 if probability_fp16 else T.bfloat16
    storage_dtype = T.float16 if dtype == DType.F16 else T.bfloat16
    qk_dtype = T.int8 if int8_fixed else storage_dtype
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    qk_quant = 16.0

    @T.prim_func
    def main(
        query: T.Tensor([1, num_tokens, heads, dim], storage_dtype),
        key: T.Tensor([1, num_tokens, heads, dim], storage_dtype),
        value: T.Tensor([1, num_tokens, heads, dim], storage_dtype),
        output: T.Tensor([1, num_tokens, heads, dim], storage_dtype),
    ):
        T.annotate_pass_configs({"tl.enable_fast_math": True})
        with T.Kernel(
            T.ceildiv(num_tokens, tile_query_tokens), heads, 1, threads=threads
        ) as (row, head, batch):
            query_shared = T.alloc_shared([tile_query_tokens, padded_dim], qk_dtype)
            key_shared = T.alloc_shared([tile_kv_tokens, padded_dim], qk_dtype)
            value_shared = T.alloc_shared(
                [tile_kv_tokens, padded_dim], probability_dtype
            )
            output_shared = T.alloc_shared(
                [tile_query_tokens, padded_dim], storage_dtype
            )
            scores = T.alloc_fragment([tile_query_tokens, tile_kv_tokens], T.float32)
            integer_scores = T.alloc_fragment(
                [tile_query_tokens, tile_kv_tokens], T.int32
            )
            probabilities = T.alloc_fragment(
                [tile_query_tokens, tile_kv_tokens], probability_dtype
            )
            accumulator = T.alloc_fragment([tile_query_tokens, padded_dim], T.float32)
            maximum = T.alloc_fragment([tile_query_tokens], T.float32)
            previous = T.alloc_fragment([tile_query_tokens], T.float32)
            correction = T.alloc_fragment([tile_query_tokens], T.float32)
            total = T.alloc_fragment([tile_query_tokens], T.float32)
            block_total = T.alloc_fragment([tile_query_tokens], T.float32)
            if padded_dim == dim and not int8_fixed:
                T.copy(
                    query[
                        batch,
                        row * tile_query_tokens : (row + 1) * tile_query_tokens,
                        head,
                        :,
                    ],
                    query_shared,
                )
            else:
                for i, j in T.Parallel(tile_query_tokens, padded_dim):
                    token = row * tile_query_tokens + i
                    if int8_fixed:
                        quantized = T.max(
                            -127.0,
                            T.min(
                                127.0,
                                query[batch, token, head, j] * qk_quant,
                            ),
                        )
                        query_shared[i, j] = T.if_then_else(
                            token < num_tokens and j < dim,
                            T.cast(quantized, T.int8),
                            0,
                        )
                    else:
                        query_shared[i, j] = T.if_then_else(
                            token < num_tokens and j < dim,
                            query[batch, token, head, j],
                            0.0,
                        )
            T.fill(accumulator, 0)
            T.fill(total, 0)
            T.fill(maximum, -T.infinity(T.float32))
            for inner in T.Pipelined(
                T.ceildiv(num_tokens, tile_kv_tokens), num_stages=stages
            ):
                if padded_dim == dim and not int8_fixed:
                    T.copy(
                        key[
                            batch,
                            inner * tile_kv_tokens : (inner + 1) * tile_kv_tokens,
                            head,
                            :,
                        ],
                        key_shared,
                    )
                else:
                    for i, j in T.Parallel(tile_kv_tokens, padded_dim):
                        token = inner * tile_kv_tokens + i
                        if int8_fixed:
                            quantized = T.max(
                                -127.0,
                                T.min(
                                    127.0,
                                    key[batch, token, head, j] * qk_quant,
                                ),
                            )
                            key_shared[i, j] = T.if_then_else(
                                token < num_tokens and j < dim,
                                T.cast(quantized, T.int8),
                                0,
                            )
                        else:
                            key_shared[i, j] = T.if_then_else(
                                token < num_tokens and j < dim,
                                key[batch, token, head, j],
                                0.0,
                            )
                if int8_fixed:
                    T.clear(integer_scores)
                    T.gemm(
                        query_shared,
                        key_shared,
                        integer_scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(tile_query_tokens, tile_kv_tokens):
                        scores[i, j] = T.cast(integer_scores[i, j], T.float32) / (
                            qk_quant * qk_quant
                        )
                else:
                    T.clear(scores)
                    T.gemm(
                        query_shared,
                        key_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                for i, j in T.Parallel(tile_query_tokens, tile_kv_tokens):
                    if inner * tile_kv_tokens + j >= num_tokens:
                        scores[i, j] = -T.infinity(T.float32)
                T.copy(maximum, previous)
                T.fill(maximum, -T.infinity(T.float32))
                T.reduce_max(scores, maximum, dim=1, clear=False)
                for i in T.Parallel(tile_query_tokens):
                    maximum[i] = T.max(maximum[i], previous[i])
                    correction[i] = T.exp2(previous[i] * scale - maximum[i] * scale)
                for i, j in T.Parallel(tile_query_tokens, tile_kv_tokens):
                    scores[i, j] = T.exp2(scores[i, j] * scale - maximum[i] * scale)
                T.reduce_sum(scores, block_total, dim=1)
                for i in T.Parallel(tile_query_tokens):
                    total[i] = total[i] * correction[i] + block_total[i]
                T.copy(scores, probabilities)
                for i, j in T.Parallel(tile_query_tokens, padded_dim):
                    accumulator[i, j] *= correction[i]
                if padded_dim == dim:
                    T.copy(
                        value[
                            batch,
                            inner * tile_kv_tokens : (inner + 1) * tile_kv_tokens,
                            head,
                            :,
                        ],
                        value_shared,
                    )
                else:
                    for i, j in T.Parallel(tile_kv_tokens, padded_dim):
                        token = inner * tile_kv_tokens + i
                        value_shared[i, j] = T.if_then_else(
                            token < num_tokens and j < dim,
                            value[batch, token, head, j],
                            0.0,
                        )
                T.gemm(
                    probabilities,
                    value_shared,
                    accumulator,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for i, j in T.Parallel(tile_query_tokens, padded_dim):
                accumulator[i, j] /= total[i]
            if padded_dim == dim:
                T.copy(accumulator, output_shared)
                T.copy(
                    output_shared,
                    output[
                        batch,
                        row * tile_query_tokens : (row + 1) * tile_query_tokens,
                        head,
                        :,
                    ],
                )
            else:
                for i, j in T.Parallel(tile_query_tokens, dim):
                    token = row * tile_query_tokens + i
                    if token < num_tokens:
                        output[batch, token, head, j] = accumulator[i, j]

    return main.with_attr(
        "global_symbol",
        "self_attention_"
        f"{heads}_{num_tokens}_{dim}_{int(probability_fp16)}_{int(int8_fixed)}_{dtype.value}_"
        f"{tile_query_tokens}_{tile_kv_tokens}_{stages}_{threads}",
    )


class SelfAttentionKernel(
    Kernel[SelfAttentionArguments, SelfAttentionWorkload, AttentionConfig]
):
    name = "self_attention"
    program = self_attention

    @classmethod
    def make_arguments(cls, workload: SelfAttentionWorkload) -> SelfAttentionArguments:
        """Describe flattened Q/K/V matrices and their matching output."""
        shape = (workload.num_tokens, workload.heads * workload.dim)
        return SelfAttentionArguments(
            query=TensorDesc.empty(workload.dtype, shape),
            key=TensorDesc.empty(workload.dtype, shape),
            value=TensorDesc.empty(workload.dtype, shape),
            output=TensorDesc.empty(workload.dtype, shape),
            heads=workload.heads,
            dim=workload.dim,
            probability_fp16=workload.probability_fp16,
            int8_fixed=workload.int8_fixed,
        )

    @classmethod
    def make_config(cls, workload: SelfAttentionWorkload) -> AttentionConfig:
        """Dispatch the measured VAE and vision-encoder attention shapes."""
        if workload.dim == 64:
            return AttentionConfig(
                stages=2, threads=256, tile_kv_tokens=64, tile_query_tokens=128
            )
        assert workload.dim == 72, (
            f"unsupported self-attention dimension: {workload.dim}"
        )
        return AttentionConfig(
            stages=1, threads=128, tile_kv_tokens=64, tile_query_tokens=64
        )

    @classmethod
    def make_workload(cls, arguments: SelfAttentionArguments) -> SelfAttentionWorkload:
        assert arguments.query.dtype in (DType.F16, DType.BF16)
        assert (
            arguments.key.dtype
            == arguments.value.dtype
            == arguments.output.dtype
            == arguments.query.dtype
        )
        assert arguments.query.dtype == DType.BF16 or arguments.probability_fp16
        assert (
            arguments.query.shape
            == arguments.key.shape
            == arguments.value.shape
            == arguments.output.shape
        )
        assert len(arguments.query.shape) == 2
        assert arguments.heads > 0 and arguments.dim > 0
        assert arguments.query.shape[1] == arguments.heads * arguments.dim
        return SelfAttentionWorkload(
            heads=arguments.heads,
            num_tokens=arguments.query.shape[0],
            dim=arguments.dim,
            probability_fp16=arguments.probability_fp16,
            int8_fixed=arguments.int8_fixed,
            dtype=arguments.query.dtype,
        )

    @classmethod
    def tops(cls, arguments: SelfAttentionArguments) -> Tops:
        workload = cls.make_workload(arguments)
        operations = 2 * workload.heads * workload.num_tokens**2 * workload.dim
        qk_type = (
            MmaType.I8I8I32
            if workload.int8_fixed
            else MmaType.F16F16F32
            if workload.dtype == DType.F16
            else MmaType.BF16BF16F32
        )
        pv_type = (
            MmaType.F16F16F32 if workload.probability_fp16 else MmaType.BF16BF16F32
        )
        result = {qk_type: operations}
        result[pv_type] = result.get(pv_type, 0) + operations
        return result

    @classmethod
    def ref_program(cls, arguments: SelfAttentionArguments) -> None:
        """Compute dense self-attention through PyTorch's mathematical operation."""
        import torch
        import torch.nn.functional as F

        workload = cls.make_workload(arguments)
        shape = (workload.num_tokens, workload.heads, workload.dim)
        query = arguments.query.as_torch().view(shape).transpose(0, 1)
        key = arguments.key.as_torch().view(shape).transpose(0, 1)
        value = arguments.value.as_torch().view(shape).transpose(0, 1)
        if workload.int8_fixed:
            # TileLang casts the clamped scaled values to INT8 before GEMM.
            # Preserve that truncation instead of merely round-tripping through
            # the original floating-point dtype.
            query = (query * 16).clamp(-127, 127).to(torch.int8).to(query.dtype) / 16
            key = (key * 16).clamp(-127, 127).to(torch.int8).to(key.dtype) / 16
        result = F.scaled_dot_product_attention(query, key, value)
        arguments.output.as_torch().view(shape).copy_(result.transpose(0, 1))
