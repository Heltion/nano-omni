"""Causal attention over one padded sequence with optional grouped KV heads."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.base import AttentionConfig


class CausalAttentionWorkload(Workload):
    """Physical row capacity, valid sequence length and Q/KV head counts."""

    dtype: DType = DType.BF16
    num_padded_tokens: int
    num_tokens: int
    heads: int
    kv_heads: int
    dim: int


@dataclasses.dataclass(frozen=True, slots=True)
class CausalAttentionArguments(Arguments):
    query: TensorDesc
    key: TensorDesc
    value: TensorDesc
    output: TensorDesc
    num_tokens: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one padded program for different valid prefix lengths."""
        return (
            ("num_tokens", self.num_tokens),
            ("num_padded_tokens", self.query.shape[0]),
        )


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def causal_attention(
    dtype,
    num_padded_tokens,
    num_tokens,
    heads,
    kv_heads,
    dim,
    tile_query_tokens=64,
    tile_kv_tokens=64,
    stages=1,
    threads=128,
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    dynamic_num_padded_tokens = T.dynamic("num_padded_tokens")
    storage_type = T.float32 if dtype == DType.F32 else T.bfloat16
    probability_dtype = T.float32 if dtype == DType.F32 else T.float16
    groups = heads // kv_heads
    scale = (1.0 / dim) ** 0.5 * 1.44269504

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        num_padded_tokens: dynamic_num_padded_tokens,
        query: T.Tensor([1, dynamic_num_padded_tokens, heads, dim], storage_type),
        key: T.Tensor([1, dynamic_num_padded_tokens, kv_heads, dim], storage_type),
        value: T.Tensor([1, dynamic_num_padded_tokens, kv_heads, dim], storage_type),
        output: T.Tensor([1, dynamic_num_padded_tokens, heads, dim], storage_type),
    ):
        T.annotate_pass_configs({"tl.enable_fast_math": True})
        with T.Kernel(
            T.ceildiv(dynamic_num_padded_tokens, tile_query_tokens),
            heads,
            1,
            threads=threads,
        ) as (row, head, batch):
            query_shared = T.alloc_shared([tile_query_tokens, dim], storage_type)
            key_shared = T.alloc_shared([tile_kv_tokens, dim], storage_type)
            value_shared = T.alloc_shared([tile_kv_tokens, dim], probability_dtype)
            scores = T.alloc_fragment([tile_query_tokens, tile_kv_tokens], T.float32)
            if dtype == DType.F32:
                probabilities = T.alloc_shared(
                    [tile_query_tokens, tile_kv_tokens], probability_dtype
                )
            else:
                probabilities = T.alloc_fragment(
                    [tile_query_tokens, tile_kv_tokens], probability_dtype
                )
            accumulator = T.alloc_fragment([tile_query_tokens, dim], T.float32)
            maximum = T.alloc_fragment([tile_query_tokens], T.float32)
            previous = T.alloc_fragment([tile_query_tokens], T.float32)
            correction = T.alloc_fragment([tile_query_tokens], T.float32)
            total = T.alloc_fragment([tile_query_tokens], T.float32)
            block_total = T.alloc_fragment([tile_query_tokens], T.float32)
            T.copy(
                query[
                    batch,
                    row * tile_query_tokens : (row + 1) * tile_query_tokens,
                    head,
                    :,
                ],
                query_shared,
            )
            T.fill(accumulator, 0)
            T.fill(total, 0)
            T.fill(maximum, -T.infinity(T.float32))
            for inner in T.Pipelined(
                T.min(
                    T.ceildiv(dynamic_num_tokens, tile_kv_tokens),
                    T.ceildiv((row + 1) * tile_query_tokens, tile_kv_tokens),
                ),
                num_stages=stages,
            ):
                for i, j in T.Parallel(tile_kv_tokens, dim):
                    key_position = inner * tile_kv_tokens + i
                    key_shared[i, j] = T.if_then_else(
                        key_position < dynamic_num_tokens,
                        key[batch, key_position, head // groups, j],
                        0.0,
                    )
                T.fill(scores, 0)
                T.gemm(
                    query_shared,
                    key_shared,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(tile_query_tokens, tile_kv_tokens):
                    query_position = row * tile_query_tokens + i
                    key_position = inner * tile_kv_tokens + j
                    scores[i, j] = T.if_then_else(
                        query_position < dynamic_num_tokens
                        and key_position < dynamic_num_tokens
                        and (query_position >= key_position),
                        scores[i, j],
                        -T.infinity(T.float32),
                    )
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
                for i, j in T.Parallel(tile_query_tokens, dim):
                    accumulator[i, j] *= correction[i]
                for i, j in T.Parallel(tile_kv_tokens, dim):
                    key_position = inner * tile_kv_tokens + i
                    value_shared[i, j] = T.if_then_else(
                        key_position < dynamic_num_tokens,
                        value[batch, key_position, head // groups, j],
                        0.0,
                    )
                T.gemm(
                    probabilities,
                    value_shared,
                    accumulator,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for i, j in T.Parallel(tile_query_tokens, dim):
                accumulator[i, j] /= total[i]
            for i, j in T.Parallel(tile_query_tokens, dim):
                position = row * tile_query_tokens + i
                if position < dynamic_num_tokens:
                    output[batch, position, head, j] = accumulator[i, j]
                elif position < dynamic_num_padded_tokens:
                    output[batch, position, head, j] = 0

    return main.with_attr(
        "global_symbol",
        "causal_attention_"
        f"{heads}_{kv_heads}_{dim}_"
        f"{tile_query_tokens}_{tile_kv_tokens}_{stages}_{threads}_{dtype.value}",
    )


class CausalAttentionKernel(
    Kernel[CausalAttentionArguments, CausalAttentionWorkload, AttentionConfig]
):
    name = "causal_attention"
    program = causal_attention

    @classmethod
    def make_arguments(
        cls, workload: CausalAttentionWorkload
    ) -> CausalAttentionArguments:
        """Describe padded Q/K/V storage and the valid causal prefix."""
        query_shape = (workload.num_padded_tokens, workload.heads, workload.dim)
        kv_shape = (workload.num_padded_tokens, workload.kv_heads, workload.dim)
        return CausalAttentionArguments(
            query=TensorDesc.empty(workload.dtype, query_shape),
            key=TensorDesc.empty(workload.dtype, kv_shape),
            value=TensorDesc.empty(workload.dtype, kv_shape),
            output=TensorDesc.empty(workload.dtype, query_shape),
            num_tokens=workload.num_tokens,
        )

    @classmethod
    def make_config(cls, workload: CausalAttentionWorkload) -> AttentionConfig:
        f32 = workload.dtype == DType.F32
        return AttentionConfig(
            tile_query_tokens=16 if f32 else 64,
            tile_kv_tokens=32,
            stages=1,
            threads=128,
        )

    @classmethod
    def make_workload(
        cls, arguments: CausalAttentionArguments
    ) -> CausalAttentionWorkload:
        query, key, value, output = (
            arguments.query,
            arguments.key,
            arguments.value,
            arguments.output,
        )
        assert all(len(tensor.shape) == 3 for tensor in (query, key, value, output))
        num_padded_tokens, heads, dim = query.shape
        assert output.shape == query.shape
        assert key.shape == value.shape
        assert key.shape[0] == num_padded_tokens and key.shape[2] == dim
        assert all(tensor.dtype == query.dtype for tensor in (key, value, output))
        assert query.dtype in (DType.BF16, DType.F32)
        assert heads % key.shape[1] == 0
        assert 0 < arguments.num_tokens <= num_padded_tokens
        return CausalAttentionWorkload(
            dtype=query.dtype,
            num_padded_tokens=num_padded_tokens,
            num_tokens=arguments.num_tokens,
            heads=heads,
            kv_heads=key.shape[1],
            dim=dim,
        )

    @classmethod
    def tops(cls, arguments: CausalAttentionArguments) -> Tops:
        workload = cls.make_workload(arguments)
        pairs = workload.num_tokens * (workload.num_tokens + 1) // 2
        # QK and PV each perform one multiply-add per valid causal pair.
        operations = 4 * workload.heads * pairs * workload.dim
        mma = (
            MmaType.TF32TF32F32 if workload.dtype == DType.F32 else MmaType.BF16BF16F32
        )
        return {mma: operations}

    @classmethod
    def ref_program(cls, arguments: CausalAttentionArguments) -> None:
        """Evaluate grouped-head causal attention over the valid token prefix."""
        import torch.nn.functional as F

        workload = cls.make_workload(arguments)
        stop = workload.num_tokens
        query = arguments.query.as_torch()[:stop].transpose(0, 1)
        key = arguments.key.as_torch()[:stop].transpose(0, 1)
        value = arguments.value.as_torch()[:stop].transpose(0, 1)
        result = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            enable_gqa=workload.heads != workload.kv_heads,
        )
        output = arguments.output.as_torch()
        output.zero_()
        output[:stop].copy_(result.transpose(0, 1))
