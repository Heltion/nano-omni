"""Dense BF16 attention for H3 head slices."""

import dataclasses
from typing import Literal

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc


class DenseBf16AttentionWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    total_heads: int = Field(gt=0)


class DenseBf16AttentionConfig(Config):
    tile_query_tokens: Literal[32, 64, 128] = 64
    tile_kv_tokens: Literal[32, 64, 128] = 64
    threads: Literal[64, 128, 256] = 128


@dataclasses.dataclass(frozen=True, slots=True)
class DenseBf16AttentionArguments(Arguments):
    query: TensorDesc
    key: TensorDesc
    value: TensorDesc
    output: TensorDesc
    head_offset: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (
            ("num_tokens", self.query.shape[0]),
            ("head_offset", self.head_offset),
        )


@tilelang.jit(
    out_idx=[],
    execution_backend="nvrtc",
    pass_configs={"tl.disable_vectorize_256": True},
    compile_flags=["--use_fast_math"],
)
def dense_bf16_attention(
    num_tokens,
    heads,
    total_heads,
    head_offset=0,
    tile_query_tokens=64,
    tile_kv_tokens=64,
    threads=128,
):
    import tilelang.language as T

    scale = 128**-0.5 * 1.4426950408889634

    @T.macro
    def attend_block(key_block, masked, head, buffers):
        (
            key,
            value,
            q,
            k,
            v,
            scores,
            probability,
            accumulator,
            maximum,
            previous,
            correction,
            total,
            subtotal,
        ) = buffers
        T.copy(
            key[key_block * tile_kv_tokens : (key_block + 1) * tile_kv_tokens, head, :],
            k,
        )
        if masked:
            for i, j in T.Parallel(tile_query_tokens, tile_kv_tokens):
                scores[i, j] = T.if_then_else(
                    key_block * tile_kv_tokens + j < num_tokens,
                    0.0,
                    -T.infinity(T.float32),
                )
        else:
            T.clear(scores)
        T.gemm(
            q,
            k,
            scores,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullRow,
        )
        T.copy(
            value[
                key_block * tile_kv_tokens : (key_block + 1) * tile_kv_tokens, head, :
            ],
            v,
        )
        T.copy(maximum, previous)
        T.fill(maximum, -T.infinity(T.float32))
        T.reduce_max(scores, maximum, dim=1, clear=False)
        for i in T.Parallel(tile_query_tokens):
            maximum[i] = T.max(maximum[i], previous[i])
            correction[i] = T.exp2(previous[i] * scale - maximum[i] * scale)
        for i, j in T.Parallel(tile_query_tokens, tile_kv_tokens):
            scores[i, j] = T.exp2(scores[i, j] * scale - maximum[i] * scale)
        T.copy(scores, probability)
        T.reduce_sum(scores, subtotal, dim=1)
        for i in T.Parallel(tile_query_tokens):
            total[i] = total[i] * correction[i] + subtotal[i]
        for i, d in T.Parallel(tile_query_tokens, 128):
            if correction[i] != 1.0:
                accumulator[i, d] *= correction[i]
        T.gemm(
            probability,
            v,
            accumulator,
            policy=T.GemmWarpPolicy.FullRow,
        )

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        head_offset: head_offset,
        query: T.Tensor((num_tokens, heads, 128), T.bfloat16),
        key: T.Tensor((num_tokens, heads, 128), T.bfloat16),
        value: T.Tensor((num_tokens, heads, 128), T.bfloat16),
        output: T.Tensor((num_tokens, total_heads, 128), T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(num_tokens, tile_query_tokens), heads, threads=threads
        ) as (
            query_block,
            head,
        ):
            q = T.alloc_shared((tile_query_tokens, 128), T.bfloat16)
            k = T.alloc_shared((tile_kv_tokens, 128), T.bfloat16)
            v = T.alloc_shared((tile_kv_tokens, 128), T.bfloat16)
            scores = T.alloc_fragment((tile_query_tokens, tile_kv_tokens), T.float32)
            probability = T.alloc_fragment(
                (tile_query_tokens, tile_kv_tokens), T.bfloat16
            )
            accumulator = T.alloc_fragment((tile_query_tokens, 128), T.float32)
            maximum = T.alloc_fragment((tile_query_tokens,), T.float32)
            previous = T.alloc_fragment((tile_query_tokens,), T.float32)
            correction = T.alloc_fragment((tile_query_tokens,), T.float32)
            total = T.alloc_fragment((tile_query_tokens,), T.float32)
            subtotal = T.alloc_fragment((tile_query_tokens,), T.float32)

            T.copy(
                query[
                    query_block * tile_query_tokens : (query_block + 1)
                    * tile_query_tokens,
                    head,
                    :,
                ],
                q,
            )
            T.clear(accumulator)
            T.clear(total)
            T.fill(maximum, -1e30)
            buffers = (
                key,
                value,
                q,
                k,
                v,
                scores,
                probability,
                accumulator,
                maximum,
                previous,
                correction,
                total,
                subtotal,
            )
            for key_block in T.serial(num_tokens // tile_kv_tokens):
                attend_block(key_block, False, head, buffers)
            if num_tokens % tile_kv_tokens:
                attend_block(num_tokens // tile_kv_tokens, True, head, buffers)
            for i in T.Parallel(tile_query_tokens):
                total[i] = 1.0 / total[i]
            for i, d in T.Parallel(tile_query_tokens, 128):
                if query_block * tile_query_tokens + i < num_tokens:
                    output[
                        query_block * tile_query_tokens + i, head + head_offset, d
                    ] = accumulator[i, d] * total[i]

    return main.with_attr(
        "global_symbol",
        f"dense_bf16_attention_{heads}_{total_heads}_{tile_query_tokens}_{tile_kv_tokens}_{threads}",
    )


class DenseBf16AttentionKernel(
    Kernel[
        DenseBf16AttentionArguments,
        DenseBf16AttentionWorkload,
        DenseBf16AttentionConfig,
    ]
):
    name = "dense_bf16_attention_d128"
    program = dense_bf16_attention

    @classmethod
    def make_arguments(
        cls, workload: DenseBf16AttentionWorkload
    ) -> DenseBf16AttentionArguments:
        shape = (workload.num_tokens, workload.heads, 128)
        return DenseBf16AttentionArguments(
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.total_heads, 128)
            ),
            0,
        )

    @classmethod
    def make_config(
        cls, workload: DenseBf16AttentionWorkload
    ) -> DenseBf16AttentionConfig:
        if workload.num_tokens >= 1024:
            return DenseBf16AttentionConfig(tile_kv_tokens=32)
        return DenseBf16AttentionConfig()

    @classmethod
    def make_workload(
        cls, arguments: DenseBf16AttentionArguments
    ) -> DenseBf16AttentionWorkload:
        num_tokens, heads, dim = arguments.query.shape
        assert dim == 128
        assert arguments.query.dtype == DType.BF16
        assert arguments.key.dtype == arguments.value.dtype == DType.BF16
        assert arguments.query.shape == arguments.key.shape == arguments.value.shape
        assert arguments.output.dtype == DType.BF16
        assert len(arguments.output.shape) == 3
        assert (
            arguments.output.shape[0] == num_tokens and arguments.output.shape[2] == 128
        )
        total_heads = arguments.output.shape[1]
        assert 0 <= arguments.head_offset <= total_heads - heads
        return DenseBf16AttentionWorkload(
            num_tokens=num_tokens, heads=heads, total_heads=total_heads
        )

    @classmethod
    def tops(cls, arguments: DenseBf16AttentionArguments) -> Tops:
        workload = cls.make_workload(arguments)
        operations = 4 * workload.heads * workload.num_tokens**2 * 128
        return {MmaType.BF16BF16F32: operations}

    @classmethod
    def ref_program(cls, arguments: DenseBf16AttentionArguments) -> None:
        import torch.nn.functional as F

        query = arguments.query.as_torch().transpose(0, 1)
        key = arguments.key.as_torch().transpose(0, 1)
        value = arguments.value.as_torch().transpose(0, 1)
        result = F.scaled_dot_product_attention(query, key, value)
        start = arguments.head_offset
        arguments.output.as_torch()[:, start : start + query.shape[0]].copy_(
            result.transpose(0, 1)
        )
