"""Approximate-block finalization for Sol NVFP4 attention."""

import dataclasses
from typing import Literal

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc


class SolNvfp4FinalizeWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    total_heads: int = Field(gt=0)


class SolNvfp4FinalizeConfig(Config):
    threads: Literal[64, 128, 256] = 128


@dataclasses.dataclass(frozen=True, slots=True)
class SolNvfp4FinalizeArguments(Arguments):
    query: TensorDesc
    pooled: TensorDesc
    selected: TensorDesc
    exact: TensorDesc
    state: TensorDesc
    output: TensorDesc
    head_offset: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.query.shape[0]), ("head_offset", self.head_offset))


@tilelang.jit(out_idx=[], execution_backend="nvrtc", compile_flags=["--use_fast_math"])
def sol_nvfp4_finalize(num_tokens, heads, total_heads, head_offset=0, threads=128):
    import tilelang.language as T

    num_blocks = (num_tokens + 63) // 64
    scale = 128**-0.5 * 1.4426950408889634

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        head_offset: head_offset,
        query: T.Tensor((num_tokens, heads, 128), T.bfloat16),
        pooled: T.Tensor((3, heads, num_blocks, 128), T.bfloat16),
        selected: T.Tensor((num_blocks, heads, num_blocks), T.uint8),
        exact: T.Tensor((num_tokens, total_heads, 128), T.bfloat16),
        state: T.Tensor((num_tokens, total_heads, 2), T.float32),
        output: T.Tensor((num_tokens, total_heads, 128), T.bfloat16),
    ):
        with T.Kernel(num_blocks, heads, threads=threads) as (query_block, head):
            q = T.alloc_shared((64, 128), T.bfloat16)
            pooled_tile = T.alloc_shared((32, 128), T.bfloat16)
            scores = T.alloc_fragment((64, 32), T.float32)
            probability = T.alloc_fragment((64, 32), T.bfloat16)
            accumulator = T.alloc_fragment((64, 128), T.float32)
            maximum = T.alloc_fragment((64,), T.float32)
            previous = T.alloc_fragment((64,), T.float32)
            correction = T.alloc_fragment((64,), T.float32)
            total = T.alloc_fragment((64,), T.float32)
            subtotal = T.alloc_fragment((64,), T.float32)

            T.copy(query[query_block * 64 : (query_block + 1) * 64, head, :], q)
            for i, d in T.Parallel(64, 128):
                accumulator[i, d] = T.if_then_else(
                    query_block * 64 + i < num_tokens,
                    exact[query_block * 64 + i, head + head_offset, d],
                    0,
                )
            for i in T.Parallel(64):
                maximum[i] = T.if_then_else(
                    query_block * 64 + i < num_tokens,
                    state[query_block * 64 + i, head + head_offset, 0],
                    -1e30,
                )
                total[i] = T.if_then_else(
                    query_block * 64 + i < num_tokens,
                    state[query_block * 64 + i, head + head_offset, 1],
                    0,
                )

            for group in T.serial(T.ceildiv(num_blocks, 32)):
                T.copy(
                    pooled[1, head, group * 32 : (group + 1) * 32, :],
                    pooled_tile,
                )
                T.clear(scores)
                T.gemm(
                    q,
                    pooled_tile,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(maximum, previous)
                for i, j in T.Parallel(64, 32):
                    block = group * 32 + j
                    scores[i, j] = T.if_then_else(
                        block < num_blocks and selected[query_block, head, block] == 0,
                        scores[i, j],
                        -1e30,
                    )
                T.reduce_max(scores, maximum, dim=1, clear=False)
                for i in T.Parallel(64):
                    correction[i] = T.exp2((previous[i] - maximum[i]) * scale)
                for i, j in T.Parallel(64, 32):
                    block = group * 32 + j
                    scores[i, j] = T.if_then_else(
                        block < num_blocks and selected[query_block, head, block] == 0,
                        T.exp2((scores[i, j] - maximum[i]) * scale),
                        0,
                    )
                T.copy(scores, probability)
                for i, j in T.Parallel(64, 32):
                    scores[i, j] *= T.min(64, num_tokens - (group * 32 + j) * 64)
                T.reduce_sum(scores, subtotal, dim=1)
                for i in T.Parallel(64):
                    total[i] = total[i] * correction[i] + subtotal[i]
                for i, d in T.Parallel(64, 128):
                    accumulator[i, d] *= correction[i]
                T.copy(
                    pooled[2, head, group * 32 : (group + 1) * 32, :],
                    pooled_tile,
                )
                T.gemm(
                    probability,
                    pooled_tile,
                    accumulator,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for i, d in T.Parallel(64, 128):
                if query_block * 64 + i < num_tokens:
                    output[query_block * 64 + i, head + head_offset, d] = (
                        accumulator[i, d] / total[i]
                    )

    return main.with_attr(
        "global_symbol",
        f"sol_nvfp4_finalize_{heads}_{total_heads}_{threads}",
    )


class SolNvfp4FinalizeKernel(
    Kernel[
        SolNvfp4FinalizeArguments,
        SolNvfp4FinalizeWorkload,
        SolNvfp4FinalizeConfig,
    ]
):
    name = "sol_nvfp4_finalize"
    program = sol_nvfp4_finalize

    @classmethod
    def make_arguments(
        cls, workload: SolNvfp4FinalizeWorkload
    ) -> SolNvfp4FinalizeArguments:
        num_blocks = -(-workload.num_tokens // 64)
        return SolNvfp4FinalizeArguments(
            TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.heads, 128)),
            TensorDesc.empty(DType.BF16, (3, workload.heads, num_blocks, 128)),
            TensorDesc.empty(DType.U8, (num_blocks, workload.heads, num_blocks)),
            TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.total_heads, 128)
            ),
            TensorDesc.empty(DType.F32, (workload.num_tokens, workload.total_heads, 2)),
            TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.total_heads, 128)
            ),
            0,
        )

    @classmethod
    def make_config(cls, workload: SolNvfp4FinalizeWorkload) -> SolNvfp4FinalizeConfig:
        del workload
        return SolNvfp4FinalizeConfig(threads=64)

    @classmethod
    def make_workload(
        cls, arguments: SolNvfp4FinalizeArguments
    ) -> SolNvfp4FinalizeWorkload:
        num_tokens, heads, dim = arguments.query.shape
        num_blocks = -(-num_tokens // 64)
        assert arguments.query.dtype == DType.BF16 and dim == 128
        assert arguments.pooled.dtype == DType.BF16
        assert arguments.pooled.shape == (3, heads, num_blocks, 128)
        assert arguments.selected.dtype == DType.U8
        assert arguments.selected.shape == (num_blocks, heads, num_blocks)
        assert arguments.exact.dtype == arguments.output.dtype == DType.BF16
        assert arguments.state.dtype == DType.F32
        assert arguments.exact.shape == arguments.output.shape
        assert arguments.state.shape[:1] == (num_tokens,)
        total_heads = arguments.output.shape[1]
        assert arguments.output.shape == (num_tokens, total_heads, 128)
        assert arguments.state.shape == (num_tokens, total_heads, 2)
        assert 0 <= arguments.head_offset <= total_heads - heads
        return SolNvfp4FinalizeWorkload(
            num_tokens=num_tokens, heads=heads, total_heads=total_heads
        )

    @classmethod
    def tops(cls, arguments: SolNvfp4FinalizeArguments) -> Tops:
        workload = cls.make_workload(arguments)
        num_blocks = -(-workload.num_tokens // 64)
        operations = 4 * workload.heads * 128 * workload.num_tokens * num_blocks
        return {MmaType.BF16BF16F32: operations}

    @classmethod
    def ref_program(cls, arguments: SolNvfp4FinalizeArguments) -> None:
        """Merge exact numerator/state with unselected pooled block estimates.

        State maximums use raw QK dot-product units; this stage applies the
        common attention scale to both exact and pooled contributions.
        """
        import torch

        workload = cls.make_workload(arguments)
        query = arguments.query.as_torch()
        pooled = arguments.pooled.as_torch()
        selected = arguments.selected.as_torch()
        exact = arguments.exact.as_torch()
        state = arguments.state.as_torch()
        output = arguments.output.as_torch()
        num_blocks = -(-workload.num_tokens // 64)
        scale = 128**-0.5 * 1.4426950408889634
        lengths = torch.full(
            (num_blocks,), 64, dtype=torch.float32, device=query.device
        )
        lengths[-1] = workload.num_tokens - (num_blocks - 1) * 64

        for query_block in range(num_blocks):
            start = query_block * 64
            stop = min(start + 64, workload.num_tokens)
            for head in range(workload.heads):
                target = head + arguments.head_offset
                block_query = query[start:stop, head].float()
                numerator = exact[start:stop, target].float()
                maximum = state[start:stop, target, 0].float()
                denominator = state[start:stop, target, 1].float()

                # Match the kernel's 32-block traversal and online-softmax update.
                for group_start in range(0, num_blocks, 32):
                    group_stop = min(group_start + 32, num_blocks)
                    approximate = (
                        selected[query_block, head, group_start:group_stop] == 0
                    )
                    scores = (
                        block_query @ pooled[1, head, group_start:group_stop].float().T
                    )
                    scores = scores.masked_fill(~approximate[None, :], -1e30)
                    next_maximum = torch.maximum(maximum, scores.amax(dim=1))
                    correction = torch.exp2((maximum - next_maximum) * scale)
                    probability = (
                        torch.exp2((scores - next_maximum[:, None]) * scale)
                        * approximate[None, :]
                    )
                    numerator = numerator * correction[:, None] + (
                        probability @ pooled[2, head, group_start:group_stop].float()
                    )
                    denominator = denominator * correction + (
                        probability * lengths[group_start:group_stop][None, :]
                    ).sum(dim=1)
                    maximum = next_maximum

                output[start:stop, target].copy_(
                    (numerator / denominator[:, None]).to(output.dtype)
                )
