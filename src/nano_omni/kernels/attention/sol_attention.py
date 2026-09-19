"""BF16 head-sliced Sol attention with pooled correction."""

import dataclasses
import hashlib
import math
from typing import Literal

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc


class SolAttentionWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    total_heads: int = Field(gt=0)
    tau: float = Field(default=1.0, allow_inf_nan=False)


class SolAttentionConfig(Config):
    blocks_per_group: Literal[16, 32, 64] = 32
    threads: Literal[64, 128, 256] = 128


@dataclasses.dataclass(frozen=True, slots=True)
class SolAttentionArguments(Arguments):
    query: TensorDesc
    key: TensorDesc
    value: TensorDesc
    pooled: TensorDesc
    stats: TensorDesc
    output: TensorDesc
    tau: float
    head_offset: int
    num_protected_blocks: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (
            ("num_tokens", self.query.shape[0]),
            ("head_offset", self.head_offset),
            ("num_protected_blocks", self.num_protected_blocks),
        )


@tilelang.jit(
    out_idx=[],
    execution_backend="nvrtc",
    pass_configs={"tl.disable_vectorize_256": True},
    compile_flags=["--use_fast_math"],
)
def sol_attention(
    num_tokens,
    heads,
    total_heads,
    tau,
    head_offset=0,
    num_protected_blocks=0,
    blocks_per_group=32,
    threads=128,
):
    import tilelang.language as T

    num_blocks = (num_tokens + 63) // 64
    scale = 128**-0.5 * 1.4426950408889634

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        head_offset: head_offset,
        num_protected_blocks: num_protected_blocks,
        query: T.Tensor((num_tokens, heads, 128), "bfloat16"),
        key: T.Tensor((num_tokens, heads, 128), "bfloat16"),
        value: T.Tensor((num_tokens, heads, 128), "bfloat16"),
        pooled: T.Tensor((2, heads, num_blocks, 128), "bfloat16"),
        stats: T.Tensor((heads, 2, 128), "float32"),
        output: T.Tensor((num_tokens, total_heads, 128), "bfloat16"),
    ):
        with T.Kernel(num_blocks, heads, threads=threads) as (qb, head):
            q = T.alloc_shared((64, 128), "bfloat16")
            k = T.alloc_shared((64, 128), "bfloat16")
            kc = T.alloc_shared((blocks_per_group, 128), "bfloat16")
            q_float = T.alloc_fragment((64, 128), "float32")
            qmean = T.alloc_fragment((128,), "float32")
            moments = T.alloc_fragment((2, 128), "float32")
            threshold_parts = T.alloc_fragment((2,), "float32")
            threshold = T.alloc_shared((1,), "float32")
            reduced_parts = T.alloc_shared((2,), "float32")
            scores = T.alloc_fragment((64, 64), "float32")
            pooled_scores = T.alloc_fragment((64, blocks_per_group), "float32")
            route_scores = T.alloc_fragment((blocks_per_group,), "float32")
            selected = T.alloc_shared((blocks_per_group,), "int32")
            selected_count = T.alloc_var("int32")
            selected_bits = T.alloc_var("uint64")
            probability = T.alloc_fragment((64, 64), "bfloat16")
            pooled_probability = T.alloc_fragment((64, blocks_per_group), "bfloat16")
            accum = T.alloc_fragment((64, 128), "float32")
            maximum = T.alloc_fragment((64,), "float32")
            previous = T.alloc_fragment((64,), "float32")
            alpha = T.alloc_fragment((64,), "float32")
            total = T.alloc_fragment((64,), "float32")
            subtotal = T.alloc_fragment((64,), "float32")
            T.copy(query[qb * 64 : (qb + 1) * 64, head, :], q)
            T.copy(q, q_float)
            T.reduce_sum(q_float, qmean, dim=0)
            for d in T.Parallel(128):
                qmean[d] /= T.min(64, num_tokens - qb * 64)
            for part, d in T.Parallel(2, 128):
                moments[part, d] = T.if_then_else(
                    part == 0,
                    qmean[d] * stats[head, 0, d] * scale,
                    qmean[d] * qmean[d] * stats[head, 1, d] * scale * scale,
                )
            T.reduce_sum(moments, threshold_parts, dim=1)
            T.copy(threshold_parts, reduced_parts)
            for i in T.Parallel(1):
                threshold[0] = reduced_parts[0] + tau * T.sqrt(
                    T.max(reduced_parts[1], 0) + 1e-6
                )
            T.clear(accum)
            T.clear(total)
            T.fill(maximum, -1e30)
            for group in T.serial(T.ceildiv(num_blocks, blocks_per_group)):
                T.copy(
                    pooled[
                        0,
                        head,
                        group * blocks_per_group : (group + 1) * blocks_per_group,
                        :,
                    ],
                    kc,
                )
                T.clear(pooled_scores)
                T.gemm(
                    q,
                    kc,
                    pooled_scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(64, blocks_per_group):
                    pooled_scores[i, j] *= scale
                T.reduce_sum(pooled_scores, route_scores, dim=0)
                for j in T.Parallel(blocks_per_group):
                    selected[j] = T.if_then_else(
                        group * blocks_per_group + j < num_blocks
                        and (
                            group * blocks_per_group + j < num_protected_blocks
                            or route_scores[j] / T.min(64, num_tokens - qb * 64)
                            > threshold[0]
                            or (
                                qb >= group * blocks_per_group + j - 1
                                and qb <= group * blocks_per_group + j + 1
                            )
                        ),
                        1,
                        0,
                    )
                # Pooled num_blocks contribute a V sum and a length-weighted denominator.
                T.copy(maximum, previous)
                for i, j in T.Parallel(64, blocks_per_group):
                    pooled_scores[i, j] = T.if_then_else(
                        group * blocks_per_group + j < num_blocks and selected[j] == 0,
                        pooled_scores[i, j],
                        -1e30,
                    )
                T.reduce_max(pooled_scores, maximum, dim=1, clear=False)
                for i in T.Parallel(64):
                    alpha[i] = T.exp2(previous[i] - maximum[i])
                for i, j in T.Parallel(64, blocks_per_group):
                    pooled_scores[i, j] = T.if_then_else(
                        group * blocks_per_group + j < num_blocks and selected[j] == 0,
                        T.exp2(pooled_scores[i, j] - maximum[i]),
                        0,
                    )
                T.copy(pooled_scores, pooled_probability)
                for i, j in T.Parallel(64, blocks_per_group):
                    pooled_scores[i, j] *= T.min(
                        64, num_tokens - (group * blocks_per_group + j) * 64
                    )
                T.reduce_sum(pooled_scores, subtotal, dim=1)
                for i in T.Parallel(64):
                    total[i] = total[i] * alpha[i] + subtotal[i]
                for i, d in T.Parallel(64, 128):
                    accum[i, d] *= alpha[i]
                T.copy(
                    pooled[
                        1,
                        head,
                        group * blocks_per_group : (group + 1) * blocks_per_group,
                        :,
                    ],
                    kc,
                )
                T.gemm(pooled_probability, kc, accum, policy=T.GemmWarpPolicy.FullRow)
                # Every warp forms the same mask; each set bit names one exact block.
                selected_bits = T.ballot_sync(
                    T.if_then_else(
                        T.get_thread_binding() % 32 < blocks_per_group,
                        selected[T.get_thread_binding() % 32] != 0,
                        False,
                    )
                )
                if blocks_per_group == 64:
                    selected_bits = selected_bits | (
                        T.ballot_sync(selected[T.get_thread_binding() % 32 + 32] != 0)
                        << 32
                    )
                selected_count = T.popcount(selected_bits)
                for sparse in T.serial(selected_count):
                    kb = group * blocks_per_group + T.__ffs(selected_bits) - 1
                    selected_bits = selected_bits & (selected_bits - 1)
                    T.copy(key[kb * 64 : (kb + 1) * 64, head, :], k)
                    T.clear(scores)
                    T.gemm(
                        q,
                        k,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(64, 64):
                        scores[i, j] = T.if_then_else(
                            kb * 64 + j < num_tokens, scores[i, j] * scale, -1e30
                        )
                    T.copy(maximum, previous)
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(64):
                        alpha[i] = T.exp2(previous[i] - maximum[i])
                    for i, j in T.Parallel(64, 64):
                        scores[i, j] = T.exp2(scores[i, j] - maximum[i])
                    T.copy(scores, probability)
                    T.reduce_sum(scores, subtotal, dim=1)
                    for i in T.Parallel(64):
                        total[i] = total[i] * alpha[i] + subtotal[i]
                    for i, d in T.Parallel(64, 128):
                        accum[i, d] *= alpha[i]
                    T.copy(value[kb * 64 : (kb + 1) * 64, head, :], k)
                    T.gemm(probability, k, accum, policy=T.GemmWarpPolicy.FullRow)
            for i, d in T.Parallel(64, 128):
                if qb * 64 + i < num_tokens:
                    output[qb * 64 + i, head + head_offset, d] = accum[i, d] / total[i]

    return main.with_attr(
        "global_symbol",
        f"sol_attention_{heads}_{total_heads}_{blocks_per_group}_{threads}_"
        + hashlib.sha256(str(tau).encode()).hexdigest()[:8],
    )


class SolAttentionKernel(
    Kernel[SolAttentionArguments, SolAttentionWorkload, SolAttentionConfig]
):
    name = "sol_attention_d128"
    program = sol_attention

    @classmethod
    def make_arguments(cls, workload: SolAttentionWorkload) -> SolAttentionArguments:
        num_blocks = -(-workload.num_tokens // 64)
        return SolAttentionArguments(
            TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.heads, 128)),
            TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.heads, 128)),
            TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.heads, 128)),
            TensorDesc.empty(DType.BF16, (2, workload.heads, num_blocks, 128)),
            TensorDesc.empty(DType.F32, (workload.heads, 2, 128)),
            TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.total_heads, 128)
            ),
            workload.tau,
            0,
            0,
        )

    @classmethod
    def make_config(cls, workload: SolAttentionWorkload) -> SolAttentionConfig:
        del workload
        return SolAttentionConfig()

    @classmethod
    def tops(cls, arguments: SolAttentionArguments) -> Tops:
        workload = cls.make_workload(arguments)
        num_blocks = -(-workload.num_tokens // 64)
        protected = min(arguments.num_protected_blocks, num_blocks)
        lengths = tuple(
            min(64, workload.num_tokens - block * 64) for block in range(num_blocks)
        )
        mandatory = workload.num_tokens * sum(lengths[:protected])
        for query_block, query_length in enumerate(lengths):
            for key_block in range(
                max(protected, query_block - 1), min(num_blocks, query_block + 2)
            ):
                mandatory += query_length * lengths[key_block]
        probability = 0.5 * math.erfc(workload.tau / math.sqrt(2))
        exact_pairs = mandatory + probability * (workload.num_tokens**2 - mandatory)
        pooled_pairs = workload.num_tokens * num_blocks
        operations = round(4 * workload.heads * 128 * (exact_pairs + pooled_pairs))
        return {MmaType.BF16BF16F32: operations}

    @classmethod
    def ref_program(cls, arguments: SolAttentionArguments) -> None:
        import torch

        workload = cls.make_workload(arguments)
        query = arguments.query.as_torch().float()
        key = arguments.key.as_torch().float()
        value = arguments.value.as_torch().float()
        pooled = arguments.pooled.as_torch().float()
        stats = arguments.stats.as_torch()
        num_blocks = -(-workload.num_tokens // 64)
        scale = 128**-0.5
        result = torch.empty_like(query)

        for query_block in range(num_blocks):
            query_start = query_block * 64
            query_end = min(query_start + 64, workload.num_tokens)
            block_query = query[query_start:query_end]
            query_mean = block_query.mean(dim=0)
            threshold = (query_mean * stats[:, 0]).sum(
                dim=1
            ) * scale + workload.tau * torch.sqrt(
                (query_mean.square() * stats[:, 1] * (scale * scale))
                .sum(dim=1)
                .clamp_min(0)
                + 1e-6
            )
            route_scores = torch.einsum("qhd,hbd->qhb", block_query, pooled[0])
            route_scores = route_scores.mean(dim=0) * scale

            for head in range(workload.heads):
                logits: list[torch.Tensor] = []
                weighted_values: list[torch.Tensor] = []
                multiplicities: list[torch.Tensor] = []
                for key_block in range(num_blocks):
                    key_start = key_block * 64
                    key_end = min(key_start + 64, workload.num_tokens)
                    selected = (
                        key_block < arguments.num_protected_blocks
                        or route_scores[head, key_block] > threshold[head]
                        or abs(query_block - key_block) <= 1
                    )
                    if selected:
                        block_logits = (
                            block_query[:, head] @ key[key_start:key_end, head].T
                        ) * scale
                        logits.append(block_logits)
                        weighted_values.append(value[key_start:key_end, head])
                        multiplicities.append(
                            torch.ones(
                                key_end - key_start,
                                device=block_query.device,
                                dtype=torch.float32,
                            )
                        )
                    else:
                        block_logits = (
                            block_query[:, head] @ pooled[0, head, key_block]
                        ).unsqueeze(1) * scale
                        logits.append(block_logits)
                        weighted_values.append(pooled[1, head, key_block].unsqueeze(0))
                        multiplicities.append(
                            torch.tensor(
                                [key_end - key_start],
                                device=block_query.device,
                                dtype=torch.float32,
                            )
                        )

                all_logits = torch.cat(logits, dim=1)
                maximum = all_logits.max(dim=1, keepdim=True).values
                probabilities = torch.exp(all_logits - maximum)
                numerator = torch.zeros(
                    (query_end - query_start, 128),
                    device=block_query.device,
                    dtype=torch.float32,
                )
                denominator = torch.zeros(
                    (query_end - query_start, 1),
                    device=block_query.device,
                    dtype=torch.float32,
                )
                offset = 0
                for block_values, lengths in zip(
                    weighted_values, multiplicities, strict=True
                ):
                    width = block_values.shape[0]
                    block_probability = probabilities[:, offset : offset + width]
                    numerator += block_probability @ block_values
                    denominator += block_probability @ lengths[:, None]
                    offset += width
                result[query_start:query_end, head] = numerator / denominator

        output = arguments.output.as_torch()
        start = arguments.head_offset
        output[:, start : start + workload.heads].copy_(result)

    @classmethod
    def make_workload(cls, arguments: SolAttentionArguments) -> SolAttentionWorkload:
        num_tokens, heads, dim = arguments.query.shape
        assert dim == 128
        assert (
            arguments.query.dtype
            == arguments.key.dtype
            == arguments.value.dtype
            == DType.BF16
        )
        assert arguments.query.shape == arguments.key.shape == arguments.value.shape
        num_blocks = -(-num_tokens // 64)
        assert arguments.pooled.dtype == DType.BF16 and arguments.pooled.shape == (
            2,
            heads,
            num_blocks,
            128,
        )
        assert arguments.stats.dtype == DType.F32 and arguments.stats.shape == (
            heads,
            2,
            128,
        )
        assert arguments.output.dtype == DType.BF16 and len(arguments.output.shape) == 3
        assert (
            arguments.output.shape[0] == num_tokens and arguments.output.shape[2] == 128
        )
        total_heads = arguments.output.shape[1]
        assert 0 <= arguments.head_offset <= total_heads - heads
        assert 0 <= arguments.num_protected_blocks <= num_blocks
        return SolAttentionWorkload(
            num_tokens=num_tokens,
            heads=heads,
            total_heads=total_heads,
            tau=arguments.tau,
        )
