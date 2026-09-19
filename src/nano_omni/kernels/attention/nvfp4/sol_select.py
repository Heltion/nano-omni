"""BF16 block routing mask for Sol NVFP4 attention."""

import dataclasses
import hashlib
import math
from typing import Literal

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc


class SolNvfp4SelectWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    tau: float = Field(default=1.0, allow_inf_nan=False)


class SolNvfp4SelectConfig(Config):
    threads: Literal[64, 128, 256] = 128


@dataclasses.dataclass(frozen=True, slots=True)
class SolNvfp4SelectArguments(Arguments):
    query: TensorDesc
    pooled_key: TensorDesc
    stats: TensorDesc
    output: TensorDesc
    tau: float
    num_protected_blocks: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (
            ("num_tokens", self.query.shape[0]),
            ("num_protected_blocks", self.num_protected_blocks),
        )


@tilelang.jit(out_idx=[], execution_backend="nvrtc", compile_flags=["--use_fast_math"])
def sol_nvfp4_select(num_tokens, heads, tau, num_protected_blocks=0, threads=128):
    import tilelang.language as T

    num_blocks = (num_tokens + 63) // 64
    scale = 128**-0.5

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        num_protected_blocks: num_protected_blocks,
        query: T.Tensor((num_tokens, heads, 128), T.bfloat16),
        pooled_key: T.Tensor((heads, num_blocks, 128), T.bfloat16),
        stats: T.Tensor((heads, 2, 128), T.float32),
        selected: T.Tensor((num_blocks, heads, num_blocks), T.uint8),
    ):
        with T.Kernel(num_blocks, heads, threads=threads) as (query_block, head):
            query_tile = T.alloc_shared((64, 128), T.bfloat16)
            key_tile = T.alloc_shared((32, 128), T.bfloat16)
            scores = T.alloc_fragment((64, 32), T.float32)
            route = T.alloc_fragment((32,), T.float32)
            query_float = T.alloc_fragment((64, 128), T.float32)
            query_mean = T.alloc_fragment((128,), T.float32)
            moments = T.alloc_fragment((2, 128), T.float32)
            threshold_parts = T.alloc_fragment((2,), T.float32)
            threshold_shared = T.alloc_shared((2,), T.float32)

            T.copy(
                query[query_block * 64 : (query_block + 1) * 64, head, :],
                query_tile,
            )
            T.copy(query_tile, query_float)
            T.reduce_sum(query_float, query_mean, dim=0)
            for d in T.Parallel(128):
                query_mean[d] /= T.min(64, num_tokens - query_block * 64)
            for part, d in T.Parallel(2, 128):
                moments[part, d] = T.if_then_else(
                    part == 0,
                    query_mean[d] * stats[head, 0, d] * scale,
                    query_mean[d] * query_mean[d] * stats[head, 1, d] * scale * scale,
                )
            T.reduce_sum(moments, threshold_parts, dim=1)
            T.copy(threshold_parts, threshold_shared)

            for group in T.serial(T.ceildiv(num_blocks, 32)):
                T.copy(
                    pooled_key[head, group * 32 : (group + 1) * 32, :],
                    key_tile,
                )
                T.clear(scores)
                T.gemm(
                    query_tile,
                    key_tile,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.reduce_sum(scores, route, dim=0)
                for block_in_group in T.Parallel(32):
                    block = group * 32 + block_in_group
                    selected[query_block, head, block] = T.if_then_else(
                        block < num_blocks
                        and (
                            block < num_protected_blocks
                            or route[block_in_group]
                            * scale
                            / T.min(64, num_tokens - query_block * 64)
                            > threshold_shared[0]
                            + tau * T.sqrt(T.max(threshold_shared[1], 0) + 1e-6)
                            or (query_block >= block - 1 and query_block <= block + 1)
                        ),
                        1,
                        0,
                    )

    return main.with_attr(
        "global_symbol",
        f"sol_nvfp4_select_{heads}_{threads}_"
        + hashlib.sha256(str(tau).encode()).hexdigest()[:8],
    )


class SolNvfp4SelectKernel(
    Kernel[
        SolNvfp4SelectArguments,
        SolNvfp4SelectWorkload,
        SolNvfp4SelectConfig,
    ]
):
    name = "sol_nvfp4_select"
    program = sol_nvfp4_select

    @classmethod
    def make_arguments(
        cls, workload: SolNvfp4SelectWorkload
    ) -> SolNvfp4SelectArguments:
        num_blocks = -(-workload.num_tokens // 64)
        return SolNvfp4SelectArguments(
            TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.heads, 128)),
            TensorDesc.empty(DType.BF16, (workload.heads, num_blocks, 128)),
            TensorDesc.empty(DType.F32, (workload.heads, 2, 128)),
            TensorDesc.empty(DType.U8, (num_blocks, workload.heads, num_blocks)),
            workload.tau,
            num_blocks,
        )

    @classmethod
    def make_config(cls, workload: SolNvfp4SelectWorkload) -> SolNvfp4SelectConfig:
        del workload
        return SolNvfp4SelectConfig(threads=256)

    @classmethod
    def make_workload(
        cls, arguments: SolNvfp4SelectArguments
    ) -> SolNvfp4SelectWorkload:
        num_tokens, heads, dim = arguments.query.shape
        num_blocks = -(-num_tokens // 64)
        assert arguments.query.dtype == DType.BF16 and dim == 128
        assert arguments.pooled_key.dtype == DType.BF16
        assert arguments.pooled_key.shape == (heads, num_blocks, 128)
        assert arguments.stats.dtype == DType.F32
        assert arguments.stats.shape == (heads, 2, 128)
        assert arguments.output.dtype == DType.U8
        assert arguments.output.shape == (num_blocks, heads, num_blocks)
        assert 0 <= arguments.num_protected_blocks <= num_blocks
        return SolNvfp4SelectWorkload(
            num_tokens=num_tokens, heads=heads, tau=arguments.tau
        )

    @classmethod
    def tops(cls, arguments: SolNvfp4SelectArguments) -> Tops:
        workload = cls.make_workload(arguments)
        num_blocks = -(-workload.num_tokens // 64)
        operations = 2 * workload.heads * 128 * workload.num_tokens * num_blocks
        return {MmaType.BF16BF16F32: operations}

    @classmethod
    def ref_program(cls, arguments: SolNvfp4SelectArguments) -> None:
        import torch

        workload = cls.make_workload(arguments)
        num_blocks = -(-workload.num_tokens // 64)
        padded = torch.nn.functional.pad(
            arguments.query.as_torch().float(),
            (0, 0, 0, 0, 0, num_blocks * 64 - workload.num_tokens),
        )
        query_mean = padded.view(num_blocks, 64, workload.heads, 128).sum(1)
        lengths = torch.full((num_blocks,), 64, device="cuda", dtype=torch.float32)
        lengths[-1] = workload.num_tokens - (num_blocks - 1) * 64
        query_mean /= lengths[:, None, None]
        centroids = arguments.pooled_key.as_torch().float()
        stats = arguments.stats.as_torch()
        route = torch.einsum("qhd,hkd->qhk", query_mean, centroids) / math.sqrt(128)
        mean = (query_mean * stats[:, 0]).sum(-1) / math.sqrt(128)
        variance = (query_mean.square() * stats[:, 1]).sum(-1) / 128
        deviation = (variance.clamp_min(0) + 1e-6).sqrt()
        mask = route > (mean + workload.tau * deviation)[:, :, None]
        mask[:, :, : arguments.num_protected_blocks] = True
        indices = torch.arange(num_blocks, device="cuda")
        mask |= (indices[:, None, None] - indices[None, None, :]).abs() <= 1
        arguments.output.as_torch().copy_(mask.to(torch.uint8))
