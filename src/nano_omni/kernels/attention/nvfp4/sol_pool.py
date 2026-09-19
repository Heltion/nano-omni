"""Block pooling for Sol NVFP4 attention."""

import dataclasses
from typing import Literal

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class SolNvfp4PoolWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)


class SolNvfp4PoolConfig(Config):
    threads: Literal[64, 128, 256] = 128


@dataclasses.dataclass(frozen=True, slots=True)
class SolNvfp4PoolArguments(Arguments):
    key: TensorDesc
    value: TensorDesc
    key_mean: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.key.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def sol_nvfp4_pool(num_tokens, heads, threads=128):
    import tilelang.language as T

    num_blocks = (num_tokens + 63) // 64

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        key: T.Tensor((num_tokens, heads, 128), T.bfloat16),
        value: T.Tensor((num_tokens, heads, 128), T.bfloat16),
        key_mean: T.Tensor((heads, 128), T.float32),
        pooled: T.Tensor((3, heads, num_blocks, 128), T.bfloat16),
    ):
        with T.Kernel(num_blocks, heads, threads=threads) as (block, head):
            values = T.alloc_fragment((64, 128), T.float32)
            reduced = T.alloc_fragment((128,), T.float32)
            for i, d in T.Parallel(64, 128):
                values[i, d] = T.if_then_else(
                    block * 64 + i < num_tokens, key[block * 64 + i, head, d], 0
                )
            T.reduce_sum(values, reduced, dim=0)
            for d in T.Parallel(128):
                block_mean = reduced[d] / T.min(64, num_tokens - block * 64)
                pooled[0, head, block, d] = block_mean
                pooled[1, head, block, d] = block_mean - key_mean[head, d]
            for i, d in T.Parallel(64, 128):
                values[i, d] = T.if_then_else(
                    block * 64 + i < num_tokens, value[block * 64 + i, head, d], 0
                )
            T.reduce_sum(values, reduced, dim=0)
            for d in T.Parallel(128):
                pooled[2, head, block, d] = reduced[d]

    return main.with_attr("global_symbol", f"sol_nvfp4_pool_{heads}_{threads}")


class SolNvfp4PoolKernel(
    Kernel[SolNvfp4PoolArguments, SolNvfp4PoolWorkload, SolNvfp4PoolConfig]
):
    name = "sol_nvfp4_pool"
    program = sol_nvfp4_pool

    @classmethod
    def make_arguments(cls, workload: SolNvfp4PoolWorkload) -> SolNvfp4PoolArguments:
        shape = (workload.num_tokens, workload.heads, 128)
        num_blocks = -(-workload.num_tokens // 64)
        return SolNvfp4PoolArguments(
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(DType.F32, (workload.heads, 128)),
            TensorDesc.empty(DType.BF16, (3, workload.heads, num_blocks, 128)),
        )

    @classmethod
    def make_config(cls, workload: SolNvfp4PoolWorkload) -> SolNvfp4PoolConfig:
        del workload
        return SolNvfp4PoolConfig()

    @classmethod
    def make_workload(cls, arguments: SolNvfp4PoolArguments) -> SolNvfp4PoolWorkload:
        num_tokens, heads, dim = arguments.key.shape
        num_blocks = -(-num_tokens // 64)
        assert arguments.key.dtype == arguments.value.dtype == DType.BF16
        assert arguments.key.shape == arguments.value.shape and dim == 128
        assert arguments.key_mean.dtype == DType.F32
        assert arguments.key_mean.shape == (heads, 128)
        assert arguments.output.dtype == DType.BF16
        assert arguments.output.shape == (3, heads, num_blocks, 128)
        return SolNvfp4PoolWorkload(num_tokens=num_tokens, heads=heads)

    @classmethod
    def ref_program(cls, arguments: SolNvfp4PoolArguments) -> None:
        import torch

        workload = cls.make_workload(arguments)
        num_blocks = -(-workload.num_tokens // 64)
        lengths = torch.full((num_blocks,), 64, device="cuda", dtype=torch.float32)
        lengths[-1] = workload.num_tokens - (num_blocks - 1) * 64

        def sums(source):
            padded = torch.nn.functional.pad(
                source.float(), (0, 0, 0, 0, 0, num_blocks * 64 - workload.num_tokens)
            )
            return padded.view(num_blocks, 64, workload.heads, 128).sum(1)

        key_mean = (sums(arguments.key.as_torch()) / lengths[:, None, None]).permute(
            1, 0, 2
        )
        output = arguments.output.as_torch()
        output[0].copy_(key_mean)
        output[1].copy_(key_mean - arguments.key_mean.as_torch()[:, None, :])
        output[2].copy_(sums(arguments.value.as_torch()).permute(1, 0, 2))
