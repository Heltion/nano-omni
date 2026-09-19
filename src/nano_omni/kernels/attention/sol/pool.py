"""Block summaries for Sol attention."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.sol.config import SolPrepareConfig


class SolPoolWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)


@dataclasses.dataclass(frozen=True, slots=True)
class SolPoolArguments(Arguments):
    key: TensorDesc
    value: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.key.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def sol_pool(num_tokens, heads, threads=128):
    import tilelang.language as T

    num_blocks = (num_tokens + 63) // 64

    @T.prim_func
    def sol_pool(
        num_tokens: num_tokens,
        key: T.Tensor((num_tokens, heads, 128), "bfloat16"),
        value: T.Tensor((num_tokens, heads, 128), "bfloat16"),
        pooled: T.Tensor((2, heads, num_blocks, 128), "bfloat16"),
    ):
        with T.Kernel(num_blocks, heads, threads=threads) as (block, head):
            values = T.alloc_fragment((64, 128), "float32")
            reduced = T.alloc_fragment((128,), "float32")
            for i, d in T.Parallel(64, 128):
                values[i, d] = T.if_then_else(
                    block * 64 + i < num_tokens, key[block * 64 + i, head, d], 0
                )
            T.reduce_sum(values, reduced, dim=0)
            for d in T.Parallel(128):
                pooled[0, head, block, d] = reduced[d] / T.min(
                    64, num_tokens - block * 64
                )
            for i, d in T.Parallel(64, 128):
                values[i, d] = T.if_then_else(
                    block * 64 + i < num_tokens, value[block * 64 + i, head, d], 0
                )
            T.reduce_sum(values, reduced, dim=0)
            # The approximate numerator uses V sums; its denominator uses block lengths.
            for d in T.Parallel(128):
                pooled[1, head, block, d] = reduced[d]

    return sol_pool.with_attr("global_symbol", f"sol_pool_{heads}_{threads}")


class SolPoolKernel(Kernel[SolPoolArguments, SolPoolWorkload, SolPrepareConfig]):
    name = "sol_pool"
    program = sol_pool

    @classmethod
    def make_arguments(cls, workload: SolPoolWorkload) -> SolPoolArguments:
        shape = (workload.num_tokens, workload.heads, 128)
        return SolPoolArguments(
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(
                DType.BF16, (2, workload.heads, -(-workload.num_tokens // 64), 128)
            ),
        )

    @classmethod
    def make_config(cls, workload: SolPoolWorkload) -> SolPrepareConfig:
        del workload
        return SolPrepareConfig()

    @classmethod
    def make_workload(cls, arguments: SolPoolArguments) -> SolPoolWorkload:
        assert arguments.key.dtype == arguments.value.dtype == DType.BF16
        assert arguments.key.shape == arguments.value.shape
        num_tokens, heads, dim = arguments.key.shape
        assert dim == 128
        assert arguments.output.dtype == DType.BF16
        assert arguments.output.shape == (2, heads, -(-num_tokens // 64), 128)
        return SolPoolWorkload(num_tokens=num_tokens, heads=heads)

    @classmethod
    def ref_program(cls, arguments: SolPoolArguments) -> None:
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

        key_sum = sums(arguments.key.as_torch())
        value_sum = sums(arguments.value.as_torch())
        output = arguments.output.as_torch()
        output[0].copy_((key_sum / lengths[:, None, None]).permute(1, 0, 2))
        output[1].copy_(value_sum.permute(1, 0, 2))
