"""Key-centroid statistics for Sol attention."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.sol.config import SolPrepareConfig


class SolStatsWorkload(Workload):
    num_blocks: int = Field(gt=0)
    heads: int = Field(gt=0)


@dataclasses.dataclass(frozen=True, slots=True)
class SolStatsArguments(Arguments):
    pooled: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_blocks", self.pooled.shape[2]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def sol_stats(num_blocks, heads, threads=128):
    import tilelang.language as T

    @T.prim_func
    def sol_stats(
        num_blocks: num_blocks,
        pooled: T.Tensor((2, heads, num_blocks, 128), "bfloat16"),
        stats: T.Tensor((heads, 2, 128), "float32"),
    ):
        with T.Kernel(heads, threads=threads) as head:
            values = T.alloc_fragment((32, 128), "float32")
            reduced = T.alloc_fragment((128,), "float32")
            total = T.alloc_fragment((128,), "float32")
            squared = T.alloc_fragment((128,), "float32")
            T.clear(total)
            T.clear(squared)
            for group in T.serial(T.ceildiv(num_blocks, 32)):
                for i, d in T.Parallel(32, 128):
                    values[i, d] = T.if_then_else(
                        group * 32 + i < num_blocks,
                        pooled[0, head, group * 32 + i, d],
                        0,
                    )
                T.reduce_sum(values, reduced, dim=0)
                for d in T.Parallel(128):
                    total[d] += reduced[d]
                for i, d in T.Parallel(32, 128):
                    values[i, d] *= values[i, d]
                T.reduce_sum(values, reduced, dim=0)
                for d in T.Parallel(128):
                    squared[d] += reduced[d]
            for d in T.Parallel(128):
                stats[head, 0, d] = total[d] / num_blocks
                stats[head, 1, d] = T.max(
                    squared[d] / num_blocks
                    - (total[d] / num_blocks) * (total[d] / num_blocks),
                    0,
                )

    return sol_stats.with_attr("global_symbol", f"sol_stats_{heads}_{threads}")


class SolStatsKernel(Kernel[SolStatsArguments, SolStatsWorkload, SolPrepareConfig]):
    name = "sol_stats"
    program = sol_stats

    @classmethod
    def make_arguments(cls, workload: SolStatsWorkload) -> SolStatsArguments:
        return SolStatsArguments(
            TensorDesc.empty(DType.BF16, (2, workload.heads, workload.num_blocks, 128)),
            TensorDesc.empty(DType.F32, (workload.heads, 2, 128)),
        )

    @classmethod
    def make_config(cls, workload: SolStatsWorkload) -> SolPrepareConfig:
        del workload
        return SolPrepareConfig()

    @classmethod
    def make_workload(cls, arguments: SolStatsArguments) -> SolStatsWorkload:
        assert arguments.pooled.dtype == DType.BF16
        assert len(arguments.pooled.shape) == 4 and arguments.pooled.shape[0] == 2
        _, heads, num_blocks, dim = arguments.pooled.shape
        assert dim == 128
        assert arguments.output.dtype == DType.F32
        assert arguments.output.shape == (heads, 2, 128)
        return SolStatsWorkload(num_blocks=num_blocks, heads=heads)

    @classmethod
    def ref_program(cls, arguments: SolStatsArguments) -> None:
        key_mean = arguments.pooled.as_torch()[0].float()
        mean = key_mean.mean(1)
        variance = (key_mean.square().mean(1) - mean.square()).clamp_min(0)
        arguments.output.as_torch().copy_(
            __import__("torch").stack((mean, variance), dim=1)
        )
