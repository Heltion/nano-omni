"""Generate F32 rotary angles for 128-wide Q/K heads."""

import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class QwenRopeWorkload(Workload):
    num_tokens: int
    theta: float = 5_000_000.0


class QwenRopeConfig(Config):
    tile_rows: int
    tile_columns: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class QwenRopeArguments(Arguments):
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.output.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def qwen_rope(num_tokens, theta, tile_rows=1, tile_columns=32, threads=128):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    theta_log = math.log(theta)
    columns = 64

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        output: T.Tensor([dynamic_num_tokens, columns], T.float32),
    ):
        with T.Kernel(
            T.ceildiv(dynamic_num_tokens, tile_rows),
            T.ceildiv(columns, tile_columns),
            threads=threads,
        ) as (row_block, column_block):
            for row, column in T.Parallel(tile_rows, tile_columns):
                position = row_block * tile_rows + row
                dimension = column_block * tile_columns + column
                if position < dynamic_num_tokens and dimension < columns:
                    frequency = T.exp(-theta_log * dimension * 2.0 / 128.0)
                    output[position, dimension] = position * frequency
    return main.with_attr(
        "global_symbol",
        f"qwen_rope_{tile_rows}_{tile_columns}_{threads}",
    )


class QwenRopeKernel(Kernel[QwenRopeArguments, QwenRopeWorkload, QwenRopeConfig]):
    name = "qwen_rope"
    program = qwen_rope

    @classmethod
    def make_arguments(cls, workload: QwenRopeWorkload) -> QwenRopeArguments:
        """Describe the generated half-head rotary angle table."""
        assert workload.theta == 5_000_000.0
        return QwenRopeArguments(TensorDesc.empty(DType.F32, (workload.num_tokens, 64)))

    @classmethod
    def make_workload(cls, arguments: QwenRopeArguments) -> QwenRopeWorkload:
        assert arguments.output.dtype == DType.F32
        assert len(arguments.output.shape) == 2 and arguments.output.shape[1] == 64
        return QwenRopeWorkload(num_tokens=arguments.output.shape[0])

    @classmethod
    def make_config(cls, workload: QwenRopeWorkload) -> QwenRopeConfig:
        del workload
        return QwenRopeConfig(threads=128, tile_columns=32, tile_rows=16)

    @classmethod
    def ref_program(cls, arguments: QwenRopeArguments) -> None:
        """Generate row-position angles for the first half of a 128-wide head."""
        import torch

        workload = cls.make_workload(arguments)
        output = arguments.output.as_torch()
        positions = torch.arange(workload.num_tokens, device=output.device).float()
        dimensions = torch.arange(64, device=output.device).float()
        frequencies = torch.exp(-math.log(workload.theta) * dimensions * 2 / 128)
        output.copy_(positions[:, None] * frequencies[None, :])
