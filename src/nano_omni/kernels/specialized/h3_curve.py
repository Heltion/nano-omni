"""Interpolate one row from a sampled H3 conditioning curve."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class H3CurveWorkload(Workload):
    """At least two sampled rows define linear interpolation for each column."""

    grid: int = Field(ge=2)
    columns: int = Field(gt=0)


class H3CurveConfig(Config):
    tile_columns: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class H3CurveArguments(Arguments):
    """One model-selected timestep and output level for a sampled FP32 curve."""

    table: TensorDesc
    output: TensorDesc
    timesteps: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def h3_curve(
    grid,
    columns,
    tile_columns=8,
    threads=128,
):
    import tilelang.language as T

    @T.prim_func
    def main(
        table: T.Tensor([grid, columns], T.float32),
        output: T.Tensor([1, columns], T.float32),
        timesteps: T.Tensor([1], T.float32),
    ):
        with T.Kernel(T.ceildiv(columns, tile_columns), threads=threads) as column_block:
            timestep = timesteps[0]
            position = T.max(
                0.0,
                T.min(1.0, timestep),
            ) * (grid - 1)
            lower = T.min(
                T.cast(T.floor(position), T.int32),
                grid - 2,
            )
            fraction = position - lower
            for local in T.Parallel(tile_columns):
                column = column_block * tile_columns + local
                if column < columns:
                    first = table[lower, column]
                    output[0, column] = first + fraction * (
                        table[lower + 1, column] - first
                    )

    return main


class H3CurveKernel(Kernel[H3CurveArguments, H3CurveWorkload, H3CurveConfig]):
    name = "h3_curve"
    program = h3_curve

    @classmethod
    def make_arguments(cls, workload: H3CurveWorkload) -> H3CurveArguments:
        """Describe the sampled curve, interpolated row, and scalar timestep."""
        return H3CurveArguments(
            table=TensorDesc.empty(DType.F32, (workload.grid, workload.columns)),
            output=TensorDesc.empty(DType.F32, (1, workload.columns)),
            timesteps=TensorDesc.empty(DType.F32, (1,)),
        )

    @classmethod
    def make_workload(cls, arguments: H3CurveArguments) -> H3CurveWorkload:
        assert all(tensor.dtype == DType.F32 for tensor in arguments.values())
        assert len(arguments.table.shape) == 2, (
            "Curve samples must form a grid-by-column matrix"
        )
        workload = H3CurveWorkload(
            grid=arguments.table.shape[0],
            columns=arguments.table.shape[1],
        )
        assert arguments.output.shape == (1, workload.columns)
        assert arguments.timesteps.shape == (1,)
        return workload

    @classmethod
    def ref_program(cls, arguments: H3CurveArguments) -> None:
        """Linearly interpolate the two curve rows surrounding the timestep."""
        import torch

        table = arguments.table.as_torch()
        timestep = arguments.timesteps.as_torch().clamp(0, 1)
        position = timestep * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        arguments.output.as_torch().copy_(
            torch.lerp(table[lower], table[lower + 1], (position - lower)[:, None])
        )

    @classmethod
    def make_config(cls, workload: H3CurveWorkload) -> H3CurveConfig:
        del workload
        return H3CurveConfig(threads=128, tile_columns=8)
