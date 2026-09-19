"""Per-channel value scaling for INT8/FP8 attention."""

import dataclasses
from typing import Literal

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


@tilelang.jit(
    out_idx=[],
    execution_backend="nvrtc",
    pass_configs={"tl.disable_vectorize_256": True},
)
def int8_fp8_value_scale(num_tokens, heads, scale_max=2.25, threads=128):
    import tilelang.language as T

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        source: T.Tensor((num_tokens, heads * 128), T.bfloat16),
        scales: T.Tensor((heads, 128), T.float32),
    ):
        with T.Kernel(heads, 8, threads=threads) as (head, channel_block):
            values = T.alloc_fragment((128, 16), T.float32)
            partial = T.alloc_fragment((16,), T.float32)
            maximum = T.alloc_fragment((16,), T.float32)
            T.clear(maximum)
            for block in T.serial(T.ceildiv(num_tokens, 128)):
                for i, j in T.Parallel(128, 16):
                    row = block * 128 + i
                    values[i, j] = T.if_then_else(
                        row < num_tokens,
                        T.abs(
                            T.cast(
                                source[row, head * 128 + channel_block * 16 + j],
                                T.float32,
                            )
                        ),
                        0,
                    )
                T.reduce_max(values, partial, dim=0)
                for j in T.Parallel(16):
                    maximum[j] = T.max(maximum[j], partial[j])
            for j in T.Parallel(16):
                scales[head, channel_block * 16 + j] = T.max(
                    maximum[j] / scale_max, 1e-10
                )

    return main.with_attr(
        "global_symbol", f"dense_int8_fp8_vscale_{heads}_{scale_max}_{threads}"
    )


class Int8Fp8ThreadConfig(Config):
    threads: Literal[64, 128, 256] = 128


class Int8Fp8ValueWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)


@dataclasses.dataclass(frozen=True, slots=True)
class Int8Fp8ValueScaleArguments(Arguments):
    source: TensorDesc
    output: TensorDesc
    heads: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.source.shape[0]),)


class Int8Fp8ValueScaleKernel(
    Kernel[Int8Fp8ValueScaleArguments, Int8Fp8ValueWorkload, Int8Fp8ThreadConfig]
):
    name = "int8_fp8_value_scale"
    program = int8_fp8_value_scale

    @classmethod
    def make_arguments(
        cls, workload: Int8Fp8ValueWorkload
    ) -> Int8Fp8ValueScaleArguments:
        """Describe BF16 V rows and their per-head, per-channel scales."""
        return Int8Fp8ValueScaleArguments(
            source=TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.heads * 128)
            ),
            output=TensorDesc.empty(DType.F32, (workload.heads, 128)),
            heads=workload.heads,
        )

    @classmethod
    def make_config(cls, workload: Int8Fp8ValueWorkload) -> Int8Fp8ThreadConfig:
        """Use the measured reduction launch shape."""
        del workload
        return Int8Fp8ThreadConfig(threads=64)

    @classmethod
    def make_workload(
        cls, arguments: Int8Fp8ValueScaleArguments
    ) -> Int8Fp8ValueWorkload:
        """Recover the workload and validate source and channel scales."""
        assert len(arguments.source.shape) == 2
        num_tokens, columns = arguments.source.shape
        assert columns == arguments.heads * 128
        assert arguments.source.dtype == DType.BF16
        assert arguments.output.dtype == DType.F32
        assert arguments.output.shape == (arguments.heads, 128)
        return Int8Fp8ValueWorkload(num_tokens=num_tokens, heads=arguments.heads)

    @classmethod
    def ref_program(cls, arguments: Int8Fp8ValueScaleArguments) -> None:
        """Reduce the maximum absolute V value for every head channel."""
        source = arguments.source.as_torch()
        arguments.output.as_torch().copy_(
            (source.float().abs().amax(dim=0) / 2.25)
            .clamp_min(1e-10)
            .reshape(arguments.heads, 128)
        )
