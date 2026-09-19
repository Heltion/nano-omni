"""Reduce a contiguous BF16 matrix to one FP32 dequantization scale."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class Fp8ScaleWorkload(Workload):
    num_tokens: int
    columns: int
    use_pre_scale: bool = False


class Fp8ScaleConfig(Config):
    tile_elements: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class Fp8ScaleArguments(Arguments):
    """BF16 input [rows, columns], optional BF16 column factors, F32 output [1].

    For finite values and a positive divisor, the output is
    max(abs(input * pre_scale)) / divisor over the whole matrix; omitted
    pre_scale uses the input directly. The reduction fragments are FP32.
    An all-zero input produces scale zero. Consumers
    that divide by this scale must handle their own nonzero-scale requirement.
    """

    input: TensorDesc
    pre_scale: TensorDesc | None
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def fp8_scale(
    num_tokens, columns, use_pre_scale=False, tile_elements=1024, threads=4
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    num_elements = dynamic_num_tokens * columns

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor([dynamic_num_tokens, columns], T.bfloat16),
        pre_scale: T.Tensor([columns], T.bfloat16),
        output: T.Tensor([1], T.float32),
    ):
        # A separate launch completes the reset before any reduction block starts.
        with T.Kernel(1, threads=32):
            for index in T.Parallel(1):
                output[index] = 0.0
        with T.Kernel(T.ceildiv(num_elements, tile_elements), threads=threads) as block:
            values = T.alloc_fragment([tile_elements], T.float32)
            maximum = T.alloc_fragment([1], T.float32)
            for index in T.Parallel(tile_elements):
                offset = block * tile_elements + index
                if use_pre_scale:
                    values[index] = T.if_then_else(
                        offset < num_elements,
                        input[offset // columns, offset % columns]
                        * pre_scale[offset % columns],
                        0.0,
                    )
                else:
                    values[index] = T.if_then_else(
                        offset < num_elements,
                        input[offset // columns, offset % columns],
                        0.0,
                    )
            T.reduce_absmax(values, maximum)
            divisor = 2688.0 if use_pre_scale else 448.0
            T.atomic_max(output[0], maximum[0] / divisor)

    return main.with_attr(
        "global_symbol",
        f"fp8_scale_{columns}_{int(use_pre_scale)}_{tile_elements}_{threads}",
    )


class Fp8ScaleKernel(Kernel[Fp8ScaleArguments, Fp8ScaleWorkload, Fp8ScaleConfig]):
    name = "fp8_scale"
    program = fp8_scale

    @classmethod
    def make_arguments(cls, workload: Fp8ScaleWorkload) -> Fp8ScaleArguments:
        """Describe BF16 input, optional column factors, and F32 scale output."""
        return Fp8ScaleArguments(
            input=TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.columns)
            ),
            pre_scale=(
                TensorDesc.empty(DType.BF16, (workload.columns,))
                if workload.use_pre_scale
                else None
            ),
            output=TensorDesc.empty(DType.F32, (1,)),
        )

    @classmethod
    def make_workload(cls, arguments: Fp8ScaleArguments) -> Fp8ScaleWorkload:
        assert arguments.input.dtype == DType.BF16
        assert len(arguments.input.shape) == 2
        assert arguments.output.dtype == DType.F32
        assert arguments.output.shape == (1,)
        if arguments.pre_scale is not None:
            assert arguments.pre_scale.dtype == DType.BF16
            assert arguments.pre_scale.shape == (arguments.input.shape[1],)
        return Fp8ScaleWorkload(
            num_tokens=arguments.input.shape[0],
            columns=arguments.input.shape[1],
            use_pre_scale=arguments.pre_scale is not None,
        )

    @classmethod
    def ref_program(cls, arguments: Fp8ScaleArguments) -> None:
        """Reduce the optional pre-scaled input to its FP8 dequantization scale."""
        value = arguments.input.as_torch()
        divisor = 448.0
        if arguments.pre_scale is not None:
            value = value * arguments.pre_scale.as_torch()
            divisor = 2688.0
        arguments.output.as_torch()[0] = value.abs().max().float() / divisor

    @classmethod
    def make_config(cls, workload: Fp8ScaleWorkload) -> Fp8ScaleConfig:
        """Select the measured launch width for this workload."""
        threads = 4 if workload.columns == 5120 and not workload.use_pre_scale else 8
        return Fp8ScaleConfig(tile_elements=1024, threads=threads)
