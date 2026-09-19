"""Quantize a contiguous BF16 matrix with one scalar scale to E4M3FN FP8."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class QuantizeFp8Workload(Workload):
    num_tokens: int
    columns: int
    dynamic_scale: bool


class QuantizeFp8Config(Config):
    tile_elements: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizeFp8Arguments(Arguments):
    """BF16 input, selected F32 scale tensor, and E4M3FN output."""

    input: TensorDesc
    dynamic_scale: TensorDesc | None
    output: TensorDesc
    static_scale: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def quantize_fp8(num_tokens, columns, dynamic_scale, tile_elements=1024, threads=256):
    import tilelang.language as T

    num_elements = num_tokens * columns

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        input: T.Tensor([num_tokens, columns], T.bfloat16),
        input_scale: T.Tensor([1], T.float32),
        output: T.Tensor([num_tokens, columns], T.float8_e4m3fn),
        static_scale: T.Tensor([1], T.float32),
    ):
        with T.Kernel(T.ceildiv(num_elements, tile_elements), threads=threads) as block:
            for index in T.Parallel(tile_elements):
                offset = block * tile_elements + index
                if offset < num_elements:
                    row = offset // columns
                    column = offset % columns
                    scale = input_scale[0] if dynamic_scale else static_scale[0]
                    inverse_scale = T.cast(1.0 / scale, T.bfloat16)
                    value = input[row, column] * inverse_scale
                    output[row, column] = T.max(-448.0, T.min(448.0, value))

    return main.with_attr(
        "global_symbol",
        f"quantize_fp8_{columns}_{int(dynamic_scale)}_{tile_elements}_{threads}",
    )


class QuantizeFp8Kernel(
    Kernel[QuantizeFp8Arguments, QuantizeFp8Workload, QuantizeFp8Config]
):
    name = "quantize_fp8"
    program = quantize_fp8

    @classmethod
    def make_arguments(cls, workload: QuantizeFp8Workload) -> QuantizeFp8Arguments:
        shape = (workload.num_tokens, workload.columns)
        return QuantizeFp8Arguments(
            input=TensorDesc.empty(DType.BF16, shape),
            dynamic_scale=(
                TensorDesc.empty(DType.F32, (1,)) if workload.dynamic_scale else None
            ),
            output=TensorDesc.empty(DType.FP8_E4M3, shape),
            static_scale=TensorDesc.empty(DType.F32, (1,)),
        )

    @classmethod
    def make_workload(cls, arguments: QuantizeFp8Arguments) -> QuantizeFp8Workload:
        assert len(arguments.input.shape) == 2
        assert arguments.output.shape == arguments.input.shape
        assert arguments.input.dtype == DType.BF16
        assert arguments.output.dtype == DType.FP8_E4M3
        assert arguments.static_scale.dtype == DType.F32
        assert arguments.static_scale.shape == (1,)
        if arguments.dynamic_scale is not None:
            assert arguments.dynamic_scale.dtype == DType.F32
            assert arguments.dynamic_scale.shape == (1,)
        return QuantizeFp8Workload(
            num_tokens=arguments.input.shape[0],
            columns=arguments.input.shape[1],
            dynamic_scale=arguments.dynamic_scale is not None,
        )

    @classmethod
    def make_config(cls, workload: QuantizeFp8Workload) -> QuantizeFp8Config:
        del workload
        assert False, "quantize_fp8 has no production configuration"

    @classmethod
    def ref_program(cls, arguments: QuantizeFp8Arguments) -> None:
        scale = (
            arguments.dynamic_scale.as_torch()
            if arguments.dynamic_scale is not None
            else arguments.static_scale.as_torch()
        )
        inverse = (1.0 / scale).to(arguments.input.as_torch().dtype)
        result = (arguments.input.as_torch() * inverse).clamp(-448, 448)
        arguments.output.as_torch().copy_(result)
