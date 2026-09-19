"""FP8 value quantization for INT8/FP8 attention."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Kernel
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.int8_fp8.value_scale import (
    Int8Fp8ThreadConfig,
    Int8Fp8ValueWorkload,
)


@tilelang.jit(
    out_idx=[],
    execution_backend="nvrtc",
    pass_configs={"tl.disable_vectorize_256": True},
)
def int8_fp8_quantize_value(num_tokens, heads, permute=True, threads=128):
    import tilelang.language as T

    num_padded_tokens = -(-num_tokens // 64) * 64

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        source: T.Tensor((num_tokens, heads * 128), T.bfloat16),
        scales: T.Tensor((heads, 128), T.float32),
        quantized: T.Tensor((heads, 128, num_padded_tokens), T.float8_e4m3fn),
    ):
        with T.Kernel(T.ceildiv(num_tokens, 64), heads, threads=threads) as (
            block,
            head,
        ):
            transposed = T.alloc_shared((128, 64), T.float8_e4m3fn)
            T.annotate_layout(
                {
                    transposed: T.Layout(
                        (128, 64),
                        lambda j, i: j * 64 + ((i // 16) ^ (j % 4)) * 16 + i % 16,
                    )
                }
            )
            for i, j in T.Parallel(64, 128):
                row = block * 64 + i
                transposed[j, i] = T.if_then_else(
                    row < num_tokens,
                    T.cast(source[row, head * 128 + j], T.float32) / scales[head, j],
                    0,
                )
            for j, packed_row in T.Parallel(128, 64):
                source_row = (
                    packed_row // 16 * 16
                    + packed_row % 16 // 4 * 2
                    + packed_row % 2
                    + packed_row % 4 // 2 * 8
                    if permute
                    else packed_row
                )
                quantized[head, j, block * 64 + packed_row] = transposed[j, source_row]

    return main.with_attr(
        "global_symbol", f"dense_int8_fp8_v_{heads}_{int(permute)}_{threads}"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Int8Fp8QuantizeValueArguments(Arguments):
    source: TensorDesc
    scale: TensorDesc
    output: TensorDesc
    heads: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.source.shape[0]),)


class Int8Fp8QuantizeValueKernel(
    Kernel[Int8Fp8QuantizeValueArguments, Int8Fp8ValueWorkload, Int8Fp8ThreadConfig]
):
    name = "int8_fp8_quantize_value"
    program = int8_fp8_quantize_value

    @classmethod
    def make_arguments(
        cls, workload: Int8Fp8ValueWorkload
    ) -> Int8Fp8QuantizeValueArguments:
        """Describe BF16 V num_padded_tokens, channel scales, and permuted FP8 output."""
        num_padded_tokens = -(-workload.num_tokens // 64) * 64
        return Int8Fp8QuantizeValueArguments(
            source=TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.heads * 128)
            ),
            scale=TensorDesc.empty(DType.F32, (workload.heads, 128)),
            output=TensorDesc.empty(
                DType.FP8_E4M3, (workload.heads, 128, num_padded_tokens)
            ),
            heads=workload.heads,
        )

    @classmethod
    def make_config(cls, workload: Int8Fp8ValueWorkload) -> Int8Fp8ThreadConfig:
        """Use the measured FP8 quantization launch shape."""
        del workload
        return Int8Fp8ThreadConfig(threads=256)

    @classmethod
    def make_workload(
        cls, arguments: Int8Fp8QuantizeValueArguments
    ) -> Int8Fp8ValueWorkload:
        """Recover the workload and validate the permuted FP8 layout."""
        assert len(arguments.source.shape) == 2
        num_tokens, columns = arguments.source.shape
        assert columns == arguments.heads * 128
        assert arguments.source.dtype == DType.BF16
        assert arguments.scale.dtype == DType.F32
        assert arguments.scale.shape == (arguments.heads, 128)
        assert arguments.output.dtype == DType.FP8_E4M3
        assert arguments.output.shape == (
            arguments.heads,
            128,
            -(-num_tokens // 64) * 64,
        )
        return Int8Fp8ValueWorkload(num_tokens=num_tokens, heads=arguments.heads)

    @classmethod
    def ref_program(cls, arguments: Int8Fp8QuantizeValueArguments) -> None:
        """Scale, transpose, and permute V into its attention FP8 layout."""
        import torch

        source = arguments.source.as_torch()
        scales = arguments.scale.as_torch()
        output = arguments.output.as_torch()
        num_padded_tokens = output.shape[2]
        order = torch.tensor(
            [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
            device=source.device,
        )
        values = torch.zeros(
            num_padded_tokens,
            arguments.heads,
            128,
            dtype=torch.float32,
            device=source.device,
        )
        values[: source.shape[0]] = source.reshape(-1, arguments.heads, 128)
        values /= scales
        packed = values.reshape(-1, 4, 16, arguments.heads, 128)[:, :, order].reshape(
            num_padded_tokens, arguments.heads, 128
        )
        output.copy_(packed.permute(1, 2, 0))
