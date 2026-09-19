"""Symmetric per-row INT8 quantization with FP32 scales."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class Int8RowwiseWorkload(Workload):
    """Dimensions of the input and quantized output matrices."""

    num_tokens: int
    columns: int


class Int8RowwiseConfig(Config):
    """CUDA threads assigned to each row."""

    threads: int = 256


@dataclasses.dataclass(frozen=True, slots=True)
class Int8RowwiseArguments(Arguments):
    """BF16 input, INT8 output, and one FP32 scale per row."""

    input: TensorDesc
    output: TensorDesc
    scales: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def int8_rowwise(num_tokens, columns, threads=256):
    """Build row-wise symmetric INT8 quantization."""
    import tilelang.language as T

    width = 1 << (columns - 1).bit_length()

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        input: T.Tensor((num_tokens, columns), T.bfloat16),
        output: T.Tensor((num_tokens, columns), T.int8),
        scales: T.Tensor((num_tokens,), T.float32),
    ):
        with T.Kernel(num_tokens, threads=threads) as row:
            values = T.alloc_fragment((width,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            for column in T.Parallel(width):
                values[column] = T.if_then_else(
                    column < columns, input[row, column], 0.0
                )
            T.reduce_absmax(values, maximum)
            scale = T.max(maximum[0] / 127.0, 1e-30)
            scales[row] = scale
            for column in T.Parallel(width):
                if column < columns:
                    output[row, column] = T.max(
                        -127.0,
                        T.min(127.0, T.nearbyint(values[column] / scale)),
                    ).astype(T.int8)

    return main


class Int8RowwiseKernel(
    Kernel[Int8RowwiseArguments, Int8RowwiseWorkload, Int8RowwiseConfig]
):
    """Quantize each row independently and expose its dequantization scale."""

    name = "int8_rowwise"
    program = int8_rowwise

    @classmethod
    def make_arguments(cls, workload: Int8RowwiseWorkload) -> Int8RowwiseArguments:
        """Describe distinct source, quantized output, and scale tensors."""
        shape = (workload.num_tokens, workload.columns)
        return Int8RowwiseArguments(
            input=TensorDesc.empty(DType.BF16, shape),
            output=TensorDesc.empty(DType.I8, shape),
            scales=TensorDesc.empty(DType.F32, (workload.num_tokens,)),
        )

    @classmethod
    def make_workload(cls, arguments: Int8RowwiseArguments) -> Int8RowwiseWorkload:
        """Validate the quantization contract and recover matrix dimensions."""
        assert arguments.input.dtype == DType.BF16
        assert len(arguments.input.shape) == 2
        assert arguments.output.dtype == DType.I8
        assert arguments.output.shape == arguments.input.shape
        assert arguments.scales.dtype == DType.F32
        assert arguments.scales.shape == (arguments.input.shape[0],)
        return Int8RowwiseWorkload(
            num_tokens=arguments.input.shape[0], columns=arguments.input.shape[1]
        )

    @classmethod
    def make_config(cls, workload: Int8RowwiseWorkload) -> Int8RowwiseConfig:
        del workload
        assert False, "int8_rowwise has no production configuration"

    @classmethod
    def ref_program(cls, arguments: Int8RowwiseArguments) -> None:
        """Evaluate per-row symmetric quantization directly with Torch."""
        import torch

        value = arguments.input.as_torch().float()
        scale = torch.clamp_min(value.abs().amax(dim=1) / 127.0, 1e-30)
        arguments.scales.as_torch().copy_(scale)
        arguments.output.as_torch().copy_(
            torch.round(value / scale[:, None]).clamp_(-127, 127).to(torch.int8)
        )
