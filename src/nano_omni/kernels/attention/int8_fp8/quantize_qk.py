"""Grouped INT8 Q/K quantization for INT8/FP8 attention."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.int8_fp8.value_scale import Int8Fp8ThreadConfig


@tilelang.jit(
    out_idx=[],
    execution_backend="nvrtc",
    pass_configs={"tl.disable_vectorize_256": True},
)
def int8_fp8_quantize_qk(num_tokens, heads, tokens_per_block, center, threads=128):
    import tilelang.language as T

    if tokens_per_block not in (32, 64):
        raise ValueError(
            "dense INT8/FP8 Q/K num_blocks must contain 32 or 64 num_tokens"
        )
    num_blocks = -(-num_tokens // tokens_per_block)
    num_padded_tokens = num_blocks * tokens_per_block

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        source: T.Tensor((num_tokens, heads * 128), T.bfloat16),
        mean: T.Tensor((1, heads * 128), T.float32),
        quantized: T.Tensor((heads, num_padded_tokens, 128), T.int8),
        scales: T.Tensor((heads, num_blocks), T.float32),
    ):
        with T.Kernel(num_blocks, heads, threads=threads) as (group, head):
            values = T.alloc_fragment((tokens_per_block, 128), T.float32)
            absolute = T.alloc_fragment((tokens_per_block, 128), T.float32)
            row_max = T.alloc_fragment((tokens_per_block,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            for i, j in T.Parallel(tokens_per_block, 128):
                row = group * tokens_per_block + i
                if center:
                    values[i, j] = T.if_then_else(
                        row < num_tokens,
                        T.cast(source[row, head * 128 + j], T.float32)
                        - mean[0, head * 128 + j],
                        0,
                    )
                else:
                    values[i, j] = T.if_then_else(
                        row < num_tokens, source[row, head * 128 + j], 0
                    )
                absolute[i, j] = T.abs(values[i, j])
            T.reduce_max(absolute, row_max, dim=1)
            T.reduce_max(row_max, maximum, dim=0)
            for i, j in T.Parallel(tokens_per_block, 128):
                quantized[head, group * tokens_per_block + i, j] = T.cast(
                    T.round(values[i, j] / T.max(maximum[0] / 127, 1e-10)),
                    T.int8,
                )
            for i in T.Parallel(1):
                scales[head, group] = T.max(maximum[0] / 127, 1e-10)

    return main.with_attr(
        "global_symbol",
        f"dense_int8_fp8_qk_{heads}_{tokens_per_block}_{int(center)}_{threads}",
    )


class Int8Fp8QkWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    tokens_per_block: int = Field(default=32, gt=0)
    center: bool = False

    @property
    def num_padded_tokens(self) -> int:
        return -(-self.num_tokens // self.tokens_per_block) * self.tokens_per_block

    @property
    def packed_bytes(self) -> int:
        return self.heads * self.num_padded_tokens * 128

    @property
    def storage_bytes(self) -> int:
        return (
            self.packed_bytes
            + self.heads * (self.num_padded_tokens // self.tokens_per_block) * 4
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Int8Fp8QuantizeQKArguments(Arguments):
    source: TensorDesc
    statistics: TensorDesc
    quantized: TensorDesc
    scales: TensorDesc
    heads: int
    tokens_per_block: int = 32
    center: bool = False

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.source.shape[0]),)


class Int8Fp8QuantizeQKKernel(
    Kernel[Int8Fp8QuantizeQKArguments, Int8Fp8QkWorkload, Int8Fp8ThreadConfig]
):
    name = "int8_fp8_quantize_qk"
    program = int8_fp8_quantize_qk

    @classmethod
    def make_arguments(cls, workload: Int8Fp8QkWorkload) -> Int8Fp8QuantizeQKArguments:
        """Describe grouped INT8 Q/K quantization inputs and outputs."""
        return Int8Fp8QuantizeQKArguments(
            source=TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.heads * 128)
            ),
            statistics=TensorDesc.empty(DType.F32, (1, workload.heads * 128)),
            quantized=TensorDesc.empty(
                DType.I8, (workload.heads, workload.num_padded_tokens, 128)
            ),
            scales=TensorDesc.empty(
                DType.F32,
                (
                    workload.heads,
                    workload.num_padded_tokens // workload.tokens_per_block,
                ),
            ),
            heads=workload.heads,
            tokens_per_block=workload.tokens_per_block,
            center=workload.center,
        )

    @classmethod
    def make_config(cls, workload: Int8Fp8QkWorkload) -> Int8Fp8ThreadConfig:
        """Use 128 threads for the smallest and largest Q/K grids."""
        if workload.num_tokens < 1024 or workload.num_tokens >= 65536:
            return Int8Fp8ThreadConfig(threads=128)
        return Int8Fp8ThreadConfig(threads=256)

    @classmethod
    def make_workload(cls, arguments: Int8Fp8QuantizeQKArguments) -> Int8Fp8QkWorkload:
        """Recover the workload and validate grouped Q/K storage."""
        assert len(arguments.source.shape) == 2
        num_tokens, columns = arguments.source.shape
        assert columns == arguments.heads * 128
        assert arguments.source.dtype == DType.BF16
        assert arguments.statistics.dtype == DType.F32
        assert arguments.statistics.shape == (1, columns)
        num_padded_tokens = (
            -(-num_tokens // arguments.tokens_per_block) * arguments.tokens_per_block
        )
        assert arguments.quantized.dtype == DType.I8
        assert arguments.quantized.shape == (arguments.heads, num_padded_tokens, 128)
        assert arguments.scales.dtype == DType.F32
        assert arguments.scales.shape == (
            arguments.heads,
            num_padded_tokens // arguments.tokens_per_block,
        )
        return Int8Fp8QkWorkload(
            num_tokens=num_tokens,
            heads=arguments.heads,
            tokens_per_block=arguments.tokens_per_block,
            center=arguments.center,
        )

    @classmethod
    def ref_program(cls, arguments: Int8Fp8QuantizeQKArguments) -> None:
        """Center optional Q/K num_blocks, quantize them, and store group scales."""
        import torch

        workload = cls.make_workload(arguments)
        source = arguments.source.as_torch()
        statistics = arguments.statistics.as_torch()
        quantized = arguments.quantized.as_torch()
        scales = arguments.scales.as_torch()
        values = source.float().reshape(workload.num_tokens, workload.heads, 128)
        if workload.center:
            values -= statistics.reshape(1, workload.heads, 128)
        if workload.num_padded_tokens != workload.num_tokens:
            values = torch.nn.functional.pad(
                values,
                (0, 0, 0, 0, 0, workload.num_padded_tokens - workload.num_tokens),
            )
        num_blocks = workload.num_padded_tokens // workload.tokens_per_block
        grouped = values.reshape(
            num_blocks, workload.tokens_per_block, workload.heads, 128
        )
        scale = (grouped.abs().amax((1, 3)) / 127).clamp_min(1e-10)
        quantized.copy_(
            (grouped / scale[:, None, :, None])
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
            .permute(2, 0, 1, 3)
            .reshape(workload.heads, workload.num_padded_tokens, 128)
        )
        scales.copy_(scale.T)
