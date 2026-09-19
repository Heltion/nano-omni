"""Per-head NVFP4 preparation for attention tensor-core operands."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class Nvfp4PrepareWorkload(Workload):
    num_tokens: int
    heads: int
    center: bool = False
    transpose: bool = False


class Nvfp4PrepareConfig(Config):
    tile_tokens: int = 16
    threads: int = 128


@dataclasses.dataclass(frozen=True, slots=True)
class Nvfp4PrepareArguments(Arguments):
    source: TensorDesc
    mean: TensorDesc | None
    quantized: TensorDesc
    scales: TensorDesc
    heads: int
    transpose: bool = False

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.source.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def nvfp4_prepare(num_tokens, heads, center, transpose, tile_tokens=16, threads=128):
    import tilelang.language as T

    num_padded_tokens = (num_tokens + 127) // 128 * 128
    num_rows = 128 if transpose else num_padded_tokens
    columns = num_padded_tokens if transpose else 128
    num_padded_scale_rows = (num_rows + 127) // 128 * 128

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        source: T.Tensor((num_tokens, heads * 128), T.bfloat16),
        mean: T.Tensor((heads, 128), T.float32),
        packed_output: T.Tensor((heads, num_rows, columns // 2), T.uint8),
        scales: T.Tensor(
            (heads, num_padded_scale_rows, columns // 16), T.float8_e4m3fn
        ),
    ):
        packed = T.view(packed_output, (heads, num_rows, columns), T.float4_e2m1fn)
        with T.Kernel(
            T.ceildiv(num_padded_scale_rows, tile_tokens),
            columns // 64,
            heads,
            threads=threads,
        ) as (row_block, column_block, head):
            values = T.alloc_fragment((tile_tokens, 4, 16), T.float32)
            maximum = T.alloc_fragment((tile_tokens, 4), T.float32)
            inverse = T.alloc_fragment((tile_tokens, 4), T.float32)
            for i, group, j in T.Parallel(tile_tokens, 4, 16):
                row = row_block * tile_tokens + i
                column = column_block * 64 + group * 16 + j
                token = column if transpose else row
                channel = row if transpose else column
                value = T.if_then_else(
                    token < num_tokens and channel < 128,
                    source[token, head * 128 + channel],
                    0.0,
                )
                if center:
                    values[i, group, j] = T.if_then_else(
                        token < num_tokens and channel < 128,
                        value - mean[head, channel],
                        0.0,
                    )
                else:
                    values[i, group, j] = value
            T.reduce_absmax(values, maximum, dim=2)
            for i, group in T.Parallel(tile_tokens, 4):
                maximum[i, group] = T.cast(
                    T.cast(T.min(maximum[i, group] / 6.0, 448.0), T.float8_e4m3fn),
                    T.float32,
                )
                inverse[i, group] = T.if_then_else(
                    maximum[i, group] == 0.0, 0.0, T.fast_rcp(maximum[i, group])
                )
                row = row_block * tile_tokens + i
                index = (
                    ((row // 128) * (columns // 64) + column_block) * 128
                    + (row % 32) * 4
                    + (row % 128) // 32
                ) * 4 + group
                if row < num_padded_scale_rows:
                    scales[
                        head,
                        index // (columns // 16),
                        index % (columns // 16),
                    ] = maximum[i, group]
            for i, group, j in T.Parallel(tile_tokens, 4, 16):
                row = row_block * tile_tokens + i
                column = column_block * 64 + group * 16 + j
                if row < num_rows:
                    packed[head, row, column] = values[i, group, j] * inverse[i, group]

    return main.with_attr(
        "global_symbol",
        f"nvfp4_prepare_{heads}_{int(center)}_{int(transpose)}_{tile_tokens}_{threads}",
    )


class Nvfp4PrepareKernel(
    Kernel[Nvfp4PrepareArguments, Nvfp4PrepareWorkload, Nvfp4PrepareConfig]
):
    name = "nvfp4_prepare"
    program = nvfp4_prepare

    @classmethod
    def make_arguments(cls, workload: Nvfp4PrepareWorkload) -> Nvfp4PrepareArguments:
        padded = -(-workload.num_tokens // 128) * 128
        num_rows = 128 if workload.transpose else padded
        columns = padded if workload.transpose else 128
        return Nvfp4PrepareArguments(
            TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.heads * 128)),
            (
                TensorDesc.empty(DType.F32, (workload.heads, 128))
                if workload.center
                else None
            ),
            TensorDesc.empty(DType.U8, (workload.heads, num_rows, columns // 2)),
            TensorDesc.empty(
                DType.FP8_UE4M3, (workload.heads, num_rows, columns // 16)
            ),
            workload.heads,
            workload.transpose,
        )

    @classmethod
    def make_workload(cls, arguments: Nvfp4PrepareArguments) -> Nvfp4PrepareWorkload:
        assert arguments.source.dtype == DType.BF16
        num_tokens, width = arguments.source.shape
        assert width == arguments.heads * 128
        padded = -(-num_tokens // 128) * 128
        num_rows = 128 if arguments.transpose else padded
        columns = padded if arguments.transpose else 128
        assert arguments.quantized.dtype == DType.U8
        assert arguments.quantized.shape == (arguments.heads, num_rows, columns // 2)
        assert arguments.scales.dtype == DType.FP8_UE4M3
        assert arguments.scales.shape == (arguments.heads, num_rows, columns // 16)
        if arguments.mean is not None:
            assert arguments.mean.dtype == DType.F32
            assert arguments.mean.shape == (arguments.heads, 128)
        return Nvfp4PrepareWorkload(
            num_tokens=num_tokens,
            heads=arguments.heads,
            center=arguments.mean is not None,
            transpose=arguments.transpose,
        )

    @classmethod
    def make_config(cls, workload: Nvfp4PrepareWorkload) -> Nvfp4PrepareConfig:
        del workload
        return Nvfp4PrepareConfig()

    @classmethod
    def ref_program(cls, arguments: Nvfp4PrepareArguments) -> None:
        import torch

        workload = cls.make_workload(arguments)
        padded = -(-workload.num_tokens // 128) * 128
        values = (
            arguments.source.as_torch()
            .float()
            .reshape(workload.num_tokens, workload.heads, 128)
        )
        if arguments.mean is not None:
            values = values - arguments.mean.as_torch()[None]
        values = torch.nn.functional.pad(
            values, (0, 0, 0, 0, 0, padded - workload.num_tokens)
        )
        values = (
            values.permute(1, 2, 0) if workload.transpose else values.permute(1, 0, 2)
        )
        num_rows, columns = values.shape[1:]
        groups = values.reshape(workload.heads, num_rows, columns // 16, 16)
        block_scale = (groups.abs().amax(-1) / 6).clamp(max=448).to(torch.float8_e4m3fn)
        inverse = torch.where(block_scale == 0, 0, block_scale.float().reciprocal())
        normalized = (groups.float() * inverse[..., None]).double()
        magnitudes = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6],
            device=values.device,
            dtype=torch.float64,
        )
        distances = (normalized.abs()[..., None] - magnitudes).abs()
        nearest = distances == distances.amin(-1, keepdim=True)
        indices = torch.arange(8, device=values.device)
        priority = indices + (indices % 2) * 16
        codes = torch.where(nearest, priority, 100).argmin(-1)
        codes |= normalized.signbit().long() * 8
        codes = codes.byte().reshape(workload.heads, num_rows, columns)
        arguments.quantized.as_torch().copy_(codes[..., ::2] | codes[..., 1::2] << 4)
        blocked = (
            block_scale.reshape(
                workload.heads, num_rows // 128, 4, 32, columns // 64, 4
            )
            .permute(0, 1, 4, 3, 2, 5)
            .contiguous()
            .reshape(arguments.scales.shape)
        )
        arguments.scales.as_torch().copy_(blocked)
