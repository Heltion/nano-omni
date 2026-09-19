"""Apply per-head RMS normalization and split-half rotary embedding."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class QkNormRopeWorkload(Workload):
    """Padded storage and valid Q/K dimensions."""

    num_padded_tokens: int
    heads: int
    num_tokens: int
    dim: int


class QkNormRopeConfig(Config):
    """Valid tokens per tile and CUDA thread count."""

    tile_tokens: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class QkNormRopeArguments(Arguments):
    """Padded Q/K input, head weights, rotary angles, and output."""

    input: TensorDesc
    weight: TensorDesc
    angles: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (
            ("num_tokens", self.angles.shape[0]),
            ("num_padded_tokens", self.input.shape[0]),
        )


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def qk_norm_rope(
    num_padded_tokens, heads, num_tokens, dim, tile_tokens=1, threads=128
):
    """Build fused weighted RMS normalization and RoPE."""
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    dynamic_num_padded_tokens = T.dynamic("num_padded_tokens")
    half_dim = dim // 2

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        num_padded_tokens: dynamic_num_padded_tokens,
        input: T.Tensor([dynamic_num_padded_tokens, heads, dim], T.bfloat16),
        weight: T.Tensor([dim], T.bfloat16),
        angles: T.Tensor([dynamic_num_tokens, half_dim], T.float32),
        output: T.Tensor([dynamic_num_padded_tokens, heads, dim], T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(dynamic_num_tokens, tile_tokens), heads, threads=threads) as (
            block,
            head,
        ):
            first = T.alloc_fragment([tile_tokens, half_dim], T.float32)
            second = T.alloc_fragment([tile_tokens, half_dim], T.float32)
            first_squares = T.alloc_fragment([tile_tokens, half_dim], T.float32)
            second_squares = T.alloc_fragment([tile_tokens, half_dim], T.float32)
            totals = T.alloc_fragment([tile_tokens], T.float32)
            inverse = T.alloc_fragment([tile_tokens], T.float32)
            for i, j in T.Parallel(tile_tokens, half_dim):
                first[i, j] = input[block * tile_tokens + i, head, j]
                first_squares[i, j] = first[i, j] * first[i, j]
                second[i, j] = input[block * tile_tokens + i, head, j + half_dim]
                second_squares[i, j] = second[i, j] * second[i, j]
            T.reduce_sum(first_squares, totals, dim=1)
            T.reduce_sum(second_squares, totals, dim=1, clear=False)
            for i in T.Parallel(tile_tokens):
                inverse[i] = T.rsqrt(totals[i] / dim + 1e-6)
            for i, j in T.Parallel(tile_tokens, half_dim):
                first[i, j] *= inverse[i] * weight[j]
                second[i, j] *= inverse[i] * weight[j + half_dim]
            for i, j in T.Parallel(tile_tokens, half_dim):
                position = block * tile_tokens + i
                angle = T.if_then_else(
                    position < dynamic_num_tokens, angles[position, j], 0.0
                )
                cosine = T.cos(angle)
                sine = T.sin(angle)
                if position < dynamic_num_tokens:
                    output[position, head, j] = (
                        first[i, j] * cosine - second[i, j] * sine
                    )
                    output[position, head, j + half_dim] = (
                        second[i, j] * cosine + first[i, j] * sine
                    )

    return main.with_attr(
        "global_symbol", f"qk_norm_rope_{heads}_{dim}_{tile_tokens}_{threads}"
    )


class QkNormRopeKernel(
    Kernel[QkNormRopeArguments, QkNormRopeWorkload, QkNormRopeConfig]
):
    """Normalize and rotate Q or K heads."""

    name = "qk_norm_rope"
    program = qk_norm_rope

    @classmethod
    def make_arguments(cls, workload: QkNormRopeWorkload) -> QkNormRopeArguments:
        """Describe padded heads, weights, angles, and output."""
        shape = (workload.num_padded_tokens, workload.heads, workload.dim)
        return QkNormRopeArguments(
            input=TensorDesc.empty(DType.BF16, shape),
            weight=TensorDesc.empty(DType.BF16, (workload.dim,)),
            angles=TensorDesc.empty(
                DType.F32, (workload.num_tokens, workload.dim // 2)
            ),
            output=TensorDesc.empty(DType.BF16, shape),
        )

    @classmethod
    def make_workload(cls, arguments: QkNormRopeArguments) -> QkNormRopeWorkload:
        """Validate tensor shapes and recover the specialization."""
        assert len(arguments.input.shape) == 3
        num_padded_tokens, heads, dim = arguments.input.shape
        assert dim % 2 == 0
        assert arguments.output.shape == arguments.input.shape
        assert arguments.weight.shape == (dim,)
        assert arguments.angles.shape[1] * 2 == dim
        assert arguments.input.dtype == arguments.output.dtype == DType.BF16
        assert arguments.weight.dtype == DType.BF16
        assert arguments.angles.dtype == DType.F32
        return QkNormRopeWorkload(
            num_padded_tokens=num_padded_tokens,
            heads=heads,
            num_tokens=arguments.angles.shape[0],
            dim=dim,
        )

    @classmethod
    def make_config(cls, workload: QkNormRopeWorkload) -> QkNormRopeConfig:
        """Use the wider tile for the small eight-head vision workload."""
        if workload.heads == 8 and workload.num_padded_tokens < 1024:
            return QkNormRopeConfig(threads=128, tile_tokens=4)
        return QkNormRopeConfig(threads=64, tile_tokens=2)

    @classmethod
    def ref_program(cls, arguments: QkNormRopeArguments) -> None:
        """Evaluate weighted RMS normalization and rotation with Torch."""
        import torch

        workload = cls.make_workload(arguments)
        value = arguments.input.as_torch()[: workload.num_tokens].float()
        normalized = value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        normalized *= arguments.weight.as_torch().float()
        first, second = normalized.chunk(2, dim=-1)
        angles = arguments.angles.as_torch()[:, None]
        cosine, sine = angles.cos(), angles.sin()
        arguments.output.as_torch()[: workload.num_tokens].copy_(
            torch.cat(
                (
                    first * cosine - second * sine,
                    second * cosine + first * sine,
                ),
                dim=-1,
            )
        )
