"""H3 head normalization, RoPE and grouped INT8 query preparation."""

import dataclasses
from typing import Literal

import tilelang

from nano_omni.core.kernel import (
    Arguments,
    Config,
    Kernel,
    Workload,
)
from nano_omni.core.tensor import DType, TensorDesc


class H3QueryPrepareWorkload(Workload):
    tokens: int
    heads: int
    use_rope: bool


class H3QueryPrepareConfig(Config):
    threads: Literal[128] = 128


@dataclasses.dataclass(frozen=True, slots=True)
class H3QueryPrepareArguments(Arguments):
    source: TensorDesc
    weight: TensorDesc
    cosines: TensorDesc | None
    sines: TensorDesc | None
    quantized: TensorDesc
    scales: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one program for different token counts."""
        return (("tokens", self.source.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def h3_query_prepare(tokens: int, heads: int, use_rope: bool, threads: int = 128):
    import tilelang.language as T

    dynamic_tokens = T.dynamic("tokens")
    groups = T.ceildiv(dynamic_tokens, 32)
    warps = threads // 32
    layout = T.Fragment(
        (32, 128),
        forward_thread_fn=lambda i, j: (i % warps) * 32 + j % 32,
        forward_index_fn=lambda i, j: (i // warps) * 4 + j // 32,
    )

    @T.prim_func
    def main(
        tokens: dynamic_tokens,
        source: T.Tensor((dynamic_tokens, heads * 128), T.bfloat16),
        weight: T.Tensor((128,), T.bfloat16),
        cosines: T.Tensor((dynamic_tokens, 48), T.float32),
        sines: T.Tensor((dynamic_tokens, 48), T.float32),
        quantized: T.Tensor((heads, groups * 32, 128), T.int8),
        scales: T.Tensor((heads, groups), T.float32),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(groups, heads, threads=threads) as (group, head):
            x = T.alloc_fragment((32, 128), T.float32)
            squared = T.alloc_fragment((32, 128), T.float32)
            normed = T.alloc_shared((32, 128), T.float32)
            values = T.alloc_fragment((32, 128), T.float32)
            absolute = T.alloc_fragment((32, 128), T.float32)
            packed = T.alloc_fragment((32, 128), T.int8)
            total = T.alloc_fragment((32,), T.float32)
            row_max = T.alloc_fragment((32,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            # Each element has one owner across loading, normalization and packing.
            T.annotate_layout(
                {
                    x: layout,
                    squared: layout,
                    values: layout,
                    absolute: layout,
                    packed: layout,
                }
            )
            T.copy(
                source[group * 32 : group * 32 + 32, head * 128 : head * 128 + 128], x
            )
            for i, j in T.Parallel(32, 128):
                squared[i, j] = x[i, j] * x[i, j]
            T.reduce_sum(squared, total, dim=1)
            for i, j in T.Parallel(32, 128):
                normed[i, j] = x[i, j] * T.rsqrt(total[i] / 128 + 1e-5) * weight[j]
            for i, j in T.Parallel(32, 128):
                if group * 32 + i < tokens:
                    # Preserve the original BF16 RoPE output before INT8 conversion.
                    if use_rope and j < 48:
                        values[i, j] = T.cast(
                            normed[i, j] * cosines[group * 32 + i, j]
                            - normed[i, j + 48] * sines[group * 32 + i, j],
                            T.bfloat16,
                        )
                    elif use_rope and j < 96:
                        values[i, j] = T.cast(
                            normed[i, j - 48] * sines[group * 32 + i, j - 48]
                            + normed[i, j] * cosines[group * 32 + i, j - 48],
                            T.bfloat16,
                        )
                    else:
                        values[i, j] = T.cast(normed[i, j], T.bfloat16)
                else:
                    values[i, j] = 0
                absolute[i, j] = T.abs(values[i, j])
            T.reduce_max(absolute, row_max, dim=1)
            T.reduce_max(row_max, maximum, dim=0)
            for i, j in T.Parallel(32, 128):
                packed[i, j] = T.cast(
                    T.round(values[i, j] / T.max(maximum[0] / 127, 1e-10)), T.int8
                )
            T.copy(packed, quantized[head, group * 32 : group * 32 + 32, :])
            for i in T.Parallel(1):
                scales[head, group] = T.max(maximum[0] / 127, 1e-10)

    return main.with_attr(
        "global_symbol", f"h3_query_prepare_{heads}_{int(use_rope)}_{threads}"
    )


class H3QueryPrepareKernel(
    Kernel[H3QueryPrepareArguments, H3QueryPrepareWorkload, H3QueryPrepareConfig]
):
    name = "h3_query_prepare"
    program = h3_query_prepare

    @classmethod
    def make_arguments(
        cls, workload: H3QueryPrepareWorkload
    ) -> H3QueryPrepareArguments:
        """Describe normalized Q input data and grouped INT8 outputs."""
        rows = -(-workload.tokens // 32) * 32
        return H3QueryPrepareArguments(
            source=TensorDesc.empty(
                DType.BF16, (workload.tokens, workload.heads * 128)
            ),
            weight=TensorDesc.empty(DType.BF16, (128,)),
            cosines=(
                TensorDesc.empty(DType.F32, (workload.tokens, 48))
                if workload.use_rope
                else None
            ),
            sines=(
                TensorDesc.empty(DType.F32, (workload.tokens, 48))
                if workload.use_rope
                else None
            ),
            quantized=TensorDesc.empty(DType.I8, (workload.heads, rows, 128)),
            scales=TensorDesc.empty(DType.F32, (workload.heads, rows // 32)),
        )

    @classmethod
    def make_workload(
        cls, arguments: H3QueryPrepareArguments
    ) -> H3QueryPrepareWorkload:
        """Recover the workload and validate normalized, rotated Q layouts."""
        assert len(arguments.source.shape) == 2
        tokens, columns = arguments.source.shape
        assert not columns % 128
        heads = columns // 128
        assert arguments.source.dtype == arguments.weight.dtype == DType.BF16
        assert arguments.weight.shape == (128,)
        assert (arguments.cosines is None) == (arguments.sines is None)
        if arguments.cosines is not None and arguments.sines is not None:
            assert arguments.cosines.dtype == arguments.sines.dtype == DType.F32
            assert arguments.cosines.shape == arguments.sines.shape == (tokens, 48)
        rows = -(-tokens // 32) * 32
        assert arguments.quantized.dtype == DType.I8
        assert arguments.quantized.shape == (heads, rows, 128)
        assert arguments.scales.dtype == DType.F32
        assert arguments.scales.shape == (heads, rows // 32)
        return H3QueryPrepareWorkload(
            tokens=tokens,
            heads=heads,
            use_rope=arguments.cosines is not None,
        )

    @classmethod
    def ref_program(cls, arguments: H3QueryPrepareArguments) -> None:
        """Normalize, rotate and quantize Q using direct tensor mathematics."""
        import torch

        workload = cls.make_workload(arguments)
        source = (
            arguments.source.as_torch()
            .view(workload.tokens, workload.heads, 128)
            .float()
        )
        values = source * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-5)
        values *= arguments.weight.as_torch().float()
        if workload.use_rope:
            assert arguments.cosines is not None and arguments.sines is not None
            first, second = values[..., :48], values[..., 48:96]
            cosine = arguments.cosines.as_torch()[:, None]
            sine = arguments.sines.as_torch()[:, None]
            values = torch.cat(
                (
                    first * cosine - second * sine,
                    first * sine + second * cosine,
                    values[..., 96:],
                ),
                dim=-1,
            )
        values = values.bfloat16().float()
        rows = arguments.quantized.shape[1]
        if rows != workload.tokens:
            values = torch.nn.functional.pad(
                values, (0, 0, 0, 0, 0, rows - workload.tokens)
            )
        grouped = values.view(-1, 32, workload.heads, 128).permute(2, 0, 1, 3)
        scales = (grouped.abs().amax((2, 3)) / 127).clamp_min(1e-10)
        arguments.quantized.as_torch().copy_(
            (grouped / scales[:, :, None, None])
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
            .reshape(workload.heads, rows, 128)
        )
        arguments.scales.as_torch().copy_(scales)

    @classmethod
    def make_config(
        cls, workload: H3QueryPrepareWorkload
    ) -> H3QueryPrepareConfig:
        del workload
        return H3QueryPrepareConfig()
