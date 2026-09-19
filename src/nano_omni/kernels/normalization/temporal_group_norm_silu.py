"""Per-frame group normalization with optional fused SiLU."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Kernel
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.normalization.base import (
    GroupNormalizationConfig,
    GroupNormalizationWorkload,
)


class TemporalGroupNormSiluWorkload(GroupNormalizationWorkload):
    # Each frame/group reduces (channels / groups) * height * width in FP32.
    # FP16/BF16 storage, FP16 affine parameters; fixed epsilon 1e-6 precedes rsqrt.
    frames: int = Field(gt=0)
    dtype: DType = DType.BF16


@dataclasses.dataclass(frozen=True, slots=True)
class TemporalGroupNormSiluArguments(Arguments):
    input: TensorDesc
    weight: TensorDesc
    bias: TensorDesc
    output: TensorDesc
    groups: int
    activate: bool


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def temporal_group_norm_silu(
    channels, frames, height, width, groups, activate, dtype, tile_elements=1024, threads=256
):
    import tilelang.language as T

    spatial = height * width
    channels_per_group = channels // groups
    group_elements = channels_per_group * spatial
    storage_type = T.float16 if dtype == DType.F16 else T.bfloat16

    @T.prim_func
    def main(
        input: T.Tensor([channels, frames, height, width], storage_type),
        weight: T.Tensor([channels], T.float16),
        bias: T.Tensor([channels], T.float16),
        output: T.Tensor([channels, frames, height, width], storage_type),
    ):
        T.annotate_pass_configs(
            {"tl.enable_fast_math": True, "tl.disable_vectorize_256": True}
        )
        with T.Kernel(groups, frames, threads=threads) as (group, frame):
            values = T.alloc_fragment([tile_elements], T.float32)
            squares = T.alloc_fragment([tile_elements], T.float32)
            total = T.alloc_fragment([1], T.float32)
            square_total = T.alloc_fragment([1], T.float32)
            T.clear(total)
            T.clear(square_total)
            for chunk in range(T.ceildiv(group_elements, tile_elements)):
                for index in T.Parallel(tile_elements):
                    offset = chunk * tile_elements + index
                    channel = group * channels_per_group + offset // spatial
                    position = offset % spatial
                    values[index] = T.if_then_else(
                        offset < group_elements,
                        input[
                            channel,
                            frame,
                            position // width,
                            position % width,
                        ],
                        0.0,
                    )
                    squares[index] = values[index] * values[index]
                T.reduce_sum(values, total, dim=0, clear=False)
                T.reduce_sum(squares, square_total, dim=0, clear=False)
            mean = total[0] / group_elements
            inverse = T.rsqrt(square_total[0] / group_elements - mean * mean + 1e-6)
            for chunk in range(T.ceildiv(group_elements, tile_elements)):
                for index in T.Parallel(tile_elements):
                    offset = chunk * tile_elements + index
                    if offset < group_elements:
                        channel = group * channels_per_group + offset // spatial
                        position = offset % spatial
                        normalized = (
                            input[
                                channel,
                                frame,
                                position // width,
                                position % width,
                            ]
                            - mean
                        ) * inverse * weight[channel] + bias[channel]
                        if activate:
                            result = normalized / (1.0 + T.exp(-normalized))
                        else:
                            result = normalized
                        output[
                            channel,
                            frame,
                            position // width,
                            position % width,
                        ] = result

    return main

class TemporalGroupNormSiluKernel(
    Kernel[
        TemporalGroupNormSiluArguments,
        TemporalGroupNormSiluWorkload,
        GroupNormalizationConfig,
    ]
):
    name = "temporal_group_norm_silu"
    program = temporal_group_norm_silu

    @classmethod
    def make_arguments(
        cls, workload: TemporalGroupNormSiluWorkload
    ) -> TemporalGroupNormSiluArguments:
        """Describe the video tensor, channel affine parameters, and output."""
        shape = (
            workload.channels,
            workload.frames,
            workload.height,
            workload.width,
        )
        return TemporalGroupNormSiluArguments(
            input=TensorDesc.empty(workload.dtype, shape),
            weight=TensorDesc.empty(DType.F16, (workload.channels,)),
            bias=TensorDesc.empty(DType.F16, (workload.channels,)),
            output=TensorDesc.empty(workload.dtype, shape),
            groups=workload.groups,
            activate=workload.activate,
        )

    @classmethod
    def make_workload(
        cls, arguments: TemporalGroupNormSiluArguments
    ) -> TemporalGroupNormSiluWorkload:
        """Recover the workload and validate video and affine tensor contracts."""
        assert len(arguments.input.shape) == 4
        channels, frames, height, width = arguments.input.shape
        assert not channels % arguments.groups
        assert arguments.input.dtype == arguments.output.dtype
        assert arguments.input.dtype in (DType.F16, DType.BF16)
        assert arguments.output.shape == arguments.input.shape
        assert arguments.weight.dtype == arguments.bias.dtype == DType.F16
        assert arguments.weight.shape == arguments.bias.shape == (channels,)
        return TemporalGroupNormSiluWorkload(
            channels=channels,
            frames=frames,
            height=height,
            width=width,
            groups=arguments.groups,
            activate=arguments.activate,
            dtype=arguments.input.dtype,
        )

    @classmethod
    def ref_program(cls, arguments: TemporalGroupNormSiluArguments) -> None:
        """Apply independent per-frame group normalization and optional SiLU."""
        import torch

        workload = cls.make_workload(arguments)
        value = torch.nn.functional.group_norm(
            arguments.input.as_torch().permute(1, 0, 2, 3).float(),
            workload.groups,
            arguments.weight.as_torch().float(),
            arguments.bias.as_torch().float(),
            1e-6,
        )
        if workload.activate:
            value = torch.nn.functional.silu(value)
        arguments.output.as_torch().copy_(value.permute(1, 0, 2, 3))

    @classmethod
    def make_config(
        cls, workload: TemporalGroupNormSiluWorkload
    ) -> GroupNormalizationConfig:
        """Use one block to reduce a complete 4096-element group tile."""
        del workload
        return GroupNormalizationConfig(threads=256, tile_elements=4096)
