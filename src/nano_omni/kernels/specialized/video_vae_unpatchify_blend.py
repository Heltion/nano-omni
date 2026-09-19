"""Unpatchify projected H3 tokens directly into a blended spatial row."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class VideoVaeUnpatchifyBlendWorkload(Workload):
    first_width: int = Field(gt=0)
    overlap: int = Field(gt=0, le=256)
    projected_dtype: DType = DType.BF16


@dataclasses.dataclass(frozen=True, slots=True)
class VideoVaeUnpatchifyBlendArguments(Arguments):
    first: TensorDesc
    projected: TensorDesc
    output: TensorDesc
    overlap: int


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def video_vae_unpatchify_blend(first_width, overlap, projected_dtype, tile_elements=512, threads=256):
    import tilelang.language as T

    channels, frames, height, tile_width = 3, 28, 256, 256
    output_width = first_width + tile_width - overlap
    elements = channels * frames * height * output_width
    boundary = first_width - overlap
    projected_type = T.float16 if projected_dtype == DType.F16 else T.bfloat16

    @T.prim_func
    def main(
        first: T.Tensor([channels, frames, height, first_width], T.float32),
        projected: T.Tensor([1792, 3072], projected_type),
        output: T.Tensor([channels, frames, height, output_width], T.float32),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    x = index % output_width
                    position = index // output_width
                    y = position % height
                    plane = position // height
                    frame = plane % frames
                    channel = plane // frames
                    value = T.alloc_var(T.float32)
                    if x < boundary:
                        value = first[channel, frame, y, x]
                    else:
                        second_x = x - boundary
                        token = (frame // 4 * 16 + y // 16) * 16 + second_x // 16
                        column = (
                            (channel * 4 + frame % 4) * 16 + y % 16
                        ) * 16 + second_x % 16
                        second = T.cast(projected[token, column], T.float32)
                        if x < first_width:
                            weight = T.cast(second_x, T.float32) / overlap
                            # The retained production SASS rounds second*weight
                            # before its FFMA of first*(1-weight). Keep that order.
                            value = T.call_extern(
                                T.float32,
                                "__fmaf_rn",
                                first[channel, frame, y, x],
                                1.0 - weight,
                                second * weight,
                            )
                        else:
                            value = second
                    output[channel, frame, y, x] = value

    return main.with_attr(
        "global_symbol",
        f"video_vae_unpatchify_blend_{first_width}_{overlap}_{projected_dtype.value}_{tile_elements}_{threads}",
    )


class VideoVaeUnpatchifyBlendKernel(
    ElementwiseKernel[
        VideoVaeUnpatchifyBlendArguments,
        VideoVaeUnpatchifyBlendWorkload,
    ]
):
    name = "video_vae_unpatchify_blend"
    program = video_vae_unpatchify_blend
    launch = (512, 256)

    @classmethod
    def make_arguments(
        cls, workload: VideoVaeUnpatchifyBlendWorkload
    ) -> VideoVaeUnpatchifyBlendArguments:
        """Describe the decoded prefix, projected tile, and blended output row."""
        return VideoVaeUnpatchifyBlendArguments(
            first=TensorDesc.empty(DType.F32, (3, 28, 256, workload.first_width)),
            projected=TensorDesc.empty(workload.projected_dtype, (1792, 3072)),
            output=TensorDesc.empty(
                DType.F32,
                (3, 28, 256, workload.first_width + 256 - workload.overlap),
            ),
            overlap=workload.overlap,
        )

    @classmethod
    def make_workload(
        cls, arguments: VideoVaeUnpatchifyBlendArguments
    ) -> VideoVaeUnpatchifyBlendWorkload:
        first_width = arguments.first.shape[3]
        assert arguments.first.shape[:3] == (3, 28, 256) and (
            arguments.projected.shape == (1792, 3072)
        ), "H3 unpatchify blend requires a decoded row and 1792x3072 projected tokens"
        assert arguments.output.shape == (
            3,
            28,
            256,
            first_width + 256 - arguments.overlap,
        ), "H3 unpatchify blend output does not match the appended row"
        return VideoVaeUnpatchifyBlendWorkload(
            first_width=first_width, overlap=arguments.overlap,
            projected_dtype=arguments.projected.dtype,
        )

    @classmethod
    def ref_program(cls, arguments: VideoVaeUnpatchifyBlendArguments) -> None:
        """Unpatchify the projected tile and blend its horizontal overlap."""
        import torch

        workload = cls.make_workload(arguments)
        first = arguments.first.as_torch()
        projected = arguments.projected.as_torch()
        output = arguments.output.as_torch()
        second = (
            projected.reshape(7, 16, 16, 3, 4, 16, 16)
            .permute(3, 0, 4, 1, 5, 2, 6)
            .reshape(3, 28, 256, 256)
            .float()
        )
        boundary = workload.first_width - workload.overlap
        weight = (
            torch.arange(workload.overlap, device=first.device).float()
            / workload.overlap
        )
        output[..., :boundary] = first[..., :boundary]
        output[..., boundary : workload.first_width] = (
            first[..., boundary:] * (1 - weight)
            + second[..., : workload.overlap] * weight
        )
        output[..., workload.first_width :] = second[..., workload.overlap :]
