"""Append a projected tile while blending horizontal and vertical boundaries."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class VideoVaeSpatialFinishWorkload(Workload):
    """Vertical-prefix height, horizontal-prefix width, and H/W overlap lengths."""

    first_height: int = Field(gt=0)
    first_width: int = Field(gt=0)
    horizontal_overlap: int = Field(gt=0, le=256)
    vertical_overlap: int = Field(gt=0, le=256)
    projected_dtype: DType = DType.BF16


@dataclasses.dataclass(frozen=True, slots=True)
class VideoVaeSpatialFinishArguments(Arguments):
    """FP32 top[3,28,H,W], first[3,28,256,X], and projected[1792,3072].

    W = X + 256 - horizontal_overlap. Output is FP32 [3,28,H+256-vertical_overlap,W].
    CFHW channel/frame axes are preserved; horizontal and vertical overlaps apply
    only to width and height. Projection rows use the decoder's 7x16x16 token
    order and 3x4x16x16 patch columns. Output must not overlap any input because
    the expanded planar strides require prefixes to remain live throughout launch.
    """

    top: TensorDesc
    first: TensorDesc
    projected: TensorDesc
    output: TensorDesc
    horizontal_overlap: int
    vertical_overlap: int


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def video_vae_spatial_finish(
    first_height,
    first_width,
    horizontal_overlap,
    vertical_overlap,
    projected_dtype,
    tile_elements=256,
    threads=128,
):
    import tilelang.language as T

    channels, frames, tile_height, tile_width = 3, 28, 256, 256
    width = first_width + tile_width - horizontal_overlap
    height = first_height + tile_height - vertical_overlap
    elements = channels * frames * height * width
    horizontal_boundary = first_width - horizontal_overlap
    vertical_boundary = first_height - vertical_overlap
    projected_type = T.float16 if projected_dtype == DType.F16 else T.bfloat16

    @T.prim_func
    def main(
        top: T.Tensor([channels, frames, first_height, width], T.float32),
        first: T.Tensor([channels, frames, tile_height, first_width], T.float32),
        projected: T.Tensor([1792, 3072], projected_type),
        output: T.Tensor([channels, frames, height, width], T.float32),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    x = index % width
                    row_index = index // width
                    y = row_index % height
                    plane_index = row_index // height
                    frame = plane_index % frames
                    channel = plane_index // frames
                    value = T.alloc_var(T.float32)
                    if y < vertical_boundary:
                        value = top[channel, frame, y, x]
                    else:
                        row_y = y - vertical_boundary
                        if x < horizontal_boundary:
                            value = first[channel, frame, row_y, x]
                        else:
                            tile_x = x - horizontal_boundary
                            token = (frame // 4 * 16 + row_y // 16) * 16 + tile_x // 16
                            column = (
                                (channel * 4 + frame % 4) * 16 + row_y % 16
                            ) * 16 + tile_x % 16
                            projected_value = T.cast(
                                projected[token, column], T.float32
                            )
                            if x < first_width:
                                weight = T.cast(tile_x, T.float32) / horizontal_overlap
                                # Match the retained horizontal boundary's explicit rounding.
                                value = T.call_extern(
                                    T.float32,
                                    "__fmaf_rn",
                                    first[channel, frame, row_y, x],
                                    1.0 - weight,
                                    projected_value * weight,
                                )
                            else:
                                value = projected_value
                        if y < first_height:
                            weight = T.cast(row_y, T.float32) / vertical_overlap
                            # The row result is FP32 before the vertical multiply/FMA.
                            value = T.call_extern(
                                T.float32,
                                "__fmaf_rn",
                                top[channel, frame, y, x],
                                1.0 - weight,
                                value * weight,
                            )
                    output[channel, frame, y, x] = value

    return main.with_attr(
        "global_symbol",
        f"video_vae_spatial_finish_{first_height}_{first_width}_{horizontal_overlap}_{vertical_overlap}_{projected_dtype.value}_{tile_elements}_{threads}",
    )


class VideoVaeSpatialFinishKernel(
    ElementwiseKernel[
        VideoVaeSpatialFinishArguments,
        VideoVaeSpatialFinishWorkload,
    ]
):
    name = "video_vae_spatial_finish"
    program = video_vae_spatial_finish
    launch = (256, 128)

    @classmethod
    def make_arguments(
        cls, workload: VideoVaeSpatialFinishWorkload
    ) -> VideoVaeSpatialFinishArguments:
        """Describe the vertical prefix, row prefix, projected tile, and output."""
        width = workload.first_width + 256 - workload.horizontal_overlap
        height = workload.first_height + 256 - workload.vertical_overlap
        return VideoVaeSpatialFinishArguments(
            top=TensorDesc.empty(DType.F32, (3, 28, workload.first_height, width)),
            first=TensorDesc.empty(DType.F32, (3, 28, 256, workload.first_width)),
            projected=TensorDesc.empty(workload.projected_dtype, (1792, 3072)),
            output=TensorDesc.empty(DType.F32, (3, 28, height, width)),
            horizontal_overlap=workload.horizontal_overlap,
            vertical_overlap=workload.vertical_overlap,
        )

    @classmethod
    def make_workload(
        cls, arguments: VideoVaeSpatialFinishArguments
    ) -> VideoVaeSpatialFinishWorkload:
        first_height = arguments.top.shape[2]
        first_width = arguments.first.shape[3]
        output_width = first_width + 256 - arguments.horizontal_overlap
        output_height = first_height + 256 - arguments.vertical_overlap
        assert arguments.first.shape[:3] == (3, 28, 256), (
            "H3 spatial finish requires a 3x28x256 row prefix"
        )
        assert arguments.top.shape == (3, 28, first_height, output_width), (
            "H3 spatial finish requires a full-width vertical prefix"
        )
        assert arguments.projected.shape == (1792, 3072), (
            "H3 spatial finish requires 1792x3072 projected tokens"
        )
        assert arguments.output.shape == (3, 28, output_height, output_width), (
            "H3 spatial finish output shape does not match its prefixes"
        )
        return VideoVaeSpatialFinishWorkload(
            first_height=first_height,
            first_width=first_width,
            horizontal_overlap=arguments.horizontal_overlap,
            vertical_overlap=arguments.vertical_overlap,
            projected_dtype=arguments.projected.dtype,
        )

    @classmethod
    def ref_program(cls, arguments: VideoVaeSpatialFinishArguments) -> None:
        """Unpatchify the final tile and blend both spatial boundaries."""
        import torch

        workload = cls.make_workload(arguments)
        top = arguments.top.as_torch()
        first = arguments.first.as_torch()
        projected = arguments.projected.as_torch()
        output = arguments.output.as_torch()
        tile = (
            projected.reshape(7, 16, 16, 3, 4, 16, 16)
            .permute(3, 0, 4, 1, 5, 2, 6)
            .reshape(3, 28, 256, 256)
            .float()
        )
        overlap = workload.horizontal_overlap
        boundary = workload.first_width - overlap
        weight = torch.arange(overlap, device=first.device).float() / overlap
        row = torch.cat(
            (
                first[..., :boundary],
                first[..., boundary:] * (1 - weight) + tile[..., :overlap] * weight,
                tile[..., overlap:],
            ),
            dim=3,
        )
        overlap = workload.vertical_overlap
        boundary = workload.first_height - overlap
        weight = (torch.arange(overlap, device=first.device).float() / overlap)[:, None]
        output.copy_(
            torch.cat(
                (
                    top[:, :, :boundary],
                    top[:, :, boundary:] * (1 - weight) + row[:, :, :overlap] * weight,
                    row[:, :, overlap:],
                ),
                dim=2,
            )
        )
