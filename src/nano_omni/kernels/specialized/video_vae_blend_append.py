"""Blend and append CFHW volumes along height or width without changing frames."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class VideoVaeBlendAppendWorkload(Workload):
    """Two CFHW extents, append axis 2 (height) or 3 (width), and overlap length."""

    channels: int
    frames: int
    first_height: int
    first_width: int
    second_height: int
    second_width: int
    axis: int
    overlap: int


@dataclasses.dataclass(frozen=True, slots=True)
class VideoVaeBlendAppendArguments(Arguments):
    """Read-only FP32 first/second[C,F,H,W] and a separate FP32 output volume.

    axis=2 appends height; axis=3 appends width. All other dimensions must match.
    The overlap lies within both input extents; its weight is index/overlap,
    followed by the remaining second-input region. Output must not overlap either
    input: expanded planar strides can overwrite values other threads still read.
    """

    first: TensorDesc
    second: TensorDesc
    output: TensorDesc
    axis: int
    overlap: int


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def video_vae_blend_append(
    channels,
    frames,
    first_height,
    first_width,
    second_height,
    second_width,
    axis,
    overlap,
    tile_elements=256,
    threads=128,
):
    import tilelang.language as T

    output_height = (
        first_height + second_height - overlap if axis == 2 else first_height
    )
    output_width = first_width + second_width - overlap if axis == 3 else first_width
    elements = channels * frames * output_height * output_width

    @T.prim_func
    def main(
        first: T.Tensor(
            [channels, frames, first_height, first_width],
            T.float32,
        ),
        second: T.Tensor(
            [channels, frames, second_height, second_width],
            T.float32,
        ),
        output: T.Tensor(
            [channels, frames, output_height, output_width],
            T.float32,
        ),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    x = index % output_width
                    y = (index // output_width) % output_height
                    frame = (index // (output_width * output_height)) % frames
                    channel = index // (output_width * output_height * frames)
                    value = T.alloc_var(T.float32)
                    if axis == 3:
                        boundary = first_width - overlap
                        if x < boundary:
                            value = first[channel, frame, y, x]
                        elif x < first_width:
                            second_x = x - boundary
                            weight = T.cast(second_x, T.float32) / overlap
                            value = (
                                first[channel, frame, y, x] * (1.0 - weight)
                                + second[channel, frame, y, second_x] * weight
                            )
                        else:
                            value = second[channel, frame, y, x - boundary]
                    else:
                        boundary = first_height - overlap
                        if y < boundary:
                            value = first[channel, frame, y, x]
                        elif y < first_height:
                            second_y = y - boundary
                            weight = T.cast(second_y, T.float32) / overlap
                            value = (
                                first[channel, frame, y, x] * (1.0 - weight)
                                + second[channel, frame, second_y, x] * weight
                            )
                        else:
                            value = second[channel, frame, y - boundary, x]
                    output[channel, frame, y, x] = value

    return main.with_attr(
        "global_symbol",
        "video_vae_blend_append_"
        f"{channels}_{frames}_{first_height}_{first_width}_{second_height}_"
        f"{second_width}_{axis}_{overlap}_"
        f"{tile_elements}_{threads}",
    )


class VideoVaeBlendAppendKernel(
    ElementwiseKernel[
        VideoVaeBlendAppendArguments,
        VideoVaeBlendAppendWorkload,
    ]
):
    name = "video_vae_blend_append"
    program = video_vae_blend_append
    launch = (256, 128)

    @classmethod
    def make_arguments(
        cls, workload: VideoVaeBlendAppendWorkload
    ) -> VideoVaeBlendAppendArguments:
        output_height = (
            workload.first_height + workload.second_height - workload.overlap
            if workload.axis == 2
            else workload.first_height
        )
        output_width = (
            workload.first_width + workload.second_width - workload.overlap
            if workload.axis == 3
            else workload.first_width
        )
        return VideoVaeBlendAppendArguments(
            TensorDesc.empty(
                DType.F32,
                (
                    workload.channels,
                    workload.frames,
                    workload.first_height,
                    workload.first_width,
                ),
            ),
            TensorDesc.empty(
                DType.F32,
                (
                    workload.channels,
                    workload.frames,
                    workload.second_height,
                    workload.second_width,
                ),
            ),
            TensorDesc.empty(
                DType.F32,
                (workload.channels, workload.frames, output_height, output_width),
            ),
            workload.axis,
            workload.overlap,
        )

    @classmethod
    def make_workload(
        cls, arguments: VideoVaeBlendAppendArguments
    ) -> VideoVaeBlendAppendWorkload:
        assert len(arguments.first.shape) == len(arguments.second.shape) == 4
        assert arguments.axis in (2, 3)
        assert arguments.first.dtype == arguments.second.dtype == DType.F32
        assert arguments.output.dtype == DType.F32
        workload = VideoVaeBlendAppendWorkload(
            channels=arguments.first.shape[0],
            frames=arguments.first.shape[1],
            first_height=arguments.first.shape[2],
            first_width=arguments.first.shape[3],
            second_height=arguments.second.shape[2],
            second_width=arguments.second.shape[3],
            axis=arguments.axis,
            overlap=arguments.overlap,
        )
        assert workload.overlap > 0
        assert workload.overlap <= (
            workload.first_height if workload.axis == 2 else workload.first_width
        )
        assert workload.overlap <= (
            workload.second_height if workload.axis == 2 else workload.second_width
        )
        assert arguments.first.shape[:2] == arguments.second.shape[:2]
        if workload.axis == 2:
            assert workload.first_width == workload.second_width
            output_shape = (
                workload.channels,
                workload.frames,
                workload.first_height + workload.second_height - workload.overlap,
                workload.first_width,
            )
        else:
            assert workload.first_height == workload.second_height
            output_shape = (
                workload.channels,
                workload.frames,
                workload.first_height,
                workload.first_width + workload.second_width - workload.overlap,
            )
        assert arguments.output.shape == output_shape
        return workload

    @classmethod
    def ref_program(cls, arguments: VideoVaeBlendAppendArguments) -> None:
        """Blend the overlap and append the remaining second volume."""
        import torch

        workload = cls.make_workload(arguments)
        first = arguments.first.as_torch()
        second = arguments.second.as_torch()
        output = arguments.output.as_torch()
        axis = workload.axis
        boundary = first.shape[axis] - workload.overlap
        first_prefix = first.narrow(axis, 0, boundary)
        first_overlap = first.narrow(axis, boundary, workload.overlap)
        second_overlap = second.narrow(axis, 0, workload.overlap)
        weight_shape = [1] * first.ndim
        weight_shape[axis] = workload.overlap
        weight = (
            torch.arange(workload.overlap, device=first.device, dtype=torch.float32)
            .div(workload.overlap)
            .view(weight_shape)
        )
        blended = first_overlap * (1.0 - weight) + second_overlap * weight
        second_suffix = second.narrow(
            axis, workload.overlap, second.shape[axis] - workload.overlap
        )
        output.copy_(torch.cat((first_prefix, blended, second_suffix), dim=axis))
