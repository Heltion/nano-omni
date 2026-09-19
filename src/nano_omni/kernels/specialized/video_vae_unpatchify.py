"""Restore projected VAE patch tokens to a contiguous FP32 CFHW volume."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel

PATCH_T = 4
PATCH_H = 16
PATCH_W = 16


class VideoVaeUnpatchifyWorkload(Workload):
    """Patch-grid dimensions and restored channel count."""

    frames: int
    height: int
    width: int
    channels: int
    input_dtype: DType = DType.BF16


@dataclasses.dataclass(frozen=True, slots=True)
class VideoVaeUnpatchifyArguments(Arguments):
    """Projected FP16/BF16 patch tokens and restored FP32 CFHW output."""

    input: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def video_vae_unpatchify(
    frames, height, width, channels, input_dtype, tile_elements=256, threads=128
):
    import tilelang.language as T

    tokens = frames * height * width
    columns = channels * PATCH_T * PATCH_H * PATCH_W
    output_frames = frames * PATCH_T
    output_height = height * PATCH_H
    output_width = width * PATCH_W
    output_spatial = output_frames * output_height * output_width
    elements = channels * output_spatial
    input_type = T.float16 if input_dtype == DType.F16 else T.bfloat16

    @T.prim_func
    def main(
        input: T.Tensor((tokens, columns), input_type),
        output: T.Tensor(
            (channels, output_frames, output_height, output_width), T.float32
        ),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    channel = index // output_spatial
                    output_index = index % output_spatial
                    frame = output_index // (output_height * output_width)
                    spatial = output_index % (output_height * output_width)
                    y = spatial // output_width
                    x = spatial % output_width
                    token = (
                        frame // PATCH_T * height + y // PATCH_H
                    ) * width + x // PATCH_W
                    column = (
                        (channel * PATCH_T + frame % PATCH_T) * PATCH_H + y % PATCH_H
                    ) * PATCH_W + x % PATCH_W
                    output[channel, frame, y, x] = input[token, column]

    return main.with_attr(
        "global_symbol",
        f"video_vae_unpatchify_{frames}_{height}_{width}_{channels}_{input_dtype.value}_{tile_elements}_{threads}",
    )


class VideoVaeUnpatchifyKernel(
    ElementwiseKernel[
        VideoVaeUnpatchifyArguments,
        VideoVaeUnpatchifyWorkload,
    ]
):
    name = "video_vae_unpatchify"
    program = video_vae_unpatchify
    launch = (1024, 256)

    @classmethod
    def make_arguments(
        cls, workload: VideoVaeUnpatchifyWorkload
    ) -> VideoVaeUnpatchifyArguments:
        """Describe projected patch tokens and the restored CFHW volume."""
        return VideoVaeUnpatchifyArguments(
            input=TensorDesc.empty(
                workload.input_dtype,
                (
                    workload.frames * workload.height * workload.width,
                    workload.channels * PATCH_T * PATCH_H * PATCH_W,
                ),
            ),
            output=TensorDesc.empty(
                DType.F32,
                (
                    workload.channels,
                    workload.frames * PATCH_T,
                    workload.height * PATCH_H,
                    workload.width * PATCH_W,
                ),
            ),
        )

    @classmethod
    def make_workload(
        cls, arguments: VideoVaeUnpatchifyArguments
    ) -> VideoVaeUnpatchifyWorkload:
        assert len(arguments.input.shape) == 2 and len(arguments.output.shape) == 4
        channels, output_frames, output_height, output_width = arguments.output.shape
        assert output_frames % PATCH_T == 0
        assert output_height % PATCH_H == 0
        assert output_width % PATCH_W == 0
        frames = output_frames // PATCH_T
        height = output_height // PATCH_H
        width = output_width // PATCH_W
        assert arguments.input.shape == (
            frames * height * width,
            channels * PATCH_T * PATCH_H * PATCH_W,
        )
        assert arguments.input.dtype in (DType.F16, DType.BF16)
        assert arguments.output.dtype == DType.F32
        return VideoVaeUnpatchifyWorkload(
            frames=frames,
            height=height,
            width=width,
            channels=channels,
            input_dtype=arguments.input.dtype,
        )

    @classmethod
    def ref_program(cls, arguments: VideoVaeUnpatchifyArguments) -> None:
        """Restore channel, frame, and spatial patch axes in CFHW order."""
        workload = cls.make_workload(arguments)
        arguments.output.as_torch().copy_(
            arguments.input.as_torch()
            .view(
                workload.frames,
                workload.height,
                workload.width,
                workload.channels,
                PATCH_T,
                PATCH_H,
                PATCH_W,
            )
            .permute(3, 0, 4, 1, 5, 2, 6)
            .reshape(arguments.output.shape)
        )
