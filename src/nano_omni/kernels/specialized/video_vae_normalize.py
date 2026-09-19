"""Normalize posterior mean channels from FP16/BF16 storage into FP32 latents."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class VideoVaeNormalizeWorkload(Workload):
    """Output channels and the unchanged temporal and spatial extent."""

    channels: int
    frames: int
    height: int
    width: int
    weight_dtype: DType
    input_dtype: DType = DType.BF16


@dataclasses.dataclass(frozen=True, slots=True)
class VideoVaeNormalizeArguments(Arguments):
    """Posterior storage, channel statistics, and normalized output."""

    input: TensorDesc
    mean: TensorDesc
    std: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def video_vae_normalize(
    channels, frames, height, width, weight_dtype, input_dtype, tile_elements=256, threads=128
):
    import tilelang.language as T

    weight_type = T.float16 if weight_dtype == DType.F16 else T.float32
    spatial = frames * height * width
    total = channels * spatial
    input_type = T.float16 if input_dtype == DType.F16 else T.bfloat16

    @T.prim_func
    def main(
        input: T.Tensor((channels * 2, spatial), input_type),
        mean: T.Tensor((channels,), weight_type),
        std: T.Tensor((channels,), weight_type),
        output: T.Tensor((channels, spatial), T.float32),
    ):
        with T.Kernel(T.ceildiv(total, tile_elements), threads=threads) as block:
            for index in T.Parallel(tile_elements):
                offset = block * tile_elements + index
                if offset < total:
                    channel = offset // spatial
                    position = offset % spatial
                    output[channel, position] = (
                        input[channel, position] - mean[channel]
                    ) / std[channel]

    return main.with_attr(
        "global_symbol",
        f"video_vae_normalize_{channels}_{frames}_{height}_{width}_{weight_dtype.value}_{input_dtype.value}_{tile_elements}_{threads}",
    )


class VideoVaeNormalizeKernel(
    ElementwiseKernel[VideoVaeNormalizeArguments, VideoVaeNormalizeWorkload]
):
    name = "video_vae_normalize"
    program = video_vae_normalize
    launch = (256, 128)

    @classmethod
    def make_arguments(
        cls, workload: VideoVaeNormalizeWorkload
    ) -> VideoVaeNormalizeArguments:
        """Describe posterior storage, channel statistics, and normalized latents."""
        spatial_shape = (workload.frames, workload.height, workload.width)
        return VideoVaeNormalizeArguments(
            input=TensorDesc.empty(workload.input_dtype, (workload.channels * 2, *spatial_shape)),
            mean=TensorDesc.empty(workload.weight_dtype, (workload.channels,)),
            std=TensorDesc.empty(workload.weight_dtype, (workload.channels,)),
            output=TensorDesc.empty(DType.F32, (workload.channels, *spatial_shape)),
        )

    @classmethod
    def make_workload(
        cls, arguments: VideoVaeNormalizeArguments
    ) -> VideoVaeNormalizeWorkload:
        assert len(arguments.output.shape) == 4
        channels, frames, height, width = arguments.output.shape
        assert arguments.input.shape == (channels * 2, frames, height, width)
        assert arguments.mean.shape == arguments.std.shape == (channels,)
        assert arguments.mean.dtype == arguments.std.dtype
        assert arguments.input.dtype in (DType.F16, DType.BF16)
        assert arguments.output.dtype == DType.F32
        return VideoVaeNormalizeWorkload(
            channels=channels,
            frames=frames,
            height=height,
            width=width,
            weight_dtype=arguments.mean.dtype,
            input_dtype=arguments.input.dtype,
        )

    @classmethod
    def ref_program(cls, arguments: VideoVaeNormalizeArguments) -> None:
        """Normalize the posterior mean channels in FP32."""
        channels = arguments.output.shape[0]
        arguments.output.as_torch().copy_(
            (
                arguments.input.as_torch()[:channels].float()
                - arguments.mean.as_torch().float()[:, None, None, None]
            )
            / arguments.std.as_torch().float()[:, None, None, None]
        )
