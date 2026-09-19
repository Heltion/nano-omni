"""Fuse filtered upsampling, logarithmic SnakeBeta, and downsampling."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class SnakeBetaWorkload(Workload):
    """Per-stream frames and channels, sampling ratios and filter tap counts."""

    channels: int
    frames: int
    stereo: int
    up_ratio: int
    down_ratio: int
    up_kernel_size: int
    down_kernel_size: int


class SnakeBetaConfig(Config):
    """Output frames and channels per tile, plus CUDA threads per block."""

    tile_frames: int
    tile_channels: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class SnakeBetaArguments(Arguments):
    """Contiguous F32 tensors with independent streams concatenated along rows.

    Input/output use [stereo, frames, channels]. Alpha/beta are logarithmic
    weights[channels], exponentiated by the activation. Filters use [1, 1, taps];
    up_ratio/down_ratio set the intermediate sampling strides.
    """

    input: TensorDesc
    alpha: TensorDesc
    beta: TensorDesc
    up_filter: TensorDesc
    down_filter: TensorDesc
    output: TensorDesc
    up_ratio: int
    down_ratio: int


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def snake_beta(
    channels,
    frames,
    stereo,
    up_ratio,
    down_ratio,
    up_kernel_size,
    down_kernel_size,
    tile_frames=64,
    tile_channels=16,
    threads=256,
):
    import tilelang.language as T

    up_pad = up_kernel_size // up_ratio - 1
    up_pad_left = up_pad * up_ratio + (up_kernel_size - up_ratio) // 2
    down_pad_left = down_kernel_size // 2 - int(down_kernel_size % 2 == 0)
    activation_frames = frames * up_ratio
    shared_frames = tile_frames * down_ratio + down_kernel_size - down_ratio

    @T.prim_func
    def main(
        input: T.Tensor([stereo, frames, channels], T.float32),
        alpha: T.Tensor([channels], T.float32),
        beta: T.Tensor([channels], T.float32),
        up_filter: T.Tensor([1, 1, up_kernel_size], T.float32),
        down_filter: T.Tensor([1, 1, down_kernel_size], T.float32),
        output: T.Tensor([stereo, frames, channels], T.float32),
    ):
        T.annotate_pass_configs({"tl.enable_fast_math": True})
        with T.Kernel(
            T.ceildiv(channels, tile_channels),
            T.ceildiv(frames, tile_frames),
            stereo,
            threads=threads,
        ) as (channel_block, frame_block, stream):
            activated = T.alloc_shared([tile_channels * shared_frames], T.float32)
            upsampled = T.alloc_shared([tile_channels * shared_frames], T.float32)
            T.clear(upsampled)
            T.sync_threads()
            for kernel_frame in range(up_kernel_size):
                for offset in T.Parallel(tile_channels * shared_frames):
                    channel_offset = offset // shared_frames
                    shared_frame = offset % shared_frames
                    channel = channel_block * tile_channels + channel_offset
                    raw_activation_frame = (
                        frame_block * tile_frames * down_ratio
                        + shared_frame
                        - down_pad_left
                    )
                    activation_frame = T.max(
                        0,
                        T.min(raw_activation_frame, activation_frames - 1),
                    )
                    transpose_frame = activation_frame + up_pad_left
                    padded_frame = (
                        transpose_frame - kernel_frame
                    ) // up_ratio
                    if (
                        (transpose_frame - kernel_frame) % up_ratio == 0
                        and padded_frame >= 0
                        and padded_frame < frames + 2 * up_pad
                        and channel < channels
                    ):
                        input_frame = T.max(
                            0,
                            T.min(padded_frame - up_pad, frames - 1),
                        )
                        upsampled[offset] += (
                            input[stream, input_frame, channel]
                            * up_filter[0, 0, kernel_frame]
                            * up_ratio
                        )
            for offset in T.Parallel(tile_channels * shared_frames):
                channel_offset = offset // shared_frames
                channel = channel_block * tile_channels + channel_offset
                alpha_value = T.exp(
                    T.if_then_else(channel < channels, alpha[channel], 0.0)
                )
                beta_value = T.exp(
                    T.if_then_else(channel < channels, beta[channel], 0.0)
                )
                value = upsampled[offset]
                sine = T.sin(alpha_value * value)
                activated[offset] = T.if_then_else(
                    channel < channels,
                    value + sine * sine / (beta_value + 1e-9),
                    0.0,
                )
            T.sync_threads()
            result = T.alloc_shared([tile_frames * tile_channels], T.float32)
            T.clear(result)
            T.sync_threads()
            for kernel_frame in range(down_kernel_size):
                for offset in T.Parallel(tile_frames * tile_channels):
                    frame_offset = offset // tile_channels
                    channel_offset = offset % tile_channels
                    result[offset] += (
                        activated[
                            channel_offset * shared_frames
                            + frame_offset * down_ratio
                            + kernel_frame
                        ]
                        * down_filter[0, 0, kernel_frame]
                    )
            T.sync_threads()
            for offset in T.Parallel(tile_frames * tile_channels):
                frame_offset = offset // tile_channels
                channel_offset = offset % tile_channels
                frame = frame_block * tile_frames + frame_offset
                channel = channel_block * tile_channels + channel_offset
                if frame < frames and channel < channels:
                    output[stream, frame, channel] = result[offset]

    return main


class SnakeBetaKernel(Kernel[SnakeBetaArguments, SnakeBetaWorkload, SnakeBetaConfig]):
    name = "snake_beta"
    program = snake_beta

    @classmethod
    def make_arguments(cls, workload: SnakeBetaWorkload) -> SnakeBetaArguments:
        """Describe activation, filters, channel weights, and output tensors."""
        activation_shape = (workload.stereo, workload.frames, workload.channels)
        channel_shape = (workload.channels,)
        return SnakeBetaArguments(
            input=TensorDesc.empty(DType.F32, activation_shape),
            alpha=TensorDesc.empty(DType.F32, channel_shape),
            beta=TensorDesc.empty(DType.F32, channel_shape),
            up_filter=TensorDesc.empty(DType.F32, (1, 1, workload.up_kernel_size)),
            down_filter=TensorDesc.empty(DType.F32, (1, 1, workload.down_kernel_size)),
            output=TensorDesc.empty(DType.F32, activation_shape),
            up_ratio=workload.up_ratio,
            down_ratio=workload.down_ratio,
        )

    @classmethod
    def make_workload(cls, arguments: SnakeBetaArguments) -> SnakeBetaWorkload:
        assert len(arguments.input.shape) == 3
        assert arguments.output.shape == arguments.input.shape
        channels = arguments.input.shape[2]
        assert arguments.alpha.shape == arguments.beta.shape == (channels,)
        assert (
            arguments.up_filter.shape[:2]
            == arguments.down_filter.shape[:2]
            == (
                1,
                1,
            )
        )
        assert all(
            tensor.dtype == DType.F32
            for tensor in (
                arguments.input,
                arguments.alpha,
                arguments.beta,
                arguments.up_filter,
                arguments.down_filter,
                arguments.output,
            )
        )
        return SnakeBetaWorkload(
            channels=channels,
            frames=arguments.input.shape[1],
            stereo=arguments.input.shape[0],
            up_ratio=arguments.up_ratio,
            down_ratio=arguments.down_ratio,
            up_kernel_size=arguments.up_filter.shape[2],
            down_kernel_size=arguments.down_filter.shape[2],
        )

    @classmethod
    def make_config(cls, workload: SnakeBetaWorkload) -> SnakeBetaConfig:
        """Select the launch shape from the stream channel count."""
        if workload.channels in (16, 512):
            return SnakeBetaConfig(
                tile_frames=64, tile_channels=16, threads=256
            )
        return SnakeBetaConfig(
            tile_frames=32 if workload.channels == 8 else 64,
            tile_channels=16,
            threads=128 if workload.channels == 8 else 256,
        )

    @classmethod
    def ref_program(cls, arguments: SnakeBetaArguments) -> None:
        """Evaluate filtered upsampling, SnakeBeta, and filtered downsampling."""
        import torch

        workload = cls.make_workload(arguments)
        source = arguments.input.as_torch().transpose(1, 2)
        up_filter = arguments.up_filter.as_torch()
        down_filter = arguments.down_filter.as_torch()
        pad = workload.up_kernel_size // workload.up_ratio - 1
        left = (
            pad * workload.up_ratio + (workload.up_kernel_size - workload.up_ratio) // 2
        )
        with torch.backends.cudnn.flags(
            benchmark=False, deterministic=True, allow_tf32=False
        ):
            expanded = (
                torch.nn.functional.conv_transpose1d(
                    torch.nn.functional.pad(source, (pad, pad), mode="replicate"),
                    up_filter.expand(workload.channels, -1, -1),
                    stride=workload.up_ratio,
                    groups=workload.channels,
                )
                * workload.up_ratio
            )
            expanded = expanded[..., left : left + workload.frames * workload.up_ratio]
            phase = arguments.alpha.as_torch().exp().reshape(1, -1, 1) * expanded
            activated = expanded + phase.sin().square() / (
                arguments.beta.as_torch().exp().reshape(1, -1, 1) + 1e-9
            )
            left = workload.down_kernel_size // 2 - int(
                workload.down_kernel_size % 2 == 0
            )
            right = workload.down_kernel_size - 1 - left
            result = torch.nn.functional.conv1d(
                torch.nn.functional.pad(activated, (left, right), mode="replicate"),
                down_filter.expand(workload.channels, -1, -1),
                stride=workload.down_ratio,
                groups=workload.channels,
            )
        arguments.output.as_torch().copy_(result.transpose(1, 2))
