"""Blend H3 temporal chunks directly into the final pitched NV12 surface."""

import dataclasses
from typing import Literal

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def video_vae_temporal_nv12(
    output_frames, has_previous, height, width, pitch, tile_pixels=256, threads=256
):
    import tilelang.language as T

    chunk_frames = 28
    main_frames = 17
    pre_padding = 3
    overlap_start = 23
    overlap_frames = 5
    pairs_per_frame = height * width // 4
    pairs = output_frames * pairs_per_frame
    surface_height = height * 3 // 2

    @T.prim_func
    def main(
        previous: T.Tensor([3, chunk_frames, height, width], T.float32),
        current: T.Tensor([3, chunk_frames, height, width], T.float32),
        nv12: T.Tensor([output_frames, surface_height, pitch], T.uint8),
    ):
        with T.Kernel(T.ceildiv(pairs, tile_pixels), threads=threads) as block:
            for local in T.Parallel(tile_pixels):
                index = block * tile_pixels + local
                if index < pairs:
                    frame = index // pairs_per_frame
                    pair = index % pairs_per_frame
                    pair_width = width // 2
                    y = (pair // pair_width) * 2
                    x = (pair % pair_width) * 2
                    # The short five-frame clip uses the same leading crop, frames 3:8.
                    source_frame = T.if_then_else(
                        frame < main_frames,
                        frame + pre_padding,
                        overlap_start + frame - main_frames,
                    )
                    red_sum = T.alloc_var(T.float32)
                    green_sum = T.alloc_var(T.float32)
                    blue_sum = T.alloc_var(T.float32)
                    red_sum = 0.0
                    green_sum = 0.0
                    blue_sum = 0.0
                    for dy in T.serial(2):
                        for dx in T.serial(2):
                            red_raw = T.alloc_var(T.float32)
                            green_raw = T.alloc_var(T.float32)
                            blue_raw = T.alloc_var(T.float32)
                            red_raw = current[0, source_frame, y + dy, x + dx]
                            green_raw = current[1, source_frame, y + dy, x + dx]
                            blue_raw = current[2, source_frame, y + dy, x + dx]
                            if has_previous:  # noqa: SIM102 -- static specialization
                                if frame < overlap_frames:
                                    weight = T.cast(frame, T.float32) / overlap_frames
                                    red_raw = (
                                        previous[
                                            0, overlap_start + frame, y + dy, x + dx
                                        ]
                                        * (1.0 - weight)
                                        + red_raw * weight
                                    )
                                    green_raw = (
                                        previous[
                                            1, overlap_start + frame, y + dy, x + dx
                                        ]
                                        * (1.0 - weight)
                                        + green_raw * weight
                                    )
                                    blue_raw = (
                                        previous[
                                            2, overlap_start + frame, y + dy, x + dx
                                        ]
                                        * (1.0 - weight)
                                        + blue_raw * weight
                                    )
                            red = T.max(0.0, T.min(1.0, red_raw * 0.229 + 0.485))
                            green = T.max(0.0, T.min(1.0, green_raw * 0.224 + 0.456))
                            blue = T.max(0.0, T.min(1.0, blue_raw * 0.225 + 0.406))
                            luma = T.floor(
                                16.0
                                + 65.481 * red
                                + 128.553 * green
                                + 24.966 * blue
                                + 0.5
                            )
                            nv12[frame, y + dy, x + dx] = T.cast(
                                T.max(16.0, T.min(235.0, luma)), T.uint8
                            )
                            red_sum += red
                            green_sum += green
                            blue_sum += blue
                    red = red_sum * 0.25
                    green = green_sum * 0.25
                    blue = blue_sum * 0.25
                    chroma_y = height + y // 2
                    u = T.floor(
                        128.0 - 37.797 * red - 74.203 * green + 112.0 * blue + 0.5
                    )
                    v = T.floor(
                        128.0 + 112.0 * red - 93.786 * green - 18.214 * blue + 0.5
                    )
                    nv12[frame, chroma_y, x] = T.cast(
                        T.max(16.0, T.min(240.0, u)), T.uint8
                    )
                    nv12[frame, chroma_y, x + 1] = T.cast(
                        T.max(16.0, T.min(240.0, v)), T.uint8
                    )

    return main.with_attr(
        "global_symbol",
        f"video_vae_temporal_nv12_{output_frames}_{int(has_previous)}_{height}_{width}_{pitch}_{tile_pixels}_{threads}",
    )


class VideoVaeTemporalNv12Workload(Workload):
    output_frames: Literal[5, 17, 22]
    has_previous: bool
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    pitch: int = Field(gt=0)


class VideoVaeTemporalNv12Config(Config):
    tile_pixels: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class VideoVaeTemporalNv12Arguments(Arguments):
    previous: TensorDesc | None
    current: TensorDesc
    output: TensorDesc


class VideoVaeTemporalNv12Kernel(
    Kernel[
        VideoVaeTemporalNv12Arguments,
        VideoVaeTemporalNv12Workload,
        VideoVaeTemporalNv12Config,
    ]
):
    name = "video_vae_temporal_nv12"
    program = video_vae_temporal_nv12

    @classmethod
    def make_arguments(
        cls, workload: VideoVaeTemporalNv12Workload
    ) -> VideoVaeTemporalNv12Arguments:
        """Describe decoded chunks and the pitched NV12 output surface."""
        chunk_shape = (3, 28, workload.height, workload.width)
        return VideoVaeTemporalNv12Arguments(
            previous=(
                TensorDesc.empty(DType.F32, chunk_shape)
                if workload.has_previous
                else None
            ),
            current=TensorDesc.empty(DType.F32, chunk_shape),
            output=TensorDesc.empty(
                DType.U8,
                (workload.output_frames, workload.height * 3 // 2, workload.pitch),
            ),
        )

    @classmethod
    def make_config(
        cls, workload: VideoVaeTemporalNv12Workload
    ) -> VideoVaeTemporalNv12Config:
        del workload
        return VideoVaeTemporalNv12Config(tile_pixels=256, threads=256)

    @classmethod
    def make_workload(
        cls, arguments: VideoVaeTemporalNv12Arguments
    ) -> VideoVaeTemporalNv12Workload:
        channels, frames, height, width = arguments.current.shape
        assert arguments.current.dtype == DType.F32
        assert channels == 3 and frames == 28
        if arguments.previous is not None:
            assert arguments.previous.dtype == DType.F32
            assert arguments.previous.shape == arguments.current.shape
        output_frames, surface_height, pitch = arguments.output.shape
        assert arguments.output.dtype == DType.U8
        assert not height % 2 and not width % 2
        assert surface_height == height * 3 // 2 and pitch >= width
        return VideoVaeTemporalNv12Workload(
            output_frames=output_frames,
            has_previous=arguments.previous is not None,
            height=height,
            width=width,
            pitch=pitch,
        )

    @classmethod
    def ref_program(cls, arguments: VideoVaeTemporalNv12Arguments) -> None:
        """Blend decoded chunks and convert visible pixels to limited-range NV12."""
        import torch

        workload = cls.make_workload(arguments)
        previous = (
            arguments.previous.as_torch() if arguments.previous is not None else None
        )
        current = arguments.current.as_torch()
        output = arguments.output.as_torch()
        mean = torch.tensor([0.485, 0.456, 0.406], device=current.device)[
            :, None, None, None
        ]
        std = torch.tensor([0.229, 0.224, 0.225], device=current.device)[
            :, None, None, None
        ]

        def average(channel: torch.Tensor) -> torch.Tensor:
            return (
                channel[:, 0::2, 0::2]
                + channel[:, 0::2, 1::2]
                + channel[:, 1::2, 0::2]
                + channel[:, 1::2, 1::2]
            ) * 0.25

        frames = torch.arange(workload.output_frames, device=current.device)
        source_frames = frames + torch.where(frames < 17, 3, 6)
        value = current[:, source_frames]
        if previous is not None:
            alpha = torch.arange(5, device=current.device) / 5
            value[:, :5] = (
                previous[:, 23:28] * (1 - alpha[None, :, None, None])
                + value[:, :5] * alpha[None, :, None, None]
            )
        red, green, blue = (value * std + mean).clamp(0, 1).unbind(0)
        output[: workload.output_frames, : workload.height, : workload.width] = (
            (16 + 65.481 * red + 128.553 * green + 24.966 * blue + 0.5)
            .floor()
            .clamp(16, 235)
        )
        red, green, blue = (average(channel) for channel in (red, green, blue))
        chroma = output[
            : workload.output_frames, workload.height :, : workload.width
        ]
        chroma[:, :, 0::2] = (
            (128 - 37.797 * red - 74.203 * green + 112 * blue + 0.5)
            .floor()
            .clamp(16, 240)
        )
        chroma[:, :, 1::2] = (
            (128 + 112 * red - 93.786 * green - 18.214 * blue + 0.5)
            .floor()
            .clamp(16, 240)
        )
