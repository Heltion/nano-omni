"""Build the H3 video VAE rotary table and suffix identity entries."""

import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class VideoVaeRopeWorkload(Workload):
    frames: int
    height: int
    width: int
    suffix_tokens: int
    frequencies: int


class VideoVaeRopeConfig(Config):
    tile_elements: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class VideoVaeRopeArguments(Arguments):
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def video_vae_rope(
    frames, height, width, suffix_tokens, frequencies, tile_elements=256, threads=128
):
    import tilelang.language as T

    patch_tokens = frames * height * width
    tokens = patch_tokens + suffix_tokens
    columns = frequencies * 3
    elements = tokens * columns

    @T.prim_func
    def main(
        output: T.Tensor([2, tokens, columns], T.float32),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    token = index // columns
                    column = index % columns
                    if token < patch_tokens:
                        spatial = token % (height * width)
                        axis = column // frequencies
                        coordinate = T.if_then_else(
                            axis == 0,
                            token // (height * width),
                            T.if_then_else(
                                axis == 1,
                                spatial // width,
                                spatial % width,
                            ),
                        )
                        size = T.if_then_else(
                            axis == 0,
                            frames,
                            T.if_then_else(
                                axis == 1,
                                height,
                                width,
                            ),
                        )
                        position = 2.0 * (coordinate + 0.5) / size - 1.0
                        frequency = column % frequencies
                        inverse = T.exp2(
                            -frequency * (math.log2(100.0) / frequencies)
                        )
                        angle = 2.0 * math.pi * position * inverse
                        output[0, token, column] = T.cos(angle)
                        output[1, token, column] = T.sin(angle)
                    else:
                        output[0, token, column] = 1.0
                        output[1, token, column] = 0.0

    return main.with_attr(
        "global_symbol",
        "video_vae_rope_"
        f"{frames}_{height}_{width}_{suffix_tokens}_{frequencies}_"
        f"{tile_elements}_{threads}",
    )


class VideoVaeRopeKernel(
    Kernel[VideoVaeRopeArguments, VideoVaeRopeWorkload, VideoVaeRopeConfig]
):
    """Generate the one rotary table shape used by the H3 video VAE."""

    name = "video_vae_rope"
    program = video_vae_rope

    @classmethod
    def make_arguments(cls, workload: VideoVaeRopeWorkload) -> VideoVaeRopeArguments:
        """Describe the generated rotary planes."""
        tokens = (
            workload.frames * workload.height * workload.width + workload.suffix_tokens
        )
        return VideoVaeRopeArguments(
            TensorDesc.empty(DType.F32, (2, tokens, workload.frequencies * 3))
        )

    @classmethod
    def make_workload(cls, arguments: VideoVaeRopeArguments) -> VideoVaeRopeWorkload:
        assert arguments.output.dtype == DType.F32
        assert arguments.output.shape == (2, 1797, 24), (
            "H3 video VAE RoPE requires the 7x16x16 patch grid and five suffix tokens"
        )
        return VideoVaeRopeWorkload(
            frames=7, height=16, width=16, suffix_tokens=5, frequencies=8
        )

    @classmethod
    def make_config(
        cls, workload: VideoVaeRopeWorkload
    ) -> VideoVaeRopeConfig:
        del workload
        return VideoVaeRopeConfig(tile_elements=512, threads=256)

    @classmethod
    def ref_program(cls, arguments: VideoVaeRopeArguments) -> None:
        """Generate normalized frame/height/width angles and identity suffix rows."""
        import torch

        workload = cls.make_workload(arguments)
        output = arguments.output.as_torch()
        coordinates = torch.cartesian_prod(
            *(
                torch.arange(size, device=output.device)
                for size in (workload.frames, workload.height, workload.width)
            )
        ).float()
        sizes = coordinates.new_tensor(
            (workload.frames, workload.height, workload.width)
        )
        positions = 2 * (coordinates + 0.5) / sizes - 1
        inverse = torch.pow(
            100.0,
            -torch.arange(workload.frequencies, device=output.device)
            / workload.frequencies,
        )
        angles = (2 * math.pi * positions[:, :, None] * inverse).flatten(1)
        output[:, : len(coordinates)].copy_(torch.stack((angles.cos(), angles.sin())))
        output[0, len(coordinates) :].fill_(1)
        output[1, len(coordinates) :].zero_()
