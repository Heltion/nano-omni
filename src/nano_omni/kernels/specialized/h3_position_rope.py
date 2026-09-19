"""Generate one H3 position region directly as cosine and sine planes."""

import dataclasses
import math
from typing import Literal

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc

type PositionKind = Literal["text", "audio", "video"]


class H3PositionRopeWorkload(Workload):
    kind: PositionKind
    num_tokens: int
    frames: int = 0
    height: int = 0
    width: int = 0
    frequencies: int = 16


class H3PositionRopeConfig(Config):
    tile_elements: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class H3PositionRopeArguments(Arguments):
    base: float
    height: int
    width: int
    inverse_frequencies: TensorDesc
    cosines: TensorDesc
    sines: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int | float], ...]:
        """Pass the token extent and position base as runtime scalars."""
        return (
            ("num_tokens", math.prod(self.cosines.shape[:-1])),
            ("base", self.base),
        )


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def h3_position_rope(
    kind,
    base,
    num_tokens,
    frames,
    height,
    width,
    frequencies,
    tile_elements=256,
    threads=256,
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    dynamic_base = T.dynamic("base", "float32")
    columns = 3 * frequencies
    num_elements = dynamic_num_tokens * columns
    if kind == "video":
        patch_height, patch_width = height // 2, width // 2
        area = (height * width) ** 0.5
        height_ratio, width_ratio = height / area, width / area
        height_start = (1.0 - height_ratio) * 16.0
        width_start = (1.0 - width_ratio) * 16.0
        height_step = height_ratio / patch_height * 32.0
        width_step = width_ratio / patch_width * 32.0
        audio_start = audio_end = 0.0
    elif kind == "audio":
        ratio = width / (height * width) ** 0.5
        audio_start = (1.0 - ratio) * 16.0
        audio_end = audio_start + (width / 2.0 - 1.0) * ratio / (width / 2.0) * 32.0
        patch_height = patch_width = 0
        height_start = width_start = height_step = width_step = 0.0
    else:
        patch_height = patch_width = 0
        height_start = width_start = height_step = width_step = 0.0
        audio_start = audio_end = 0.0

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        base: dynamic_base,
        inverse: T.Tensor([frequencies], T.float32),
        cosines: T.Tensor([dynamic_num_tokens, columns], T.float32),
        sines: T.Tensor([dynamic_num_tokens, columns], T.float32),
    ):
        with T.Kernel(T.ceildiv(num_elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < num_elements:
                    token = index // columns
                    column = index % columns
                    axis = column // frequencies
                    frequency = column % frequencies
                    position = T.alloc_var(T.float32)
                    position = 0.0
                    if kind == "text":
                        if axis == 0:
                            position = base + token
                    elif kind == "audio":
                        if axis == 0:
                            position = base + token % frames
                        elif axis == 2:
                            position = T.if_then_else(
                                token < frames,
                                audio_start,
                                audio_end,
                            )
                    else:
                        frame_area = patch_height * patch_width
                        frame = token // frame_area
                        spatial = token % frame_area
                        if axis == 0:
                            groups, remainder = frame // 5, frame % 5
                            position = (
                                base
                                + groups * 85.0 / 3.0
                                + T.if_then_else(
                                    remainder == 0,
                                    0.0,
                                    5.0 / 3.0 + (remainder - 1) * 20.0 / 3.0,
                                )
                            )
                        elif axis == 1:
                            position = (
                                height_start + spatial // patch_width * height_step
                            )
                        else:
                            position = width_start + spatial % patch_width * width_step
                    angle = position * inverse[frequency]
                    cosines[token, column] = T.cos(angle)
                    sines[token, column] = T.sin(angle)

    return main.with_attr(
        "global_symbol",
        f"h3_position_rope_{kind}_{frames}_{height}_{width}_"
        f"{frequencies}_{tile_elements}_{threads}",
    )


class H3PositionRopeKernel(
    Kernel[H3PositionRopeArguments, H3PositionRopeWorkload, H3PositionRopeConfig]
):
    name = "h3_position_rope"
    program = h3_position_rope

    @classmethod
    def make_arguments(
        cls, workload: H3PositionRopeWorkload
    ) -> H3PositionRopeArguments:
        shape: tuple[int, ...]
        if workload.kind == "text":
            shape = (workload.num_tokens, 3 * workload.frequencies)
        elif workload.kind == "audio":
            assert workload.num_tokens == 2 * workload.frames
            shape = (2, workload.frames, 3 * workload.frequencies)
        else:
            assert (
                workload.num_tokens
                == workload.frames * workload.height * workload.width // 4
            )
            shape = (
                workload.frames,
                workload.height // 2,
                workload.width // 2,
                3 * workload.frequencies,
            )
        return H3PositionRopeArguments(
            base=0.0,
            height=workload.height,
            width=workload.width,
            inverse_frequencies=TensorDesc.empty(DType.F32, (workload.frequencies,)),
            cosines=TensorDesc.empty(DType.F32, shape),
            sines=TensorDesc.empty(DType.F32, shape),
        )

    @classmethod
    def make_workload(
        cls, arguments: H3PositionRopeArguments
    ) -> H3PositionRopeWorkload:
        assert arguments.inverse_frequencies.dtype == DType.F32
        assert len(arguments.inverse_frequencies.shape) == 1
        assert arguments.cosines.dtype == arguments.sines.dtype == DType.F32
        assert arguments.cosines.shape == arguments.sines.shape
        shape = arguments.cosines.shape
        frequencies = arguments.inverse_frequencies.shape[0]
        assert shape[-1] == 3 * frequencies
        if len(shape) == 2:
            kind: PositionKind = "text"
            num_tokens, frames, height, width = shape[0], 0, 0, 0
        elif len(shape) == 3:
            assert shape[0] == 2
            kind = "audio"
            frames, height, width = shape[1], arguments.height, arguments.width
            num_tokens = 2 * frames
        else:
            assert len(shape) == 4
            kind = "video"
            frames, height, width = shape[0], shape[1] * 2, shape[2] * 2
            num_tokens = frames * shape[1] * shape[2]
        return H3PositionRopeWorkload(
            kind=kind,
            num_tokens=num_tokens,
            frames=frames,
            height=height,
            width=width,
            frequencies=frequencies,
        )

    @classmethod
    def make_config(cls, workload: H3PositionRopeWorkload) -> H3PositionRopeConfig:
        del workload
        return H3PositionRopeConfig(tile_elements=256, threads=256)

    @classmethod
    def ref_program(cls, arguments: H3PositionRopeArguments) -> None:
        """Build logical coordinates and apply the shared inverse frequencies."""
        import math

        import torch

        workload = cls.make_workload(arguments)
        base = arguments.base
        device = arguments.inverse_frequencies.as_torch().device
        positions = torch.zeros(
            (workload.num_tokens, 3), device=device, dtype=torch.float32
        )
        if workload.kind == "text":
            positions[:, 0] = base + torch.arange(workload.num_tokens, device=device)
        elif workload.kind == "audio":
            spatial_height, spatial_width = workload.height, workload.width
            ratio = spatial_width / math.sqrt(spatial_height * spatial_width)
            patch_width = spatial_width / 2
            start = (1 - ratio) * 16
            step = ratio / patch_width * 32
            positions[:, 0] = (
                torch.arange(workload.frames, device=positions.device) + base
            ).repeat(2)
            positions[: workload.frames, 2] = start
            positions[workload.frames :, 2] = start + (patch_width - 1) * step
        else:
            height, width = workload.height // 2, workload.width // 2
            area = math.sqrt(workload.height * workload.width)
            axes = tuple(
                (
                    torch.arange(size, device=device) * (extent / area / size)
                    + (1 - extent / area) / 2
                )
                * 32
                for size, extent in ((height, workload.height), (width, workload.width))
            )
            yy, xx = torch.meshgrid(*axes, indexing="ij")
            spatial = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
            intervals = positions.new_tensor(
                [5 / 3, 20 / 3, 20 / 3, 20 / 3, 20 / 3]
            ).repeat((workload.frames + 4) // 5)
            times = torch.cat(
                (positions.new_zeros(1), intervals[: workload.frames - 1].cumsum(0))
            )
            result = positions.view(workload.frames, height * width, 3)
            result[:, :, 0] = (times + base)[:, None]
            result[:, :, 1:] = spatial
        angles = positions[:, :, None] * arguments.inverse_frequencies.as_torch()
        arguments.cosines.as_torch().view(workload.num_tokens, -1).copy_(
            angles.cos().flatten(1)
        )
        arguments.sines.as_torch().view(workload.num_tokens, -1).copy_(
            angles.sin().flatten(1)
        )
