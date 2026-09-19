"""H3 FL2VA diffusion; empty keyframe inputs select text-to-video-and-audio."""

from __future__ import annotations

import dataclasses
import math

from nano_omni.core.model import ModelMetadata
from nano_omni.core.op import Op
from nano_omni.core.planning.scheduling import Action
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.models.h3.diffusion import (
    H3_DYNAMIC_ACTIVATION_BASE,
    ConditionPlan,
    H3Diffusion,
    H3DiffusionSpec,
)
from nano_omni.ops.minimax_h3.input import Input
from nano_omni.ops.minimax_h3.positions import PositionRegion, Positions
from nano_omni.ops.minimax_h3.sampling import Combine


@dataclasses.dataclass(frozen=True, slots=True)
class H3Fl2vaDiffusionSpec(H3DiffusionSpec):
    condition_frames: tuple[int, ...] = ()
    condition_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.condition_frames) != len(self.condition_indices):
            raise ValueError("Each keyframe condition requires a frame index")
        if len(self.condition_frames) > 2 or any(n <= 0 for n in self.condition_frames):
            raise ValueError("FL2VA accepts up to two nonempty keyframe conditions")
        frames = sum(1 if index % 5 == 0 else 4 for index in range(self.video_frames))
        if any(index < 0 or index >= frames for index in self.condition_indices):
            raise ValueError("Keyframe index is outside the generated video")


@dataclasses.dataclass(frozen=True, slots=True)
class H3Fl2vaDiffusionArgs:
    """Video/audio buffers contain initial FP32 noise and receive final latents."""

    context: TensorDesc
    output_video: TensorDesc
    output_audio: TensorDesc
    runtime: CudaRuntime
    condition_video: tuple[TensorDesc, ...] = ()
    condition_noise: tuple[TensorDesc, ...] = ()


class KeyframeConditionPlan(ConditionPlan):
    def __init__(self, spec: H3Fl2vaDiffusionSpec) -> None:
        self.spec = spec
        self.shapes = tuple(
            (24, frames, spec.video_height, spec.video_width)
            for frames in spec.condition_frames
        )
        base = H3_DYNAMIC_ACTIVATION_BASE
        self.inputs = tuple(base + i for i in range(len(self.shapes)))
        self.noise = tuple(base + len(self.inputs) + i for i in range(len(self.shapes)))
        self.tokens = (
            sum(spec.condition_frames) * spec.video_height * spec.video_width // 4
        )
        self.levels = 3 if self.tokens else 2
        self.segments = ()

    def bindings(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (identity, math.prod(shape) * 4)
            for identity, shape in zip((*self.inputs, *self.noise), self.shapes * 2)
        )

    def prepare(self) -> list[Op]:
        return [
            Combine(
                identity, noise, identity, shape=shape, scales=(0.999, 0.001, 0.0)
            ).to(identity)
            for identity, noise, shape in zip(self.inputs, self.noise, self.shapes)
        ]

    def input(
        self,
        video: int | tuple[int, int],
        audio: int | tuple[int, int],
        context: int | tuple[int, int],
        *,
        curve_timesteps: tuple[TensorDesc, ...],
    ) -> Input:
        return Input(
            video,
            audio,
            context,
            *self.inputs,
            spec=self.spec,
            curve_timesteps=curve_timesteps,
        )

    def curve_timesteps(
        self, timesteps: TensorDesc, endpoint: TensorDesc
    ) -> tuple[TensorDesc, ...]:
        del endpoint
        levels = 3 if self.tokens else 2
        return tuple(
            timesteps.view((1,), level * DType.F32.itemsize) for level in range(levels)
        )

    def positions(self) -> Positions:
        spec = self.spec
        regions: list[PositionRegion] = []
        start = 0

        regions.append(PositionRegion(start, spec.text_tokens, 0.0, "text"))
        start += spec.text_tokens
        for frames, index in zip(
            spec.condition_frames, spec.condition_indices, strict=True
        ):
            rows = frames * spec.video_height * spec.video_width // 4
            regions.append(
                PositionRegion(
                    start,
                    rows,
                    spec.text_tokens + 5.0 / 3.0 * index,
                    "video",
                    frames,
                    spec.video_height,
                    spec.video_width,
                )
            )
            start += rows
        audio_rows = 2 * spec.audio_frames
        regions.append(
            PositionRegion(
                start,
                audio_rows,
                float(spec.text_tokens),
                "audio",
                spec.audio_frames,
                spec.video_height,
                spec.video_width,
            )
        )
        start += audio_rows
        video_rows = spec.video_frames * spec.video_height * spec.video_width // 4
        regions.append(
            PositionRegion(
                start,
                video_rows,
                float(spec.text_tokens),
                "video",
                spec.video_frames,
                spec.video_height,
                spec.video_width,
            )
        )
        return Positions(tuple(regions), rows=start + video_rows)


@dataclasses.dataclass(slots=True)
class H3Fl2vaDiffusion(H3Diffusion[H3Fl2vaDiffusionSpec, H3Fl2vaDiffusionArgs]):
    @classmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: H3Fl2vaDiffusionSpec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[None, list[Action], int]:
        from nano_omni.models.h3 import fixed

        capacity = cls.workspace_capacity(total_memory * 4 // 5, workspace_limit)
        return fixed.plan(metadata, spec, capacity, KeyframeConditionPlan(spec))

    def run(self, args: H3Fl2vaDiffusionArgs) -> None:
        inputs = tuple(
            TensorDesc.from_pointer(tensor.data_ptr(), tensor.num_bytes)
            for tensor in (
                args.context,
                args.output_video,
                args.output_audio,
                *args.condition_video,
                *args.condition_noise,
            )
        )
        outputs = tuple(
            TensorDesc.from_pointer(tensor.data_ptr(), tensor.num_bytes)
            for tensor in (args.output_video, args.output_audio)
        )
        self.execute(args.runtime, inputs, outputs)
