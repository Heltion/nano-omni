"""INT8 reference-conditioned H3 DiT with explicit reference activations."""

from __future__ import annotations

import dataclasses
import math

from nano_omni.core.model import ModelMetadata
from nano_omni.core.op import Op
from nano_omni.core.planning.scheduling import Action
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.models.h3 import fixed, ref_layout
from nano_omni.models.h3.diffusion import (
    H3_DYNAMIC_ACTIVATION_BASE,
    ConditionPlan,
    H3Diffusion,
    H3DiffusionSpec,
)
from nano_omni.models.h3.ref_layout import ReferenceBlock
from nano_omni.ops.minimax_h3.position_input import PositionInput
from nano_omni.ops.minimax_h3.reference_input import ReferenceInput
from nano_omni.ops.minimax_h3.sampling import Combine


@dataclasses.dataclass(frozen=True, slots=True)
class H3Ref2vaDiffusionSpec(H3DiffusionSpec):
    references: tuple[ReferenceBlock, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class H3Ref2vaDiffusionArgs:
    context: TensorDesc
    output_video: TensorDesc
    output_audio: TensorDesc
    references: tuple[TensorDesc, ...]
    noises: tuple[TensorDesc, ...]
    positions: TensorDesc
    runtime: CudaRuntime


class ReferenceConditionPlan(ConditionPlan):
    def __init__(self, spec: H3Ref2vaDiffusionSpec) -> None:
        self.spec = spec
        self.shapes: list[tuple[int, ...]] = []
        for block in spec.references:
            if block.audio_frames:
                self.shapes.append((32, 2, block.audio_frames))
            if block.frames:
                self.shapes.append((24, block.frames, block.height, block.width))
        base = H3_DYNAMIC_ACTIVATION_BASE
        self.inputs = tuple(base + index for index in range(len(self.shapes)))
        visuals = (
            (identity, shape)
            for identity, shape in zip(self.inputs, self.shapes)
            if len(shape) == 4
        )
        # Each noise binding carries its destination and shape directly.
        self.noises = tuple(
            (identity, base + len(self.inputs) + index, shape)
            for index, (identity, shape) in enumerate(visuals)
        )
        self.position_id = base + len(self.inputs) + len(self.noises)
        self.tokens = sum(block.tokens for block in spec.references)
        self.rows = (
            spec.text_tokens
            + self.tokens
            + 2 * spec.audio_frames
            + spec.video_frames * spec.video_height * spec.video_width // 4
        )
        self.levels = 4
        _, self.segments = ref_layout.positions(
            spec.text_tokens,
            spec.video_frames,
            spec.video_height,
            spec.video_width,
            spec.audio_frames,
            spec.references,
        )

    def bindings(self) -> tuple[tuple[int, int], ...]:
        return (
            *(
                (identity, math.prod(shape) * 4)
                for identity, shape in zip(self.inputs, self.shapes)
            ),
            *((noise, math.prod(shape) * 4) for _, noise, shape in self.noises),
            (self.position_id, self.rows * 3 * 4),
        )

    def prepare(self) -> list[Op]:
        return [
            Combine(
                identity,
                noise,
                identity,
                shape=shape,
                scales=(0.999, 0.001, 0.0),
            ).to(identity)
            for identity, noise, shape in self.noises
        ]

    def input(
        self,
        video: int | tuple[int, int],
        audio: int | tuple[int, int],
        context: int | tuple[int, int],
        *,
        curve_timesteps: tuple[TensorDesc, ...],
    ) -> ReferenceInput:
        return ReferenceInput(
            video,
            audio,
            context,
            *self.inputs,
            spec=self.spec,
            shapes=self.shapes,
            curve_timesteps=curve_timesteps,
        )

    def curve_timesteps(
        self, timesteps: TensorDesc, endpoint: TensorDesc
    ) -> tuple[TensorDesc, ...]:
        return (
            *(timesteps.view((1,), level * DType.F32.itemsize) for level in range(3)),
            endpoint,
        )

    def positions(self) -> PositionInput:
        return PositionInput(self.position_id, rows=self.rows)


class H3Ref2vaDiffusion(H3Diffusion[H3Ref2vaDiffusionSpec, H3Ref2vaDiffusionArgs]):
    @classmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: H3Ref2vaDiffusionSpec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[None, list[Action], int]:
        capacity = cls.workspace_capacity(total_memory * 4 // 5, workspace_limit)
        return fixed.plan(metadata, spec, capacity, ReferenceConditionPlan(spec))

    def run(self, args: H3Ref2vaDiffusionArgs) -> None:
        inputs = (
            args.context,
            args.output_video,
            args.output_audio,
            *args.references,
            *args.noises,
            args.positions,
        )
        self.execute(
            args.runtime,
            tuple(
                TensorDesc.from_pointer(value.data_ptr(), value.num_bytes)
                for value in inputs
            ),
            tuple(
                TensorDesc.from_pointer(value.data_ptr(), value.num_bytes)
                for value in (args.output_video, args.output_audio)
            ),
        )
