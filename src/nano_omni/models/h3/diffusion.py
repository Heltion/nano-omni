"""Shared H3 diffusion metadata, sampling schedule and conditioning contract."""

from __future__ import annotations

import abc
import dataclasses
import math
import struct
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path

from nano_omni.core.layout import ActivationLayout
from nano_omni.core.model import (
    FileMetadata,
    ModelMetadata,
    PlannedModel,
)
from nano_omni.core.op import Op
from nano_omni.core.planning.scheduling import Action, Copy
from nano_omni.core.tensor import DType, TensorDesc, TensorKind

H3_BLOCKS = 50
H3_VIDEO_SHIFT = 12.0
H3_AUDIO_SHIFT = 3.0
H3_VISUAL_CONDITION_TIMESTEP = 0.999


class Act(IntEnum):
    """Stable activation identities shared by every H3 diffusion plan."""

    CONTEXT_INPUT = 0
    CONTEXT = 1
    VIDEO = 2
    AUDIO = 3
    MODEL_AUDIO = 4
    HIDDEN = 5
    CURVE = 6
    MODULATION = 7
    ROPE = 8
    VIDEO_VELOCITY = 9
    AUDIO_VELOCITY = 10
    VIDEO_CLEAN_0 = 11
    VIDEO_CLEAN_1 = 12
    AUDIO_CLEAN_0 = 13
    AUDIO_CLEAN_1 = 14
    KEY = 15
    KEY_SCALE = 16
    VALUE = 17
    VALUE_SCALE = 18


H3_DYNAMIC_ACTIVATION_BASE = max(Act) + 1


@dataclasses.dataclass(frozen=True, slots=True)
class H3DiffusionSpec:
    text_tokens: int = 32
    video_frames: int = 17
    video_height: int = 22
    video_width: int = 38
    audio_frames: int = 93
    steps: int = 20
    seed: int = 0
    text_visual_spans: tuple[tuple[int, int], ...] = ()
    video_shift: float = H3_VIDEO_SHIFT
    audio_shift: float = H3_AUDIO_SHIFT
    attention: str = "dense"
    attention_precision: str = "int8_fp8"
    lora_strength: float = 1.0
    mlp_chunk_tokens: int = 0


def read_h3_diffusion_metadata(path: Path) -> ModelMetadata:
    metadata = ModelMetadata((FileMetadata.read(0, path),))
    validate_h3_diffusion_metadata(metadata)
    return metadata


def validate_h3_diffusion_metadata(metadata: ModelMetadata) -> None:
    weights = metadata.files[0].weights
    assert weights["adaln_t_table"].shape == (1025, 8)
    assert weights["condition_proj.weight"].shape == (5376, 5120)
    for index in range(H3_BLOCKS):
        prefix = f"blocks.{index}"
        expected = {
            f"{prefix}.attn.qkv_proj.weight": (21504, 5376),
            f"{prefix}.attn.out_proj.weight": (5376, 7168),
            f"{prefix}.mlp.fc1.weight": (28672, 5376),
            f"{prefix}.mlp.fc2.weight": (5376, 14336),
        }
        for name, shape in expected.items():
            value = weights[name]
            if value.dtype != DType.I8:
                raise ValueError(f"H3 DiT requires INT8 scaled weight: {name}")
            assert value.shape == shape
            scale = weights[f"{name}_scale"]
            expected_scale = (shape[0], 1)
            assert scale.dtype == DType.F32 and scale.shape == expected_scale


def validate_h3_lora_metadata(metadata: ModelMetadata) -> None:
    prefixes = tuple(f"blocks.{index}" for index in range(H3_BLOCKS)) + tuple(
        f"token_refiner.blocks.{index}" for index in range(2)
    )
    for file in metadata.files[1:]:
        weights = file.weights
        expected_names: set[str] = set()
        with file.path.open("rb") as source:
            for block in prefixes:
                for suffix, input_rows, output_rows in (
                    ("attn.qkv_proj", 5376, 21504),
                    ("attn.out_proj", 7168, 5376),
                    ("mlp.fc1", 5376, 28672),
                    ("mlp.fc2", 14336, 5376),
                ):
                    prefix = f"diffusion_model.{block}.{suffix}"
                    names = tuple(
                        f"{prefix}.{suffix}"
                        for suffix in ("lora_A.weight", "lora_B.weight", "alpha")
                    )
                    expected_names.update(names)
                    down = weights[names[0]]
                    up = weights[names[1]]
                    alpha = weights[names[2]]
                    rank = down.shape[0]
                    assert down.dtype == up.dtype == DType.BF16
                    assert down.shape == (rank, input_rows)
                    assert up.shape == (output_rows, rank)
                    assert alpha.dtype == DType.F32
                    assert alpha.shape == ()
                    source.seek(alpha.info[1])
                    scale = struct.unpack("<f", source.read(4))[0] / rank
                    assert scale in (1.0, 1.0 / 16.0)
        assert set(weights) == expected_names


def shifted_sigma(base: float, shift: float) -> float:
    return shift * base / (1.0 + (shift - 1.0) * base)


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return shifted_sigma(base, to_shift)


def h3_sigmas(steps: int, shift: float = H3_VIDEO_SHIFT) -> tuple[float, ...]:
    assert steps > 0
    return tuple(shifted_sigma(1.0 - step / steps, shift) for step in range(steps)) + (
        0.0,
    )


def res_multistep_scales(
    sigmas: tuple[float, ...], step: int
) -> tuple[float, float, float]:
    sigma = sigmas[step]
    following = sigmas[step + 1]
    if step == 0 or following == 0.0:
        ratio = following / sigma
        return ratio, 1.0 - ratio, 0.0
    previous = sigmas[step - 1]
    h = math.log(sigma / following)
    c2 = math.log(sigma / previous) / h
    phi1 = math.expm1(-h) / -h
    phi2 = (phi1 - 1.0) / -h
    return following / sigma, h * (phi1 - phi2 / c2), h * phi2 / c2


class H3Diffusion[Spec: H3DiffusionSpec, Args](PlannedModel[Spec, Args, None]):
    @classmethod
    def read_metadata(
        cls, path: Path, additional_weights: Sequence[Path] = ()
    ) -> ModelMetadata:
        base = read_h3_diffusion_metadata(path)
        metadata = ModelMetadata(
            (
                *base.files,
                *(
                    FileMetadata.read(index + 1, weight_path)
                    for index, weight_path in enumerate(additional_weights)
                ),
            )
        )
        validate_h3_lora_metadata(metadata)
        return metadata


class ConditionPlan(abc.ABC):
    """Condition-specific operators plus one ordered input-to-activation binding list."""

    tokens: int
    levels: int
    segments: tuple[tuple[int, int, int, int], ...]

    @abc.abstractmethod
    def bindings(self) -> tuple[tuple[int, int], ...]:
        """Return activation IDs and byte sizes in runtime input order."""
        ...

    def declare(self, storage: ActivationLayout) -> None:
        for identity, size in self.bindings():
            storage.require(identity, size)

    def copies(self, offsets: dict[int, int]) -> list[Action]:
        return [
            Copy(
                TensorDesc.bytes(TensorKind.INPUT, index + 3, 0, size),
                TensorDesc.bytes(TensorKind.WORKSPACE, 0, offsets[identity], size),
                "compute",
            )
            for index, (identity, size) in enumerate(self.bindings())
        ]

    @abc.abstractmethod
    def prepare(self) -> list[Op]: ...

    @abc.abstractmethod
    def input(
        self,
        video: int | tuple[int, int],
        audio: int | tuple[int, int],
        context: int | tuple[int, int],
        *,
        curve_timesteps: tuple[TensorDesc, ...],
    ) -> Op[tuple[TensorDesc, TensorDesc, TensorDesc, TensorDesc, TensorDesc]]: ...

    @abc.abstractmethod
    def curve_timesteps(
        self, timesteps: TensorDesc, endpoint: TensorDesc
    ) -> tuple[TensorDesc, ...]:
        """Bind each output curve level to one model-selected scalar."""
        ...

    @abc.abstractmethod
    def positions(self) -> Op[TensorDesc]: ...
