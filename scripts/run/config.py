from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from nano_omni.core.runtime.synchronization import Synchronization

PATH_FIELDS = (
    "prompt_file", "hardware", "first_frame", "last_frame", "reference_video",
    "reference_audio", "latent_output", "latent_reference", "quality_reference",
    "output",
)


def relative_path(value: str | Path) -> Path:
    """Normalize a repository-relative path on either host platform."""
    if not isinstance(value, (str, Path)):
        raise TypeError("paths must be text")
    text = str(value).replace("\\", "/")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text or text.startswith("~") or posix.is_absolute() or windows.anchor
        or ".." in posix.parts or ":" in text or "\x00" in text
        or posix == PurePosixPath(".")
    ):
        raise ValueError("paths must be repository-relative without parent traversal")
    return Path(*posix.parts)


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    pipeline: Literal["h3_fl2va", "h3_ref2va"]
    prompt: str | None = None
    prompt_file: Path | None = None
    expected_tokens: int | None = Field(default=None, gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    seconds: float = Field(default=4.0, gt=0, allow_inf_nan=False)
    steps: int = Field(gt=0)
    seed: int = 0
    lora: str | None = None
    lora_strength: float = Field(default=1.0, allow_inf_nan=False)
    video_shift: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    audio_shift: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    attention: Literal["dense", "sol"] = "dense"
    attention_precision: Literal["bf16", "int8_fp8", "nvfp4"] = "int8_fp8"
    mlp_chunk_tokens: int = Field(default=0, ge=0)
    working_set_gib: int = Field(default=24, ge=8, le=24)
    pinned_gib: int = Field(default=4, ge=2, le=8)
    maximum_gpu_temperature_celsius: int | None = Field(default=None, gt=0)
    synchronization: Synchronization = Field(default_factory=Synchronization)
    hardware: Path | None = None
    workspace_gib: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    first_frame: Path | None = None
    last_frame: Path | None = None
    reference_video: Path | None = None
    reference_audio: Path | None = None
    latent_output: Path | None = None
    latent_reference: Path | None = None
    quality_reference: Path | None = None
    output: Path | None = None

    @field_validator(*PATH_FIELDS, mode="before")
    @classmethod
    def validate_paths(cls, value: str | Path | None) -> str | None:
        return None if value is None else relative_path(value).as_posix()

    @field_serializer(*PATH_FIELDS, when_used="json")
    def serialize_paths(self, value: Path | None) -> str | None:
        return None if value is None else value.as_posix()

    @field_validator("lora")
    @classmethod
    def validate_lora_path(cls, value: str | None) -> str | None:
        if value in (None, "none", "turbo_4step", "turbo_8step"):
            return value
        return relative_path(value).as_posix()

    @model_validator(mode="after")
    def validate_inputs(self) -> "RunConfig":
        from nano_omni.models import h3

        if (self.prompt is None) == (self.prompt_file is None):
            raise ValueError("provide exactly one of prompt and prompt_file")
        if self.last_frame is not None and self.first_frame is None:
            raise ValueError("last_frame requires first_frame")
        if self.pipeline == "h3_ref2va":
            if self.reference_video is None or self.reference_audio is None:
                raise ValueError("REF2VA requires reference_video and reference_audio")
            if self.first_frame is not None or self.last_frame is not None:
                raise ValueError("Keyframes belong to the FL2VA pipeline")
        elif self.reference_video is not None or self.reference_audio is not None:
            raise ValueError("Reference media belongs to the REF2VA pipeline")
        if self.latent_reference is not None and self.latent_output is None:
            raise ValueError("latent_reference requires latent_output")
        if self.width < 256 or self.width % 32 or self.height < 256 or self.height % 32:
            raise ValueError(
                "MiniMax H3 dimensions must be at least 256 and divisible by 32"
            )
        if h3.requested_frame_count(self.seconds) < 5:
            raise ValueError("MiniMax H3 requires at least five requested frames")
        if self.pipeline == "h3_fl2va" and self.lora == "turbo_4step" and (
            self.width != 1344 or self.height != 768 or self.steps != 4
        ):
            raise ValueError("FL2VA turbo_4step requires 1344x768 and four steps")
        if self.lora == "turbo_8step" and self.steps != 8:
            raise ValueError("turbo_8step requires eight steps")
        return self

    @property
    def workspace_nbytes(self) -> int | None:
        return (
            None if self.workspace_gib is None else int(self.workspace_gib * (1 << 30))
        )

    def read_prompt(self, root: Path) -> str:
        if self.prompt is not None:
            return self.prompt
        assert self.prompt_file is not None
        return (root / self.prompt_file).read_text(encoding="utf-8")

    def lora_value(self, root: Path) -> str | Path:
        if self.lora in (None, "none", "turbo_4step", "turbo_8step"):
            return self.lora or "none"
        return root / self.lora


def load(path: Path, overrides: dict[str, object] | None = None) -> RunConfig:
    """Paths in configuration are relative to the repository root, not the YAML."""
    values = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(values, dict):
        raise TypeError("configuration must be a YAML mapping")
    if overrides:
        values.update(overrides)
        if "prompt" in overrides and "prompt_file" not in overrides:
            values.pop("prompt_file", None)
        if "prompt_file" in overrides and "prompt" not in overrides:
            values.pop("prompt", None)
    return RunConfig.model_validate(values)


def save(config: RunConfig, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
