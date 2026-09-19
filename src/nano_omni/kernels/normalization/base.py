"""Workload and launch schemas shared by row normalization algorithms."""

from typing import Self

from pydantic import Field, model_validator

from nano_omni.core.kernel import Config, Workload


class RowNormalizationShape(Workload):
    # Rows are independent; reductions exclude padded lanes beyond columns.
    num_tokens: int = Field(gt=0)
    columns: int = Field(gt=0)


class RowNormalizationWorkload(RowNormalizationShape):
    # Add epsilon inside rsqrt, after dividing the row's reduction by columns.
    epsilon: float = Field(gt=0, allow_inf_nan=False)


class RowNormalizationConfig(Config):
    tile_tokens: int = Field(gt=0)
    threads: int = Field(ge=32, le=1024, multiple_of=32)


class GroupNormalizationWorkload(Workload):
    channels: int = Field(gt=0)
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    groups: int = Field(gt=0)
    activate: bool

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        assert not self.channels % self.groups, "channels must be divisible by groups"
        return self


class GroupNormalizationConfig(Config):
    tile_elements: int = Field(gt=0)
    threads: int = Field(ge=32, le=1024, multiple_of=32)
