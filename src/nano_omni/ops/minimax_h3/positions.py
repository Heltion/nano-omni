"""Generate model-selected H3 position regions directly as rotary planes."""

from __future__ import annotations

import dataclasses
import itertools

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.specialized.h3_position_rope import (
    H3PositionRopeArguments,
    H3PositionRopeKernel,
    PositionKind,
)


@dataclasses.dataclass(frozen=True, slots=True)
class PositionRegion:
    """One contiguous output slice whose shape selects its coordinate algorithm."""

    start: int
    rows: int
    base: float
    kind: PositionKind
    frames: int = 0
    height: int = 0
    width: int = 0

    @property
    def output_shape(self) -> tuple[int, ...]:
        if self.kind == "text":
            return self.rows, 48
        if self.kind == "audio":
            assert self.rows == 2 * self.frames
            return 2, self.frames, 48
        assert self.height % 2 == 0 and self.width % 2 == 0
        assert self.rows == self.frames * self.height * self.width // 4
        return self.frames, self.height // 2, self.width // 2, 48


class Positions(Op[TensorDesc]):
    def __init__(self, regions: tuple[PositionRegion, ...], *, rows: int) -> None:
        assert regions and regions[0].start == 0
        assert all(region.rows > 0 for region in regions)
        assert all(
            first.start + first.rows == second.start
            for first, second in itertools.pairwise(regions)
        )
        assert regions[-1].start + regions[-1].rows == rows
        self.regions = regions
        self.rows = rows
        super().__init__(outputs=((DType.F32, (2, rows, 48)),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel, ...]:
        del scratch
        (output,) = self.bound_outputs
        plane_bytes = self.rows * 48 * DType.F32.itemsize
        return tuple(
            H3PositionRopeKernel(
                H3PositionRopeArguments(
                    base=region.base,
                    height=region.height,
                    width=region.width,
                    inverse_frequencies=self.bound_weights,
                    cosines=output.view(
                        region.output_shape,
                        region.start * 48 * DType.F32.itemsize,
                    ),
                    sines=output.view(
                        region.output_shape,
                        plane_bytes + region.start * 48 * DType.F32.itemsize,
                    ),
                )
            )
            for region in self.regions
        )
