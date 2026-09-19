"""Project reference audio/video and target streams into one ordered sequence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.layout.concat_rows import ConcatRowsArguments, ConcatRowsKernel
from nano_omni.kernels.matmul.matmul_f32 import MatmulF32Arguments, MatmulF32Kernel
from nano_omni.kernels.specialized.h3_audio_pack import (
    H3AudioPackArguments,
    H3AudioPackKernel,
)
from nano_omni.kernels.specialized.h3_curve import H3CurveArguments, H3CurveKernel
from nano_omni.kernels.specialized.h3_video_patch import (
    H3VideoPatchArguments,
    H3VideoPatchKernel,
)
from nano_omni.ops.matmul.sequence import KernelSequence

if TYPE_CHECKING:
    from nano_omni.models.h3.diffusion import H3DiffusionSpec


class ReferenceInput(
    Op[tuple[TensorDesc, TensorDesc, TensorDesc, TensorDesc, TensorDesc]]
):
    def __init__(
        self,
        video: int | tuple[int, int],
        audio: int | tuple[int, int],
        context: int | tuple[int, int],
        *references: int | tuple[int, int],
        spec: H3DiffusionSpec,
        shapes: Sequence[tuple[int, ...]],
        curve_timesteps: tuple[TensorDesc, ...],
    ) -> None:
        self.spec, self.shapes = spec, shapes
        self.curve_timesteps = curve_timesteps
        assert len(curve_timesteps) == 4
        assert all(
            value.dtype == DType.F32 and value.shape == (1,)
            for value in curve_timesteps
        )
        stream_shapes = (
            *shapes,
            (32, 2, spec.audio_frames),
            (24, spec.video_frames, spec.video_height, spec.video_width),
        )
        stream_positions = (*references, audio, video)
        inputs = (
            TensorDesc.activation(context, DType.BF16, (spec.text_tokens, 5376)),
            *(
                TensorDesc.activation(position, DType.F32, shape)
                for position, shape in zip(stream_positions, stream_shapes, strict=True)
            ),
        )
        rows = spec.text_tokens + sum(
            math.prod(shape[1:]) // (1 if len(shape) == 3 else 4)
            for shape in stream_shapes
        )
        super().__init__(
            *inputs,
            outputs=((DType.BF16, (rows, 5376)), (DType.F32, (4, 8))),
        )

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        output_bindings = self.bound_outputs
        sequence = KernelSequence(scratch)
        video_weight, video_bias, audio_weight, audio_bias, table = self.bound_weights
        prefix = self.inputs[0]
        # References retain their supplied order, followed by generated audio/video.
        streams = self.inputs[1:]
        for index, source in enumerate(streams):
            shape = source.shape
            is_audio = len(shape) == 3
            channels = shape[0]
            patch_size = 1 if is_audio else 4
            count = math.prod(shape[1:]) // patch_size
            packed = sequence.temporary(
                f"packed_{index}", DType.F32, (count, channels * patch_size)
            )
            if is_audio:
                sequence.emit(
                    H3AudioPackKernel,
                    H3AudioPackArguments(source, packed),
                )
                matrix, bias = audio_weight, audio_bias
            else:
                sequence.emit(
                    H3VideoPatchKernel,
                    H3VideoPatchArguments(source, packed),
                )
                matrix, bias = video_weight, video_bias
            projected = sequence.temporary(
                f"projected_{index}", DType.BF16, (count, 5376)
            )
            sequence.emit(
                MatmulF32Kernel,
                MatmulF32Arguments(packed, matrix, bias, projected, True),
            )
            joined_shape = (prefix.shape[0] + count, 5376)
            joined = (
                output_bindings[0]
                if index == len(streams) - 1
                else sequence.temporary(f"prefix_{index}", DType.BF16, joined_shape)
            )
            sequence.emit(
                ConcatRowsKernel, ConcatRowsArguments(prefix, projected, joined)
            )
            prefix = joined
        curve = output_bindings[1]
        for level, timestep in enumerate(self.curve_timesteps):
            sequence.emit(
                H3CurveKernel,
                H3CurveArguments(
                    table,
                    curve.view((1, 8), level * 8 * DType.F32.itemsize),
                    timestep,
                ),
            )
        return sequence.calls
