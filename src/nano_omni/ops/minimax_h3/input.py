"""Build the H3 token sequence and timestep curve from latent activations."""

from __future__ import annotations

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
    from nano_omni.models.h3.fl2va_diffusion import H3Fl2vaDiffusionSpec


class Input(Op[tuple[TensorDesc, TensorDesc, TensorDesc, TensorDesc, TensorDesc]]):
    def __init__(
        self,
        video: int | tuple[int, int],
        audio: int | tuple[int, int],
        context: int | tuple[int, int],
        *conditions: int | tuple[int, int],
        spec: H3Fl2vaDiffusionSpec,
        curve_timesteps: tuple[TensorDesc, ...],
    ) -> None:
        self.spec = spec
        self.curve_timesteps = curve_timesteps
        frames, height, width = spec.video_frames, spec.video_height, spec.video_width
        condition_tokens = sum(spec.condition_frames) * height * width // 4
        audio_tokens = spec.audio_frames * 2
        video_tokens = frames * height * width // 4
        inputs = (
            TensorDesc.activation(video, DType.F32, (24, frames, height, width)),
            TensorDesc.activation(audio, DType.F32, (32, 2, spec.audio_frames)),
            TensorDesc.activation(context, DType.BF16, (spec.text_tokens, 5376)),
            *(
                TensorDesc.activation(value, DType.F32, (24, count, height, width))
                for value, count in zip(conditions, spec.condition_frames, strict=True)
            ),
        )
        outputs = (
            (
                DType.BF16,
                (
                    spec.text_tokens + condition_tokens + audio_tokens + video_tokens,
                    5376,
                ),
            ),
            (DType.F32, (3 if condition_tokens else 2, 8)),
        )
        assert len(curve_timesteps) == outputs[1][1][0]
        assert all(
            value.dtype == DType.F32 and value.shape == (1,)
            for value in curve_timesteps
        )
        super().__init__(*inputs, outputs=outputs)

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        spec = self.spec
        output_bindings = self.bound_outputs
        if len(self.inputs) - 3 != len(spec.condition_frames):
            raise ValueError("one activation is required for each condition")
        condition_tokens = (
            sum(spec.condition_frames) * spec.video_height * spec.video_width // 4
        )
        video_weight, video_bias, audio_weight, audio_bias, table = self.bound_weights
        sequence = KernelSequence(scratch)
        frames, height, width = spec.video_frames, spec.video_height, spec.video_width
        video_tokens, audio_tokens = frames * height * width // 4, spec.audio_frames * 2
        video, audio, context = self.inputs[:3]
        hidden, curve = output_bindings
        video_packed = sequence.temporary("video_packed", DType.F32, (video_tokens, 96))
        audio_packed = sequence.temporary("audio_packed", DType.F32, (audio_tokens, 32))
        video_rows = sequence.temporary("video_rows", DType.BF16, (video_tokens, 5376))
        audio_rows = sequence.temporary("audio_rows", DType.BF16, (audio_tokens, 5376))
        prefix = sequence.temporary(
            "prefix",
            DType.BF16,
            (spec.text_tokens + condition_tokens + audio_tokens, 5376),
        )
        sequence.emit(
            H3VideoPatchKernel,
            H3VideoPatchArguments(video, video_packed),
        )
        sequence.emit(
            H3AudioPackKernel,
            H3AudioPackArguments(audio, audio_packed),
        )
        # Preserve video-then-audio projection order and shared video weights.
        for packed, weight, bias, projected in (
            (video_packed, video_weight, video_bias, video_rows),
            (audio_packed, audio_weight, audio_bias, audio_rows),
        ):
            sequence.emit(
                MatmulF32Kernel,
                MatmulF32Arguments(packed, weight, bias, projected, True),
            )
        # Conditions follow text and precede generated audio/video tokens.
        for index, condition_frames in enumerate(spec.condition_frames):
            count = condition_frames * height * width // 4
            condition = self.inputs[3 + index]
            packed = sequence.temporary(
                f"condition_packed_{index}", DType.F32, (count, 96)
            )
            projected = sequence.temporary(
                f"condition_rows_{index}", DType.BF16, (count, 5376)
            )
            sequence.emit(
                H3VideoPatchKernel,
                H3VideoPatchArguments(condition, packed),
            )
            sequence.emit(
                MatmulF32Kernel,
                MatmulF32Arguments(packed, video_weight, video_bias, projected, True),
            )
            joined = sequence.temporary(
                f"condition_prefix_{index}",
                DType.BF16,
                (context.shape[0] + count, 5376),
            )
            sequence.emit(
                ConcatRowsKernel, ConcatRowsArguments(context, projected, joined)
            )
            context = joined
        sequence.emit(
            ConcatRowsKernel, ConcatRowsArguments(context, audio_rows, prefix)
        )
        sequence.emit(ConcatRowsKernel, ConcatRowsArguments(prefix, video_rows, hidden))
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
