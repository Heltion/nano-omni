"""H3 output normalization, projections and audio/video unpacking."""

from __future__ import annotations

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.matmul_f16 import MatmulF16Arguments, MatmulF16Kernel
from nano_omni.kernels.matmul.matmul_f32 import MatmulF32Arguments, MatmulF32Kernel
from nano_omni.kernels.normalization.adaptive_rms_norm import (
    AdaptiveRmsNormArguments,
    AdaptiveRmsNormKernel,
)
from nano_omni.kernels.specialized.h3_audio_pack import (
    H3AudioPackArguments,
    H3AudioPackKernel,
)
from nano_omni.kernels.specialized.h3_video_patch import (
    H3VideoPatchArguments,
    H3VideoPatchKernel,
)
from nano_omni.ops.matmul.sequence import KernelSequence


class Output(
    Op[
        tuple[
            TensorDesc,
            TensorDesc,
            TensorDesc,
            TensorDesc,
            TensorDesc,
            TensorDesc,
            TensorDesc,
        ]
    ]
):
    def __init__(
        self,
        hidden: int | tuple[int, int],
        curve: int | tuple[int, int],
        *,
        text_tokens: int,
        video_frames: int,
        height: int,
        width: int,
        audio_frames: int,
        condition_tokens: int = 0,
    ) -> None:
        self.text_tokens = text_tokens
        self.video_frames, self.height, self.width = video_frames, height, width
        self.audio_frames = audio_frames
        self.condition_tokens = condition_tokens
        audio_tokens = audio_frames * 2
        video_tokens = video_frames * height * width // 4
        rows = text_tokens + condition_tokens + audio_tokens + video_tokens
        groups = 3 if condition_tokens else 2
        inputs = (
            TensorDesc.activation(hidden, DType.BF16, (rows, 5376)),
            TensorDesc.activation(curve, DType.F32, (groups, 8)),
        )
        outputs = (
            (DType.F32, (24, video_frames, height, width)),
            (DType.F32, (32, 2, audio_frames)),
        )
        super().__init__(*inputs, outputs=outputs)

    def kernels(self, scratch: ScratchLayout) -> list[Kernel]:
        output_bindings = self.bound_outputs
        assert len(output_bindings) == 2, "output head requires video and audio outputs"
        (
            modulation_weight,
            modulation_bias,
            norm,
            audio_weight,
            audio_bias,
            video_weight,
            video_bias,
        ) = self.bound_weights
        audio_tokens = self.audio_frames * 2
        video_tokens = self.video_frames * self.height * self.width // 4
        groups = 3 if self.condition_tokens else 2
        hidden, curve = self.inputs
        video, audio = output_bindings
        sequence = KernelSequence(scratch)
        modulation = sequence.temporary("modulation", DType.BF16, (groups, 2 * 5376))
        sequence.emit(
            MatmulF16Kernel,
            MatmulF16Arguments(
                curve,
                modulation_weight,
                modulation_bias,
                None,
                None,
                None,
                modulation,
                False,
            ),
        )
        modulation = modulation.view((groups, 2, 5376))
        normalized = sequence.temporary(
            "normalized", DType.F32, (audio_tokens + video_tokens, 5376)
        )
        prefix_rows = self.text_tokens + self.condition_tokens
        audio_input = normalized.view((audio_tokens, 5376))
        video_input = normalized.view(
            (video_tokens, 5376),
            audio_tokens * 5376 * DType.F32.itemsize,
        )
        for timestep, rows, source, destination in (
            (
                1,
                audio_tokens,
                hidden.view(
                    (audio_tokens, 5376),
                    prefix_rows * 5376 * DType.BF16.itemsize,
                ),
                audio_input,
            ),
            (
                0,
                video_tokens,
                hidden.view(
                    (video_tokens, 5376),
                    (prefix_rows + audio_tokens) * 5376 * DType.BF16.itemsize,
                ),
                video_input,
            ),
        ):
            assert source.shape == (rows, 5376)
            sequence.emit(
                AdaptiveRmsNormKernel,
                AdaptiveRmsNormArguments(
                    source,
                    norm,
                    modulation.view((5376,), timestep * 2 * 5376 * DType.BF16.itemsize),
                    modulation.view(
                        (5376,), (timestep * 2 + 1) * 5376 * DType.BF16.itemsize
                    ),
                    destination,
                ),
            )
        audio_rows = sequence.temporary("audio_rows", DType.F32, (audio_tokens, 32))
        video_rows = sequence.temporary("video_rows", DType.F32, (video_tokens, 96))
        for source, weight, bias, projected in (
            (audio_input, audio_weight, audio_bias, audio_rows),
            (video_input, video_weight, video_bias, video_rows),
        ):
            sequence.emit(
                MatmulF32Kernel,
                MatmulF32Arguments(source, weight, bias, projected, False),
            )
        sequence.emit(
            H3VideoPatchKernel,
            H3VideoPatchArguments(video_rows, video),
        )
        sequence.emit(
            H3AudioPackKernel,
            H3AudioPackArguments(audio_rows, audio),
        )
        return sequence.calls
