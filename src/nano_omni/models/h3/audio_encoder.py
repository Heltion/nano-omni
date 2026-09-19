"""DAC waveform encoder preceding the H3 audio posterior projection."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection

from nano_omni.core.layout import ActivationLayout
from nano_omni.core.model import ModelMetadata, PlannedModel
from nano_omni.core.planning import operators
from nano_omni.core.planning.scheduling import Action, Copy
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import DType, TensorDesc, TensorKind
from nano_omni.core.weights import WeightRegistry
from nano_omni.ops.activation.snake import SnakeOp
from nano_omni.ops.convolution.conv1d import Conv1dOp


@dataclasses.dataclass(frozen=True, slots=True)
class AudioEncoderSpec:
    """Stereo waveform length covering complete 800-sample downsampling periods."""

    samples: int
    stereo: int = 2

    def __post_init__(self) -> None:
        if self.samples <= 0 or self.samples % 800 or self.stereo != 2:
            raise ValueError("stereo audio must be padded to a multiple of 800 samples")


@dataclasses.dataclass(frozen=True, slots=True)
class AudioEncoderArgs:
    waveform: TensorDesc
    output: TensorDesc
    runtime: CudaRuntime


class AudioEncoder(PlannedModel[AudioEncoderSpec, AudioEncoderArgs, None]):
    @classmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: AudioEncoderSpec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[None, list[Action], int]:
        capacity = cls.workspace_capacity(total_memory * 4 // 5, workspace_limit)
        registry = WeightRegistry(metadata)
        storage = ActivationLayout()
        ops: list[Conv1dOp | SnakeOp] = []
        shapes: dict[int, tuple[int, int]] = {}
        cursor = 0

        # Slot zero holds the input before the three-slot rotation begins.
        hidden = 0
        shapes[hidden] = (spec.stereo * spec.samples, 1)
        storage.require(hidden, spec.stereo * spec.samples * 4)

        def allocate(shape: tuple[int, int], excluded: Collection[int | None]) -> int:
            """Advance the ring and reserve the selected slot for its new shape."""
            nonlocal cursor
            for _ in range(3):
                cursor = (cursor + 1) % 3
                if cursor not in excluded:
                    shapes[cursor] = shape
                    storage.require(cursor, shape[0] * shape[1] * 4)
                    return cursor
            raise RuntimeError("audio encoder activation rotation exhausted")

        def conv(
            input_id: int,
            prefix: str,
            *,
            stride: int = 1,
            dilation: int = 1,
            padding: int = 0,
            residual: int | None = None,
        ) -> int:
            weight = registry.reference(prefix + ".weight")
            bias = registry.reference(prefix + ".bias")
            input_shape = shapes[input_id]
            # Rows concatenate stereo channels; each channel is strided independently.
            frames = input_shape[0] // spec.stereo
            kernel_size = weight.shape[2]
            output_frames = (
                frames + 2 * padding - dilation * (kernel_size - 1) - 1
            ) // stride + 1
            output_shape = (spec.stereo * output_frames, weight.shape[0])
            if residual is not None and shapes[residual] != output_shape:
                raise ValueError("Conv1d residual shape mismatch")
            result = allocate(output_shape, {input_id, residual})
            ops.append(
                Conv1dOp(
                    input_id,
                    residual,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    stereo=spec.stereo,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                )
                .with_weights((weight, bias))
                .to(result)
            )
            return result

        def snake(input_id: int, prefix: str) -> int:
            shape = shapes[input_id]
            alpha = registry.reference(prefix + ".alpha")
            alpha = dataclasses.replace(alpha, dtype=DType.F32, shape=(shape[1],))
            result = allocate(shape, {input_id})
            ops.append(
                SnakeOp(input_id, rows=shape[0], columns=shape[1])
                .with_weights(alpha)
                .to(result)
            )
            return result

        hidden = conv(hidden, "encoder.block.0", padding=3)
        for stage, stride in enumerate((2, 4, 4, 5, 5), 1):
            prefix = f"encoder.block.{stage}.block"
            for block, dilation in enumerate((1, 3, 9)):
                unit = f"{prefix}.{block}.block"
                residual = hidden
                hidden = snake(hidden, unit + ".0")
                hidden = conv(
                    hidden, unit + ".1", dilation=dilation, padding=3 * dilation
                )
                hidden = snake(hidden, unit + ".2")
                hidden = conv(hidden, unit + ".3", residual=residual)
            hidden = snake(hidden, prefix + ".3")
            hidden = conv(
                hidden, prefix + ".4", stride=stride, padding=(stride + 1) // 2
            )
        hidden = snake(hidden, "encoder.block.6")
        hidden = conv(hidden, "encoder.block.7", padding=1)

        commands, stats = operators.commands(ops, storage, registry, capacity)
        offsets = stats["activation_offsets"]
        input_bytes = spec.stereo * spec.samples * 4
        output_bytes = shapes[hidden][0] * shapes[hidden][1] * 4
        commands.insert(
            0,
            Copy(
                TensorDesc.bytes(TensorKind.INPUT, 0, 0, input_bytes),
                TensorDesc.bytes(TensorKind.WORKSPACE, 0, offsets[0], input_bytes),
                "compute",
            ),
        )
        commands.append(
            Copy(
                TensorDesc.bytes(
                    TensorKind.WORKSPACE, 0, offsets[hidden], output_bytes
                ),
                TensorDesc.bytes(TensorKind.OUTPUT, 0, 0, output_bytes),
                "compute",
            )
        )
        return None, commands, capacity

    def run(self, args: AudioEncoderArgs) -> None:
        self.execute(
            args.runtime,
            (TensorDesc.from_pointer(args.waveform.data_ptr(), args.waveform.num_bytes),),
            (TensorDesc.from_pointer(args.output.data_ptr(), args.output.num_bytes),),
        )
