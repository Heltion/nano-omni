"""Plan one or more H3 video VAE encoder volumes with shared weights."""

import dataclasses
from collections.abc import Sequence
from pathlib import Path

from nano_omni.core.layout import ActivationLayout, aligned
from nano_omni.core.model import (
    FileMetadata,
    ModelMetadata,
    PlannedModel,
)
from nano_omni.core.planning import operators
from nano_omni.core.planning.scheduling import Action
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import DType, TensorDesc, TensorKind
from nano_omni.core.weights import WeightRegistry
from nano_omni.ops.minimax_h3.video_vae import VideoVaeEncode

VIDEO_VAE_MEMORY_GRANULARITY = 64 << 20
VIDEO_VAE_MEMORY_RESERVE = 1 << 30


@dataclasses.dataclass(frozen=True, slots=True)
class H3VideoVaeEncoderSpec:
    """One input volume shape and its repetitions under a shared weight plan."""

    shape: tuple[int, int, int]
    count: int = 1


@dataclasses.dataclass(frozen=True, slots=True)
class H3VideoVaeEncoderArgs:
    video: TensorDesc
    output: TensorDesc
    runtime: CudaRuntime


@dataclasses.dataclass(slots=True)
class H3VideoVaeEncoder(
    PlannedModel[H3VideoVaeEncoderSpec, H3VideoVaeEncoderArgs, None]
):
    @classmethod
    def read_metadata(
        cls, path: Path, additional_weights: Sequence[Path] = ()
    ) -> ModelMetadata:
        """Read checkpoint tensor shapes, dtypes, and byte ranges."""
        assert not additional_weights
        return ModelMetadata((FileMetadata.read(0, path),))

    @classmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: H3VideoVaeEncoderSpec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[None, list[Action], int]:
        """Reserve 1 GiB and round the remaining workspace down to 64 MiB."""
        available = cls.workspace_capacity(
            total_memory - VIDEO_VAE_MEMORY_RESERVE, workspace_limit
        )
        if available <= 0:
            raise MemoryError("H3 Video VAE encoder requires a 1 GiB reserve")
        capacity = available - available % VIDEO_VAE_MEMORY_GRANULARITY
        if capacity == 0:
            raise MemoryError("H3 Video VAE encoder has no allocatable VRAM")
        commands, required = build_encoder_plan(metadata, spec, capacity)
        workspace_nbytes = aligned(required, VIDEO_VAE_MEMORY_GRANULARITY)
        return (
            None,
            commands,
            workspace_nbytes,
        )

    def run(self, args: H3VideoVaeEncoderArgs) -> None:
        """Execute using caller-owned F32 video and normalized-latent buffers."""
        input = TensorDesc.from_pointer(args.video.data_ptr(), args.video.num_bytes)
        output = TensorDesc.from_pointer(args.output.data_ptr(), args.output.num_bytes)
        self.execute(args.runtime, (input,), (output,))


def encoder_weight_refs(registry: WeightRegistry) -> dict[str, TensorDesc]:
    """Bind encoder, quantization, and latent-statistic weights in existing order."""
    # Preserve set iteration here: registry IDs follow first-reference order.
    names = {
        name
        for name in registry.metadata.files[0].weights
        if name.startswith(("encoder.", "quant_conv."))
        or name in ("latents_mean", "latents_std")
    }
    return {name: registry.reference(name) for name in names}


def build_encoder_plan(
    metadata: ModelMetadata, spec: H3VideoVaeEncoderSpec, capacity: int
) -> tuple[list[Action], int]:
    """Lower repeated encoder Ops into one global weight plan."""
    registry = WeightRegistry(metadata)
    weights = encoder_weight_refs(registry)
    storage = ActivationLayout()
    assert spec.count > 0, "video VAE encoder requires at least one volume"
    ops = []
    input_offset = 0
    output_offset = 0
    frames, height, width = spec.shape
    assert frames > 0 and height % 16 == 0 and width % 16 == 0
    for _ in range(spec.count):
        input_shape = (3, frames, height, width)
        output_shape = (24, (frames + 3) // 4, height // 16, width // 16)
        input = TensorDesc(DType.F32, input_shape, TensorKind.INPUT, (0, input_offset))
        output = TensorDesc(
            DType.F32, output_shape, TensorKind.OUTPUT, (0, output_offset)
        )
        ops.append(VideoVaeEncode(input).with_weights(weights).to(output))
        input_offset += input.num_bytes
        output_offset += output.num_bytes
    commands, statistics = operators.commands(ops, storage, registry, capacity)
    return commands, statistics["workspace_bytes"]
