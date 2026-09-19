"""Plan H3 patch encoding and its final and deepstack feature mergers."""

import dataclasses
from collections.abc import Sequence
from pathlib import Path

from nano_omni.core.layout import ActivationLayout
from nano_omni.core.model import ModelMetadata, PlannedModel
from nano_omni.core.planning import operators
from nano_omni.core.planning.scheduling import Action, Copy
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import TensorDesc, TensorKind
from nano_omni.core.weights import WeightRegistry
from nano_omni.models.h3.text_encoder import read_h3_text_metadata
from nano_omni.ops.minimax_h3.vision import (
    VisionLayer,
    VisionLayerWeights,
    VisionMerger,
    VisionMergerWeights,
    VisionPatch,
    VisionRope,
)

VISION_WIDTH = 1152
VISION_DIM = 72
DEEPSTACK_LAYERS = (8, 16, 24)


@dataclasses.dataclass(frozen=True, slots=True)
class H3VisionEncoderSpec:
    """Patch-token count (a positive multiple of four) and the fixed 27 layers."""

    tokens: int
    layers: int = 27


@dataclasses.dataclass(frozen=True, slots=True)
class H3VisionEncoderArgs:
    """Caller-owned device inputs and BF16 merged/deepstack output buffers."""

    patches: TensorDesc
    position_indices: TensorDesc
    position_weights: TensorDesc
    rope_positions: TensorDesc
    inverse_frequencies: TensorDesc
    merged: TensorDesc
    deepstack_0: TensorDesc
    deepstack_1: TensorDesc
    deepstack_2: TensorDesc
    runtime: CudaRuntime


@dataclasses.dataclass(slots=True)
class H3VisionEncoder(PlannedModel[H3VisionEncoderSpec, H3VisionEncoderArgs, None]):
    """Encode patches and expose the final merger followed by three deepstacks."""

    @classmethod
    def read_metadata(
        cls, path: Path, additional_weights: Sequence[Path] = ()
    ) -> ModelMetadata:
        """Read the shared text/vision checkpoint's header without tensor data."""
        assert not additional_weights
        return read_h3_text_metadata(path)

    @classmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: H3VisionEncoderSpec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[None, list[Action], int]:
        """Plan all layers and feature mergers; memory capacities are bytes."""
        if spec.tokens <= 0 or spec.tokens % 4 or spec.layers != 27:
            raise ValueError(
                "vision encoder requires a positive multiple of four tokens and 27 layers"
            )
        capacity = cls.workspace_capacity(total_memory * 4 // 5, workspace_limit)
        registry = WeightRegistry(metadata)
        storage = ActivationLayout()
        patches, indices, position_weights, positions, frequencies = range(5)
        hidden, target, rope = 5, 6, 7
        output_ids = (8, 9, 10, 11)
        input_sizes = (
            spec.tokens * 1536 * 2,
            4 * spec.tokens * 4,
            4 * spec.tokens * 4,
            spec.tokens * 2 * 4,
            (VISION_DIM // 4) * 4,
        )
        for identity, size in enumerate(input_sizes):
            storage.require(identity, size)
        storage.require(hidden, spec.tokens * VISION_WIDTH * 2)
        storage.require(target, spec.tokens * VISION_WIDTH * 2)
        storage.require(rope, 2 * spec.tokens * (VISION_DIM // 2) * 4)
        patch_weight = registry.reference("visual.patch_embed.proj.weight")
        ops: list[VisionPatch | VisionRope | VisionLayer | VisionMerger] = [
            VisionPatch(patches, indices, position_weights, tokens=spec.tokens)
            .with_weights(
                (
                    dataclasses.replace(patch_weight, shape=(VISION_WIDTH, 1536)),
                    registry.reference("visual.patch_embed.proj.bias"),
                    registry.reference("visual.pos_embed.weight"),
                )
            )
            .to(hidden),
            VisionRope(positions, frequencies, tokens=spec.tokens).to(rope),
        ]
        # Each checkpoint names its output and normalization placement. The final
        # merger normalizes before merging patches; deepstack mergers normalize after.
        mergers = {
            layer: (
                output_ids[index + 1],
                f"visual.deepstack_merger_list.{index}",
                False,
            )
            for index, layer in enumerate(DEEPSTACK_LAYERS)
        }
        mergers[spec.layers - 1] = (output_ids[0], "visual.merger", True)
        for index in range(spec.layers):
            ops.append(
                VisionLayer(hidden, rope, tokens=spec.tokens)
                .with_weights(vision_layer_weights(registry, index))
                .to(target)
            )
            hidden, target = target, hidden
            if index in mergers:
                output, prefix, pre_merge_norm = mergers[index]
                weights = vision_merger_weights(registry, prefix)
                storage.require(
                    output, (spec.tokens // 4) * weights.fc2_weight.shape[0] * 2
                )
                ops.append(
                    VisionMerger(
                        hidden, tokens=spec.tokens, pre_merge_norm=pre_merge_norm
                    )
                    .with_weights(weights)
                    .to(output)
                )

        commands, stats = operators.commands(ops, storage, registry, capacity)
        offsets = stats["activation_offsets"]
        commands[:0] = [
            Copy(
                TensorDesc.bytes(TensorKind.INPUT, index, 0, size),
                TensorDesc.bytes(TensorKind.WORKSPACE, 0, offsets[index], size),
                "compute",
            )
            for index, size in enumerate(input_sizes)
        ]
        commands.extend(
            Copy(
                TensorDesc.bytes(
                    TensorKind.WORKSPACE,
                    0,
                    offsets[identity],
                    storage.requirements[identity][0],
                ),
                TensorDesc.bytes(
                    TensorKind.OUTPUT,
                    index,
                    0,
                    storage.requirements[identity][0],
                ),
                "compute",
            )
            for index, identity in enumerate(output_ids)
        )
        return None, commands, capacity

    def run(self, args: H3VisionEncoderArgs) -> None:
        """Bind the five patch/position inputs and four merger outputs in order."""
        inputs = (
            args.patches,
            args.position_indices,
            args.position_weights,
            args.rope_positions,
            args.inverse_frequencies,
        )
        outputs = (
            args.merged,
            args.deepstack_0,
            args.deepstack_1,
            args.deepstack_2,
        )
        self.execute(
            args.runtime,
            tuple(
                TensorDesc.from_pointer(tensor.data_ptr(), tensor.num_bytes)
                for tensor in inputs
            ),
            tuple(
                TensorDesc.from_pointer(tensor.data_ptr(), tensor.num_bytes)
                for tensor in outputs
            ),
        )


def vision_layer_weights(registry: WeightRegistry, index: int) -> VisionLayerWeights:
    """Bind a vision block's norms, attention and MLP weights in argument order."""
    prefix = f"visual.blocks.{index}"
    names = (
        "norm1",
        "attn.qkv",
        "attn.proj",
        "norm2",
        "mlp.linear_fc1",
        "mlp.linear_fc2",
    )
    return VisionLayerWeights(
        *(
            registry.reference(f"{prefix}.{name}.{suffix}")
            for name in names
            for suffix in ("weight", "bias")
        )
    )


def vision_merger_weights(registry: WeightRegistry, prefix: str) -> VisionMergerWeights:
    """Bind one merger's normalization and two projection weight/bias pairs."""
    return VisionMergerWeights(
        *(
            registry.reference(f"{prefix}.{name}.{suffix}")
            for name in ("norm", "linear_fc1", "linear_fc2")
            for suffix in ("weight", "bias")
        )
    )
