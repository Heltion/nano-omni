"""Plan H3 text encoding with optional visual and deepstack feature injection."""

import dataclasses
import itertools
from collections.abc import Sequence
from pathlib import Path

from nano_omni.core.layout import ActivationLayout
from nano_omni.core.model import (
    FileMetadata,
    ModelMetadata,
    PlannedModel,
)
from nano_omni.core.op import Op
from nano_omni.core.planning import operators
from nano_omni.core.planning.scheduling import Action, Copy
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import DType, TensorDesc, TensorKind
from nano_omni.core.weights import WeightRegistry
from nano_omni.ops.elementwise.copy import CopyOp
from nano_omni.ops.elementwise.residual import ResidualOp
from nano_omni.ops.minimax_h3.text import (
    Fp4Projection,
    TextEmbedding,
    TextLayer,
    TextLayerWeights,
    TextRope,
)


@dataclasses.dataclass(frozen=True, slots=True)
class H3TextEncoderSpec:
    """Token count, encoded layer count and (start, length) visual token spans."""

    sequence: int
    layers: int = 50
    vision_spans: tuple[tuple[int, int], ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class H3TextEncoderArgs:
    """Caller-owned device buffers; visual inputs are required for visual spans."""

    tokens: TensorDesc
    output: TensorDesc
    runtime: CudaRuntime
    visual: TensorDesc | None = None
    deepstack_0: TensorDesc | None = None
    deepstack_1: TensorDesc | None = None
    deepstack_2: TensorDesc | None = None
    mrope_positions: TensorDesc | None = None
    inverse_frequencies: TensorDesc | None = None


def visual_regions(
    rows: int, spans: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int, int], ...]:
    """Return destination start, source start and length with last-span priority."""
    source_starts: list[int] = []
    source = 0
    boundaries = {0, rows}
    for start, length in spans:
        assert length > 0 and 0 <= start < start + length <= rows
        source_starts.append(source)
        source += length
        boundaries.update((start, start + length))
    result: list[tuple[int, int, int]] = []
    points = sorted(boundaries)
    mapped_spans = tuple(zip(spans, source_starts, strict=True))
    for start, stop in itertools.pairwise(points):
        selected = next(
            (
                (span_start, source_start)
                for (span_start, length), source_start in reversed(mapped_spans)
                if span_start <= start < span_start + length
            ),
            None,
        )
        if selected is None:
            continue
        span_start, source_start = selected
        mapped_source = source_start + start - span_start
        if (
            result
            and result[-1][0] + result[-1][2] == start
            and result[-1][1] + result[-1][2] == mapped_source
        ):
            destination, previous_source, length = result[-1]
            result[-1] = (destination, previous_source, length + stop - start)
        else:
            result.append((start, mapped_source, stop - start))
    return tuple(result)


def read_h3_text_metadata(path: Path) -> ModelMetadata:
    """Read the H3 safetensors header without changing stored dtypes."""
    return ModelMetadata((FileMetadata.read(0, path),))


@dataclasses.dataclass(slots=True)
class H3TextEncoder(PlannedModel[H3TextEncoderSpec, H3TextEncoderArgs, None]):
    """Encode text into 5120-wide BF16 hidden states, without a final norm."""

    @classmethod
    def read_metadata(
        cls, path: Path, additional_weights: Sequence[Path] = ()
    ) -> ModelMetadata:
        """Describe the single text/vision checkpoint without loading its tensors."""
        assert not additional_weights
        return read_h3_text_metadata(path)

    @classmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: H3TextEncoderSpec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[None, list[Action], int]:
        """Plan the requested layers within a workspace budget measured in bytes."""
        assert 0 <= spec.layers <= 50
        sequence = spec.sequence
        rows = (sequence + 127) // 128 * 128
        capacity = cls.workspace_capacity(total_memory * 4 // 5, workspace_limit)
        registry = WeightRegistry(metadata)
        tokens, hidden, target, angles = range(4)
        sizes = {
            tokens: sequence * 4,
            hidden: rows * 5120 * 2,
            target: rows * 5120 * 2,
            angles: sequence * 64 * 4,
        }
        input_ids = [tokens]
        visual_inputs: tuple[int, ...] = ()
        visual_rows = sum(length for _, length in spec.vision_spans)
        ops: list[Op] = []
        regions = visual_regions(rows, spec.vision_spans)
        if spec.vision_spans:
            visual_inputs = (4, 5, 6, 7)
            positions, frequencies = 8, 9
            sizes.update(
                (identity, visual_rows * 5120 * 2) for identity in visual_inputs
            )
            sizes.update({positions: sequence * 3 * 4, frequencies: 64 * 4})
            input_ids.extend((*visual_inputs, positions, frequencies))
            ops.append(
                TextRope(positions, frequencies, sequence=sequence, mrope=True).to(
                    angles
                )
            )
        else:
            ops.append(TextRope(sequence=sequence, mrope=False).to(angles))

        storage = ActivationLayout()
        for identity, size in sizes.items():
            storage.require(identity, size)
        ops.append(
            TextEmbedding(tokens, sequence=sequence, rows=rows, width=5120)
            .with_weights(
                (
                    registry.reference("model.embed_tokens.weight"),
                    registry.reference("model.embed_tokens.weight_scale"),
                )
            )
            .to(hidden)
        )
        # Boundary 0 replaces visual tokens in the embedding; boundaries 1..3
        # add deepstack features after the corresponding encoded layer.
        for boundary in range(spec.layers + 1):
            if boundary:
                ops.append(
                    TextLayer(hidden, angles, rows=rows, sequence=sequence)
                    .with_weights(text_layer_weights(registry, boundary - 1))
                    .to(target)
                )
                hidden, target = target, hidden
            if boundary < len(visual_inputs):
                shape = (rows, 5120)
                ops.append(CopyOp(hidden, dtype=DType.BF16, shape=shape).to(target))
                row_bytes = 5120 * DType.BF16.itemsize
                for destination, source, length in regions:
                    region_shape = (length, 5120)
                    destination_view = (target, destination * row_bytes)
                    source_view = (visual_inputs[boundary], source * row_bytes)
                    op = (
                        ResidualOp(
                            destination_view,
                            source_view,
                            dtype=DType.BF16,
                            shape=region_shape,
                        )
                        if boundary
                        else CopyOp(source_view, dtype=DType.BF16, shape=region_shape)
                    )
                    ops.append(op.to(destination_view))
                hidden, target = target, hidden

        commands, stats = operators.commands(ops, storage, registry, capacity)
        offsets = stats["activation_offsets"]
        commands[:0] = [
            Copy(
                TensorDesc.bytes(TensorKind.INPUT, index, 0, sizes[identity]),
                TensorDesc.bytes(
                    TensorKind.WORKSPACE, 0, offsets[identity], sizes[identity]
                ),
                "compute",
            )
            for index, identity in enumerate(input_ids)
        ]
        commands.append(
            Copy(
                TensorDesc.bytes(
                    TensorKind.WORKSPACE, 0, offsets[hidden], sequence * 5120 * 2
                ),
                TensorDesc.bytes(TensorKind.OUTPUT, 0, 0, sequence * 5120 * 2),
                "compute",
            )
        )
        return None, commands, capacity

    def run(self, args: H3TextEncoderArgs) -> None:
        """Bind token and optional visual buffers in the plan's input order."""
        input_tensors = [args.tokens]
        if self.spec.vision_spans:
            for tensor in (
                args.visual,
                args.deepstack_0,
                args.deepstack_1,
                args.deepstack_2,
                args.mrope_positions,
                args.inverse_frequencies,
            ):
                assert tensor is not None
                input_tensors.append(tensor)
        inputs = tuple(
            TensorDesc.from_pointer(tensor.data_ptr(), tensor.num_bytes)
            for tensor in input_tensors
        )
        outputs = (TensorDesc.from_pointer(args.output.data_ptr(), args.output.num_bytes),)
        self.execute(args.runtime, inputs, outputs)


def text_layer_weights(registry: WeightRegistry, index: int) -> TextLayerWeights:
    """Bind one text layer's norms and FP4 projections in kernel argument order."""
    prefix = f"model.layers.{index}"
    return TextLayerWeights(
        input_norm=registry.reference(f"{prefix}.input_layernorm.weight"),
        query_norm=registry.reference(f"{prefix}.self_attn.q_norm.weight"),
        key_norm=registry.reference(f"{prefix}.self_attn.k_norm.weight"),
        query=projection(registry, f"{prefix}.self_attn.q_proj"),
        key=projection(registry, f"{prefix}.self_attn.k_proj"),
        value=projection(registry, f"{prefix}.self_attn.v_proj"),
        output=projection(registry, f"{prefix}.self_attn.o_proj"),
        post_norm=registry.reference(f"{prefix}.post_attention_layernorm.weight"),
        gate=projection(registry, f"{prefix}.mlp.gate_proj"),
        up=projection(registry, f"{prefix}.mlp.up_proj"),
        down=projection(registry, f"{prefix}.mlp.down_proj"),
    )


def projection(registry: WeightRegistry, prefix: str) -> Fp4Projection:
    """Bind packed weights and scales, preserving optional dynamic input scaling."""
    pre_scale_name = f"{prefix}.pre_quant_scale"
    names = registry.metadata.files[0].weights
    scales = registry.reference(f"{prefix}.weight_scale")
    # Safetensors has no UE4M3 dtype tag, so NVFP4 scales arrive under the
    # signed E4M3 storage tag and are reinterpreted by their model contract.
    scales = dataclasses.replace(scales, dtype=DType.FP8_UE4M3)
    return Fp4Projection(
        registry.reference(f"{prefix}.weight"),
        scales,
        registry.reference(f"{prefix}.weight_scale_2"),
        registry.reference(f"{prefix}.input_scale")
        if f"{prefix}.input_scale" in names
        else None,
        registry.reference(pre_scale_name) if pre_scale_name in names else None,
    )
