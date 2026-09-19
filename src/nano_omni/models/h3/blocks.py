"""Model-owned H3 block connections and token-slice expansion."""

from __future__ import annotations

import dataclasses
import itertools

from nano_omni.core.layout import aligned
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.core.weights import WeightRegistry
from nano_omni.models.h3.diffusion import H3_BLOCKS
from nano_omni.ops.elementwise.residual_gate import ResidualGate
from nano_omni.ops.minimax_h3.attention import AttentionWeights
from nano_omni.ops.minimax_h3.mlp import MLP, MLPWeights, Projection
from nano_omni.ops.normalization.adaptive_rms_norm import AdaptiveRmsNorm


@dataclasses.dataclass(frozen=True, slots=True)
class TokenRegion:
    """One contiguous token interval with a single modulation selection."""

    start: int
    length: int
    modality: int
    timestep: int


def token_regions(
    rows: int,
    text_tokens: int,
    condition_tokens: int,
    audio_tokens: int,
    visual_spans: tuple[tuple[int, int], ...],
    segments: tuple[tuple[int, int, int, int], ...],
) -> tuple[TokenRegion, ...]:
    """Normalize ordered H3 route overrides into disjoint contiguous regions."""
    assert rows > 0, "token sequence must be nonempty"
    assert 0 <= text_tokens <= rows, "text region exceeds the token sequence"
    assert condition_tokens >= 0, "condition token count must be nonnegative"
    assert audio_tokens >= 0, "audio token count must be nonnegative"
    assert text_tokens + condition_tokens + audio_tokens <= rows, (
        "base token regions exceed the token sequence"
    )
    for start, length in visual_spans:
        assert length > 0, "visual spans must be nonempty"
        assert 0 <= start < start + length <= rows, (
            "visual span exceeds the token sequence"
        )
    for start, length, modality, timestep in segments:
        assert length > 0, "route segments must be nonempty"
        assert 0 <= start < start + length <= rows, (
            "route segment exceeds the token sequence"
        )
        assert 0 <= modality < 3, "route modality is outside the modulation table"
        assert timestep >= 0, "route timestep must be nonnegative"
    boundaries = {0, rows, text_tokens, text_tokens + condition_tokens}
    boundaries.add(text_tokens + condition_tokens + audio_tokens)
    for start, length in (*visual_spans, *(item[:2] for item in segments)):
        boundaries.add(start)
        boundaries.add(start + length)
    points = sorted(boundaries)
    regions: list[TokenRegion] = []
    condition_end = text_tokens + condition_tokens
    audio_end = condition_end + audio_tokens
    for start, end in itertools.pairwise(points):
        if start == end:
            continue
        if start < text_tokens:
            modality, timestep = 1, 0
        elif start < condition_end:
            modality, timestep = 0, 2
        elif start < audio_end:
            modality, timestep = 2, 1
        else:
            modality, timestep = 0, 0
        for span_start, span_length in visual_spans:
            if span_start <= start < span_start + span_length:
                modality = 0
        for (
            segment_start,
            segment_length,
            segment_modality,
            segment_timestep,
        ) in segments:
            if segment_start <= start < segment_start + segment_length:
                modality, timestep = segment_modality, segment_timestep
        region = TokenRegion(start, end - start, modality, timestep)
        if (
            regions
            and regions[-1].start + regions[-1].length == start
            and regions[-1].modality == modality
            and regions[-1].timestep == timestep
        ):
            previous = regions.pop()
            region = dataclasses.replace(previous, length=previous.length + end - start)
        regions.append(region)
    assert regions and regions[0].start == 0, "token regions must start at zero"
    assert all(region.length > 0 for region in regions), (
        "token regions must be nonempty"
    )
    assert all(
        first.start + first.length == second.start
        for first, second in itertools.pairwise(regions)
    ), "token regions must be contiguous"
    assert regions[-1].start + regions[-1].length == rows, (
        "token regions must cover every token"
    )
    return tuple(regions)


def modulation_view(
    modulation: TensorDesc, region: TokenRegion, phase: int
) -> TensorDesc:
    """Select one contiguous 5376-column modulation vector for a token region."""
    row = region.timestep * 18 + region.modality * 6 + phase
    return modulation.view((5376,), row * 5376 * DType.BF16.itemsize)


def projection(
    registry: WeightRegistry,
    prefix: str,
    strength: float,
    *,
    part: int = 0,
    parts: int = 1,
) -> Projection:
    """Bind a checkpoint row slice with its quantization and LoRA metadata."""
    weight = registry.reference(prefix + ".weight", part=part, parts=parts)
    if weight.dtype not in (DType.I8, DType.BF16):
        raise ValueError(f"Unsupported H3 projection weight dtype: {weight.dtype}")
    scale: float | TensorDesc = 1.0
    rotation_group = 0
    if weight.dtype == DType.I8:
        quantization = registry.quantization(prefix)
        if quantization.get("format") != "int8_tensorwise":
            raise ValueError(f"Unsupported INT8 format for {prefix}")
        if quantization.get("convrot", False):
            group_size = quantization.get("convrot_groupsize", 256)
            if not isinstance(group_size, (int, float, str)):
                raise TypeError(f"Invalid ConvRot group size for {prefix}")
            rotation_group = int(group_size)
        scale = registry.reference(prefix + ".weight_scale", part=part, parts=parts)
    input_scale = None
    loras: list[tuple[TensorDesc, TensorDesc, float]] = []
    target = "diffusion_model." + prefix
    for index, file in enumerate(registry.metadata.files[1:], 1):
        down_name, up_name = target + ".lora_A.weight", target + ".lora_B.weight"
        if down_name not in file.weights or up_name not in file.weights:
            continue
        down = registry.reference(down_name, file_index=index)
        up = registry.reference(up_name, file_index=index, part=part, parts=parts)
        alpha = registry.scalar(target + ".alpha", file_index=index)
        loras.append((down, up, strength * alpha / down.shape[0]))
    return Projection(weight, scale, input_scale, tuple(loras), rotation_group)


def _attention_weights(
    registry: WeightRegistry, prefix: str, strength: float
) -> AttentionWeights:
    # Register Q/K/V first to preserve weight IDs and allocation order.
    query, key, value = (
        projection(registry, prefix + ".attn.qkv_proj", strength, part=part, parts=3)
        for part in range(3)
    )
    return AttentionWeights(
        registry.reference(prefix + ".norm1.weight"),
        registry.reference(prefix + ".attn.q_norm.weight"),
        registry.reference(prefix + ".attn.k_norm.weight"),
        query,
        key,
        value,
        projection(registry, prefix + ".attn.out_proj", strength),
    )


def dense_bf16_slice_heads(
    rows: int,
    capacity: int,
    activation_bytes: int,
) -> int:
    """Choose the widest head slice from a closed-form scratch estimate."""
    # The head-independent projection, LoRA, and full attended-output regions use
    # 81,672 bytes per token. Q/K/V add three BF16 vectors per selected head.
    fixed_bytes_per_row = 81_672
    bytes_per_head_row = 3 * 128 * DType.BF16.itemsize
    resident_weight_floor = max(2 << 30, capacity // 4)
    scratch_budget = (
        capacity - activation_bytes - resident_weight_floor
    ) // 256 * 256
    fixed_scratch = rows * fixed_bytes_per_row
    max_heads = (scratch_budget - fixed_scratch) // (rows * bytes_per_head_row)
    legal_heads = (1, 2, 4, 7, 8, 14, 28, 56)
    slice_heads = max((heads for heads in legal_heads if heads <= max_heads), default=0)
    if slice_heads == 0:
        raise MemoryError("dense BF16 attention does not fit with a one-head slice")
    scratch_bytes = aligned(
        rows * (fixed_bytes_per_row + slice_heads * bytes_per_head_row), 256
    )
    assert activation_bytes + scratch_bytes + resident_weight_floor <= capacity
    return slice_heads


def _mlp_weights(registry: WeightRegistry, prefix: str, strength: float) -> MLPWeights:
    return MLPWeights(
        registry.reference(prefix + ".norm2.weight"),
        projection(registry, prefix + ".mlp.fc1", strength, part=0, parts=2),
        projection(registry, prefix + ".mlp.fc1", strength, part=1, parts=2),
        projection(registry, prefix + ".mlp.fc2", strength),
    )


def block_ops(
    registry: WeightRegistry,
    index: int,
    hidden: int | tuple[int, int],
    modulation: int | tuple[int, int],
    normalized: int | tuple[int, int],
    cosines: int | tuple[int, int],
    sines: int | tuple[int, int],
    *,
    rows: int,
    text_tokens: int,
    audio_tokens: int,
    chunk_tokens: int = 4096,
    lora_strength: float = 1.0,
    condition_tokens: int = 0,
    visual_spans: tuple[tuple[int, int], ...] = (),
    segments: tuple[tuple[int, int, int, int], ...] = (),
    attention: str = "dense",
    attention_precision: str = "int8_fp8",
    attention_slice_heads: int = 14,
    kv_activations: tuple[
        int | tuple[int, int],
        int | tuple[int, int],
        int | tuple[int, int],
        int | tuple[int, int],
    ],
) -> list[Op]:
    """Expand one block into attention and MLP ops with in-place token slices.

    KV bindings enable query slicing. Activation offsets are byte offsets.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    source = (hidden, 0) if isinstance(hidden, int) else hidden
    levels = max(
        3 if condition_tokens else 2,
        max((segment[3] + 1 for segment in segments), default=0),
    )
    hidden_tensor = TensorDesc.activation(source, DType.BF16, (rows, 5376))
    normalized_tensor = TensorDesc.activation(normalized, DType.BF16, (rows, 5376))
    modulation_tensor = TensorDesc.activation(
        modulation, DType.BF16, (levels, 18, 5376)
    )
    regions = token_regions(
        rows,
        text_tokens,
        condition_tokens,
        audio_tokens,
        visual_spans,
        segments,
    )
    # Query and MLP slices share the same byte offsets and tail length.
    chunks = tuple(
        (
            start,
            min(chunk_tokens, rows - start),
            (source[0], source[1] + start * 5376 * 2),
        )
        for start in range(0, rows, chunk_tokens)
    )
    prefix = f"blocks.{index}"
    attention_weights = _attention_weights(registry, prefix, lora_strength)
    from nano_omni.ops.minimax_h3.attention_slices import (
        PrepareKeyValue,
        QuerySlice,
    )

    ops: list[Op] = []
    for region in regions:
        byte_offset = region.start * 5376 * DType.BF16.itemsize
        shape = (region.length, 5376)
        hidden_slice = hidden_tensor.view(shape, byte_offset)
        normalized_slice = normalized_tensor.view(shape, byte_offset)
        ops.append(
            AdaptiveRmsNorm(
                hidden_slice,
                modulation_view(modulation_tensor, region, 0),
                modulation_view(modulation_tensor, region, 1),
            )
            .with_weights(attention_weights.norm)
            .to(normalized_slice)
        )
    cosine_position = (cosines, 0) if isinstance(cosines, int) else cosines
    sine_position = (sines, 0) if isinstance(sines, int) else sines
    cosine_tensor = TensorDesc.activation(cosine_position, DType.F32, (rows, 48))
    sine_tensor = TensorDesc.activation(sine_position, DType.F32, (rows, 48))
    key: TensorDesc | None = None
    key_scale: TensorDesc | None = None
    value: TensorDesc | None = None
    value_scale: TensorDesc | None = None
    if attention == "dense" and attention_precision == "bf16":
        from nano_omni.ops.minimax_h3.dense_bf16_attention import DenseBf16Attention

        ops.append(
            DenseBf16Attention(
                normalized_tensor,
                cosine_tensor,
                sine_tensor,
                slice_heads=attention_slice_heads,
            )
            .with_weights(attention_weights)
            .to(normalized_tensor)
        )
    elif attention == "dense" and attention_precision == "nvfp4":
        from nano_omni.ops.minimax_h3.dense_nvfp4_attention import (
            DenseNvfp4Attention,
        )

        ops.append(
            DenseNvfp4Attention(normalized_tensor, cosine_tensor, sine_tensor)
            .with_weights(attention_weights)
            .to(normalized_tensor)
        )
    elif attention == "sol":
        if attention_precision == "bf16":
            from nano_omni.ops.minimax_h3.sol_attention import SolAttention

            attention_op = SolAttention(
                normalized_tensor,
                cosine_tensor,
                sine_tensor,
                tau=1.0,
                protected_tokens=text_tokens + condition_tokens,
            )
        elif attention_precision == "int8_fp8":
            from nano_omni.ops.minimax_h3.sol_int8_fp8_attention import (
                SolInt8Fp8Attention,
            )

            attention_op = SolInt8Fp8Attention(
                normalized_tensor,
                cosine_tensor,
                sine_tensor,
                tau=1.0,
                protected_tokens=text_tokens + condition_tokens,
            )
        else:
            assert attention_precision == "nvfp4", "unsupported Sol attention precision"
            from nano_omni.ops.minimax_h3.sol_nvfp4_attention import (
                SolNvfp4Attention,
            )

            attention_op = SolNvfp4Attention(
                normalized_tensor,
                cosine_tensor,
                sine_tensor,
                tau=1.0,
                protected_tokens=text_tokens + condition_tokens,
            )
        ops.append(attention_op.with_weights(attention_weights).to(normalized_tensor))
    elif attention == "dense" and attention_precision == "int8_fp8":
        key_position, key_scale_position, value_position, value_scale_position = (
            kv_activations
        )
        padded_rows = -(-rows // 64) * 64
        key = TensorDesc.activation(key_position, DType.I8, (56, padded_rows, 128))
        key_scale = TensorDesc.activation(
            key_scale_position, DType.F32, (56, padded_rows // 64)
        )
        value = TensorDesc.activation(
            value_position, DType.FP8_E4M3, (56, 128, padded_rows)
        )
        value_scale = TensorDesc.activation(value_scale_position, DType.F32, (56, 128))
        ops.append(
            PrepareKeyValue(normalized_tensor, cosine_tensor, sine_tensor)
            .with_weights(attention_weights)
            .to(key, key_scale, value, value_scale)
        )
    for start, count, _ in chunks:
        byte_offset = start * 5376 * DType.BF16.itemsize
        shape = (count, 5376)
        normalized_slice = normalized_tensor.view(shape, byte_offset)
        if attention == "dense" and attention_precision == "int8_fp8":
            assert key is not None and key_scale is not None
            assert value is not None and value_scale is not None
            ops.append(
                QuerySlice(
                    normalized_slice,
                    cosine_tensor.view((count, 48), start * 48 * DType.F32.itemsize),
                    sine_tensor.view((count, 48), start * 48 * DType.F32.itemsize),
                    key,
                    key_scale,
                    value,
                    value_scale,
                    total_tokens=rows,
                )
                .with_weights(attention_weights)
                .to(normalized_slice)
            )
        stop = start + count
        for region in regions:
            region_start = max(start, region.start)
            region_stop = min(stop, region.start + region.length)
            if region_start >= region_stop:
                continue
            region_offset = region_start * 5376 * DType.BF16.itemsize
            region_shape = (region_stop - region_start, 5376)
            ops.append(
                ResidualGate(
                    hidden_tensor.view(region_shape, region_offset),
                    normalized_tensor.view(region_shape, region_offset),
                    modulation_view(modulation_tensor, region, 2),
                ).to(hidden_tensor.view(region_shape, region_offset))
            )
    mlp_weights = _mlp_weights(registry, prefix, lora_strength)
    for region in regions:
        end = region.start + region.length
        start = region.start
        while start < end:
            stop = min(end, (start // chunk_tokens + 1) * chunk_tokens)
            count = stop - start
            hidden_slice = hidden_tensor.view(
                (count, 5376), start * 5376 * DType.BF16.itemsize
            )
            ops.append(
                MLP(
                    hidden_slice,
                    modulation_view(modulation_tensor, region, 3),
                    modulation_view(modulation_tensor, region, 4),
                    modulation_view(modulation_tensor, region, 5),
                )
                .with_weights(mlp_weights)
                .to(hidden_slice)
            )
            start = stop
    return ops


def transformer_ops(
    registry: WeightRegistry,
    hidden: int | tuple[int, int],
    curve: int | tuple[int, int],
    modulation: int | tuple[int, int],
    normalized: int | tuple[int, int],
    cosines: int | tuple[int, int],
    sines: int | tuple[int, int],
    *,
    rows: int,
    text_tokens: int,
    audio_tokens: int,
    chunk_tokens: int = 4096,
    lora_strength: float = 1.0,
    condition_tokens: int = 0,
    visual_spans: tuple[tuple[int, int], ...] = (),
    segments: tuple[tuple[int, int, int, int], ...] = (),
    attention: str = "dense",
    attention_precision: str = "int8_fp8",
    attention_slice_heads: int = 14,
    kv_activations: tuple[
        int | tuple[int, int],
        int | tuple[int, int],
        int | tuple[int, int],
        int | tuple[int, int],
    ],
) -> list[Op]:
    """Emit modulation and block ops in checkpoint layer order."""
    from nano_omni.ops.minimax_h3.modulation import Modulation

    ops: list[Op] = []
    modulation_rows = max(
        3 if condition_tokens else 2,
        max((segment[3] + 1 for segment in segments), default=0),
    )
    for index in range(H3_BLOCKS):
        prefix = f"blocks.{index}.adaln_proj.linear"
        ops.append(
            Modulation(
                curve,
                rows=modulation_rows,
            )
            .with_weights(
                (
                    registry.reference(prefix + ".weight"),
                    registry.reference(prefix + ".bias"),
                )
            )
            .to(modulation)
        )
        ops.extend(
            block_ops(
                registry,
                index,
                hidden,
                modulation,
                normalized,
                cosines,
                sines,
                rows=rows,
                text_tokens=text_tokens,
                audio_tokens=audio_tokens,
                chunk_tokens=chunk_tokens,
                lora_strength=lora_strength,
                condition_tokens=condition_tokens,
                visual_spans=visual_spans,
                segments=segments,
                attention=attention,
                attention_precision=attention_precision,
                attention_slice_heads=attention_slice_heads,
                kv_activations=kv_activations,
            )
        )
    return ops


def context_ops(
    registry: WeightRegistry,
    input: int | tuple[int, int],
    hidden: int | tuple[int, int],
    *,
    rows: int,
    chunk_tokens: int = 4096,
    lora_strength: float = 1.0,
) -> list[Op]:
    """Build context projection and refinement with token-sliced MLPs."""
    from nano_omni.ops.minimax_h3.refiner import (
        ContextProjection,
        Normalize,
        RefinerAttention,
        RefinerMLP,
    )

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    activation = (hidden, 0) if isinstance(hidden, int) else hidden
    chunks = tuple(
        (
            (activation[0], activation[1] + start * 5376 * 2),
            min(chunk_tokens, rows - start),
        )
        for start in range(0, rows, chunk_tokens)
    )
    ops: list[Op] = [
        ContextProjection(input, rows=rows)
        .with_weights(
            (
                registry.reference("condition_proj.weight"),
                registry.reference("condition_proj.bias"),
            )
        )
        .to(hidden)
    ]
    for index in range(2):
        prefix = f"token_refiner.blocks.{index}"
        attention_weights = _attention_weights(registry, prefix, lora_strength)
        ops.append(
            RefinerAttention(hidden, rows=rows)
            .with_weights(attention_weights)
            .to(hidden)
        )
        mlp_weights = _mlp_weights(registry, prefix, lora_strength)
        for chunk, count in chunks:
            ops.append(
                RefinerMLP(chunk, rows=count).with_weights(mlp_weights).to(chunk)
            )
    ops.append(
        Normalize(hidden, rows=rows)
        .with_weights(registry.reference("token_refiner.final_norm.weight"))
        .to(hidden)
    )
    return ops
