"""H3 DiT as an explicit activation declaration and ordered operator list."""

from __future__ import annotations

import json
import math
import struct

from nano_omni.core.layout import ActivationLayout, ScalarLayout
from nano_omni.core.model import ModelMetadata
from nano_omni.core.op import Op
from nano_omni.core.planning import operators
from nano_omni.core.planning.scheduling import Action, Copy
from nano_omni.core.tensor import DType, TensorDesc, TensorKind
from nano_omni.core.weights import WeightRegistry
from nano_omni.models.h3 import blocks, diffusion
from nano_omni.models.h3.diffusion import Act, ConditionPlan, H3DiffusionSpec
from nano_omni.ops.minimax_h3.output import Output
from nano_omni.ops.minimax_h3.sampling import Combine, Denoise


def plan(
    metadata: ModelMetadata,
    spec: H3DiffusionSpec,
    capacity: int,
    conditions: ConditionPlan,
) -> tuple[None, list[Action], int]:
    assert spec.attention in ("dense", "sol"), "unsupported H3 attention routing"
    assert spec.attention_precision in ("bf16", "int8_fp8", "nvfp4"), (
        "unsupported H3 attention precision"
    )
    if spec.attention == "dense":
        assert spec.attention_precision in ("bf16", "int8_fp8", "nvfp4")
    if spec.attention == "sol":
        assert spec.attention_precision in ("bf16", "int8_fp8", "nvfp4")
    registry = WeightRegistry(metadata)
    storage = ActivationLayout()
    scalars = ScalarLayout()
    curve_endpoint = scalars.reserve(DType.F32, (1,), struct.pack("<f", 1.0))
    video_shape = (24, spec.video_frames, spec.video_height, spec.video_width)
    audio_shape = (32, 2, spec.audio_frames)
    video_bytes, audio_bytes = math.prod(video_shape) * 4, math.prod(audio_shape) * 4
    condition_tokens = conditions.tokens
    groups = conditions.levels
    fixed_activations = {int(identity) for identity in Act}
    condition_activations = {identity for identity, _ in conditions.bindings()}
    assert fixed_activations.isdisjoint(condition_activations), (
        "condition activations overlap fixed H3 activations"
    )
    conditions.declare(storage)
    rows = (
        spec.text_tokens
        + condition_tokens
        + spec.audio_frames * 2
        + spec.video_frames * spec.video_height * spec.video_width // 4
    )
    for identity in (
        Act.VIDEO,
        Act.VIDEO_VELOCITY,
        Act.VIDEO_CLEAN_0,
        Act.VIDEO_CLEAN_1,
    ):
        storage.require(identity, video_bytes)
    for identity in (
        Act.AUDIO,
        Act.MODEL_AUDIO,
        Act.AUDIO_VELOCITY,
        Act.AUDIO_CLEAN_0,
        Act.AUDIO_CLEAN_1,
    ):
        storage.require(identity, audio_bytes)
    for identity, size in (
        (Act.CONTEXT_INPUT, spec.text_tokens * 5120 * 2),
        (Act.CONTEXT, spec.text_tokens * 5376 * 2),
        (Act.HIDDEN, rows * 5376 * 2),
        (Act.CURVE, groups * 8 * 4),
        (Act.MODULATION, groups * 18 * 5376 * 2),
        (Act.ROPE, 2 * rows * 48 * 4),
    ):
        storage.require(identity, size)
    normalized_activation = max(
        diffusion.H3_DYNAMIC_ACTIVATION_BASE,
        max(condition_activations, default=-1) + 1,
    )
    storage.require(normalized_activation, rows * 5376 * 2)
    if spec.attention == "dense" and spec.attention_precision == "int8_fp8":
        key_rows = -(-rows // 64) * 64
        storage.require(Act.KEY, 56 * key_rows * 128)
        storage.require(Act.KEY_SCALE, 56 * (key_rows // 64) * 4)
        storage.require(Act.VALUE, 56 * 128 * key_rows)
        storage.require(Act.VALUE_SCALE, 56 * 128 * 4)
    attention_slice_heads = 14
    if spec.attention == "dense" and spec.attention_precision == "bf16":
        attention_slice_heads = blocks.dense_bf16_slice_heads(
            rows,
            capacity,
            storage.layout()[1],
        )
        print(f"dense_bf16_slice_heads={attention_slice_heads}", flush=True)
    ops: list[Op] = conditions.prepare()
    ops.extend(
        blocks.context_ops(
            registry,
            Act.CONTEXT_INPUT,
            Act.CONTEXT,
            rows=spec.text_tokens,
            lora_strength=spec.lora_strength,
        )
    )
    ops.append(
        conditions.positions()
        .with_weights(registry.reference("rope.inv_freq"))
        .to(Act.ROPE)
    )
    input_weights = (
        registry.reference("video_patch_proj.weight"),
        registry.reference("video_patch_proj.bias"),
        registry.reference("audio_patch_proj.weight"),
        registry.reference("audio_patch_proj.bias"),
        registry.reference("adaln_t_table"),
    )
    output_weights = (
        registry.reference("final_layer.adaln_proj.linear.weight"),
        registry.reference("final_layer.adaln_proj.linear.bias"),
        registry.reference("final_layer.norm.weight"),
        registry.reference("final_layer.audio_out.weight"),
        registry.reference("final_layer.audio_out.bias"),
        registry.reference("final_layer.video_out.weight"),
        registry.reference("final_layer.video_out.bias"),
    )
    sigmas = diffusion.h3_sigmas(spec.steps, spec.video_shift)
    sampling_activation = max(storage.requirements) + 1
    sampling_data = bytearray(struct.pack(f"<{spec.steps}f", *sigmas[:-1]))
    storage.require(sampling_activation, len(sampling_data))
    old_video, old_audio = Act.VIDEO, Act.AUDIO
    for step, sigma in enumerate(sigmas[:-1]):
        sigma_audio = diffusion.time_shift_sigma(
            sigma, spec.video_shift, spec.audio_shift
        )
        carry = sigma_audio / sigma
        ops.append(
            Combine(
                Act.AUDIO,
                Act.AUDIO,
                Act.AUDIO,
                shape=audio_shape,
                scales=(carry, 0.0, 0.0),
            ).to(Act.MODEL_AUDIO)
        )
        timesteps = TensorDesc.activation(
            (sampling_activation, len(sampling_data)), DType.F32, (3,)
        )
        sampling_data.extend(struct.pack("<3f", 1.0 - sigma, 1.0 - sigma_audio, 0.999))
        ops.append(
            conditions.input(
                Act.VIDEO,
                Act.MODEL_AUDIO,
                Act.CONTEXT,
                curve_timesteps=conditions.curve_timesteps(timesteps, curve_endpoint),
            )
            .with_weights(input_weights)
            .to(Act.HIDDEN, Act.CURVE)
        )
        ops.extend(
            blocks.transformer_ops(
                registry,
                Act.HIDDEN,
                Act.CURVE,
                Act.MODULATION,
                normalized_activation,
                Act.ROPE,
                (Act.ROPE, rows * 48 * 4),
                rows=rows,
                text_tokens=spec.text_tokens,
                audio_tokens=spec.audio_frames * 2,
                condition_tokens=condition_tokens,
                visual_spans=spec.text_visual_spans,
                segments=conditions.segments,
                chunk_tokens=spec.mlp_chunk_tokens or 4096,
                attention=spec.attention,
                attention_precision=spec.attention_precision,
                attention_slice_heads=attention_slice_heads,
                kv_activations=(Act.KEY, Act.KEY_SCALE, Act.VALUE, Act.VALUE_SCALE),
                lora_strength=spec.lora_strength,
            )
        )
        ops.append(
            Output(
                Act.HIDDEN,
                Act.CURVE,
                text_tokens=spec.text_tokens,
                video_frames=spec.video_frames,
                height=spec.video_height,
                width=spec.video_width,
                audio_frames=spec.audio_frames,
                condition_tokens=condition_tokens,
            )
            .with_weights(output_weights)
            .to(Act.VIDEO_VELOCITY, Act.AUDIO_VELOCITY)
        )
        ops.append(
            Combine(
                Act.VIDEO_VELOCITY,
                Act.VIDEO_VELOCITY,
                Act.VIDEO_VELOCITY,
                shape=video_shape,
                scales=(-1.0, 0.0, 0.0),
            ).to(Act.VIDEO_VELOCITY)
        )
        ops.append(
            Combine(
                Act.AUDIO,
                Act.AUDIO_VELOCITY,
                Act.AUDIO,
                shape=audio_shape,
                scales=(
                    (1.0 - spec.video_shift / spec.audio_shift) * carry,
                    -(1.0 + (spec.video_shift / spec.audio_shift - 1.0) * sigma_audio),
                    0.0,
                ),
            ).to(Act.AUDIO_VELOCITY)
        )
        clean_video = Act.VIDEO_CLEAN_0 if step % 2 == 0 else Act.VIDEO_CLEAN_1
        clean_audio = Act.AUDIO_CLEAN_0 if step % 2 == 0 else Act.AUDIO_CLEAN_1
        ops.extend(
            [
                Denoise(
                    Act.VIDEO,
                    Act.VIDEO_VELOCITY,
                    shape=video_shape,
                    sigma=(sampling_activation, step * 4),
                ).to(clean_video),
                Denoise(
                    Act.AUDIO,
                    Act.AUDIO_VELOCITY,
                    shape=audio_shape,
                    sigma=(sampling_activation, step * 4),
                ).to(clean_audio),
            ]
        )
        scales = diffusion.res_multistep_scales(sigmas, step)
        ops.extend(
            [
                Combine(
                    Act.VIDEO, clean_video, old_video, shape=video_shape, scales=scales
                ).to(Act.VIDEO),
                Combine(
                    Act.AUDIO, clean_audio, old_audio, shape=audio_shape, scales=scales
                ).to(Act.AUDIO),
            ]
        )
        old_video, old_audio = clean_video, clean_audio
    ops.append(
        Combine(
            Act.AUDIO,
            Act.AUDIO,
            Act.AUDIO,
            shape=audio_shape,
            scales=(spec.audio_shift / spec.video_shift, 0.0, 0.0),
        ).to(Act.AUDIO)
    )
    for op in ops:
        if isinstance(op, Combine):
            assert isinstance(op.scales, tuple), "sampling scales must be bound once"
            values = struct.pack("<5f", *op.scales, 1.0, 0.0)
            op.scales = (sampling_activation, len(sampling_data))
            sampling_data.extend(values)
    storage.require(sampling_activation, len(sampling_data))
    result, stats = operators.commands(ops, storage, registry, capacity, scalars)
    offsets = stats["activation_offsets"]
    # Preserve the original prefix order: constants, conditions, audio, video, context.
    # Runtime inputs 1/2 own initial noise until the final output copies overwrite it.
    result[0:0] = [
        Copy(
            bytes(sampling_data),
            TensorDesc.bytes(
                TensorKind.WORKSPACE,
                0,
                offsets[sampling_activation],
                len(sampling_data),
            ),
            "compute",
        ),
        *conditions.copies(offsets),
        Copy(
            TensorDesc.bytes(TensorKind.INPUT, 2, 0, audio_bytes),
            TensorDesc.bytes(TensorKind.WORKSPACE, 0, offsets[Act.AUDIO], audio_bytes),
            "compute",
        ),
        Copy(
            TensorDesc.bytes(TensorKind.INPUT, 1, 0, video_bytes),
            TensorDesc.bytes(TensorKind.WORKSPACE, 0, offsets[Act.VIDEO], video_bytes),
            "compute",
        ),
        Copy(
            TensorDesc.bytes(TensorKind.INPUT, 0, 0, spec.text_tokens * 5120 * 2),
            TensorDesc.bytes(
                TensorKind.WORKSPACE,
                0,
                offsets[Act.CONTEXT_INPUT],
                spec.text_tokens * 5120 * 2,
            ),
            "compute",
        ),
    ]
    result.extend(
        [
            Copy(
                TensorDesc.bytes(
                    TensorKind.WORKSPACE, 0, offsets[Act.VIDEO], video_bytes
                ),
                TensorDesc.bytes(TensorKind.OUTPUT, 0, 0, video_bytes),
                "compute",
            ),
            Copy(
                TensorDesc.bytes(
                    TensorKind.WORKSPACE, 0, offsets[Act.AUDIO], audio_bytes
                ),
                TensorDesc.bytes(TensorKind.OUTPUT, 1, 0, audio_bytes),
                "compute",
            ),
        ]
    )
    print("fixed_plan=" + json.dumps(stats), flush=True)
    return None, result, capacity
