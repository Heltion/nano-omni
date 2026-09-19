"""Causal attention posterior mean for reference audio latents."""

from __future__ import annotations

import dataclasses

from nano_omni.core.layout import ActivationLayout
from nano_omni.core.model import ModelMetadata, PlannedModel
from nano_omni.core.op import Op
from nano_omni.core.planning import operators
from nano_omni.core.planning.scheduling import Action, Copy
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import DType, TensorDesc, TensorKind
from nano_omni.core.weights import WeightRegistry
from nano_omni.ops.activation.gelu_multiply import GeluMultiplyOp
from nano_omni.ops.attention.causal_attention_f32 import CausalAttentionF32Op
from nano_omni.ops.attention.head_average import HeadAverageOp
from nano_omni.ops.elementwise.channel_normalize_transpose import (
    ChannelNormalizeTransposeOp,
)
from nano_omni.ops.elementwise.residual import ResidualOp
from nano_omni.ops.layout.concat_rows import ConcatRowsOp
from nano_omni.ops.matmul.matmul_f32 import MatmulF32Op
from nano_omni.ops.normalization.layer_norm import LayerNormOp


@dataclasses.dataclass(frozen=True, slots=True)
class AudioPosteriorSpec:
    frames: int


@dataclasses.dataclass(frozen=True, slots=True)
class AudioPosteriorArgs:
    hidden: TensorDesc
    output: TensorDesc
    runtime: CudaRuntime


class AudioPosterior(PlannedModel[AudioPosteriorSpec, AudioPosteriorArgs, None]):
    @classmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: AudioPosteriorSpec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[None, list[Action], int]:
        capacity = cls.workspace_capacity(total_memory * 4 // 5, workspace_limit)
        registry = WeightRegistry(metadata)
        storage = ActivationLayout()
        ops: list[Op] = []
        shapes: dict[int, tuple[int, int]] = {}
        next_id = 0

        def allocate(shape: tuple[int, int]) -> int:
            nonlocal next_id
            identity = next_id
            next_id += 1
            shapes[identity] = shape
            storage.require(identity, 4 * shape[0] * shape[1])
            return identity

        hidden = allocate((2 * spec.frames, 2048))

        def norm(value: int, prefix: str) -> int:
            result = allocate(shapes[value])
            ops.append(
                LayerNormOp(value, dtype=DType.F32, shape=shapes[value], epsilon=1e-5)
                .with_weights(
                    (
                        registry.reference(prefix + ".weight"),
                        registry.reference(prefix + ".bias"),
                    )
                )
                .to(result)
            )
            return result

        def linear(
            value: int,
            prefix: str,
            *,
            matrix: TensorDesc | None = None,
            bias: TensorDesc | None = None,
        ) -> int:
            input_shape = shapes[value]
            matrix = matrix or registry.reference(prefix + ".weight")
            bias = bias or registry.reference(prefix + ".bias")
            columns = matrix.shape[0]
            result = allocate((input_shape[0], columns))
            ops.append(
                MatmulF32Op(
                    value,
                    rows=input_shape[0],
                    input_columns=input_shape[1],
                    output_columns=columns,
                )
                .with_weights((matrix, bias))
                .to(result)
            )
            return result

        normalized = norm(hidden, "pre_block.norm1")
        packed = registry.reference("pre_block.attn.qkv.weight")
        matrix_bytes = 2048 * 2048 * 4
        # The packed checkpoint stores Q, K and V matrices in this order.
        parts = tuple(
            linear(
                normalized,
                "",
                matrix=packed.view((2048, 2048), part * matrix_bytes),
                bias=registry.reference("pre_block.attn." + bias_name),
            )
            for part, bias_name in enumerate(("q_bias", "zero_k_bias", "v_bias"))
        )
        channels: list[int] = []
        channel_bytes = spec.frames * 2048 * 4
        # Attend within each stereo channel before concatenating their latent rows.
        for channel in range(2):
            q, k, v = ((value, channel * channel_bytes) for value in parts)
            attended = allocate((spec.frames, 2048))
            ops.append(
                CausalAttentionF32Op(q, k, v, rows=spec.frames, width=2048, dim=256).to(
                    attended
                )
            )
            averaged = allocate((spec.frames, 32))
            ops.append(
                HeadAverageOp(
                    attended,
                    rows=spec.frames,
                    input_columns=2048,
                    heads=8,
                    output_columns=32,
                ).to(averaged)
            )
            channels.append(averaged)
        concatenated = allocate((2 * spec.frames, 32))
        ops.append(
            ConcatRowsOp(
                *channels,
                dtype=DType.F32,
                first_shape=(spec.frames, 32),
                second_shape=(spec.frames, 32),
            ).to(concatenated)
        )
        attention = linear(concatenated, "pre_block.attn.proj")
        projected = linear(norm(hidden, "pre_block.norm3"), "pre_block.proj")
        if shapes[projected] != shapes[attention]:
            raise ValueError("posterior projection and attention shapes differ")
        residual = allocate(shapes[projected])
        ops.append(
            ResidualOp(
                projected, attention, dtype=DType.F32, shape=shapes[projected]
            ).to(residual)
        )
        normalized = norm(norm(residual, "pre_block.norm2"), "pre_block.mlp.norm")
        gate = linear(normalized, "pre_block.mlp.w0")
        value = linear(normalized, "pre_block.mlp.w1")
        gated = allocate(shapes[gate])
        ops.append(GeluMultiplyOp(gate, value, shape=shapes[gate]).to(gated))
        update = linear(gated, "pre_block.mlp.w2")
        hidden = allocate(shapes[residual])
        ops.append(
            ResidualOp(residual, update, dtype=DType.F32, shape=shapes[residual]).to(
                hidden
            )
        )
        mean_weight = registry.reference("mean_proj.weight")
        mean_weight = dataclasses.replace(
            mean_weight, dtype=DType.F32, shape=(32, 32)
        )
        mean = linear(
            hidden,
            "",
            matrix=mean_weight,
            bias=registry.reference("mean_proj.bias"),
        )
        output = allocate((32, 2 * spec.frames))
        ops.append(
            ChannelNormalizeTransposeOp(
                mean, rows=2 * spec.frames, columns=32, stereo=2
            )
            .with_weights(
                (registry.reference("latents_mean"), registry.reference("latents_std"))
            )
            .to(output)
        )
        commands, stats = operators.commands(ops, storage, registry, capacity)
        offsets = stats["activation_offsets"]
        commands.insert(
            0,
            Copy(
                TensorDesc.bytes(
                    TensorKind.INPUT, 0, 0, 2 * spec.frames * 2048 * 4
                ),
                TensorDesc.bytes(
                    TensorKind.WORKSPACE,
                    0,
                    offsets[0],
                    2 * spec.frames * 2048 * 4,
                ),
                "compute",
            ),
        )
        commands.append(
            Copy(
                TensorDesc.bytes(
                    TensorKind.WORKSPACE,
                    0,
                    offsets[output],
                    32 * 2 * spec.frames * 4,
                ),
                TensorDesc.bytes(
                    TensorKind.OUTPUT, 0, 0, 32 * 2 * spec.frames * 4
                ),
                "compute",
            )
        )
        return None, commands, capacity

    def run(self, args: AudioPosteriorArgs) -> None:
        self.execute(
            args.runtime,
            (TensorDesc.from_pointer(args.hidden.data_ptr(), args.hidden.num_bytes),),
            (TensorDesc.from_pointer(args.output.data_ptr(), args.output.num_bytes),),
        )
