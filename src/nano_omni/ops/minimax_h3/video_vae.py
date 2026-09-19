"""Explicit top-level operators for the H3 video VAE."""

import math
from collections.abc import Iterable

from nano_omni.core.kernel import Arguments, Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.self_attention import (
    SelfAttentionArguments,
    SelfAttentionKernel,
)
from nano_omni.kernels.convolution.causal_conv3d import (
    CausalConv3dArguments,
    CausalConv3dKernel,
)
from nano_omni.kernels.elementwise.channel_affine import (
    ChannelAffineArguments,
    ChannelAffineKernel,
)
from nano_omni.kernels.layout.volume_to_tokens import (
    VolumeToTokensArguments,
    VolumeToTokensKernel,
    strided_span,
)
from nano_omni.kernels.matmul.matmul_f16 import MatmulF16Arguments, MatmulF16Kernel
from nano_omni.kernels.normalization.layer_norm import (
    LayerNormArguments,
    LayerNormKernel,
)
from nano_omni.kernels.normalization.rms_norm import RmsNormArguments, RmsNormKernel
from nano_omni.kernels.normalization.temporal_group_norm_silu import (
    TemporalGroupNormSiluArguments,
    TemporalGroupNormSiluKernel,
)
from nano_omni.kernels.specialized.h3_qkv_rope import (
    H3QkvRopeArguments,
    H3QkvRopeKernel,
)
from nano_omni.kernels.specialized.video_vae_blend_append import (
    VideoVaeBlendAppendArguments,
    VideoVaeBlendAppendKernel,
)
from nano_omni.kernels.specialized.video_vae_normalize import (
    VideoVaeNormalizeArguments,
    VideoVaeNormalizeKernel,
)
from nano_omni.kernels.specialized.video_vae_rope import (
    VideoVaeRopeArguments,
    VideoVaeRopeKernel,
)
from nano_omni.kernels.specialized.video_vae_spatial_finish import (
    VideoVaeSpatialFinishArguments,
    VideoVaeSpatialFinishKernel,
)
from nano_omni.kernels.specialized.video_vae_suffix import (
    VideoVaeSuffixArguments,
    VideoVaeSuffixKernel,
)
from nano_omni.kernels.specialized.video_vae_temporal_nv12 import (
    VideoVaeTemporalNv12Arguments,
    VideoVaeTemporalNv12Kernel,
)
from nano_omni.kernels.specialized.video_vae_unpatchify import (
    VideoVaeUnpatchifyArguments,
    VideoVaeUnpatchifyKernel,
)
from nano_omni.kernels.specialized.video_vae_unpatchify_blend import (
    VideoVaeUnpatchifyBlendArguments,
    VideoVaeUnpatchifyBlendKernel,
)

TOKENS, PATCH_TOKENS, WIDTH, HEADS, DIM, ROPE_DIM = 1797, 1792, 2048, 32, 64, 48


class VideoVaeSetup(Op[tuple[TensorDesc, TensorDesc]]):
    """Denormalize the latent channels and build the shared tile rotary table."""

    def __init__(
        self, latent: int | tuple[int, int], *, shape: tuple[int, ...]
    ) -> None:
        super().__init__(
            TensorDesc.activation(latent, DType.F32, shape),
            outputs=((DType.F16, shape), (DType.F32, (2, TOKENS, 24))),
        )

    def kernels(self, scratch: ScratchLayout) -> Iterable[Kernel]:
        mean, std = self.bound_weights
        channels = self.inputs[0].shape[0]
        spatial = math.prod(self.inputs[0].shape[1:])
        return (
            ChannelAffineKernel(
                ChannelAffineArguments(
                    self.inputs[0].view((channels, spatial)),
                    mean,
                    std,
                    self.bound_outputs[0].view((channels, spatial)),
                )
            ),
            VideoVaeRopeKernel(VideoVaeRopeArguments(self.bound_outputs[1])),
        )


class VideoVaeDecodeTile(Op[dict[str, TensorDesc]]):
    """Decode a fixed latent tile and append its projection directly to a spatial row."""

    def __init__(
        self,
        latent: int | tuple[int, int],
        rope: int | tuple[int, int],
        *,
        latent_shape: tuple[int, ...],
        previous: int | tuple[int, int] | None = None,
        previous_shape: tuple[int, ...] | None = None,
        overlap: int = 0,
        vertical: int | tuple[int, int] | None = None,
        vertical_shape: tuple[int, ...] | None = None,
        vertical_overlap: int = 0,
    ) -> None:
        assert (previous is None) == (previous_shape is None), "incomplete row prefix"
        assert (vertical is None) == (vertical_shape is None), (
            "incomplete vertical prefix"
        )
        assert vertical is None or previous is not None, (
            "vertical prefix needs row prefix"
        )
        self.latent_shape = latent_shape
        self.previous_shape, self.overlap = previous_shape, overlap
        self.vertical_shape, self.vertical_overlap = vertical_shape, vertical_overlap
        input_strides = (
            math.prod(latent_shape[1:]),
            math.prod(latent_shape[2:]),
            latent_shape[3],
            1,
        )
        self.input_strides = input_strides
        input_dimensions = (24, min(7, latent_shape[1]), 16, 16)
        input_span = strided_span(input_dimensions, input_strides)
        inputs = [
            TensorDesc.activation(latent, DType.F16, (input_span,)),
            TensorDesc.activation(rope, DType.F32, (2, TOKENS, 24)),
        ]
        if previous is not None:
            assert previous_shape is not None
            inputs.append(TensorDesc.activation(previous, DType.F32, previous_shape))
        if vertical is not None:
            assert vertical_shape is not None
            inputs.append(TensorDesc.activation(vertical, DType.F32, vertical_shape))
        super().__init__(*inputs, outputs=((DType.F32, self.output_shape),))

    @property
    def output_shape(self) -> tuple[int, ...]:
        width = (
            256
            if self.previous_shape is None
            else self.previous_shape[3] + 256 - self.overlap
        )
        height = (
            256
            if self.vertical_shape is None
            else self.vertical_shape[2] + 256 - self.vertical_overlap
        )
        return (3, 28, height, width)

    def kernels(self, scratch: ScratchLayout) -> Iterable[Kernel]:
        weights = self.bound_weights
        calls: list[Kernel] = []

        def reserve(name: str, dtype: DType, shape: tuple[int, ...]) -> TensorDesc:
            return scratch.reserve(name, dtype, shape)

        def emit(kernel: type[Kernel], arguments: Arguments) -> None:
            calls.append(kernel(arguments))

        def linear(
            source: TensorDesc,
            matrix: TensorDesc,
            bias: TensorDesc | None,
            output: TensorDesc,
            *,
            scale: TensorDesc | None = None,
            residual: TensorDesc | None = None,
            multiply: TensorDesc | None = None,
            silu: bool = False,
        ) -> None:
            emit(
                MatmulF16Kernel,
                MatmulF16Arguments(
                    source, matrix, bias, scale, residual, multiply, output, silu
                ),
            )

        patches = reserve("patches", DType.F16, (PATCH_TOKENS, 24))
        emit(
            VolumeToTokensKernel,
            VolumeToTokensArguments(
                self.inputs[0],
                patches,
                7,
                16,
                16,
                self.latent_shape[0],
                self.latent_shape[1],
                self.latent_shape[2],
                self.latent_shape[3],
                self.input_strides,
            ),
        )
        embedded = reserve("embedded", DType.F16, (PATCH_TOKENS, WIDTH))
        post = weights["post_quant_conv.weight"].view((24, 24))
        intermediate = reserve("post_quant", DType.F16, (PATCH_TOKENS, 24))
        linear(patches, post, weights["post_quant_conv.bias"], intermediate)
        linear(
            intermediate,
            weights["decoder.x_embedder.weight"],
            weights["decoder.x_embedder.bias"],
            embedded,
        )
        hidden_a = reserve("hidden_a", DType.F16, (TOKENS, WIDTH))
        registers = weights["decoder.register_tokens"].view((4, WIDTH))
        emit(
            VideoVaeSuffixKernel,
            VideoVaeSuffixArguments(embedded, registers, hidden_a),
        )
        hidden_b = reserve("hidden_b", DType.F16, hidden_a.shape)
        normalized = reserve("normalized", DType.F16, hidden_a.shape)
        qkv = reserve("qkv", DType.F16, (TOKENS, WIDTH * 3))
        packed = reserve("packed", DType.F16, (3, TOKENS, HEADS, DIM))
        context = reserve("context", DType.F16, hidden_a.shape)
        gate = reserve("gate", DType.F16, (TOKENS, 8192))
        gated = reserve("gated", DType.F16, gate.shape)
        rope_bytes = TOKENS * 24 * DType.F32.itemsize
        rope_position = self.inputs[1].info
        assert len(rope_position) == 2, "RoPE tensor requires identity and offset"
        cosines = TensorDesc.activation(
            (rope_position[0], rope_position[1]), DType.F32, (TOKENS, 24)
        )
        sines = TensorDesc.activation(
            (rope_position[0], rope_position[1] + rope_bytes),
            DType.F32,
            cosines.shape,
        )
        hidden = hidden_a
        for index in range(36):
            target = hidden_b if hidden == hidden_a else hidden_a
            prefix = f"decoder.transformer_blocks.{index}"
            emit(
                RmsNormKernel,
                RmsNormArguments(
                    hidden, weights[f"{prefix}.norm1.weight"], normalized, 1e-5
                ),
            )
            linear(
                normalized,
                weights[f"{prefix}.attn.to_qkv.weight"],
                weights[f"{prefix}.attn.to_qkv.bias"],
                qkv,
            )
            emit(
                H3QkvRopeKernel,
                H3QkvRopeArguments(
                    qkv,
                    None,
                    None,
                    cosines,
                    sines,
                    packed,
                    HEADS,
                    DIM,
                    ROPE_DIM,
                    False,
                    True,
                    True,
                ),
            )
            plane = TOKENS * WIDTH * DType.F16.itemsize
            query = packed.view((TOKENS, WIDTH))
            key = packed.view(query.shape, plane)
            value = packed.view(query.shape, 2 * plane)
            emit(
                SelfAttentionKernel,
                SelfAttentionArguments(
                    query, key, value, context, HEADS, DIM, True, False
                ),
            )
            linear(
                context,
                weights[f"{prefix}.attn.to_out.weight"],
                weights[f"{prefix}.attn.to_out.bias"],
                target,
                scale=weights[f"{prefix}.scale1"],
                residual=hidden,
            )
            emit(
                RmsNormKernel,
                RmsNormArguments(
                    target, weights[f"{prefix}.norm2.weight"], normalized, 1e-5
                ),
            )
            w1 = weights[f"{prefix}.ff.w1.weight"]
            b1 = weights[f"{prefix}.ff.w1.bias"]
            split_weight = 8192 * WIDTH * DType.F16.itemsize
            split_bias = 8192 * DType.F16.itemsize
            linear(
                normalized,
                w1.view((8192, WIDTH)),
                b1.view((8192,)),
                gate,
                silu=True,
            )
            linear(
                normalized,
                w1.view((8192, WIDTH), split_weight),
                b1.view((8192,), split_bias),
                gated,
                multiply=gate,
            )
            final = hidden
            linear(
                gated,
                weights[f"{prefix}.ff.w2.weight"],
                weights[f"{prefix}.ff.w2.bias"],
                final,
                scale=weights[f"{prefix}.scale2"],
                residual=target,
            )
            hidden = final
        emit(
            LayerNormKernel,
            LayerNormArguments(
                hidden,
                weights["decoder.norm_out.weight"],
                weights["decoder.norm_out.bias"],
                normalized,
                1e-5,
            ),
        )
        projected = reserve("projected", DType.F16, (TOKENS, 3072))
        linear(
            normalized,
            weights["decoder.proj_out.weight"],
            weights["decoder.proj_out.bias"],
            projected,
        )
        patch_projection = TensorDesc.scratch(
            projected.info[1], DType.F16, (PATCH_TOKENS, 3072)
        )
        output = self.bound_outputs[0]
        if self.vertical_shape is not None:
            assert self.previous_shape is not None
            emit(
                VideoVaeSpatialFinishKernel,
                VideoVaeSpatialFinishArguments(
                    self.inputs[3],
                    self.inputs[2],
                    patch_projection,
                    output,
                    self.overlap,
                    self.vertical_overlap,
                ),
            )
        elif self.previous_shape is None:
            # The first tile establishes a row; later tiles write only the fused row result.
            emit(
                VideoVaeUnpatchifyKernel,
                VideoVaeUnpatchifyArguments(
                    patch_projection,
                    output,
                ),
            )
        else:
            emit(
                VideoVaeUnpatchifyBlendKernel,
                VideoVaeUnpatchifyBlendArguments(
                    self.inputs[2],
                    patch_projection,
                    output,
                    self.overlap,
                ),
            )
        return calls


class VideoVaeBlend(Op[None]):
    """Blend one spatial overlap and append the new non-overlapping region."""

    def __init__(
        self,
        first: int | tuple[int, int],
        second: int | tuple[int, int],
        *,
        first_shape: tuple[int, ...],
        second_shape: tuple[int, ...],
        axis: int,
        overlap: int,
    ) -> None:
        self.first_shape, self.second_shape = first_shape, second_shape
        self.axis, self.overlap = axis, overlap
        inputs = (
            TensorDesc.activation(first, DType.F32, first_shape),
            TensorDesc.activation(second, DType.F32, second_shape),
        )
        super().__init__(*inputs, outputs=((DType.F32, self.output_shape),))

    @property
    def output_shape(self) -> tuple[int, ...]:
        shape = list(self.first_shape)
        shape[self.axis] += self.second_shape[self.axis] - self.overlap
        return tuple(shape)

    def kernels(self, scratch: ScratchLayout) -> Iterable[Kernel]:
        return (
            VideoVaeBlendAppendKernel(
                VideoVaeBlendAppendArguments(
                    self.inputs[0],
                    self.inputs[1],
                    self.bound_outputs[0],
                    self.axis,
                    self.overlap,
                )
            ),
        )


class VideoVaeTemporalStore(Op[None]):
    """Write one temporal segment directly into its final NV12 byte region."""

    def __init__(
        self,
        previous: int | tuple[int, int] | None,
        current: int | tuple[int, int],
        *,
        chunk_shape: tuple[int, ...],
        output_frames: int,
        pitch: int,
    ) -> None:
        inputs = tuple(
            TensorDesc.activation(position, DType.F32, chunk_shape)
            for position in (
                (previous, current) if previous is not None else (current,)
            )
        )
        output_shape = (output_frames, chunk_shape[2] * 3 // 2, pitch)
        super().__init__(*inputs, outputs=((DType.U8, output_shape),))

    def kernels(self, scratch: ScratchLayout) -> Iterable[Kernel]:
        return (
            VideoVaeTemporalNv12Kernel(
                VideoVaeTemporalNv12Arguments(
                    self.inputs[0] if len(self.inputs) == 2 else None,
                    self.inputs[-1],
                    self.bound_outputs[0],
                )
            ),
        )


class VideoVaeEncode(Op[dict[str, TensorDesc]]):
    """Encode one volume using the existing four-slot convolution schedule."""

    def __init__(self, video: TensorDesc) -> None:
        shape = video.shape
        output_shape = (24, (shape[1] + 3) // 4, shape[2] // 16, shape[3] // 16)
        assert video.dtype == DType.F32, "video VAE encoder input must be F32"
        super().__init__(video, outputs=((DType.F32, output_shape),))

    def kernels(self, scratch: ScratchLayout) -> Iterable[Kernel]:
        weights = self.bound_weights
        calls: list[Kernel] = []
        maximum = _encoder_maximum_nbytes(weights, self.inputs[0].shape)
        buffers = tuple(
            scratch.reserve(f"activation_{index}", DType.U8, (maximum,))
            for index in range(4)
        )

        def ref(slot: int, shape: tuple[int, ...]) -> TensorDesc:
            return TensorDesc.scratch(buffers[slot].info[1], DType.F16, shape)

        def conv(
            source: TensorDesc,
            source_shape: tuple[int, ...],
            prefix: str,
            target: int,
            *,
            stride: tuple[int, int, int] = (1, 1, 1),
            padding: tuple[int, int, int] = (1, 1, 1),
            pad_end: tuple[int, int] = (0, 0),
            normalize: bool = False,
            residual: TensorDesc | None = None,
        ) -> tuple[TensorDesc, tuple[int, int, int, int]]:
            weight, bias = weights[prefix + ".weight"], weights[prefix + ".bias"]
            shape = _conv_shape(source_shape, weight.shape, stride, padding, pad_end)
            output = ref(target, shape)
            calls.append(
                CausalConv3dKernel(
                    CausalConv3dArguments(
                        source,
                        weight,
                        bias,
                        residual,
                        output,
                        normalize,
                        *stride,
                        *padding,
                    )
                )
            )
            return output, shape

        hidden_slot = 0
        hidden, shape = conv(
            self.inputs[0],
            self.inputs[0].shape,
            "encoder.conv_in",
            0,
            normalize=True,
        )
        time_strides = (1, 2, 2, 1, 1, 1)
        space_strides = (2, 2, 2, 2, 1, 1)
        for level, (time_stride, space_stride) in enumerate(
            zip(time_strides, space_strides, strict=True)
        ):
            for block in range(2):
                prefix = f"encoder.down.{level}.block.{block}"
                next_slot = 2 if hidden_slot == 0 else 0
                norm1 = ref(1, shape)
                calls.append(
                    TemporalGroupNormSiluKernel(
                        TemporalGroupNormSiluArguments(
                            hidden,
                            weights[f"{prefix}.norm1.weight"],
                            weights[f"{prefix}.norm1.bias"],
                            norm1,
                            32,
                            True,
                        )
                    )
                )
                first, first_shape = conv(norm1, shape, f"{prefix}.conv1", next_slot)
                norm2 = ref(1, first_shape)
                calls.append(
                    TemporalGroupNormSiluKernel(
                        TemporalGroupNormSiluArguments(
                            first,
                            weights[f"{prefix}.norm2.weight"],
                            weights[f"{prefix}.norm2.bias"],
                            norm2,
                            32,
                            True,
                        )
                    )
                )
                residual = hidden
                if f"{prefix}.nin_shortcut.weight" in weights:
                    residual, _ = conv(
                        hidden, shape, f"{prefix}.nin_shortcut", 3, padding=(0, 0, 0)
                    )
                hidden, shape = conv(
                    norm2, first_shape, f"{prefix}.conv2", next_slot, residual=residual
                )
                hidden_slot = next_slot
            if time_stride * space_stride > 1:
                next_slot = 2 if hidden_slot == 0 else 0
                hidden, shape = conv(
                    hidden,
                    shape,
                    f"encoder.down.{level}.downsample.conv",
                    next_slot,
                    stride=(time_stride, space_stride, space_stride),
                    padding=(1, 0, 0),
                    pad_end=(int(space_stride == 2), int(space_stride == 2)),
                )
                hidden_slot = next_slot
        normalized = ref(1, shape)
        calls.append(
            TemporalGroupNormSiluKernel(
                TemporalGroupNormSiluArguments(
                    hidden,
                    weights["encoder.norm_out.weight"],
                    weights["encoder.norm_out.bias"],
                    normalized,
                    32,
                    True,
                )
            )
        )
        next_slot = 2 if hidden_slot == 0 else 0
        hidden, shape = conv(normalized, shape, "encoder.conv_out", next_slot)
        hidden_slot = next_slot
        next_slot = 2 if hidden_slot == 0 else 0
        hidden, shape = conv(hidden, shape, "quant_conv", next_slot, padding=(0, 0, 0))
        calls.append(
            VideoVaeNormalizeKernel(
                VideoVaeNormalizeArguments(
                    hidden,
                    weights["latents_mean"],
                    weights["latents_std"],
                    self.bound_outputs[0],
                )
            )
        )
        return calls


def _conv_shape(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    pad_end: tuple[int, int],
) -> tuple[int, int, int, int]:
    return (
        weight_shape[0],
        (input_shape[1] + 2 * padding[0] - weight_shape[2]) // stride[0] + 1,
        (input_shape[2] + 2 * padding[1] + pad_end[0] - weight_shape[3]) // stride[1]
        + 1,
        (input_shape[3] + 2 * padding[2] + pad_end[1] - weight_shape[4]) // stride[2]
        + 1,
    )


def _encoder_maximum_nbytes(
    weights: dict[str, TensorDesc], input_shape: tuple[int, ...]
) -> int:
    # The first full-resolution convolution is the largest encoder activation.
    shape = _conv_shape(
        input_shape,
        weights["encoder.conv_in.weight"].shape,
        (1, 1, 1),
        (1, 1, 1),
        (0, 0),
    )
    return max(
        DType.F32.itemsize * math.prod(input_shape),
        DType.F16.itemsize * math.prod(shape),
    )
