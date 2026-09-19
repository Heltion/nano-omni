"""Audio VAE decoder topology and its protected eight-slot activation layout."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path

from nano_omni.core.layout import ActivationLayout
from nano_omni.core.model import ModelMetadata, PlannedModel
from nano_omni.core.op import Op
from nano_omni.core.planning import operators
from nano_omni.core.planning.scheduling import Action, Copy
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import DType, TensorDesc, TensorKind
from nano_omni.core.weights import WeightRegistry
from nano_omni.ops.activation.snake_beta import SnakeBetaOp
from nano_omni.ops.convolution.conv1d import Conv1dOp
from nano_omni.ops.convolution.conv_transpose1d import ConvTranspose1dOp
from nano_omni.ops.elementwise.channel_affine_transpose import ChannelAffineTransposeOp
from nano_omni.ops.elementwise.mean3 import Mean3Op
from nano_omni.ops.matmul.matmul_f32 import MatmulF32Op

AUDIO_VAE_STEREO = 2
AUDIO_VAE_LATENT_CHANNELS = 32
AUDIO_VAE_LATENT_FRAMES = 93
AUDIO_VAE_UPSAMPLE_RATES = (5, 5, 2, 2, 2, 2, 2)
AUDIO_VAE_UPSAMPLE_KERNELS = (9, 9, 4, 4, 4, 4, 4)
AUDIO_VAE_CHANNELS = (512, 256, 128, 64, 32, 16, 8)
AUDIO_VAE_RESBLOCK_KERNELS = (3, 7, 11)
AUDIO_VAE_DILATIONS = (1, 3, 5)
AUDIO_VAE_MEMORY_RESERVE = 1 << 30


@dataclasses.dataclass(frozen=True, slots=True)
class H3AudioVaeDecoderSpec:
    latent_frames: int = AUDIO_VAE_LATENT_FRAMES
    stereo: int = AUDIO_VAE_STEREO


@dataclasses.dataclass(frozen=True, slots=True)
class H3AudioVaeDecoderArgs:
    latent: TensorDesc
    output: TensorDesc
    runtime: CudaRuntime


def read_h3_audio_vae_metadata(path: Path) -> ModelMetadata:
    metadata = ModelMetadata.read((path,))
    validate_h3_audio_vae_metadata(metadata)
    return metadata


def validate_h3_audio_vae_metadata(metadata: ModelMetadata) -> None:
    weights = metadata.files[0].weights
    expected: dict[str, tuple[int, ...]] = {
        "latents_mean": (32,),
        "latents_std": (32,),
        "dec_in_proj.weight": (2048, 32, 1),
        "dec_in_proj.bias": (2048,),
        "decoder.conv_pre.weight": (1024, 2048, 7),
        "decoder.conv_pre.bias": (1024,),
        "decoder.activation_post.act.alpha": (8,),
        "decoder.activation_post.act.beta": (8,),
        "decoder.activation_post.upsample.filter": (1, 1, 12),
        "decoder.activation_post.downsample.lowpass.filter": (1, 1, 12),
        "decoder.conv_post.weight": (1, 8, 7),
    }
    for stage, (channels, rate, kernel_size) in enumerate(
        zip(
            AUDIO_VAE_CHANNELS,
            AUDIO_VAE_UPSAMPLE_RATES,
            AUDIO_VAE_UPSAMPLE_KERNELS,
            strict=True,
        )
    ):
        input_channels = channels * 2
        expected[f"decoder.ups.{stage}.0.weight"] = (
            input_channels,
            channels,
            kernel_size,
        )
        expected[f"decoder.ups.{stage}.0.bias"] = (channels,)
        assert rate in (2, 5)
        for branch, residual_kernel in enumerate(AUDIO_VAE_RESBLOCK_KERNELS):
            block = stage * 3 + branch
            prefix = f"decoder.resblocks.{block}"
            for activation in range(6):
                expected[f"{prefix}.activations.{activation}.act.alpha"] = (channels,)
                expected[f"{prefix}.activations.{activation}.act.beta"] = (channels,)
                expected[f"{prefix}.activations.{activation}.upsample.filter"] = (
                    1,
                    1,
                    12,
                )
                expected[
                    f"{prefix}.activations.{activation}.downsample.lowpass.filter"
                ] = (1, 1, 12)
            for pair in range(3):
                expected[f"{prefix}.convs1.{pair}.weight"] = (
                    channels,
                    channels,
                    residual_kernel,
                )
                expected[f"{prefix}.convs1.{pair}.bias"] = (channels,)
                expected[f"{prefix}.convs2.{pair}.weight"] = (
                    channels,
                    channels,
                    residual_kernel,
                )
                expected[f"{prefix}.convs2.{pair}.bias"] = (channels,)
    for name, shape in expected.items():
        assert weights[name].dtype == DType.F32
        assert weights[name].shape == shape


class AudioVaeDecoderGraph:
    """Build the decoder with explicit reusable activation slots."""

    def __init__(self, metadata: ModelMetadata, stereo: int) -> None:
        self.stereo = stereo
        self.registry = WeightRegistry(metadata)
        self.storage = ActivationLayout()
        self.ops: list[Op] = []
        self.shapes: dict[int, tuple[int, ...]] = {}
        self.cursor = 0

    def allocate(
        self, shape: tuple[int, ...], protected: Sequence[int | None] = ()
    ) -> int:
        """Reserve the next unprotected slot; prior occupants may enlarge its size."""
        excluded = set(protected)
        for _ in range(8):
            identity = self.cursor
            self.cursor = (self.cursor + 1) % 8
            if identity not in excluded:
                self.shapes[identity] = shape
                self.storage.require(identity, math.prod(shape) * 4)
                return identity
        raise RuntimeError("Audio VAE activation slots exhausted")

    def weight(self, name: str, shape: tuple[int, ...] | None = None) -> TensorDesc:
        value = self.registry.reference(name)
        return (
            value
            if shape is None
            else dataclasses.replace(value, shape=shape)
        )

    def snake(self, value: int, prefix: str, hold: Sequence[int] = ()) -> int:
        rows, channels = self.shapes[value]
        shape = (rows, channels)
        result = self.allocate(shape, (*hold, value))
        self.ops.append(
            SnakeBetaOp(value, shape=shape, stereo=self.stereo)
            .with_weights(
                (
                    self.weight(prefix + ".act.alpha", (channels,)),
                    self.weight(prefix + ".act.beta", (channels,)),
                    self.weight(prefix + ".upsample.filter"),
                    self.weight(prefix + ".downsample.lowpass.filter"),
                )
            )
            .to(result)
        )
        return result

    def conv(
        self,
        value: int,
        prefix: str,
        *,
        residual: int | None = None,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        clamp: bool = False,
        bias: bool = True,
        hold: Sequence[int] = (),
    ) -> int:
        rows, channels = self.shapes[value]
        input_shape = (rows, channels)
        weight = self.weight(prefix + ".weight")
        frames = input_shape[0] // self.stereo
        output_frames = (
            frames + 2 * padding - dilation * (weight.shape[2] - 1) - 1
        ) // stride + 1
        output_shape = (self.stereo * output_frames, weight.shape[0])
        result = self.allocate(output_shape, (*hold, value, residual))
        self.ops.append(
            Conv1dOp(
                value,
                residual,
                input_shape=input_shape,
                output_shape=output_shape,
                stereo=self.stereo,
                stride=stride,
                dilation=dilation,
                padding=padding,
                clamp=clamp,
            )
            .with_weights((weight, self.weight(prefix + ".bias") if bias else None))
            .to(result)
        )
        return result

    def transpose(
        self,
        value: int,
        prefix: str,
        *,
        stride: int,
        padding: int,
        hold: Sequence[int] = (),
    ) -> int:
        rows, channels = self.shapes[value]
        input_shape = (rows, channels)
        weight = self.weight(prefix + ".weight")
        frames = input_shape[0] // self.stereo
        output_frames = (frames - 1) * stride - 2 * padding + weight.shape[2]
        output_shape = (self.stereo * output_frames, weight.shape[1])
        result = self.allocate(output_shape, (*hold, value))
        self.ops.append(
            ConvTranspose1dOp(
                value,
                input_shape=input_shape,
                output_shape=output_shape,
                stereo=self.stereo,
                stride=stride,
                padding=padding,
            )
            .with_weights((weight, self.weight(prefix + ".bias")))
            .to(result)
        )
        return result

    def residual_block(
        self, value: int, stage: int, branch: int, hold: Sequence[int] = ()
    ) -> int:
        """Apply three dilation pairs, keeping each pair's residual live until addition."""
        prefix = f"decoder.resblocks.{stage * 3 + branch}"
        kernel_size = AUDIO_VAE_RESBLOCK_KERNELS[branch]
        hidden = value
        for pair, dilation in enumerate(AUDIO_VAE_DILATIONS):
            residual = hidden
            protected = (*hold, residual)
            hidden = self.snake(hidden, f"{prefix}.activations.{pair * 2}", protected)
            hidden = self.conv(
                hidden,
                f"{prefix}.convs1.{pair}",
                dilation=dilation,
                padding=(kernel_size * dilation - dilation) // 2,
                hold=protected,
            )
            hidden = self.snake(
                hidden, f"{prefix}.activations.{pair * 2 + 1}", protected
            )
            hidden = self.conv(
                hidden,
                f"{prefix}.convs2.{pair}",
                residual=residual,
                padding=(kernel_size - 1) // 2,
                hold=hold,
            )
        return hidden


@dataclasses.dataclass(slots=True)
class H3AudioVaeDecoder(
    PlannedModel[H3AudioVaeDecoderSpec, H3AudioVaeDecoderArgs, None]
):
    @classmethod
    def read_metadata(
        cls, path: Path, additional_weights: Sequence[Path] = ()
    ) -> ModelMetadata:
        assert not additional_weights
        return read_h3_audio_vae_metadata(path)

    @classmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: H3AudioVaeDecoderSpec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[None, list[Action], int]:
        capacity = cls.workspace_capacity(
            total_memory - AUDIO_VAE_MEMORY_RESERVE, workspace_limit
        )
        if capacity <= 0:
            raise MemoryError("H3 Audio VAE requires a 1 GiB VRAM reserve")
        graph = AudioVaeDecoderGraph(metadata, spec.stereo)
        latent = graph.allocate(
            (AUDIO_VAE_LATENT_CHANNELS, spec.stereo, spec.latent_frames)
        )
        hidden = graph.allocate((spec.stereo * spec.latent_frames, 32), (latent,))
        graph.ops.append(
            ChannelAffineTransposeOp(
                latent,
                channels=32,
                stereo=spec.stereo,
                frames=spec.latent_frames,
            )
            .with_weights((graph.weight("latents_mean"), graph.weight("latents_std")))
            .to(hidden)
        )
        projected = graph.allocate((spec.stereo * spec.latent_frames, 2048), (hidden,))
        graph.ops.append(
            MatmulF32Op(
                hidden,
                rows=spec.stereo * spec.latent_frames,
                input_columns=32,
                output_columns=2048,
            )
            .with_weights(
                (
                    graph.weight("dec_in_proj.weight", (2048, 32)),
                    graph.weight("dec_in_proj.bias"),
                )
            )
            .to(projected)
        )
        hidden = graph.conv(projected, "decoder.conv_pre", padding=3)
        for stage, (rate, kernel_size) in enumerate(
            zip(AUDIO_VAE_UPSAMPLE_RATES, AUDIO_VAE_UPSAMPLE_KERNELS, strict=True)
        ):
            hidden = graph.transpose(
                hidden,
                f"decoder.ups.{stage}.0",
                stride=rate,
                padding=(kernel_size - rate) // 2,
            )
            branches: list[int] = []
            # Keep the shared input and earlier branches live until Mean3 consumes them.
            for branch in range(3):
                branches.append(
                    graph.residual_block(hidden, stage, branch, (hidden, *branches))
                )
            result = graph.allocate(graph.shapes[hidden], (*branches, hidden))
            graph.ops.append(
                Mean3Op(*branches, shape=graph.shapes[hidden]).to(result)
            )
            hidden = result
        hidden = graph.snake(hidden, "decoder.activation_post")
        hidden = graph.conv(
            hidden, "decoder.conv_post", padding=3, clamp=True, bias=False
        )
        commands, stats = operators.commands(
            graph.ops, graph.storage, graph.registry, capacity
        )
        offsets = stats["activation_offsets"]
        input_bytes = 32 * spec.stereo * spec.latent_frames * 4
        output_bytes = math.prod(graph.shapes[hidden]) * 4
        commands.insert(
            0,
            Copy(
                TensorDesc.bytes(TensorKind.INPUT, 0, 0, input_bytes),
                TensorDesc.bytes(
                    TensorKind.WORKSPACE, 0, offsets[latent], input_bytes
                ),
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

    def run(self, args: H3AudioVaeDecoderArgs) -> None:
        input = TensorDesc.from_pointer(args.latent.data_ptr(), args.latent.num_bytes)
        output = TensorDesc.from_pointer(args.output.data_ptr(), args.output.num_bytes)
        self.execute(args.runtime, (input,), (output,))
