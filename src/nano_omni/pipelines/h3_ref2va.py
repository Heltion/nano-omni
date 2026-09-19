"""INT8 REF2VA: reference video, its soundtrack, and an independent voice clip."""

import math
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import av
import numpy
from PIL import Image
from tokenizers import Tokenizer

from nano_omni.core import model
from nano_omni.core.pipeline import Pipeline
from nano_omni.core.runtime.buffers import PipelineMemory
from nano_omni.core.tensor import TensorDesc
from nano_omni.models import h3
from nano_omni.models.h3 import noise, ref_conditioning, ref_layout, stages
from nano_omni.models.h3.audio_encoder import (
    AudioEncoder,
    AudioEncoderArgs,
    AudioEncoderSpec,
)
from nano_omni.models.h3.audio_posterior import (
    AudioPosterior,
    AudioPosteriorArgs,
    AudioPosteriorSpec,
)
from nano_omni.models.h3.audio_vae_decoder import (
    H3AudioVaeDecoder,
    H3AudioVaeDecoderSpec,
)
from nano_omni.models.h3.conditioning import (
    H3Presentation,
    H3VisionInput,
    prepare_vision_input,
)
from nano_omni.models.h3.ref2va_diffusion import (
    H3Ref2vaDiffusion,
    H3Ref2vaDiffusionArgs,
    H3Ref2vaDiffusionSpec,
)
from nano_omni.models.h3.ref_video_encoder import (
    ReferenceVideoEncoder,
    reference_batch_counts,
)
from nano_omni.models.h3.text_encoder import (
    H3TextEncoder,
    H3TextEncoderSpec,
)
from nano_omni.models.h3.video_vae_decoder import (
    H3VideoVaeDecoder,
    H3VideoVaeDecoderSpec,
)
from nano_omni.models.h3.video_vae_encoder import (
    H3VideoVaeEncoder,
    H3VideoVaeEncoderSpec,
)
from nano_omni.models.h3.vision_encoder import H3VisionEncoder, H3VisionEncoderSpec

DIFFUSION = (
    h3.MODEL_ROOT / "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
)
LORA = (
    h3.MODEL_ROOT / "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
)

REFERENCE_CANVAS_MULTIPLE = 32
REFERENCE_BASE_SHORT_EDGE = 768
REFERENCE_MAX_PIXELS = 768 * 1344


def reference_canvas(width: int, height: int) -> tuple[int, int]:
    """Match the official Ref2VA canvas chosen from the reference aspect ratio."""
    ratio = width / height
    if ratio >= 1.0:
        canvas_width, canvas_height = REFERENCE_BASE_SHORT_EDGE * ratio, 768.0
    else:
        canvas_width, canvas_height = 768.0, REFERENCE_BASE_SHORT_EDGE / ratio
    if canvas_width * canvas_height > REFERENCE_MAX_PIXELS:
        scale = math.sqrt(REFERENCE_MAX_PIXELS / (canvas_width * canvas_height))
        canvas_width *= scale
        canvas_height *= scale
    multiple = REFERENCE_CANVAS_MULTIPLE
    canvas = (
        max(multiple, round(canvas_width / multiple) * multiple),
        max(multiple, round(canvas_height / multiple) * multiple),
    )
    if width * height < canvas[0] * canvas[1]:
        return (
            max(multiple, round(width / multiple) * multiple),
            max(multiple, round(height / multiple) * multiple),
        )
    return canvas


def read_video(path: Path, max_frames: int) -> numpy.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            width, height = reference_canvas(image.width, image.height)
            frames.append(
                numpy.asarray(image.resize((width, height), Image.Resampling.LANCZOS))
            )
            if len(frames) == max_frames:
                break
    count = len(frames)
    while count >= 5 and count % 17 != 5:
        count -= 1
    if count < 5:
        raise ValueError("reference video requires at least five frames")
    return numpy.stack(frames[:count]).astype(numpy.float32) / 255.0


def read_audio(path: Path) -> numpy.ndarray:
    chunks = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=32_000)
        for frame in container.decode(audio=0):
            chunks.extend(item.to_ndarray() for item in resampler.resample(frame))
        chunks.extend(item.to_ndarray() for item in resampler.resample(None))
    if not chunks:
        raise ValueError("reference contains no audio")
    waveform = numpy.concatenate(chunks, axis=1)
    samples = math.ceil(waveform.shape[1] / 800) * 800
    return numpy.pad(waveform, ((0, 0), (0, samples - waveform.shape[1])))


def reference_encoder_geometry(
    frames: int, height: int, width: int, workspace_nbytes: int | None
) -> tuple[int, tuple[int, int, int], tuple[int, ...]]:
    """Select the widest reference tile supported by the workspace budget."""
    tile_size = 512 if (workspace_nbytes or 0) >= 5 << 30 else 256
    shape = (17, min(tile_size, height), min(tile_size, width))
    counts = reference_batch_counts(frames, height, width, tile_size)
    return tile_size, shape, counts


class H3Ref2vaPipeline(Pipeline):
    def model_requests(self) -> Iterator[model.ModelRequest]:
        """Describe every model specialization selected by the reference media."""
        frames, soundtrack, voice, visions, presentation, spec = self.prepare_inputs()
        for vision in visions:
            yield model.ModelRequest(
                H3VisionEncoder,
                h3.TEXT_ENCODER,
                H3VisionEncoderSpec(vision.patches.shape[0]),
            )
        yield model.ModelRequest(
            H3TextEncoder,
            h3.TEXT_ENCODER,
            H3TextEncoderSpec(
                spec.text_tokens, vision_spans=presentation.scatter_spans
            ),
        )
        _, shape, counts = reference_encoder_geometry(
            len(frames),
            frames.shape[1],
            frames.shape[2],
            self.config.workspace_nbytes,
        )
        for count in dict.fromkeys(counts):
            yield model.ModelRequest(
                H3VideoVaeEncoder,
                h3.VIDEO_VAE,
                H3VideoVaeEncoderSpec(shape, count),
            )
        for waveform in (soundtrack, voice):
            yield model.ModelRequest(
                AudioEncoder, h3.AUDIO_VAE, AudioEncoderSpec(waveform.shape[1])
            )
            yield model.ModelRequest(
                AudioPosterior,
                h3.AUDIO_VAE,
                AudioPosteriorSpec(waveform.shape[1] // 800),
            )
        yield model.ModelRequest(H3Ref2vaDiffusion, DIFFUSION, spec, (LORA,))
        yield model.ModelRequest(
            H3VideoVaeDecoder,
            h3.VIDEO_VAE,
            H3VideoVaeDecoderSpec(
                spec.video_frames, spec.video_height, spec.video_width
            ),
        )
        yield model.ModelRequest(
            H3AudioVaeDecoder,
            h3.AUDIO_VAE,
            H3AudioVaeDecoderSpec(spec.audio_frames, stereo=2),
        )

    def workload(self) -> dict[str, object]:
        frames = h3.frame_count(self.config.seconds)
        return {
            "pipeline": self.config.pipeline,
            "mode": "ref2va",
            "width": self.config.width,
            "height": self.config.height,
            "requested_frames": h3.requested_frame_count(self.config.seconds),
            "frames": frames,
            "video_latent_frames": h3.video_latent_frames(frames),
            "audio_latent_frames": h3.audio_latent_frames(frames),
            "steps": self.config.steps,
            "seed": self.config.seed,
            "lora": "turbo_4step",
            "lora_strength": self.config.lora_strength,
            "video_shift": self.config.video_shift or h3.VIDEO_SHIFT,
            "audio_shift": self.config.audio_shift or h3.AUDIO_SHIFT,
        }

    def prepare(self) -> None:
        for checkpoint in (
            h3.TEXT_ENCODER,
            h3.VIDEO_VAE,
            h3.AUDIO_VAE,
            DIFFUSION,
            LORA,
        ):
            self.prepare_file(
                h3.MODEL_ID,
                checkpoint.relative_to(h3.MODEL_ROOT).as_posix(),
                checkpoint,
            )
        for name in h3.TOKENIZER_FILES:
            self.prepare_file(h3.TOKENIZER_ID, name, h3.TOKENIZER / name)
        for path in (self.config.reference_video, self.config.reference_audio):
            assert path is not None
            if not (self.root / path).is_file():
                raise FileNotFoundError(self.root / path)
        if self.config.lora not in (None, "turbo_4step"):
            raise ValueError("REF2VA uses its own turbo_4step v0.1 LoRA")
        if self.config.steps != 4:
            raise ValueError("The REF2VA turbo checkpoint requires four steps")

    def encode_audio(
        self,
        waveform: numpy.ndarray,
        encoder: AudioEncoder,
        posterior: AudioPosterior,
        memory: PipelineMemory,
    ) -> TensorDesc:
        samples = waveform.shape[1]
        frames = samples // 800
        with self.stage("reference_audio_encoder"):
            source = memory.upload(waveform)
            hidden = memory.empty((2 * frames, 2048), numpy.float32)
            encoder.run(AudioEncoderArgs(source, hidden, memory.runtime))
            memory.release(source)
            output = memory.empty((32, 2, frames), numpy.float32)
            posterior.run(AudioPosteriorArgs(hidden, output, memory.runtime))
            memory.release(hidden)
        return output

    def prepare_inputs(
        self,
    ) -> tuple[
        numpy.ndarray,
        numpy.ndarray,
        numpy.ndarray,
        tuple[H3VisionInput, ...],
        H3Presentation,
        H3Ref2vaDiffusionSpec,
    ]:
        """Build the same media, presentation and shapes for inference and tuning."""
        config = self.config
        assert config.reference_video is not None and config.reference_audio is not None
        frame_count = h3.frame_count(config.seconds)
        with self.stage("reference_media"):
            frames = read_video(
                self.root / config.reference_video,
                frame_count,
            )
            soundtrack = read_audio(self.root / config.reference_video)
            voice = read_audio(self.root / config.reference_audio)
            samples = list(range(0, len(frames), h3.FPS // 2))
            timestamps = [index / 2 for index in range(len(samples))]
            if len(samples) % 2:
                samples.append(samples[-1])
                timestamps.append(timestamps[-1])
            visions = tuple(
                prepare_vision_input(frames[samples[index : index + 2]])
                for index in range(0, len(samples), 2)
            )
            items = (
                ref_conditioning.ReferenceItem("audio"),
                ref_conditioning.ReferenceItem(
                    "video",
                    visions,
                    tuple(
                        (timestamps[i] + timestamps[i + 1]) / 2
                        for i in range(0, len(samples), 2)
                    ),
                ),
                ref_conditioning.ReferenceItem("audio"),
            )
            presentation = ref_conditioning.build_presentation(
                Tokenizer.from_file(str(h3.TOKENIZER / "tokenizer.json")),
                config.read_prompt(self.root),
                items,
            )
        references = (
            ref_layout.ReferenceBlock(
                "video_audio",
                h3.video_latent_frames(len(frames)),
                frames.shape[1] // 16,
                frames.shape[2] // 16,
                soundtrack.shape[1] // 800,
            ),
            ref_layout.ReferenceBlock("audio", audio_frames=voice.shape[1] // 800),
        )
        spec = H3Ref2vaDiffusionSpec(
            text_tokens=len(presentation.tokens),
            references=references,
            video_frames=h3.video_latent_frames(frame_count),
            video_height=config.height // 16,
            video_width=config.width // 16,
            audio_frames=h3.audio_latent_frames(frame_count),
            steps=config.steps,
            seed=config.seed,
            lora_strength=config.lora_strength,
            video_shift=config.video_shift or h3.VIDEO_SHIFT,
            audio_shift=config.audio_shift or h3.AUDIO_SHIFT,
            text_visual_spans=presentation.modality_spans,
            attention=config.attention,
            attention_precision=config.attention_precision,
            mlp_chunk_tokens=config.mlp_chunk_tokens,
        )
        if (
            config.expected_tokens is not None
            and spec.text_tokens != config.expected_tokens
        ):
            raise ValueError(
                f"expected {config.expected_tokens} text tokens, got {spec.text_tokens}"
            )
        return frames, soundtrack, voice, visions, presentation, spec

    def execute(self) -> Path:
        config = self.config
        frames, soundtrack, voice, visions, presentation, spec = self.prepare_inputs()
        references = spec.references
        print(f"reference_spec={spec}", flush=True)
        threshold = config.working_set_gib << 30
        total = stages.cuda_total_memory()
        capacity = min(total * 4 // 5, config.workspace_nbytes or total)
        pinned_nbytes = config.pinned_gib << 30
        video_vae_spec = H3VideoVaeDecoderSpec(
            spec.video_frames, spec.video_height, spec.video_width
        )
        audio_vae_spec = H3AudioVaeDecoderSpec(spec.audio_frames, stereo=2)
        reference_tile_size, reference_shape, reference_counts = (
            reference_encoder_geometry(
                len(frames),
                frames.shape[1],
                frames.shape[2],
                config.workspace_nbytes,
            )
        )
        encoder_specs = tuple(
            H3VideoVaeEncoderSpec(reference_shape, count)
            for count in dict.fromkeys(reference_counts)
        )

        with (
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="nano-noise"
            ) as noise_pool,
            ProcessPoolExecutor(max_workers=4) as prepare_pool,
            PipelineMemory(
                pinned_nbytes,
                capacity,
                config.synchronization,
                vram_fraction=0.85,
            ) as memory,
        ):
            noise_future = noise_pool.submit(
                noise.generate,
                (24, spec.video_frames, spec.video_height, spec.video_width),
                spec.audio_frames,
                spec.seed,
            )
            vision_plans = tuple(
                prepare_pool.submit(
                    H3VisionEncoder.prepare,
                    f"vision_encoder_{index}",
                    h3.TEXT_ENCODER,
                    H3VisionEncoderSpec(vision.patches.shape[0]),
                    total,
                    (),
                    pinned_nbytes,
                    config.workspace_nbytes,
                )
                for index, vision in enumerate(visions)
            )
            text_plan = prepare_pool.submit(
                H3TextEncoder.prepare,
                "text_encoder",
                h3.TEXT_ENCODER,
                H3TextEncoderSpec(
                    spec.text_tokens, vision_spans=presentation.scatter_spans
                ),
                total,
                (),
                pinned_nbytes,
                config.workspace_nbytes,
            )
            video_encoder_plans = {
                encoder_spec.count: prepare_pool.submit(
                    H3VideoVaeEncoder.prepare,
                    "reference_video_encoder",
                    h3.VIDEO_VAE,
                    encoder_spec,
                    total,
                    (),
                    pinned_nbytes,
                    config.workspace_nbytes,
                )
                for encoder_spec in encoder_specs
            }
            audio_plans = tuple(
                (
                    prepare_pool.submit(
                        AudioEncoder.prepare,
                        f"reference_audio_encoder_{index}",
                        h3.AUDIO_VAE,
                        AudioEncoderSpec(waveform.shape[1]),
                        total,
                        (),
                        pinned_nbytes,
                        config.workspace_nbytes,
                    ),
                    prepare_pool.submit(
                        AudioPosterior.prepare,
                        f"reference_audio_posterior_{index}",
                        h3.AUDIO_VAE,
                        AudioPosteriorSpec(waveform.shape[1] // 800),
                        total,
                        (),
                        pinned_nbytes,
                        config.workspace_nbytes,
                    ),
                )
                for index, waveform in enumerate((soundtrack, voice))
            )
            diffusion_plan = prepare_pool.submit(
                H3Ref2vaDiffusion.prepare,
                "av_diffusion",
                DIFFUSION,
                spec,
                total,
                (LORA,),
                pinned_nbytes,
                config.workspace_nbytes,
            )
            video_vae_plan = prepare_pool.submit(
                H3VideoVaeDecoder.prepare,
                "video_vae",
                h3.VIDEO_VAE,
                video_vae_spec,
                total,
                (),
                pinned_nbytes,
                config.workspace_nbytes,
            )
            audio_vae_plan = prepare_pool.submit(
                H3AudioVaeDecoder.prepare,
                "audio_vae",
                h3.AUDIO_VAE,
                audio_vae_spec,
                total,
                (),
                pinned_nbytes,
                config.workspace_nbytes,
            )
            prepare_pool.shutdown(wait=False)
            vision_outputs = tuple(
                stages.encode_vision(
                    vision,
                    model.bind_model(plan, memory.runtime),
                    memory,
                    memory.runtime,
                    threshold,
                )
                for vision, plan in zip(visions, vision_plans, strict=True)
            )
            level0, level1, level2, level3 = tuple(
                memory.concatenate_rows(tuple(value[index] for value in vision_outputs))
                for index in range(4)
            )
            visual = (level0, level1, level2, level3)
            for outputs in vision_outputs:
                memory.release(*outputs)
            memory.trim()
            text_model = model.bind_model(text_plan, memory.runtime)
            context = stages.encode_text(
                presentation.tokens,
                text_model,
                memory,
                memory.runtime,
                threshold,
                visual,
                presentation,
            )
            memory.release(*visual)
            del text_model, vision_outputs, visual
            with self.stage("reference_video_encoder"):
                video_encoder = ReferenceVideoEncoder(
                    {
                        count: model.bind_model(plan, memory.runtime)
                        for count, plan in video_encoder_plans.items()
                    },
                    memory,
                    reference_tile_size,
                )
                encoded_video = video_encoder.encode(frames)
                del video_encoder, frames
                memory.trim()
                reference_video = memory.upload(encoded_video)
                del encoded_video
            encoded_audio = tuple(
                self.encode_audio(
                    waveform,
                    model.bind_model(encoder_plan, memory.runtime),
                    model.bind_model(posterior_plan, memory.runtime),
                    memory,
                )
                for waveform, (encoder_plan, posterior_plan) in zip(
                    (soundtrack, voice), audio_plans, strict=True
                )
            )
            reference_soundtrack, reference_voice = encoded_audio
            position_values, _ = ref_layout.positions(
                spec.text_tokens,
                spec.video_frames,
                spec.video_height,
                spec.video_width,
                spec.audio_frames,
                references,
            )
            positions = memory.upload(position_values)
            channels, reference_frames, reference_height, reference_width = (
                reference_video.shape
            )
            video_noise = memory.upload(
                noise.condition_video(
                    (channels, reference_frames, reference_height, reference_width),
                    config.seed,
                )
            )
            video, audio = tuple(
                memory.upload(value) for value in noise_future.result()
            )
            del noise_future
            diffusion = model.bind_model(diffusion_plan, memory.runtime)
            with self.stage("av_diffusion"):
                diffusion.run(
                    H3Ref2vaDiffusionArgs(
                        context,
                        video,
                        audio,
                        (reference_soundtrack, reference_video, reference_voice),
                        (video_noise,),
                        positions,
                        memory.runtime,
                    )
                )
            memory.release(
                context,
                reference_soundtrack,
                reference_video,
                reference_voice,
                video_noise,
                positions,
            )
            del diffusion
            video_vae = model.bind_model(video_vae_plan, memory.runtime)
            packets = stages.decode_video(
                video,
                video_vae,
                memory,
                memory.runtime,
                threshold,
                h3.requested_frame_count(config.seconds),
            )
            del video_vae
            audio_vae = model.bind_model(audio_vae_plan, memory.runtime)
            waveform = stages.decode_audio(
                audio, audio_vae, memory, memory.runtime, threshold
            )
        output = self.root / (config.output or "outputs/h3_ref2va.mp4")
        with self.stage("mux"):
            stages.h264_aac(
                packets,
                waveform,
                output,
                fps=h3.FPS,
                sample_rate=h3.AUDIO_SAMPLE_RATE,
            )
        return output
