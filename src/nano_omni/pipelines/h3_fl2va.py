# cspell:ignore rrmse savez

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy
from tokenizers import Tokenizer

from nano_omni.core import model
from nano_omni.core.pipeline import Pipeline
from nano_omni.core.runtime import observation
from nano_omni.core.runtime.buffers import PipelineMemory
from nano_omni.core.runtime.execution import CudaRuntime
from nano_omni.core.tensor import TensorDesc
from nano_omni.models import h3
from nano_omni.models.h3 import conditioning, noise
from nano_omni.models.h3.audio_vae_decoder import (
    H3AudioVaeDecoder,
    H3AudioVaeDecoderSpec,
)
from nano_omni.models.h3.conditioning import H3Presentation, H3VisionInput
from nano_omni.models.h3.fl2va_diffusion import (
    H3Fl2vaDiffusion,
    H3Fl2vaDiffusionArgs,
    H3Fl2vaDiffusionSpec,
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
    H3VideoVaeEncoderArgs,
    H3VideoVaeEncoderSpec,
)
from nano_omni.models.h3.vision_encoder import H3VisionEncoder, H3VisionEncoderSpec

if TYPE_CHECKING:
    from config import RunConfig

DIFFUSION = (
    h3.MODEL_ROOT / "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
)
LORAS = {
    "turbo_4step": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "turbo_8step": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
}


def lora_file(lora: str | Path) -> Path | None:
    if isinstance(lora, Path):
        return lora
    return None if lora == "none" else h3.MODEL_ROOT / "loras" / LORAS[lora]


from nano_omni.models.h3.stages import (
    cuda_total_memory,
    decode_audio,
    decode_video,
    encode_text,
    encode_vision,
    h264_aac,
)


def tokenize(prompt: str) -> numpy.ndarray:
    tokenizer = Tokenizer.from_file(str(h3.TOKENIZER / "tokenizer.json"))
    token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not token_ids:
        token_ids = [151643]
    return numpy.asarray(token_ids, dtype=numpy.uint32)


def prepare_visions(generation: RunConfig) -> tuple[H3VisionInput, ...]:
    return tuple(
        conditioning.prepare_vision_input(
            conditioning.load_keyframe(
                path, generation.width, generation.height, cover=cover
            )
        )
        for path, cover in (
            (generation.first_frame, False),
            (generation.last_frame, True),
        )
        if path is not None
    )


def prepare_text(
    prompt: str, visions: tuple[H3VisionInput, ...], expected_tokens: int | None
) -> tuple[numpy.ndarray, H3Presentation | None]:
    presentation = None
    if visions:
        tokenizer = Tokenizer.from_file(str(h3.TOKENIZER / "tokenizer.json"))
        presentation = conditioning.build_presentation(tokenizer, prompt, visions)
        tokens = presentation.tokens
    else:
        tokens = tokenize(prompt)
    if expected_tokens is not None and len(tokens) != expected_tokens:
        raise ValueError(f"expected {expected_tokens} tokens, got {len(tokens)}")
    return tokens, presentation


def diffusion_spec(
    generation: RunConfig,
    tokens: int,
    visions: tuple[H3VisionInput, ...],
    presentation: H3Presentation | None,
) -> H3Fl2vaDiffusionSpec:
    frames = h3.frame_count(generation.seconds)
    return H3Fl2vaDiffusionSpec(
        text_tokens=tokens,
        lora_strength=generation.lora_strength,
        video_frames=h3.video_latent_frames(frames),
        video_height=generation.height // 16,
        video_width=generation.width // 16,
        audio_frames=h3.audio_latent_frames(frames),
        steps=generation.steps,
        seed=generation.seed,
        video_shift=generation.video_shift or h3.VIDEO_SHIFT,
        audio_shift=generation.audio_shift or h3.AUDIO_SHIFT,
        condition_frames=(1,) * len(visions),
        condition_indices=tuple(
            index
            for index, path in (
                (0, generation.first_frame),
                (frames - 1, generation.last_frame),
            )
            if path is not None
        ),
        text_visual_spans=presentation.modality_spans if presentation else (),
        attention=generation.attention,
        attention_precision=generation.attention_precision,
        mlp_chunk_tokens=generation.mlp_chunk_tokens,
    )


def encode_keyframe(
    vision: H3VisionInput,
    encoder: H3VideoVaeEncoder,
    pipeline: PipelineMemory,
    runtime: CudaRuntime,
    working_set_threshold: int,
) -> TensorDesc:
    height, width = vision.image.shape[1:3]
    with observation.stage("video_vae_encoder", working_set_threshold):
        output = pipeline.empty((24, 1, height // 16, width // 16), numpy.float32)
        input_buffer = pipeline.upload(
            conditioning.prepare_video_vae_input(vision.image)
        )
        encoder.run(
            H3VideoVaeEncoderArgs(
                input_buffer,
                output,
                runtime,
            )
        )
        pipeline.release(input_buffer)
    return output


def sample_latents(
    context: TensorDesc,
    diffusion: H3Fl2vaDiffusion,
    pipeline: PipelineMemory,
    runtime: CudaRuntime,
    working_set_threshold: int,
    condition_video: tuple[TensorDesc, ...] = (),
    initial_latents: tuple[numpy.ndarray, numpy.ndarray] | None = None,
) -> tuple[TensorDesc, TensorDesc]:
    spec = diffusion.spec
    with observation.stage("diffusion", working_set_threshold):
        if initial_latents is None:
            initial_latents = noise.generate(
                (24, spec.video_frames, spec.video_height, spec.video_width),
                spec.audio_frames,
                spec.seed,
            )
        with observation.stage("latent_upload", working_set_threshold):
            output_video, output_audio = (
                pipeline.upload(value) for value in initial_latents
            )
        condition_noise = tuple(
            pipeline.upload(
                noise.condition_video(
                    (24, frames, spec.video_height, spec.video_width), spec.seed
                )
            )
            for frames in spec.condition_frames
        )
        diffusion.run(
            H3Fl2vaDiffusionArgs(
                context,
                output_video,
                output_audio,
                runtime,
                condition_video,
                condition_noise,
            )
        )
        pipeline.release(*condition_noise)
    return output_video, output_audio


def save_latents(
    path: Path,
    video: numpy.ndarray,
    audio: numpy.ndarray,
    reference: Path | None,
    reference_label: str | None = None,
) -> None:
    """Save exact latents and their numerical difference from one fixed reference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez(path, video=video, audio=audio)
    if reference is None:
        expected_video, expected_audio = video, audio
    else:
        with numpy.load(reference) as expected:
            expected_video = expected["video"]
            expected_audio = expected["audio"]

    def metrics(actual: numpy.ndarray, expected: numpy.ndarray) -> dict[str, float | int]:
        assert actual.shape == expected.shape
        left = actual.astype(numpy.float64, copy=False).reshape(-1)
        right = expected.astype(numpy.float64, copy=False).reshape(-1)
        difference = left - right
        denominator = numpy.sqrt(numpy.mean(right * right))
        cosine_denominator = numpy.linalg.norm(left) * numpy.linalg.norm(right)
        return {
            "rrmse": float(numpy.sqrt(numpy.mean(difference * difference)) / denominator),
            "cosine": float(numpy.dot(left, right) / cosine_denominator),
            "max_abs": float(numpy.max(numpy.abs(difference))),
            "nonfinite": int(numpy.count_nonzero(~numpy.isfinite(left))),
        }

    video_metrics = metrics(video, expected_video)
    audio_metrics = metrics(audio, expected_audio)
    result = {
        "reference": reference_label,
        **{f"video_latent_{name}": value for name, value in video_metrics.items()},
        **{f"audio_latent_{name}": value for name, value in audio_metrics.items()},
    }
    path.with_name("error.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def generate(
    generation: RunConfig,
    root: Path,
    output: Path,
) -> None:
    prompt = generation.read_prompt(root)
    seconds = generation.seconds
    working_set_threshold = generation.working_set_gib * (1 << 30)
    pinned_nbytes = generation.pinned_gib * (1 << 30)
    vision_started = perf_counter()
    vision_inputs = prepare_visions(generation)
    observation.record_timing("vision_preprocess", vision_started, perf_counter())
    tokenizer_started = perf_counter()
    tokens, presentation = prepare_text(
        prompt, vision_inputs, generation.expected_tokens
    )
    observation.record_timing(
        "tokenizer", tokenizer_started, perf_counter(), tokens=len(tokens)
    )
    spec = diffusion_spec(
        generation,
        len(tokens),
        vision_inputs,
        presentation,
    )
    cuda_query_started = perf_counter()
    total_memory = cuda_total_memory()
    observation.record_timing("cuda_memory_query", cuda_query_started, perf_counter())
    lora_path = lora_file(generation.lora_value(root))
    additional_weights = () if lora_path is None else (lora_path,)

    text_encoder_spec = H3TextEncoderSpec(
        len(tokens),
        vision_spans=presentation.scatter_spans if presentation else (),
    )
    video_vae_spec = H3VideoVaeDecoderSpec(
        spec.video_frames,
        spec.video_height,
        spec.video_width,
    )
    audio_vae_spec = H3AudioVaeDecoderSpec(spec.audio_frames, stereo=2)

    with (
        ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nano-noise"
        ) as noise_pool,
        ProcessPoolExecutor(max_workers=4) as plan_pool,
    ):
        noise_future = noise_pool.submit(
            noise.generate,
            (24, spec.video_frames, spec.video_height, spec.video_width),
            spec.audio_frames,
            spec.seed,
        )
        vision_plans = tuple(
            plan_pool.submit(
                H3VisionEncoder.prepare,
                f"vision_encoder_{index}",
                h3.TEXT_ENCODER,
                H3VisionEncoderSpec(vision.patches.shape[0]),
                total_memory,
                (),
                pinned_nbytes,
                generation.workspace_nbytes,
            )
            for index, vision in enumerate(vision_inputs)
        )
        keyframe_plans = tuple(
            plan_pool.submit(
                H3VideoVaeEncoder.prepare,
                f"video_vae_encoder_{index}",
                h3.VIDEO_VAE,
                H3VideoVaeEncoderSpec((1, *vision.image.shape[1:3])),
                total_memory,
                (),
                pinned_nbytes,
                generation.workspace_nbytes,
            )
            for index, vision in enumerate(vision_inputs)
        )
        text_plan = plan_pool.submit(
            H3TextEncoder.prepare,
            "text_encoder",
            h3.TEXT_ENCODER,
            text_encoder_spec,
            total_memory,
            (),
            pinned_nbytes,
            generation.workspace_nbytes,
            label="text_encoder",
        )
        diffusion_plan = plan_pool.submit(
            H3Fl2vaDiffusion.prepare,
            "av_diffusion",
            DIFFUSION,
            spec,
            total_memory,
            additional_weights,
            pinned_nbytes,
            generation.workspace_nbytes,
        )
        video_vae_plan = plan_pool.submit(
            H3VideoVaeDecoder.prepare,
            "video_vae",
            h3.VIDEO_VAE,
            video_vae_spec,
            total_memory,
            (),
            pinned_nbytes,
            generation.workspace_nbytes,
        )
        audio_vae_plan = plan_pool.submit(
            H3AudioVaeDecoder.prepare,
            "audio_vae",
            h3.AUDIO_VAE,
            audio_vae_spec,
            total_memory,
            (),
            pinned_nbytes,
            generation.workspace_nbytes,
        )
        # No more CPU requests: finish queued work, then release the workers
        # during GPU execution instead of delaying pipeline teardown.
        plan_pool.shutdown(wait=False)
        pipeline_enter_started = perf_counter()
        with (
            PipelineMemory(
                pinned_nbytes,
                min(total_memory * 4 // 5, generation.workspace_nbytes)
                if generation.workspace_nbytes is not None
                else total_memory * 4 // 5,
                synchronization=generation.synchronization,
                vram_fraction=0.85,
            ) as pipeline,
            # Binding workers must finish before their CUDA context is released.
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="nano-compile"
            ) as compile_pool,
        ):
            observation.record_timing(
                "pipeline_enter", pipeline_enter_started, perf_counter()
            )
            runtime = pipeline.runtime
            assert runtime.total_memory == total_memory
            vision_futures = tuple(
                compile_pool.submit(model.bind_model, plan, runtime)
                for plan in vision_plans
            )
            keyframe_futures = tuple(
                compile_pool.submit(model.bind_model, plan, runtime)
                for plan in keyframe_plans
            )
            diffusion_future = compile_pool.submit(
                model.bind_model, diffusion_plan, runtime
            )
            video_vae_future = compile_pool.submit(
                model.bind_model, video_vae_plan, runtime
            )
            audio_vae_future = compile_pool.submit(
                model.bind_model, audio_vae_plan, runtime
            )
            encoded_vision = tuple(
                encode_vision(
                    vision,
                    future.result(),
                    pipeline,
                    runtime,
                    working_set_threshold,
                )
                for vision, future in zip(vision_inputs, vision_futures, strict=True)
            )
            if encoded_vision:
                level0, level1, level2, level3 = tuple(
                    pipeline.concatenate_rows(
                        tuple(item[index] for item in encoded_vision)
                    )
                    for index in range(4)
                )
                visual = (level0, level1, level2, level3)
            else:
                visual = None
            for item in encoded_vision:
                pipeline.release(*item)
            condition_video = tuple(
                encode_keyframe(
                    vision,
                    future.result(),
                    pipeline,
                    runtime,
                    working_set_threshold,
                )
                for vision, future in zip(vision_inputs, keyframe_futures, strict=True)
            )
            text_encoder = model.bind_model(text_plan, runtime)
            context = encode_text(
                tokens,
                text_encoder,
                pipeline,
                runtime,
                working_set_threshold,
                visual,
                presentation,
            )
            if visual is not None:
                pipeline.release(*visual)
            diffusion = diffusion_future.result()
            video_latent, audio_latent = sample_latents(
                context,
                diffusion,
                pipeline,
                runtime,
                working_set_threshold,
                condition_video,
                noise_future.result(),
            )
            del noise_future
            pipeline.release(context, *condition_video)
            if generation.latent_output is not None:
                latent_output = root / generation.latent_output
                save_latents(
                    latent_output,
                    pipeline.download(video_latent),
                    pipeline.download(audio_latent),
                    None
                    if generation.latent_reference is None
                    else root / generation.latent_reference,
                    None
                    if generation.latent_reference is None
                    else generation.latent_reference.as_posix(),
                )
            video_vae = video_vae_future.result()
            audio_vae = audio_vae_future.result()
            requested_frames = h3.requested_frame_count(seconds)
            video_packets = decode_video(
                video_latent,
                video_vae,
                pipeline,
                runtime,
                working_set_threshold,
                requested_frames,
            )
            del video_latent
            waveform = decode_audio(
                audio_latent,
                audio_vae,
                pipeline,
                runtime,
                working_set_threshold,
            )
            del audio_latent
            pipeline_exit_started = perf_counter()
        observation.record_timing(
            "pipeline_exit", pipeline_exit_started, perf_counter()
        )
    mux_started = perf_counter()
    h264_aac(
        video_packets, waveform, output, fps=h3.FPS, sample_rate=h3.AUDIO_SAMPLE_RATE
    )
    observation.record_timing("mux", mux_started, perf_counter())
    print(f"output={output}", flush=True)


class H3Fl2vaPipeline(Pipeline):
    def model_requests(self) -> Iterator[model.ModelRequest]:
        """Describe every model specialization selected by the pipeline inputs."""
        config = self.config
        visions = prepare_visions(config)
        token_ids, presentation = prepare_text(
            config.read_prompt(self.root), visions, config.expected_tokens
        )
        tokens = len(token_ids)
        for vision in visions:
            yield model.ModelRequest(
                H3VisionEncoder,
                h3.TEXT_ENCODER,
                H3VisionEncoderSpec(vision.patches.shape[0]),
            )
            height, width = vision.image.shape[1:3]
            yield model.ModelRequest(
                H3VideoVaeEncoder,
                h3.VIDEO_VAE,
                H3VideoVaeEncoderSpec((1, height, width)),
            )
        yield model.ModelRequest(
            H3TextEncoder,
            h3.TEXT_ENCODER,
            H3TextEncoderSpec(
                tokens,
                vision_spans=presentation.scatter_spans if presentation else (),
            ),
        )
        spec = diffusion_spec(config, tokens, visions, presentation)
        lora = lora_file(config.lora_value(self.root))
        yield model.ModelRequest(
            H3Fl2vaDiffusion,
            DIFFUSION,
            spec,
            () if lora is None else (lora,),
        )
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
            "mode": "fl2va" if self.config.first_frame is not None else "t2va",
            "width": self.config.width,
            "height": self.config.height,
            "requested_frames": h3.requested_frame_count(self.config.seconds),
            "frames": frames,
            "video_latent_frames": h3.video_latent_frames(frames),
            "audio_latent_frames": h3.audio_latent_frames(frames),
            "steps": self.config.steps,
            "seed": self.config.seed,
            "lora": self.config.lora or "none",
            "lora_strength": self.config.lora_strength,
            "video_shift": self.config.video_shift or h3.VIDEO_SHIFT,
            "audio_shift": self.config.audio_shift or h3.AUDIO_SHIFT,
        }

    def prepare(self) -> None:
        config = self.config
        for checkpoint in (h3.TEXT_ENCODER, DIFFUSION, h3.VIDEO_VAE, h3.AUDIO_VAE):
            self.prepare_file(
                h3.MODEL_ID,
                checkpoint.relative_to(h3.MODEL_ROOT).as_posix(),
                checkpoint,
            )
        for name in h3.TOKENIZER_FILES:
            self.prepare_file(h3.TOKENIZER_ID, name, h3.TOKENIZER / name)
        lora = config.lora_value(self.root)
        path = lora_file(lora)
        if isinstance(lora, Path):
            if not lora.is_file():
                raise FileNotFoundError(lora)
        elif path is not None:
            name = LORAS[lora]
            self.prepare_file(h3.MODEL_ID, f"loras/{name}", path)

    def execute(self) -> Path:
        output = self.config.output or self.root / "outputs/h3_fl2va.mp4"
        generate(
            self.config,
            self.root,
            output,
        )
        return output
