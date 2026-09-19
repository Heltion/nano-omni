"""Shared checkpoints and tokenizer for the MiniMax H3 pipeline family."""

from pathlib import Path

MODEL_ID = "Comfy-Org/MiniMax-H3"


TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"


MODEL_ROOT = Path(__file__).resolve().parents[4] / "models"


TEXT_ENCODER = MODEL_ROOT / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"


VIDEO_VAE = MODEL_ROOT / "vae/minimax_h3_video_vae_fp16.safetensors"


AUDIO_VAE = MODEL_ROOT / "vae/minimax_h3_audio_vae_fp32.safetensors"


TOKENIZER = MODEL_ROOT / "tokenizers/minimax_h3"


TOKENIZER_FILES = (
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


FPS = 24
AUDIO_SAMPLE_RATE = 32_000
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def requested_frame_count(seconds: float) -> int:
    return round(seconds * FPS)


def frame_count(seconds: float) -> int:
    frames = max(5, requested_frame_count(seconds))
    return frames + (5 - frames % 17) % 17


def video_latent_frames(frames: int) -> int:
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def audio_latent_frames(frames: int) -> int:
    return round(frames / FPS * 40)
