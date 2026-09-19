"""CPU latent initialization matching the official SGLang H3 recipe."""

import hashlib
import json

import numpy


def generate(
    video_shape: tuple[int, ...], audio_frames: int, seed: int = 0
) -> tuple[numpy.ndarray, numpy.ndarray]:
    # Import in the CPU worker: Torch startup overlaps the text encoder.
    import torch

    video = torch.randn(
        (1, *video_shape),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
        device="cpu",
    )[0]
    # Each modality restarts its generator. Audio is drawn in token-row order,
    # then stored as [latent channel, stereo channel, time] for both runtimes.
    rows = torch.randn(
        (2 * audio_frames, 32),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
        device="cpu",
    )
    audio = rows.reshape(2, audio_frames, 32).permute(2, 0, 1).contiguous()
    arrays = (video.numpy(), audio.numpy())
    print(
        "initial_latents="
        + json.dumps(
            {
                "seed": seed,
                "method": "sglang_h3_cpu_fp32_independent_generators",
                "video_shape": list(arrays[0].shape),
                "audio_shape": list(arrays[1].shape),
                "video_sha256": hashlib.sha256(arrays[0].tobytes()).hexdigest(),
                "audio_sha256": hashlib.sha256(arrays[1].tobytes()).hexdigest(),
            }
        ),
        flush=True,
    )
    return arrays


def condition_video(shape: tuple[int, int, int, int], seed: int) -> numpy.ndarray:
    """Draw visual conditioning noise in patch-row order, then unpack it."""
    import torch

    channels, frames, height, width = shape
    rows = torch.randn(
        (frames * (height // 2) * (width // 2), channels * 4),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    )
    return (
        rows.reshape(frames, height // 2, width // 2, channels, 2, 2)
        .permute(3, 0, 1, 4, 2, 5)
        .reshape(shape)
        .contiguous()
        .numpy()
    )
