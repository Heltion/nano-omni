# Getting started

[中文](../zh/getting-started.md)

## Requirements

Nano-Omni provides MiniMax H3 text-to-audio/video (T2VA), first/last-frame
conditioning (FL2VA), and reference-audio/video conditioning (REF2VA).
It requires an NVIDIA GPU, a compatible driver and CUDA development environment,
Python 3.12 or newer, Git, and uv. Kernels compile through NVRTC; there is no
CPU inference backend. The benchmark environment uses Windows, RTX 5060 Ti,
CUDA 13.0, and TileLang 0.1.14. Exact runtime versions are in each measurement JSON.

## Install and run

From PowerShell:

```powershell
git clone https://github.com/Heltion/nano-omni.git
cd nano-omni
uv sync --locked
uv run scripts/run/nano-omni.py --help
uv run scripts/run/nano-omni.py --config perf/h3-fl2va-official-dense-bf16-4step/config.yaml --output outputs/example.mp4
```

Run from the repository root. Paths in YAML are relative to that root, not to the
YAML directory. `--output` selects the destination; a normal run writes an MP4
and a sibling JSON report. The example selects dense BF16 DiT attention, not an
all-BF16 model. Other attention configurations are listed in the homepage table.

Missing model files download automatically through ModelScope into `models/`.
Existing files are reused. The configured sources are `Comfy-Org/MiniMax-H3`
and `Qwen/Qwen2.5-7B-Instruct`. A custom LoRA path must already exist.
The FL2VA example uses:

```text
models/
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
  vae/minimax_h3_video_vae_fp16.safetensors
  vae/minimax_h3_audio_vae_fp32.safetensors
  loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
  tokenizers/minimax_h3/{merges.txt,tokenizer.json,tokenizer_config.json,vocab.json}
```

Allow disk space for model assets and RAM for pinned buffers and activations.
The first invocation can include downloads and compilation. `--warmup 1`
prepares assets and compiles selected kernels in a separate child before the
measured generation; it is not a download-only command.

## Configuration

| Task | Configuration |
| --- | --- |
| T2VA | `perf/h3-t2va-official-dense-bf16-4step/config.yaml` |
| FL2VA | `perf/h3-fl2va-official-dense-bf16-4step/config.yaml` |
| REF2VA | `perf/h3-ref2va-0.4mp-dense-bf16-4step/config.yaml` |

Provide exactly one of `prompt` and `prompt_file`. FL2VA accepts a first frame
and optionally a last frame; a last frame alone is rejected. REF2VA requires
both reference video and reference audio. Width and height must be at least
256 and multiples of 32. The named FL2VA `turbo_4step` LoRA requires 1344x768
and four steps; `turbo_8step` requires eight steps.

`attention` selects `dense` or `sol`; `attention_precision` selects `bf16`,
`int8_fp8`, or `nvfp4`. Use `--attention-precision bf16` to override the precision.
Memory settings include `working_set_gib`, `pinned_gib`, and `workspace_gib`.
See the [scripts guide](scripts.md) for runner options.

## ComfyUI and profiling

The ComfyUI baseline requires the submodule and the experiment dependency group:

```powershell
git submodule update --init --recursive
uv run --group experiment scripts/run/comfy-ui.py --config perf/h3-fl2va-official-dense-bf16-4step/config.yaml
```

The baseline executes a complete API prompt through ComfyUI's `PromptExecutor`.
For Nano-Omni profiling, add `--nsys`; this requires Nsight Systems on PATH.
Ordinary generation does not require Nsight Systems.
[Benchmarking](benchmarking.md) defines the timing boundary and metrics.

## Development

The [core guide](core/README.md) describes planning and runtime contracts.
Edit `hooks/readme.py` to change the generated homepage and run
`python hooks/readme.py`. English and Chinese documentation are maintained
together. Run tests in the project environment with
`uv run python -m unittest discover -s tests -v`.

On failure, consult the runner's log and check assets, CUDA compatibility,
memory settings, and configuration. Use `outputs/` for local experiments:
benchmark commands overwrite the tracked results beside their configurations.
See [licensing](licensing.md) for code and asset terms.
