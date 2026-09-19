# Benchmarking

[中文](../zh/benchmarking.md)

## Workloads and timing

Each configuration under `perf/` specifies the input, dimensions, seed, sampling
settings, and output path. Main comparisons use dense BF16 DiT attention on
both sides. This does not make every model stage or weight BF16. Requested
frames and internally padded frames are recorded separately.

Published latency is one complete process run per configuration. The maintainer
reports run-to-run variation of at most 0.1% on the benchmark machine, so repeated
runs are not required for this measurement procedure. Speedup is ComfyUI time
divided by Nano-Omni time; values below one indicate that Nano-Omni is slower.

Both ordinary runners measure child-process wall time from launch through exit,
including setup, model loading, planning, generation, decoding, output, and
teardown. Missing Nano-Omni assets download within that boundary. Prepare assets
before comparing generation times. The temperature check before launch is outside
the timer. These are end-to-end times, not attention-only or sampling-only times.

The runners use fresh child processes and existing OS/compiler caches.
`warmup=0` skips extra warmup; it does not clear caches. Nano-Omni's `--warmup 1`
prepares assets and compiles kernels without a warmup generation. ComfyUI warmup
executes a complete unmeasured generation. Ordinary benchmark runs use
`--warmup 0 --repetitions 1`.

## ComfyUI execution

The baseline uses the repository's ComfyUI submodule and executes one complete
API prompt through `PromptExecutor`. It uses the default attention dispatcher;
the actual implementation and runtime environment are recorded in the report.
Runtime metadata also records dynamic VRAM, fast-disk, H3 modality-specific
noise, pinned-memory state, and the host working-set limit. The runner maps the
configured seed, LoRA strength, and sampling shifts to the nodes, and rejects
unsupported attention or LoRA selections before measurement.

Both implementations use the configured host-memory budget with their own
memory management. A shared budget is not a claim of identical allocation or
offloading policies. The resolved configuration, workload, environment, and
backend fields describe the executed workload.

## Reports and metrics

`nano.json` and `comfy.json` contain ordinary timing, resolved configuration,
workload, and runtime metadata. Each run has a random `run_id`; quality reports
use the corresponding Nano-Omni ID. The homepage is generated from these reports,
`nano-nsys.json`, and `error.json`. Unsuccessful reports, invalid times, and
mismatched comparison workloads are excluded. `mean_seconds` is the single elapsed time when
`repetitions=1`.

`nano-nsys.json` describes a separate profiling run. Kernel% is the union of CUDA
kernel intervals divided by capture duration; GPU% includes captured copy,
memset, and video-engine work as well. `gpu_complete` records coverage. Keep
profiling times separate from ordinary latency.

`Tensor%` (`tensor_percent` in JSON) is an estimate:

```text
100 * sum(logical_MMA_operations / nominal_dense_operations_per_second)
    / ordinary_wall_seconds
```

Nominal rates come from `perf/hardware.json`. Sol uses a statistical estimate for
dynamic block selection. This is not measured Tensor Core utilization. It cannot
be subtracted from a separate profile's Kernel% or GPU% to obtain CPU time.

The FL2VA attention table compares decoded RGB frames against the seed-0
Nano-Omni dense BF16 output. PSNR uses aggregate RGB mean squared error; SSIM uses
an 11x11 Gaussian window with sigma 1.5 and is averaged across frames. These
quantify output agreement, not absolute generation quality, audio quality, or
quality parity with ComfyUI. The BF16 row compares the reference with itself. Encoded-video comparisons include compression differences.

## Run the suite

From the repository root with dependencies, model assets, and Nsight Systems
available:

```powershell
.\scripts\measure.ps1
```

For each main configuration the suite runs Nano-Omni once, ComfyUI once, then
Nano-Omni once under NSys. Each remaining attention configuration has one ordinary
Nano-Omni run and one NSys run. The script stops at the first failed command,
writes results beside the configurations, and regenerates the homepage. It
overwrites those result files; use a separate output path for local experiments.
See the [scripts guide](scripts.md) for individual commands.
