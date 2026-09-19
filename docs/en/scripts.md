# Scripts

[中文](../zh/scripts.md)

See [getting started](getting-started.md) for installation and
[benchmarking](benchmarking.md) for timing and metric definitions.

## Run

| Script | Purpose |
| --- | --- |
| [`scripts/measure.ps1`](../../scripts/measure.ps1) | Run the measurement suite and regenerate the homepage. |
| [`scripts/run/config.py`](../../scripts/run/config.py) | Load and validate shared YAML configuration. |
| [`scripts/run/nano-omni.py`](../../scripts/run/nano-omni.py) | Run Nano-Omni and write ordinary or NSys measurement JSON. |
| [`scripts/run/comfy-ui.py`](../../scripts/run/comfy-ui.py) | Execute a complete ComfyUI API prompt and write its measurement JSON. |
| [`scripts/run/common.py`](../../scripts/run/common.py) | Provide shared configuration, process, timing, video-metric, and profiling utilities. |

Run commands from the repository root. Both runners accept `--config` and
configuration overrides such as `--output`. Paths in YAML are relative to the
repository root and serialize with forward slashes. Absolute, drive-relative,
home-relative, and parent-traversal paths are rejected, including custom LoRA
paths. Ordinary runs default to `--warmup 0 --repetitions 1`.

```powershell
uv run scripts/run/nano-omni.py --config perf/h3-fl2va-official-dense-bf16-4step/config.yaml --output outputs/example.mp4
```

The ComfyUI runner requires the submodule and experiment dependencies:

```powershell
git submodule update --init --recursive
uv run --group experiment scripts/run/comfy-ui.py --config perf/h3-fl2va-official-dense-bf16-4step/config.yaml
```

Each ordinary sample measures a fresh child process through exit.
Nano-Omni's `--warmup 1` compiles selected kernels without a warmup generation;
ComfyUI warmup executes full unmeasured generations. Neither option clears
OS or compiler caches. Optional `--repetitions` and `--append` support local
diagnostics; the published benchmark uses one ordinary run per configuration.
Append requires a complete report with matching configuration, workload,
environment, and backend, and is unavailable with NSys.

Nano-Omni's `--hardware` supplies nominal throughput for `tensor_percent`.
`--nsys` profiles one Nano-Omni child and writes a separate `-nsys.json` report;
it requires `--repetitions 1` and Nsight Systems on PATH. Compilation warmup
runs outside NSys. The ComfyUI report records its selected attention backend.

Reports contain a random run ID, resolved configuration, timing, and runtime
metadata. A new report starts as `incomplete`; unsuccessful execution cannot
leave a previous successful report current. ComfyUI accepts output from the
current child and propagates monitor-thread failures. Nano-Omni quality reports
share the run ID and contain decoded-video PSNR and SSIM when
`quality_reference` is configured. `memory_samples` records per-run memory
observations, with `memory` containing the last observation.

Run the complete suite:

```powershell
.\scripts\measure.ps1
```

Each main configuration runs Nano-Omni once and ComfyUI once, followed by a
separate Nano-Omni NSys run. The remaining attention configurations each run
Nano-Omni once normally and once under NSys. Commands execute serially, stop on
failure, and overwrite results beside configurations under `perf/`.
Regenerate only the homepage with `python hooks/readme.py`.

[`scripts/experiment/`](../../scripts/experiment/) is ignored except for its
`.gitignore` and holds temporary local experiments.

## Tune

| Script | Purpose |
| --- | --- |
| [`scripts/tune.py`](../../scripts/tune.py) | Check and measure an explicit kernel workload and configuration. |
| [`scripts/update.py`](../../scripts/update.py) | Update a model inventory or kernel measurements. |

Tune accepts `--kernel`, JSON `--workload`, and a complete JSON `--config`.
It checks the candidate against its reference on workload-shaped random inputs,
then measures it. `--record perf/<instance>/tuning.csv` inserts or replaces a
candidate keyed by kernel, workload, and configuration. `--trials` optionally
repeats kernel measurements; `--dynamic` overrides supported runtime scalars
such as `num_protected_blocks`. Effective scalar values are part of the candidate
identity. Kernel tuning and end-to-end benchmark timing are separate operations.

Update selects `kernels.csv` with `--instance`:

| Subcommand | Operation |
| --- | --- |
| `model` | Rebuild the specified model's inventory, preserve matching measurements and other models, and initialize new measurements to zero. |
| `kernel` | Measure the row identified by model, kernel, and workload using production `make_config(workload)`. |
| `all-kernels` | Measure rows with zero latency. |

CSV columns are `model`, `kernel`, `percent`, `latency`, `launch`, `mem%`, `ncu%`,
`tops`, `workload`, and `config`. Latency is per-launch milliseconds; contribution
percentages include launch counts. Each update recalculates percentages and
sorts rows by contribution.

Kernel inventory and tuning CSV files are local outputs, not published benchmark
artifacts. Generate an inventory with `scripts/update.py model` before measuring
its rows. Only README configurations, inputs, source notes, and result files are
tracked under `perf/`.
