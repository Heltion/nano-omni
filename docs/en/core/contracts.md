# Layer contracts

[中文](../../zh/core/contracts.md) · [Core overview](README.md)

The public contracts carry shape-dependent work through Kernel, Op, Model, and Pipeline. Each layer owns one decision: Kernel defines one launch, Op binds semantic inputs and expands to launches, Model orders operations and plans memory, and Pipeline coordinates complete generation.

[`core/__init__.py`](../../../src/nano_omni/core/__init__.py) defines the package boundary for the four inference layers.

## Kernel

[`core/kernel.py`](../../../src/nano_omni/core/kernel.py) defines `Arguments`, `Workload`, `Config`, and the only invocation entity, `Kernel`.

A Kernel instance owns four values:

| Field | Meaning |
| --- | --- |
| `arguments` | Invocation tensors plus semantic values such as stride, padding, epsilon, and fused-operation switches |
| `workload` | Immutable dimensions and compile-time branches validated and copied from arguments |
| `config` | The Config selected for this workload |
| `compiled` | The prepared TileLang specialization |

A concrete Kernel exposes a stable `name`, stores its direct TileLang callable in `program`, and implements:

| Method | Responsibility |
| --- | --- |
| `make_arguments(workload)` | Build unaddressed descriptors used by isolated measurement |
| `make_workload(arguments)` | Validate arguments and recover the specialization |
| `make_config(workload)` | Select the measured production Config for the workload |
| `ref_program(arguments)` | Write the mathematical reference result into addressed arguments |
| `tops(arguments)` | Estimate logical MMA work by complete `MmaType`; the base result is empty |

Compile-time semantic values intentionally appear in both `Arguments` and `Workload`. The Op constructs only `Arguments`; `make_workload()` validates and copies the specialization values. This keeps `ref_program(arguments)` self-contained while giving compilation and cache identity an immutable Workload.

For a slice with explicit strides, `Arguments` and `Workload` both retain the logical dimensions and strides. Its `TensorDesc` describes only the byte span reachable from the shifted pointer. For example, `volume_to_tokens` lowers a CFHW tile to a one-dimensional span and reconstructs CFHW indexing from the duplicated strides. This keeps generic allocation bounds exact without adding a slice wrapper or a per-origin specialization.

`Arguments.map()` rebuilds the same argument structure while replacing every `TensorDesc`. `Arguments.values(include_absent=True)` emits tensor ABI leaves in field order and retains `None` tensor slots. `Arguments.dynamic_parameters()` supplies named runtime scalars separately from tensor storage. The Kernel base uses these operations for addressing, compilation placeholders, and submission.

`Kernel.compile()` expands `workload.model_dump()` and `config.model_dump()` into `program.get_tir()` and compiles that TIR. `Kernel.submit()` only submits an already compiled invocation on the active CUDA stream.

## TensorDesc

[`core/tensor.py`](../../../src/nano_omni/core/tensor.py) provides the single tensor descriptor used by every layer. It contains `dtype`, `shape`, a `TensorKind`, and a variable-length `info` tuple. `EMPTY` describes measurement shapes; `ACTIVATION`, `SCRATCH`, `SCALAR`, and `WEIGHT` describe logical model regions; `INPUT`, `OUTPUT`, and `WORKSPACE` describe runtime bindings; `HOST` and `DEVICE` carry addressed memory.

`TensorDesc.view()` changes shape and byte offset while retaining storage identity. `num_bytes` checks extents, `data_ptr()` resolves an addressed device pointer, and `as_torch()` creates a zero-copy CUDA view for reference programs.

`FP8_E4M3` represents signed E4M3 tensor values, while `FP8_UE4M3` represents NVFP4 block scales. Both occupy one byte but have different numeric semantics. Because safetensors has no UE4M3 tag, H3 weight binding reinterprets block scales as `FP8_UE4M3` according to the FP4 projection contract.

Descriptor object identity is meaningful during isolated measurement: repeated use of the same descriptor denotes intentional aliasing. Independent inputs and outputs therefore require independently constructed descriptors even when their dtype and shape match.

## Op

[`core/op.py`](../../../src/nano_omni/core/op.py) binds model activations and weights to an operation. Construction receives input descriptors, `with_weights()` attaches its weight structure, and `to()` finalizes output descriptors. A concrete `kernels(scratch)` method returns ordered Kernel instances and reserves its private temporary regions from the supplied `ScratchLayout`.

`Op.prepare()` checks that bindings are complete, creates one op-local scratch layout, obtains the Kernel instances, and returns them with the required scratch extent. It does not compile or execute them.

An activation slice is expressed directly in `TensorDesc.info=(identity, byte_offset)`. This lets a Model split tokens or write multiple outputs without introducing slice wrapper classes.

## Model

[`core/model.py`](../../../src/nano_omni/core/model.py) owns checkpoint metadata and the executable action list. `FileMetadata` reads safetensors headers into `FILE TensorDesc` values, while `ModelMetadata` preserves file order.

A `PlannedModel` implements:

| Method | Result |
| --- | --- |
| `plan(metadata, spec, total_memory, workspace_limit)` | Model config, ordered actions, and workspace bytes |
| `run(args)` | Model-specific execution result |

`plan()` reserves activation IDs, builds Ops, binds registered weights, and calls `planning.operators.commands()`. That planner lays out activation, shared scratch, immutable scalar data, and the weight arena; it also inserts transfer dependencies. The resulting action list directly contains Kernel instances, copies, and synchronization operations.

`kernel_inventory()` reruns the same plan and returns its Kernel instances in execution order. Measurement code groups those instances by concrete type and workload when it needs launch counts. No second inventory-record or command wrapper represents a Kernel.

`PlannedModel.prepare()` plans staging before returning `ModelPreparation`. `execute()` resolves the saved submission plan against the current inputs, outputs, checkpoint paths, and workspace, then submits it. Pipeline warmup calls `runtime.compilation.prepare()` separately.

## Pipeline

[`core/pipeline.py`](../../../src/nano_omni/core/pipeline.py) defines the generation lifecycle. A concrete Pipeline implements `prepare()`, `execute()`, `workload()`, and `model_requests()`.

`run()` applies the configured working-set limit, prepares missing local assets, and executes generation. `warmup()` prepares assets, obtains each distinct Model request, collects their Kernel inventories, and compiles each specialization once across the whole pipeline without staging weights, submitting a Kernel, or producing media. Model preparation may overlap across pipeline stages during normal execution.

## Metrics

`Kernel.tops(arguments)` returns estimated logical operations keyed by full MMA signatures such as `F4F4F32`, `F8F8F16`, and `F8F8F32`. Dynamic Sol masks are unavailable during planning, so their configured statistical selection probability is used. `PlannedModel.tops` sums these counts across the ordered Kernel instances. Pipeline reports combine those logical counts with the normal end-to-end interval; NSys-derived kernel and GPU percentages use the profiled interval.

