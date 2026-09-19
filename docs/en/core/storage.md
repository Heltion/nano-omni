# Tensor storage and layout

[中文](../../zh/core/storage.md) · [Core overview](README.md)

The core uses one immutable `TensorDesc` from model construction through runtime
execution. A descriptor contains its `dtype`, `shape`, storage `kind`, and
`info`. It describes a view; the object that owns the underlying file, host
memory, or device allocation remains outside the descriptor.

## Position and size

An `EMPTY` descriptor has `info=()`. Every other current kind uses
`(identifier_or_base, byte_offset)`. Before addressing, the
first component identifies a file, input, output, activation slot, weight, or
workspace. After addressing it is a host or CUDA base pointer. The second
component is always a byte offset.

`num_bytes` is the product of `shape` and `DType.itemsize`. Packed FP4 data uses
`DType.FP4`, whose descriptor shape represents the physical bytes consumed by the
kernel.

`data_ptr()` is valid only after addressing produced a `DEVICE` descriptor. It
adds the byte offset to the device base pointer. `device` returns the shared CUDA
device view used by generated TileLang launchers; this project targets device 0.
`as_torch()` exposes the same addressed range as a zero-copy contiguous Torch
view. Reference programs and validation use this single conversion point.

## Tensor kinds

| Kind | Meaning |
| --- | --- |
| `EMPTY` | Declares only dtype and shape; storage is not assigned. |
| `FILE` | A byte range in an indexed checkpoint file. |
| `INPUT`, `OUTPUT` | A caller-owned runtime buffer. |
| `ACTIVATION` | A model activation slot before workspace placement. |
| `SCRATCH` | An op-local temporary before workspace placement. |
| `WEIGHT` | A registered checkpoint tensor before residency planning. |
| `WORKSPACE` | A planned byte range in the model workspace. |
| `PINNED` | A planned byte range in the pinned host staging pool. |
| `HOST`, `DEVICE` | A fully addressed host or CUDA view. |

Planning changes `kind` and `info` with `dataclasses.replace`; it preserves
the dtype, shape, and byte count. No stage wraps a descriptor in another tensor
reference type.

## Activation and scratch layout

[`core/layout.py`](../../../src/nano_omni/core/layout.py) contains the two logical
layout builders.

`ActivationLayout.require(position, num_bytes, alignment)` merges repeated
requirements for an activation slot. A sliced position contributes
`byte_offset + num_bytes` to that slot's required extent. `layout()` places slots
in first-declaration order, honors each slot's largest alignment, and returns the
slot offsets plus the total activation extent.

`ScratchLayout.reserve(name, dtype, shape, alignment)` appends one uniquely named
temporary and returns its `SCRATCH TensorDesc`. Every op owns a fresh scratch
layout. The model planner reserves the largest op scratch extent, so sequential
ops reuse the same workspace region.

All sizes and offsets in these APIs are bytes. Internal alignment, uniqueness,
and bounds conditions are asserted where the descriptor is created or bound.

## Checkpoint weights

`core/model.py` reads safetensors headers into `FILE TensorDesc` values. The file
index selects an entry in the model's
ordered file list, and the info byte offset points directly at the tensor payload.
Header byte ranges are checked against `TensorDesc.num_bytes`.

[`core/weights.py`](../../../src/nano_omni/core/weights.py) assigns stable weight
IDs in first-reference order. `WeightRegistry.reference()` returns a `WEIGHT
TensorDesc`; repeated references to the same file range share the same ID. Row
partitions preserve the tensor dtype and trailing shape while adjusting the
leading dimension and byte offset.

The registry describes weights without loading their payload. The planner later
chooses resident workspace ranges and schedules transfers for the remaining
weights. Addressing finally replaces planned positions with live host or device
pointers, as described in [planning](planning.md) and [runtime](runtime.md).

## Resource boundaries

Pipelines and models also pass `TensorDesc` directly. Allocation capacities and
active state stay in the allocator.
[`core/runtime/synchronization.py`](../../../src/nano_omni/core/runtime/synchronization.py)
selects the CUDA dependency mechanism without changing tensor ownership.

The Windows resource boundary is implemented by
[`core/platform/__init__.py`](../../../src/nano_omni/core/platform/__init__.py),
[`core/platform/memory.py`](../../../src/nano_omni/core/platform/memory.py), and
[`core/platform/process.py`](../../../src/nano_omni/core/platform/process.py).
These modules limit the process working set and track child processes; their
limits are separate from the CUDA workspace and pinned staging layouts.
