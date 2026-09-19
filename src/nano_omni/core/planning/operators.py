"""Build executable commands from explicit operators and fixed storage regions."""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Iterable
from typing import TypedDict

from nano_omni.core.kernel import Kernel, intern_kernels
from nano_omni.core.layout import ActivationLayout, ScalarLayout, aligned
from nano_omni.core.op import Op
from nano_omni.core.planning.scheduling import Action, Copy, RecordEvent, WaitEvent
from nano_omni.core.planning.weights import Upload, WeightPlan, plan_weights
from nano_omni.core.tensor import TensorDesc, TensorKind
from nano_omni.core.weights import WeightRegistry


class CommandStatistics(TypedDict):
    workspace_bytes: int
    activation_bytes: int
    scratch_bytes: int
    scalar_bytes: int
    weight_bytes: int
    h2d_bytes: int
    h2d_count: int
    kernel_count: int
    activation_offsets: dict[int, int]


def commands(
    operators: Iterable[Op],
    activations: ActivationLayout,
    registry: WeightRegistry,
    capacity: int,
    scalars: ScalarLayout | None = None,
) -> tuple[list[Action], CommandStatistics]:
    with intern_kernels():
        prepared = [op.prepare() for op in operators]
    offsets, activation_bytes = activations.layout()
    scratch_bytes = aligned(max((size for _, size in prepared), default=0), 256)
    scalar_bytes = aligned(scalars.num_bytes if scalars is not None else 0, 256)
    scalar_base = activation_bytes + scratch_bytes
    weight_base = scalar_base + scalar_bytes
    if weight_base >= capacity:
        raise MemoryError(
            f"activation/scratch/scalars {weight_base} exceeds workspace {capacity}"
        )
    calls = [call for op_calls, _ in prepared for call in op_calls]
    weights: WeightPlan = plan_weights(calls, registry.entries, capacity - weight_base)
    uploads: collections.defaultdict[int, list[tuple[int, Upload]]] = (
        collections.defaultdict(list)
    )
    waits: collections.defaultdict[int, list[int]] = collections.defaultdict(list)
    for index, upload in enumerate(weights.uploads):
        uploads[upload.after_kernel].append((index, upload))
        waits[upload.before_kernel].append(index)
    output: list[Action] = []
    event_base = len(calls)
    if scalars is not None and scalars.regions:
        scalar_event = event_base + len(weights.uploads)
        for region in scalars.regions:
            output.append(
                Copy(
                    region.data,
                    dataclasses.replace(
                        region.tensor,
                        kind=TensorKind.WORKSPACE,
                        info=(0, scalar_base + region.tensor.info[1]),
                    ),
                )
            )
        output.append(RecordEvent(scalar_event, "copy"))
        output.append(WaitEvent(scalar_event, "compute"))

    def emit_uploads(after: int) -> None:
        for index, upload in uploads[after]:
            if after >= 0:
                output.append(WaitEvent(after, "copy"))
            entry = registry.entries[upload.weight]
            output.append(
                Copy(
                    entry,
                    TensorDesc(
                        entry.dtype,
                        entry.shape,
                        TensorKind.WORKSPACE,
                        (0, weight_base + upload.offset),
                    ),
                )
            )
            output.append(RecordEvent(event_base + index, "copy"))

    emit_uploads(-1)
    bound_calls: dict[tuple[int, tuple[tuple[int, int], ...]], Kernel] = {}
    for index, call in enumerate(calls):
        for upload in waits[index]:
            output.append(WaitEvent(event_base + upload, "compute"))

        def bind(ref: TensorDesc, index: int = index) -> TensorDesc:
            identity, relative = ref.info
            if ref.kind == TensorKind.ACTIVATION:
                offset = offsets[identity] + relative
                limit = activations.requirements[identity][0]
            elif ref.kind == TensorKind.SCRATCH:
                offset = activation_bytes + relative
                limit = scratch_bytes
            elif ref.kind == TensorKind.WEIGHT:
                offset = weight_base + weights.bindings[index][identity] + relative
                limit = registry.entries[identity].num_bytes
            elif ref.kind == TensorKind.SCALAR:
                offset = scalar_base + relative
                limit = scalar_bytes
            else:
                return ref
            assert relative >= 0 and relative + ref.num_bytes <= limit, (
                f"kernel {index} reference exceeds region: {ref}"
            )
            return dataclasses.replace(ref, kind=TensorKind.WORKSPACE, info=(0, offset))

        key = id(call), tuple(weights.bindings[index].items())
        bound_call = bound_calls.get(key)
        if bound_call is None:
            bound_call = call.map_arguments(bind)
            bound_calls[key] = bound_call
        output.append(bound_call)
        if uploads[index]:
            output.append(RecordEvent(index, "compute"))
            emit_uploads(index)
    used_weight_bytes = max(
        (upload.offset + aligned(upload.nbytes, 256) for upload in weights.uploads),
        default=0,
    )
    return output, {
        "workspace_bytes": weight_base + used_weight_bytes,
        "activation_bytes": activation_bytes,
        "scratch_bytes": scratch_bytes,
        "scalar_bytes": scalar_bytes,
        "weight_bytes": capacity - weight_base,
        "h2d_bytes": sum(upload.nbytes for upload in weights.uploads),
        "h2d_count": len(weights.uploads),
        "kernel_count": len(calls),
        "activation_offsets": offsets,
    }
