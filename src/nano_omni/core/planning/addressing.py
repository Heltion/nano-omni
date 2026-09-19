"""Resolve logical tensor positions against the active runtime."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable

import numpy

from nano_omni.core.kernel import Kernel
from nano_omni.core.planning.scheduling import Action, Copy
from nano_omni.core.planning.submission import StreamPlan
from nano_omni.core.runtime.execution import current
from nano_omni.core.tensor import TensorDesc, TensorKind


def device(value: TensorDesc) -> TensorDesc:
    """Resolve an input, output, or workspace slice to a device address."""
    assert len(value.info) == 2, "logical device tensor requires identity and offset"
    identity, offset = value.info[0], value.info[1]
    state = current()
    if value.kind == TensorKind.INPUT:
        buffer = state.inputs[identity]
    elif value.kind == TensorKind.OUTPUT:
        buffer = state.outputs[identity]
    else:
        assert value.kind == TensorKind.WORKSPACE
        buffer = state.pointer
    assert 0 <= offset and offset + value.num_bytes <= buffer.num_bytes
    return dataclasses.replace(
        value,
        kind=TensorKind.DEVICE,
        info=(buffer.data_ptr(), offset),
    )


def address_copy(
    item: Copy, resolve: Callable[[TensorDesc], TensorDesc] = device
) -> Copy:
    """Resolve both endpoints without changing the command type."""
    state = current()
    owner: object | None = None
    file_position: tuple[int, int] | None = None
    if isinstance(item.source, bytes):
        owner = numpy.frombuffer(item.source, dtype=numpy.uint8)
        source = TensorDesc(
            item.destination.dtype,
            item.destination.shape,
            TensorKind.HOST,
            (owner.ctypes.data, 0),
        )
    elif item.source.kind == TensorKind.FILE:
        assert len(item.source.info) == 2, "file tensor requires file index and offset"
        file_position = (item.source.info[0], item.source.info[1])
        source = dataclasses.replace(
            item.source,
            kind=TensorKind.HOST,
            info=(0, 0),
        )
    elif item.source.kind == TensorKind.PINNED:
        source = dataclasses.replace(
            item.source,
            kind=TensorKind.HOST,
            info=(state.pinned_address, item.source.info[1]),
        )
    else:
        source = resolve(item.source)

    if item.destination.kind == TensorKind.PINNED:
        destination = dataclasses.replace(
            item.destination,
            kind=TensorKind.HOST,
            info=(state.pinned_address, item.destination.info[1]),
        )
    else:
        destination = resolve(item.destination)
    return dataclasses.replace(
        item,
        source=source,
        destination=destination,
        owner=owner,
        file_position=file_position,
    )


def address_action(
    item: Action, resolve: Callable[[TensorDesc], TensorDesc] = device
) -> Action:
    if isinstance(item, Copy):
        return address_copy(item, resolve)
    if isinstance(item, Kernel):
        return item.map_arguments(resolve)
    return item


def address_plan(plan: StreamPlan) -> StreamPlan:
    """Resolve stream actions in the active runtime without changing dependencies."""
    current()
    addressed: dict[TensorDesc, TensorDesc] = {}
    addressed_kernels: dict[int, Kernel] = {}

    def resolve(value: TensorDesc) -> TensorDesc:
        result = addressed.get(value)
        if result is None:
            result = device(value)
            addressed[value] = result
        return result

    def address(item: Action) -> Action:
        if not isinstance(item, Kernel):
            return address_action(item, resolve)
        key = id(item)
        cached = addressed_kernels.get(key)
        if cached is not None:
            return cached
        result = copy.copy(item)
        result.arguments = item.arguments.map(resolve)
        addressed_kernels[key] = result
        return result

    return StreamPlan(
        [address(item) for item in plan.host],
        [address(item) for item in plan.copy],
        [address(item) for item in plan.compute],
        plan.dependencies,
    )
