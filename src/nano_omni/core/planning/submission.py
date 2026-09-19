"""Partition actions into three ordered streams and record their dependencies."""

import dataclasses

from nano_omni.core.kernel import Kernel
from nano_omni.core.planning.scheduling import (
    Action,
    Copy,
    RecordEvent,
    WaitEvent,
)


@dataclasses.dataclass(frozen=True, slots=True)
class Dependency:
    producer: str
    consumers: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class StreamPlan:
    host: list[Action]
    copy: list[Action]
    compute: list[Action]
    dependencies: dict[int, Dependency]


def build(actions: list[Action]) -> StreamPlan:
    """Route each original action once; synchronization actions retain their identity."""
    streams: dict[str, list[Action]] = {
        name: [] for name in ("memmove", "copy", "compute")
    }
    producers: dict[int, str] = {}
    consumers: dict[int, set[str]] = {}
    kernels: dict[tuple[type[Kernel], object, object], Kernel] = {}
    kernel_identities: dict[int, Kernel] = {}
    for action in actions:
        if isinstance(action, Kernel):
            identity = id(action)
            interned = kernel_identities.get(identity)
            if interned is None:
                key = type(action), action.arguments, action.config
                interned = kernels.setdefault(key, action)
                kernel_identities[identity] = interned
            action = interned
            stream = "compute"
        elif isinstance(action, (Copy, RecordEvent, WaitEvent)):
            stream = action.stream
        else:
            raise TypeError(type(action))
        if isinstance(action, RecordEvent):
            if action.event in producers:
                raise ValueError("each signal must have exactly one publication")
            producers[action.event] = stream
            consumers[action.event] = set()
        elif isinstance(action, WaitEvent):
            if action.event not in producers:
                raise ValueError("signal publication must precede its wait")
            consumers[action.event].add(stream)
        streams[stream].append(action)
    return StreamPlan(
        streams["memmove"],
        streams["copy"],
        streams["compute"],
        {
            event: Dependency(producer, tuple(sorted(consumers[event])))
            for event, producer in producers.items()
        },
    )
