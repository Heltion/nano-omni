"""Schedule copy, kernel, and synchronization actions."""

from __future__ import annotations

import dataclasses
from enum import StrEnum

from nano_omni.core.kernel import Kernel
from nano_omni.core.planning import staging
from nano_omni.core.tensor import TensorDesc, TensorKind


class CopyKind(StrEnum):
    H2H = "h2h"
    H2D = "h2d"
    D2D = "d2d"


@dataclasses.dataclass(frozen=True, slots=True)
class Copy:
    """One copy whose tensor endpoints are resolved in place by addressing."""

    source: TensorDesc | bytes
    destination: TensorDesc
    stream: str = "copy"
    owner: object | None = None
    file_position: tuple[int, int] | None = None

    @property
    def num_bytes(self) -> int:
        size = (
            len(self.source)
            if isinstance(self.source, bytes)
            else self.source.num_bytes
        )
        assert size == self.destination.num_bytes, "copy endpoint sizes differ"
        return size

    @property
    def kind(self) -> CopyKind:
        assert isinstance(self.source, TensorDesc), "copy source has not been addressed"
        assert self.source.kind in (TensorKind.HOST, TensorKind.DEVICE)
        assert self.destination.kind in (TensorKind.HOST, TensorKind.DEVICE)
        if self.source.kind == TensorKind.DEVICE:
            assert self.destination.kind == TensorKind.DEVICE
            return CopyKind.D2D
        return (
            CopyKind.H2D if self.destination.kind == TensorKind.DEVICE else CopyKind.H2H
        )


@dataclasses.dataclass(frozen=True, slots=True)
class H2DUse:
    ordinal: int
    command_index: int
    source: TensorDesc
    destination: TensorDesc
    stream: str = "copy"


@dataclasses.dataclass(frozen=True, slots=True)
class RecordEvent:
    event: int
    stream: str


@dataclasses.dataclass(frozen=True, slots=True)
class WaitEvent:
    event: int
    stream: str


type Action = Copy | Kernel | RecordEvent | WaitEvent


def extract_h2d_uses(actions: list[Action]) -> list[H2DUse]:
    uses: list[H2DUse] = []
    for command_index, operation in enumerate(actions):
        if (
            isinstance(operation, Copy)
            and isinstance(operation.source, TensorDesc)
            and operation.source.kind == TensorKind.FILE
            and operation.destination.kind == TensorKind.WORKSPACE
        ):
            uses.append(
                H2DUse(
                    len(uses),
                    command_index,
                    operation.source,
                    operation.destination,
                    operation.stream,
                )
            )
    return uses


def next_event_id(actions: list[Action]) -> int:
    used = [
        operation.event
        for operation in actions
        if isinstance(operation, (RecordEvent, WaitEvent)) and operation.event >= 0
    ]
    return max(used, default=-1) + 1


def plan_staging(actions: list[Action], pinned_num_bytes: int) -> list[Action]:
    uses = extract_h2d_uses(actions)
    if not uses:
        return actions
    keys: list[staging.StagingKey] = []
    for use in uses:
        assert len(use.source.info) == 2, "staging source requires identity and offset"
        keys.append(((use.source.info[0], use.source.info[1]), use.source.num_bytes))
    assigned = staging.allocate_staging(keys, pinned_num_bytes)
    bindings: dict[int, int] = {}
    fills: dict[int, tuple[tuple[int, ...], int]] = {}
    records: dict[int, list[int]] = {}
    event = next_event_id(actions)
    for use, assignment in zip(uses, assigned, strict=True):
        bindings[use.ordinal] = assignment.offset
        if not assignment.fill:
            continue
        latest_by_stream: dict[str, int] = {}
        for last_use in assignment.evicted:
            stream = uses[last_use].stream
            latest_by_stream[stream] = max(last_use, latest_by_stream.get(stream, -1))
        release_events: list[int] = []
        for last_use in latest_by_stream.values():
            release_event = event
            event += 1
            records.setdefault(last_use, []).append(release_event)
            release_events.append(release_event)
        fills[use.ordinal] = (tuple(release_events), assignment.offset)

    uses_by_command = {use.command_index: use for use in uses}
    output: list[Action] = []
    for command_index, action in enumerate(actions):
        use = uses_by_command.get(command_index)
        if use is None:
            output.append(action)
            continue
        fill = fills.get(use.ordinal)
        if fill is not None:
            wait_events, offset = fill
            for wait_event in wait_events:
                output.append(WaitEvent(wait_event, "memmove"))
            output.append(
                Copy(
                    use.source,
                    TensorDesc(
                        use.source.dtype,
                        use.source.shape,
                        TensorKind.PINNED,
                        (0, offset),
                    ),
                    "memmove",
                )
            )
            ready_event = event
            event += 1
            output.append(RecordEvent(ready_event, "memmove"))
            output.append(WaitEvent(ready_event, use.stream))
        output.append(
            Copy(
                TensorDesc(
                    use.source.dtype,
                    use.source.shape,
                    TensorKind.PINNED,
                    (0, bindings[use.ordinal]),
                ),
                use.destination,
                use.stream,
            )
        )
        output.extend(
            RecordEvent(release_event, use.stream)
            for release_event in records.get(use.ordinal, ())
        )
    return output


def host_transfer_statistics(
    original: list[Action], planned: list[Action]
) -> dict[str, int]:
    uses = extract_h2d_uses(original)
    keys: set[staging.StagingKey] = set()
    for use in uses:
        assert len(use.source.info) == 2, "staging source requires identity and offset"
        keys.add(((use.source.info[0], use.source.info[1]), use.source.num_bytes))
    fills = [
        operation
        for operation in planned
        if isinstance(operation, Copy)
        and operation.destination.kind == TensorKind.PINNED
    ]
    requested = sum(use.source.num_bytes for use in uses)
    filled = sum(operation.num_bytes for operation in fills)
    return {
        "commands_before": len(original),
        "commands_after": len(planned),
        "h2d_uses": len(uses),
        "h2d_requested_bytes": requested,
        "unique_host_slices": len(keys),
        "unique_weight_bytes": sum(key[1] for key in keys),
        "staging_fills": len(fills),
        "staging_fill_bytes": filled,
        "cache_hits": len(uses) - len(fills),
        "cache_hit_bytes": requested - filled,
    }
