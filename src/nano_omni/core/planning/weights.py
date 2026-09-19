"""Plan weight-only residency from explicit per-kernel weight references."""

from __future__ import annotations

import collections
import dataclasses
import heapq
from collections.abc import Sequence
from typing import TYPE_CHECKING

from nano_omni.core.layout import aligned
from nano_omni.core.planning.staging import allocate_interval, release_interval
from nano_omni.core.tensor import TensorDesc, TensorKind

if TYPE_CHECKING:
    from nano_omni.core.kernel import Kernel

# Negative next-kernel ordinal, residency version, weight ID.
type WeightEvictionEntry = tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class Upload:
    """Weight bytes to upload between two kernel ordinals; -1 means before all kernels."""

    weight: int
    offset: int
    nbytes: int
    after_kernel: int
    before_kernel: int


@dataclasses.dataclass(frozen=True)
class WeightPlan:
    """Uploads and per-kernel bindings from weight IDs to arena byte offsets."""

    uploads: tuple[Upload, ...]
    bindings: tuple[dict[int, int], ...]


def plan_weights(
    calls: Sequence[Kernel],
    entries: Sequence[TensorDesc],
    capacity: int,
) -> WeightPlan:
    """Fit referenced weights in a byte-sized arena, evicting the farthest next use."""
    access_cache: dict[int, tuple[int, ...]] = {}
    accesses: list[tuple[int, ...]] = []
    for call in calls:
        key = id(call)
        identities = access_cache.get(key)
        if identities is None:
            identities = tuple(
                dict.fromkeys(
                    ref.info[0]
                    for ref in call.arguments.values()
                    if ref.kind == TensorKind.WEIGHT
                )
            )
            access_cache[key] = identities
        accesses.append(identities)
    future: collections.defaultdict[int, collections.deque[int]] = (
        collections.defaultdict(collections.deque)
    )
    for index, identities in enumerate(accesses):
        for identity in identities:
            future[identity].append(index)
    # Weight ID -> (byte offset, aligned byte length, last kernel ordinal).
    resident: dict[int, tuple[int, int, int]] = {}
    versions: collections.Counter[int] = collections.Counter()
    heap: list[WeightEvictionEntry] = []
    free: list[tuple[int, int]] = [(0, capacity)]
    uploads: list[Upload] = []
    bindings: list[dict[int, int]] = []
    last_released = -1
    for index, identities in enumerate(accesses):
        required = set(identities)
        for identity in identities:
            if identity in resident:
                continue
            size = aligned(entries[identity].num_bytes, 256)
            offset = allocate_interval(free, size)
            held: list[WeightEvictionEntry] = []
            while offset is None:
                while heap:
                    candidate = heapq.heappop(heap)
                    _, version, victim = candidate
                    if victim not in resident or versions[victim] != version:
                        continue
                    if victim in required:
                        held.append(candidate)
                        continue
                    break
                else:
                    raise MemoryError(
                        f"kernel {index} weights cannot fit in {capacity} bytes"
                    )
                address, length, last_use = resident.pop(victim)
                last_released = max(last_released, last_use)
                release_interval(free, address, length)
                offset = allocate_interval(free, size)
            for candidate in held:
                heapq.heappush(heap, candidate)
            resident[identity] = (offset, size, index)
            uploads.append(
                Upload(
                    identity,
                    offset,
                    entries[identity].num_bytes,
                    last_released,
                    index,
                )
            )
        binding: dict[int, int] = {}
        for identity in identities:
            offset, size, _ = resident[identity]
            binding[identity] = offset
            resident[identity] = (offset, size, index)
            versions[identity] += 1
            future[identity].popleft()
            next_use = future[identity][0] if future[identity] else len(calls)
            heapq.heappush(heap, (-next_use, versions[identity], identity))
        bindings.append(binding)
    return WeightPlan(tuple(uploads), tuple(bindings))
