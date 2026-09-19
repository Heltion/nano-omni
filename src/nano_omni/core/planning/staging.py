import collections
import dataclasses
import heapq
import itertools

# The same host slice with a different byte length is a distinct cache entry.
type StagingKey = tuple[tuple[int, int], int]
# Negative next-use ordinal, insertion sequence, staging key.
type EvictionEntry = tuple[int, int, StagingKey]


class EvictionQueue:
    """Evict the farthest next use, breaking ties by insertion order."""

    def __init__(self) -> None:
        self.heap: list[EvictionEntry] = []
        self.entries: dict[StagingKey, EvictionEntry] = {}
        self.sequence: itertools.count[int] = itertools.count()

    def add(self, key: StagingKey, next_use: int) -> None:
        """Refresh a key using H2D-use ordinals, independent of byte offsets."""
        entry = (-next_use, next(self.sequence), key)
        self.entries[key] = entry
        heapq.heappush(self.heap, entry)
        if len(self.heap) > 2 * len(self.entries) + 64:
            self.heap = list(self.entries.values())
            heapq.heapify(self.heap)

    def pop(self) -> StagingKey:
        while self.heap:
            entry = heapq.heappop(self.heap)
            if self.entries.get(entry[2]) == entry:
                del self.entries[entry[2]]
                return entry[2]
        raise KeyError("no evictable allocation")


def allocate_interval(free: list[tuple[int, int]], nbytes: int) -> int | None:
    """Consume the first fitting (byte offset, byte length) interval."""
    for index, (offset, size) in enumerate(free):
        if size < nbytes:
            continue
        if size == nbytes:
            free.pop(index)
        else:
            free[index] = (offset + nbytes, size - nbytes)
        return offset
    return None


def release_interval(free: list[tuple[int, int]], offset: int, nbytes: int) -> None:
    """Return a byte interval and merge adjacent free intervals."""
    free.append((offset, nbytes))
    free.sort()
    merged: list[tuple[int, int]] = []
    for current_offset, current_nbytes in free:
        if merged and merged[-1][0] + merged[-1][1] == current_offset:
            previous_offset, previous_nbytes = merged[-1]
            merged[-1] = (previous_offset, previous_nbytes + current_nbytes)
        else:
            merged.append((current_offset, current_nbytes))
    free[:] = merged


@dataclasses.dataclass(frozen=True)
class StagingAllocation:
    """Pinned byte offset, fill requirement, and evicted allocations' last H2D-use ordinals."""

    offset: int
    fill: bool
    evicted: tuple[int, ...] = ()


def allocate_staging(keys: list[StagingKey], capacity: int) -> list[StagingAllocation]:
    """Assign each H2D use a cached byte interval in the pinned arena."""
    future: collections.defaultdict[StagingKey, collections.deque[int]] = (
        collections.defaultdict(collections.deque)
    )
    for index, key in enumerate(keys):
        future[key].append(index)
    # Staging key -> (pinned byte offset, last H2D-use ordinal).
    cache: dict[StagingKey, tuple[int, int]] = {}
    eviction = EvictionQueue()
    free: list[tuple[int, int]] = [(0, capacity)]
    output: list[StagingAllocation] = []
    for index, key in enumerate(keys):
        size = key[1]
        if size > capacity:
            raise ValueError(
                f"H2D use {index} requires {size} bytes, larger than the {capacity}-byte pinned arena"
            )
        future[key].popleft()
        next_use = future[key][0] if future[key] else len(keys)
        cached = cache.get(key)
        evicted: list[int] = []
        if cached is None:
            offset = allocate_interval(free, size)
            while offset is None:
                victim = eviction.pop()
                address, last_use = cache.pop(victim)
                evicted.append(last_use)
                release_interval(free, address, victim[1])
                offset = allocate_interval(free, size)
        else:
            offset, _ = cached
        cache[key] = (offset, index)
        eviction.add(key, next_use)
        output.append(StagingAllocation(offset, cached is None, tuple(evicted)))
    return output
