"""Stage timing and working-set observations shared by pipelines."""

import contextlib
import json
import os
import threading
from collections.abc import Generator
from pathlib import Path
from time import perf_counter

from nano_omni.core.platform import memory

# An optional perf_counter_ns origin keeps records on one process timeline.
# Without that environment value, offsets begin when this module is imported.
MODULE_IMPORT_FINISHED = perf_counter()
PROCESS_STARTED = (
    int(
        os.environ.get(
            "NANO_OMNI_PROCESS_START_NS", str(int(MODULE_IMPORT_FINISHED * 1e9))
        )
    )
    / 1e9
)

TIMING_OUTPUT = os.environ.get("NANO_OMNI_TIMING_OUTPUT")
TIMING_LOCK = threading.Lock()


def record_timing(
    name: str,
    started: float,
    finished: float,
    **details: str | float | bool | None,
) -> None:
    """Emit a host-clock interval to stdout and the optional JSON-lines file.

    ``started`` and ``finished`` are perf_counter timestamps in seconds. The
    start/end fields are offsets from PROCESS_STARTED; ``seconds`` is their
    difference. Extra details are merged last, preserving any explicit overrides.
    """
    record = {
        "name": name,
        "start_seconds": started - PROCESS_STARTED,
        "end_seconds": finished - PROCESS_STARTED,
        "seconds": finished - started,
        "thread": threading.current_thread().name,
        **details,
    }
    encoded = json.dumps(record, sort_keys=True)
    print(f"timing={encoded}", flush=True)
    if TIMING_OUTPUT is not None:
        with TIMING_LOCK, Path(TIMING_OUTPUT).open("a", encoding="utf-8") as output:
            output.write(encoded + "\n")


@contextlib.contextmanager
def stage(name: str, working_set_threshold: int) -> Generator[None, None, None]:
    """Time a host scope and report its sampled process-tree peak resident memory.

    The threshold is bytes; the printed peak is GiB (bytes / 2**30). The interval
    starts before entering the memory guard and ends after peak observation,
    before guard teardown. Timing is emitted even if observation or the body
    raises, once the guard has been entered; this scope adds no CUDA synchronization.
    """
    started = perf_counter()
    with memory.ensure_limit(working_set_threshold) as guard:
        peak = None
        try:
            with guard.observe() as peak:
                yield
        finally:
            finished = perf_counter()
            record_timing(name, started, finished)
            if peak is not None:
                print(
                    f"stage={name} elapsed={finished - started:.3f}s "
                    f"peak_tree_rss={peak.peak_bytes / (1 << 30):.2f}GiB",
                    flush=True,
                )
