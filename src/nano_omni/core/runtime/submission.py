"""Three ordered submission lists with numbered, single-publication signals."""

import contextvars
import ctypes
import dataclasses
import threading

from cuda.bindings import (
    driver,  # pyrefly: ignore[missing-module-attribute]  # Cython extension has no Python stub.
)

from nano_omni.core.kernel import Kernel
from nano_omni.core.planning.scheduling import (
    Action,
    Copy,
    CopyKind,
    RecordEvent,
    WaitEvent,
)
from nano_omni.core.planning.submission import StreamPlan
from nano_omni.core.runtime import execution
from nano_omni.core.runtime.execution import Stream


@dataclasses.dataclass(slots=True)
class Signal:
    # True selects a mapped value32 wait; False selects a CUDA event.
    consumers: dict[Stream, bool]
    submitted: threading.Event = dataclasses.field(default_factory=threading.Event)
    event: driver.CUevent | None = None
    flag: int = 0


def run(plan: StreamPlan) -> None:
    """Submit one addressed plan on host, copy and compute threads."""
    state = execution.current()
    synchronization = state.cuda.synchronization
    directions: dict[tuple[str, str], str] = {
        ("memmove", "copy"): synchronization.host_to_copy,
        ("copy", "memmove"): synchronization.copy_to_host,
        ("copy", "compute"): synchronization.copy_to_compute,
        ("compute", "copy"): synchronization.compute_to_copy,
    }
    signals: dict[int, Signal] = {}
    for event, dependency in plan.dependencies.items():
        producer = Stream(dependency.producer)
        consumers = {
            Stream(consumer): directions.get((producer, consumer), "event") == "value32"
            for consumer in dependency.consumers
        }
        # H2D completion already owns an event for the host consumer. Compute
        # waits on the same event instead of publishing a second flag.
        if producer == Stream.COPY and len(consumers) == 2:
            consumers = dict.fromkeys(consumers, False)
        signals[event] = Signal(consumers)

    cuda = execution.driver
    check = execution.check_cuda
    values: list[Signal] = [
        signal for signal in signals.values() if any(signal.consumers.values())
    ]
    if values:
        # Runtime owns this mapped allocation until all submission streams finish.
        host = int(check(cuda.cuMemHostAlloc(4 * len(values), 2)))
        state.mapped_hosts.append(host)
        device = int(check(cuda.cuMemHostGetDevicePointer(host, 0)))
        ctypes.memset(host, 0, 4 * len(values))
        for index, signal in enumerate(values):
            signal.flag = device + 4 * index
    for signal in signals.values():
        if False in signal.consumers.values():
            signal.event = check(
                cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING)
            )
            state.events[len(state.events)] = signal.event
    errors: list[BaseException] = []

    def submit(kind: Stream, items: list[Action]) -> None:
        try:
            check(cuda.cuCtxSetCurrent(state.cuda.context))
            token = execution._active_stream.set(kind)
            try:
                stream = state.cuda.streams[kind]
                for item in items:
                    if errors:
                        return
                    if isinstance(item, WaitEvent):
                        signal = signals[item.event]
                        # Wait for API submission, not GPU completion. This also
                        # prevents CUDA event waits from seeing an unrecorded event.
                        signal.submitted.wait()
                        if errors:
                            return
                        if kind == Stream.MEMMOVE:
                            check(cuda.cuEventSynchronize(signal.event))
                        elif signal.consumers[kind]:
                            check(
                                cuda.cuStreamWaitValue32(
                                    stream,
                                    signal.flag,
                                    1,
                                    cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
                                )
                            )
                        else:
                            check(cuda.cuStreamWaitEvent(stream, signal.event, 0))
                    elif isinstance(item, RecordEvent):
                        signal = signals[item.event]
                        if signal.flag:
                            check(
                                cuda.cuStreamWriteValue32(
                                    stream,
                                    signal.flag,
                                    1,
                                    cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                                )
                            )
                        if signal.event is not None:
                            check(cuda.cuEventRecord(signal.event, stream))
                        signal.submitted.set()
                    elif isinstance(item, Kernel):
                        item.submit()
                    elif isinstance(item, Copy):
                        assert not isinstance(item.source, bytes)
                        source_address = sum(item.source.info)
                        destination_address = sum(item.destination.info)
                        if item.kind == CopyKind.H2H:
                            assert item.file_position is not None, (
                                "H2H copy requires a checkpoint file position"
                            )
                            file_index, byte_offset = item.file_position
                            state.cuda.host_transfers.read(
                                state.paths[file_index],
                                byte_offset,
                                destination_address,
                                item.num_bytes,
                            )
                        elif item.kind == CopyKind.H2D:
                            check(
                                cuda.cuMemcpyHtoDAsync(
                                    destination_address,
                                    source_address,
                                    item.num_bytes,
                                    stream,
                                )
                            )
                        else:
                            check(
                                cuda.cuMemcpyDtoDAsync(
                                    destination_address,
                                    source_address,
                                    item.num_bytes,
                                    stream,
                                )
                            )
                    else:
                        raise TypeError(type(item))
            finally:
                execution._active_stream.reset(token)
        except BaseException as error:  # noqa: BLE001 -- propagate worker failures after waking dependent submitters
            errors.append(error)
            for signal in signals.values():
                signal.submitted.set()

    threads: list[threading.Thread] = [
        threading.Thread(
            # Each worker inherits the invocation but selects its own active stream.
            target=contextvars.copy_context().run,
            args=(submit, kind, items),
            name=f"submit-{kind}",
        )
        for kind, items in (
            (Stream.MEMMOVE, plan.host),
            (Stream.COPY, plan.copy),
            (Stream.COMPUTE, plan.compute),
        )
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
