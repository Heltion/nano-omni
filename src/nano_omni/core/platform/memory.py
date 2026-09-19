"""Sampled Windows process-tree working-set enforcement."""

import atexit
import contextlib
import ctypes
import dataclasses
import os
import sys
import threading
from collections.abc import Callable, Generator
from contextvars import ContextVar, Token
from typing import Literal, Self

import psutil

from nano_omni.core.platform.process import ProcessJob

_tracking_job: ProcessJob | None = None
_tracking_initialized = False
_tracking_lock = threading.Lock()


def tracking_job() -> ProcessJob | None:
    """Track early-created guards with a job; late guards retain tree scans.

    Existing children cannot always join the root's new nested job. Never use
    partial job membership to enforce a tree limit.
    """
    global _tracking_job, _tracking_initialized
    with _tracking_lock:
        if not _tracking_initialized:
            job = ProcessJob(kill_on_close=False)
            try:
                job.assign(os.getpid())
                descendants = psutil.Process().children(recursive=True)
                enrolled = set(job.pids())
                if any(child.pid not in enrolled for child in descendants):
                    job.close()
                else:
                    _tracking_job = job
                    atexit.register(job.close)
                _tracking_initialized = True
            except BaseException:
                job.close()
                raise
        return _tracking_job


@dataclasses.dataclass
class MemoryUsage:
    limit_bytes: int
    membership_backend: Literal["process_tree", "windows_job"] = "process_tree"
    peak_bytes: int = 0
    trims: int = 0
    exceeded: bool = False
    monitor_error: str | None = None


@dataclasses.dataclass(eq=False)
class MemoryPeak:
    peak_bytes: int = 0


ACTIVE_LIMIT: ContextVar["WorkingSetLimit | None"] = ContextVar(
    "working_set_limit", default=None
)


@contextlib.contextmanager
def ensure_limit(limit_bytes: int) -> Generator["WorkingSetLimit", None, None]:
    """Reuse an active stricter guard in this process, or create a new one."""
    active = ACTIVE_LIMIT.get()
    if (
        active is not None
        and active.pid == os.getpid()
        and active.usage.limit_bytes <= limit_bytes
    ):
        yield active
    else:
        with WorkingSetLimit(limit_bytes) as guard:
            yield guard


def processes(pid: int) -> list[psutil.Process]:
    root = psutil.Process(pid)
    return [root, *root.children(recursive=True)]


def resident(items: list[psutil.Process]) -> int:
    total = 0
    for item in items:
        try:
            total += item.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def trim(items: list[psutil.Process]) -> None:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
    for item in items:
        handle = kernel.OpenProcess(0x0400 | 0x0100, False, item.pid)
        if not handle:
            if not item.is_running():
                continue
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not psapi.EmptyWorkingSet(handle):
                if not item.is_running():
                    continue
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)


def terminate(pid: int, code: int) -> None:
    try:
        children = psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            continue
    if pid == os.getpid():
        os._exit(code)
    try:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        return


class WorkingSetLimit:
    """Sample resident memory and stop the tracked process tree at its limit."""

    def __init__(
        self,
        limit_bytes: int,
        pid: int | None = None,
        interval: float = 0.05,
        on_limit: Callable[[MemoryUsage], None] | None = None,
    ) -> None:
        if sys.platform != "win32":
            raise NotImplementedError(
                "working-set enforcement is implemented for Windows only"
            )
        if limit_bytes <= 0 or interval <= 0:
            raise ValueError("working-set limit and sample interval must be positive")
        self.pid = os.getpid() if pid is None else pid
        self.interval = interval
        self.usage = MemoryUsage(limit_bytes)
        self.on_limit = on_limit
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.token: Token[WorkingSetLimit | None] | None = None
        self.peaks: list[MemoryPeak] = []
        self.observation_lock = threading.Lock()
        self.job: ProcessJob | None = None

    def members(self) -> list[psutil.Process]:
        if self.job is None:
            return processes(self.pid)
        items = []
        for pid in self.job.pids():
            try:
                items.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
        return items

    def observe_size(self, size: int) -> None:
        with self.observation_lock:
            self.usage.peak_bytes = max(self.usage.peak_bytes, size)
            for peak in self.peaks:
                peak.peak_bytes = max(peak.peak_bytes, size)

    @contextlib.contextmanager
    def observe(self) -> Generator[MemoryPeak, None, None]:
        peak = MemoryPeak()
        with self.observation_lock:
            self.peaks.append(peak)
        try:
            self.observe_size(resident(self.members()))
            yield peak
        finally:
            try:
                self.observe_size(resident(self.members()))
            finally:
                with self.observation_lock:
                    self.peaks.remove(peak)

    def sample(self) -> None:
        items = self.members()
        size = resident(items)
        self.observe_size(size)
        if size >= self.usage.limit_bytes:
            self.usage.trims += 1
            trim(items)
            size = resident(self.members())
            if size >= self.usage.limit_bytes:
                self.usage.exceeded = True
                if self.on_limit is not None:
                    self.on_limit(self.usage)
                self.terminate(137)

    def terminate(self, code: int) -> None:
        if self.job is not None:
            self.job.terminate(code)
        else:
            terminate(self.pid, code)

    def watch(self) -> None:
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except psutil.NoSuchProcess:
                return
            except (OSError, psutil.Error) as error:
                self.usage.monitor_error = str(error)
                if self.on_limit is not None:
                    self.on_limit(self.usage)
                self.terminate(138)
                return

    def __enter__(self) -> Self:
        if self.pid == os.getpid():
            self.job = tracking_job()
        self.usage.membership_backend = "windows_job" if self.job else "process_tree"
        self.sample()
        self.thread = threading.Thread(
            target=self.watch, name="working-set", daemon=True
        )
        self.thread.start()
        self.token = ACTIVE_LIMIT.set(self)
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        assert self.thread is not None
        self.thread.join()
        assert self.token is not None
        ACTIVE_LIMIT.reset(self.token)
