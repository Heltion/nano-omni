"""Windows process membership and optional descendant lifetime control."""

import ctypes
import ctypes.wintypes

import psutil


class IoCounters(ctypes.Structure):
    read_operation_count: int
    write_operation_count: int
    other_operation_count: int
    read_transfer_count: int
    write_transfer_count: int
    other_transfer_count: int

    _fields_ = [
        ("read_operation_count", ctypes.c_uint64),
        ("write_operation_count", ctypes.c_uint64),
        ("other_operation_count", ctypes.c_uint64),
        ("read_transfer_count", ctypes.c_uint64),
        ("write_transfer_count", ctypes.c_uint64),
        ("other_transfer_count", ctypes.c_uint64),
    ]


class BasicLimitInformation(ctypes.Structure):
    per_process_user_time_limit: int
    per_job_user_time_limit: int
    limit_flags: int
    minimum_working_set_size: int
    maximum_working_set_size: int
    active_process_limit: int
    affinity: int
    priority_class: int
    scheduling_class: int

    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", ctypes.wintypes.DWORD),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.wintypes.DWORD),
        ("scheduling_class", ctypes.wintypes.DWORD),
    ]


class ExtendedLimitInformation(ctypes.Structure):
    basic_limit_information: BasicLimitInformation
    io_info: IoCounters
    process_memory_limit: int
    job_memory_limit: int
    peak_process_memory_used: int
    peak_job_memory_used: int

    _fields_ = [
        ("basic_limit_information", BasicLimitInformation),
        ("io_info", IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class ProcessJob:
    """Own a Windows job handle and manage its assigned process membership."""

    def __init__(self, *, kill_on_close: bool = True) -> None:
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p),
            "SetInformationJobObject": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32],
                ctypes.c_int,
            ),
            "AssignProcessToJobObject": (
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "QueryInformationJobObject": (
                [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                ],
                ctypes.c_int,
            ),
            "TerminateJobObject": ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
            "OpenProcess": (
                [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32],
                ctypes.c_void_p,
            ),
            "OpenThread": (
                [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32],
                ctypes.c_void_p,
            ),
            "ResumeThread": ([ctypes.c_void_p], ctypes.c_uint32),
            "CloseHandle": ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes = arguments
            function.restype = result
        self.handle: int | None = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimitInformation()
        limits.basic_limit_information.limit_flags = 0x2000 if kill_on_close else 0
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, pid: int) -> None:
        handle = self.kernel.OpenProcess(0x0100 | 0x0001, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(handle)

    def pids(self) -> list[int]:
        """Read all assigned process IDs, retrying an incomplete membership list."""
        capacity = 16
        while True:
            buffer = ctypes.create_string_buffer(
                8 + capacity * ctypes.sizeof(ctypes.c_size_t)
            )
            success = self.kernel.QueryInformationJobObject(
                self.handle, 3, buffer, ctypes.sizeof(buffer), None
            )
            assigned = ctypes.c_uint32.from_buffer(buffer).value
            count = ctypes.c_uint32.from_buffer(buffer, 4).value
            if success and count == assigned:
                return list((ctypes.c_size_t * count).from_buffer(buffer, 8))
            if not success:
                error = ctypes.get_last_error()
                if error != 234:  # ERROR_MORE_DATA
                    raise ctypes.WinError(error)
            # A successful query can also return an incomplete process list.
            capacity = max(capacity * 2, assigned)

    def start(self, pid: int) -> None:
        # Enroll before resuming so every subsequently spawned child inherits it.
        self.assign(pid)
        threads = psutil.Process(pid).threads()
        for thread in threads:
            handle = self.kernel.OpenThread(0x0002, False, thread.id)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if self.kernel.ResumeThread(handle) == 0xFFFFFFFF:
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                self.kernel.CloseHandle(handle)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def terminate(self, code: int) -> None:
        if not self.kernel.TerminateJobObject(self.handle, code):
            raise ctypes.WinError(ctypes.get_last_error())
