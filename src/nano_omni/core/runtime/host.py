"""Host transfers owned by one CUDA runtime, including cached file handles."""

import concurrent.futures
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from io import FileIO
from pathlib import Path


class HostTransfers:
    threads = 8

    def __init__(self) -> None:
        self.pool = ThreadPoolExecutor(
            max_workers=self.threads, thread_name_prefix="nano-copy"
        )
        self.local = threading.local()
        self.lock = threading.Lock()
        self.files: list[FileIO] = []

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for file in self.files:
            file.close()
        self.files.clear()
        self.local = threading.local()

    def read_chunk(
        self, path: Path, offset: int, destination: int, nbytes: int
    ) -> None:
        files: dict[Path, FileIO] | None = getattr(self.local, "files", None)
        if files is None:
            files = {}
            self.local.files = files
        file = files.get(path)
        if file is None:
            file = path.open("rb", buffering=0)
            files[path] = file
            with self.lock:
                self.files.append(file)
        file.seek(offset)
        target = (ctypes.c_ubyte * nbytes).from_address(destination)
        with memoryview(target) as view:
            done = 0
            while done < nbytes:
                read = file.readinto(view[done:])
                if not read:
                    raise EOFError(
                        "unexpected end of checkpoint during pinned staging read"
                    )
                done += read

    def read(self, path: Path, offset: int, destination: int, nbytes: int) -> None:
        # Small file reads do not amortize dozens of futures and file seeks.
        threads = min(self.threads, nbytes // (1 << 20))
        if threads < 2:
            self.read_chunk(path, offset, destination, nbytes)
            return
        chunk = nbytes // threads
        futures = [
            self.pool.submit(
                self.read_chunk,
                path,
                offset + index * chunk,
                destination + index * chunk,
                nbytes - index * chunk if index == threads - 1 else chunk,
            )
            for index in range(threads)
        ]
        concurrent.futures.wait(futures)
        for future in futures:
            future.result()
