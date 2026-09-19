"""Shared pipeline lifecycle, local model assets, and stage timing."""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import json
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from nano_omni.core.platform import memory

if TYPE_CHECKING:
    from config import RunConfig

    from nano_omni.core.model import ModelRequest


@dataclasses.dataclass(frozen=True)
class StageResult:
    """Stage timestamps in seconds relative to pipeline construction."""

    name: str
    start_seconds: float
    end_seconds: float

    @property
    def seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class Pipeline(abc.ABC):
    """Prepare model files, execute generation and describe its workload."""

    def __init__(self, config: RunConfig, root: Path) -> None:
        self.config = config
        self.root = root
        self.started = time.perf_counter()
        self.stages: list[StageResult] = []

    @abc.abstractmethod
    def prepare(self) -> None:
        """Resolve the local files required by this pipeline."""
        ...

    @abc.abstractmethod
    def execute(self) -> Path:
        """Generate the configured media and return its output path."""
        ...

    @abc.abstractmethod
    def workload(self) -> dict[str, object]:
        """Return the pipeline dimensions and mode used in measurement reports."""
        ...

    @abc.abstractmethod
    def model_requests(self) -> Iterator[ModelRequest]:
        """Yield every model specialization selected by the pipeline inputs."""
        ...

    def run(self) -> Path:
        """Prepare files and generate media under the configured working-set limit."""
        with memory.ensure_limit(self.config.working_set_gib << 30):
            with self.stage("assets"):
                self.prepare()
            return self.execute()

    def warmup(self) -> None:
        """Compile each selected model specialization once without executing it."""
        from cuda.bindings import (  # pyrefly: ignore [missing-module-attribute]
            driver,  # pyrefly: ignore
        )

        from nano_omni.core.runtime import compilation, execution

        with memory.ensure_limit(self.config.working_set_gib << 30):
            with self.stage("assets"):
                self.prepare()
            execution.check_cuda(driver.cuInit(0))
            device = execution.check_cuda(driver.cuDeviceGet(0))
            total_memory = int(execution.check_cuda(driver.cuDeviceTotalMem(device)))
            requests = dict.fromkeys(self.model_requests())
            kernels = [
                kernel
                for request in requests
                for kernel in request.kernel_inventory(
                    total_memory,
                    self.config.workspace_nbytes,
                )
            ]
            compilation.prepare(kernels)

    def prepare_file(self, repository: str, remote_path: str, path: Path) -> Path:
        """Return a fixed local model path, downloading it only when absent."""
        if path.is_file():
            return path
        import shutil
        import tempfile

        from modelscope.hub import file_download

        downloaded = Path(
            file_download.model_file_download(
                model_id=repository,
                file_path=remote_path,
                local_dir=str(path.parents[len(Path(remote_path).parts) - 1]),
            )
        )
        if downloaded.resolve() != path.resolve():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
            try:
                shutil.copyfile(downloaded, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if not path.is_file():
            raise FileNotFoundError(f"Download did not provide required file: {path}")
        return path

    @contextlib.contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Record and print elapsed stage time, including stages that raise."""
        started = time.perf_counter() - self.started
        try:
            yield
        finally:
            result = StageResult(name, started, time.perf_counter() - self.started)
            self.stages.append(result)
            print(
                "timing="
                + json.dumps({**dataclasses.asdict(result), "seconds": result.seconds}),
                flush=True,
            )
