from __future__ import annotations

import abc
import collections
import dataclasses
import json
from collections.abc import Iterable, Sequence
from concurrent.futures import Future
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Self

from nano_omni.core.kernel import Kernel, MmaType, Tops
from nano_omni.core.tensor import DType, TensorDesc, TensorKind

if TYPE_CHECKING:
    from nano_omni.core.planning.scheduling import Action
    from nano_omni.core.planning.submission import StreamPlan
    from nano_omni.core.runtime.execution import CudaRuntime


@dataclasses.dataclass(frozen=True, slots=True)
class FileMetadata:
    path: Path
    weights: dict[str, TensorDesc]

    @classmethod
    def read(cls, file_index: int, path: Path) -> FileMetadata:
        """Read weight shapes and absolute byte offsets from a safetensors header."""
        with path.open("rb") as source:
            header_size = int.from_bytes(source.read(8), "little")
            entries = json.loads(source.read(header_size))
        data_offset = 8 + header_size
        weights = {}
        for name, entry in entries.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            tensor = TensorDesc(
                DType.from_safetensors(entry["dtype"]),
                tuple(entry["shape"]),
                TensorKind.FILE,
                (file_index, data_offset + begin),
            )
            assert tensor.num_bytes == end - begin, "checkpoint tensor size mismatch"
            weights[name] = tensor
        return cls(path, weights)


@dataclasses.dataclass(frozen=True, slots=True)
class ModelMetadata:
    files: tuple[FileMetadata, ...]

    @classmethod
    def read(cls, paths: Iterable[Path]) -> ModelMetadata:
        """Read file headers in order; their indices identify weight locations."""
        return cls(
            tuple(FileMetadata.read(index, path) for index, path in enumerate(paths))
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ModelPreparation[Model]:
    """A planned model and its worker timing, ready for CUDA binding."""

    model: Model
    stage: str
    label: str
    started: float
    planned: float


@dataclasses.dataclass(slots=True)
class PlannedModel[Spec, Args, Config](abc.ABC):
    """Commands for fixed model shapes, with workspace capacity measured in bytes."""

    metadata: ModelMetadata
    spec: Spec
    config: Config
    commands: list[Action]
    workspace_nbytes: int
    submission_plan: StreamPlan = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        from nano_omni.core.planning import submission

        self.submission_plan = submission.build(self.commands)

    @classmethod
    def prepare(
        cls,
        stage: str,
        path: Path,
        spec: Spec,
        total_memory: int,
        additional_weights: tuple[Path, ...],
        pinned_nbytes: int,
        workspace_limit: int | None = None,
        *,
        label: str = "",
    ) -> ModelPreparation[Self]:
        """Plan transfers in a worker while the parent executes earlier models."""
        started = perf_counter()
        model = cls.read(
            path,
            spec,
            total_memory,
            additional_weights,
            workspace_limit=workspace_limit,
            pinned_nbytes=pinned_nbytes,
            stage=stage,
        )
        planned = perf_counter()
        return ModelPreparation(model, stage, label, started, planned)

    @abc.abstractmethod
    def run(self, args: Args) -> object:
        """Adapt model-specific arguments to the planned execution."""
        ...

    def execute(
        self,
        runtime: CudaRuntime,
        inputs: tuple[TensorDesc, ...],
        outputs: tuple[TensorDesc, ...],
    ) -> None:
        """Bind caller-owned inputs and outputs, then run the plan to completion.

        Weight files and the workspace are bound for this invocation. The caller
        owns the CUDA runtime, and every Kernel must already be compiled and bound.
        """
        from nano_omni.core.planning import addressing
        from nano_omni.core.runtime import execution

        paths = tuple(file.path for file in self.metadata.files)
        with execution.Runtime(runtime, inputs, outputs, paths, self.workspace_nbytes):
            execution.run(addressing.address_plan(self.submission_plan))

    @staticmethod
    def workspace_capacity(default: int, limit: int | None) -> int:
        """Cap a default byte capacity by an optional positive byte limit."""
        if limit is not None and limit <= 0:
            raise ValueError("workspace_limit must be positive")
        return default if limit is None else min(default, limit)

    @classmethod
    def read(
        cls,
        path: Path,
        spec: Spec,
        total_memory: int,
        additional_weights: Sequence[Path] = (),
        *,
        workspace_limit: int | None = None,
        pinned_nbytes: int | None = None,
        stage: str = "",
    ) -> Self:
        """Read weight metadata and build a model plan for the supplied shapes.

        Memory capacities are bytes. Kernel preparation and CUDA execution are
        separate steps owned by the pipeline.
        """
        metadata = cls.read_metadata(path, additional_weights)
        config, commands, workspace_nbytes = cls.plan(
            metadata, spec, total_memory, workspace_limit
        )
        if pinned_nbytes is not None:
            from nano_omni.core.planning import scheduling

            started = perf_counter()
            staged = scheduling.plan_staging(commands, pinned_nbytes)
            statistics = scheduling.host_transfer_statistics(commands, staged)
            statistics["host_plan_us"] = round((perf_counter() - started) * 1_000_000)
            print(
                f"host_transfer_plan={json.dumps({**statistics, 'stage': stage}, sort_keys=True)}",
                flush=True,
            )
            commands = staged
        return cls(metadata, spec, config, commands, workspace_nbytes)

    @classmethod
    def read_metadata(
        cls, path: Path, additional_weights: Sequence[Path] = ()
    ) -> ModelMetadata:
        """Describe the main weight file followed by additional weight files."""
        return ModelMetadata.read((path, *additional_weights))

    @classmethod
    @abc.abstractmethod
    def plan(
        cls,
        metadata: ModelMetadata,
        spec: Spec,
        total_memory: int,
        workspace_limit: int | None = None,
    ) -> tuple[Config, list[Action], int]:
        """Return model configuration, scheduled commands and workspace bytes.

        Use metadata and shapes to choose allocations within the memory budget.
        """
        ...

    @classmethod
    def kernel_inventory(
        cls,
        metadata: ModelMetadata,
        spec: Spec,
        total_memory: int,
        *,
        workspace_limit: int | None = None,
    ) -> list[Kernel]:
        """Return the planned Kernel instances in execution order."""
        _, commands, _ = cls.plan(metadata, spec, total_memory, workspace_limit)
        return [operation for operation in commands if isinstance(operation, Kernel)]

    @property
    def tops(self) -> Tops:
        """Return raw MMA operation counts for one execution."""
        counts: collections.Counter[MmaType] = collections.Counter()
        for operation in self.commands:
            if isinstance(operation, Kernel):
                counts.update(operation.tops(operation.arguments))
        return dict(counts)


@dataclasses.dataclass(frozen=True)
class ModelRequest:
    """A model type, weight paths and shapes passed to planning workers."""

    model: type[PlannedModel]
    path: Path
    spec: object
    additional_weights: tuple[Path, ...] = ()

    def kernel_inventory(
        self, total_memory: int, workspace_limit: int | None
    ) -> list[Kernel]:
        """Plan this request and return its Kernel instances."""
        metadata = self.model.read_metadata(self.path, self.additional_weights)
        return self.model.kernel_inventory(
            metadata, self.spec, total_memory, workspace_limit=workspace_limit
        )

    def read(self, total_memory: int, workspace_limit: int | None) -> PlannedModel:
        """Create the planned model in the process handling this request."""
        return self.model.read(
            self.path,
            self.spec,
            total_memory,
            self.additional_weights,
            workspace_limit=workspace_limit,
        )


def bind_model[Model: PlannedModel](
    future: Future[ModelPreparation[Model]],
    runtime: CudaRuntime,
) -> Model:
    """Receive a worker result and bind its kernels on the caller's CUDA context."""
    from nano_omni.core.runtime import compilation, observation

    waiting = perf_counter()
    prepared = future.result()
    ready = perf_counter()
    observation.record_timing(f"{prepared.stage}_plan_receive_wait", waiting, ready)
    observation.record_timing(
        f"{prepared.stage}_cpu_plan", prepared.started, prepared.planned
    )
    compilation.bind_kernel_modules(prepared.model, runtime, label=prepared.label)
    observation.record_timing(f"{prepared.stage}_cuda_bind", ready, perf_counter())
    return prepared.model
