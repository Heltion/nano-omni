"""Separate kernel specialization metadata from runtime argument bindings."""

from __future__ import annotations

import abc
import copy
import dataclasses
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from functools import cache
from typing import Any, ClassVar, Literal, Self, cast, overload

from pydantic import BaseModel, ConfigDict

from nano_omni.core.tensor import TensorDesc


class MmaType(StrEnum):
    """Complete tensor-core operand and accumulator type signature."""

    F4F4F32 = "F4F4F32"
    F8F8F16 = "F8F8F16"
    F8F8F32 = "F8F8F32"
    F16F16F32 = "F16F16F32"
    BF16BF16F32 = "BF16BF16F32"
    TF32TF32F32 = "TF32TF32F32"
    I8I8I32 = "I8I8I32"


type Tops = dict[MmaType, int]
type DynamicParameter = tuple[str, int | float]


_planning_pool: ContextVar[dict[tuple[type[object], object, object], object] | None] = (
    ContextVar("kernel_planning_pool", default=None)
)


@cache
def _field_names(value_type: Any) -> tuple[str, ...]:
    return tuple(field.name for field in dataclasses.fields(value_type))


@contextmanager
def intern_kernels() -> Iterator[None]:
    """Reuse equal kernel invocations while one model plan is being expanded."""
    token = _planning_pool.set({})
    try:
        yield
    finally:
        _planning_pool.reset(token)


class Workload(BaseModel):
    """Logical dimensions and attributes that determine a kernel specialization."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Config(BaseModel):
    """Compile-time choices for a workload; tensor addresses belong to arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Arguments:
    """Nested dataclass/tuple arguments shared by planning and execution."""

    def dynamic_parameters(self) -> tuple[DynamicParameter, ...]:
        """Return named scalar values passed to the compiled kernel at runtime."""
        return ()

    def map(self, function: Callable[[TensorDesc], TensorDesc]) -> Self:
        """Transform tensor-value leaves, rebuilding tuple/dataclass containers."""

        def visit(item: object) -> object:
            if isinstance(item, TensorDesc):
                return function(item)
            if isinstance(item, tuple):
                return tuple(visit(child) for child in item)
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                return type(item)(
                    **{
                        name: visit(getattr(item, name))
                        for name in _field_names(type(item))
                    }
                )
            return item

        # Traversal replaces leaves but preserves the outer dataclass type.
        return cast(Self, visit(self))

    @overload
    def values(self) -> tuple[TensorDesc, ...]: ...

    @overload
    def values(
        self, *, include_absent: Literal[True]
    ) -> tuple[TensorDesc | None, ...]: ...

    def values(self, *, include_absent: bool = False) -> tuple[TensorDesc | None, ...]:
        """Return tensor ABI fields in order, optionally retaining empty slots."""
        values: list[TensorDesc | None] = []

        def visit(item: object) -> None:
            if isinstance(item, TensorDesc):
                values.append(item)
            elif item is None and include_absent:
                values.append(None)
            elif isinstance(item, tuple):
                for child in item:
                    visit(child)
            elif dataclasses.is_dataclass(item) and not isinstance(item, type):
                for name in _field_names(type(item)):
                    visit(getattr(item, name))

        visit(self)
        return tuple(values)


class Kernel[ArgumentsT: Arguments, WorkloadT: Workload, ConfigT: Config](abc.ABC):
    """One kernel invocation and its compile-time specialization."""

    name: ClassVar[str] = ""
    program: ClassVar[Any]
    reference_rtol: ClassVar[float] = 0.02
    reference_atol: ClassVar[float] = 0.01

    def __new__(
        cls,
        arguments: ArgumentsT | None = None,
        config: ConfigT | None = None,
        compiled: Any | None = None,
    ) -> Self:
        pool = _planning_pool.get()
        if pool is None or arguments is None or compiled is not None:
            return super().__new__(cls)
        key = (cls, arguments, config)
        cached = pool.get(key)
        if cached is not None:
            return cast(Self, cached)
        result = super().__new__(cls)
        pool[key] = result
        return result

    def __init__(
        self,
        arguments: ArgumentsT,
        config: ConfigT | None = None,
        compiled: Any | None = None,
    ) -> None:
        if hasattr(self, "arguments"):
            return
        self.arguments = arguments
        self.workload = self.make_workload(arguments)
        self.config = (
            config if config is not None else type(self).make_config(self.workload)
        )
        self.compiled = compiled

    @classmethod
    @abc.abstractmethod
    def make_arguments(cls, workload: WorkloadT) -> ArgumentsT:
        """Describe the empty tensor arguments required by a workload."""
        ...

    @classmethod
    def tops(cls, arguments: ArgumentsT) -> Tops:
        """Estimate logical MMA operations by complete instruction type."""
        return {}

    @classmethod
    @abc.abstractmethod
    def ref_program(cls, arguments: ArgumentsT) -> None:
        """Write reference results into the addressed output arguments."""
        ...

    @classmethod
    @abc.abstractmethod
    def make_config(cls, workload: WorkloadT) -> ConfigT:
        """Select the production Config for this concrete workload."""
        ...

    @classmethod
    @abc.abstractmethod
    def make_workload(cls, arguments: ArgumentsT) -> WorkloadT:
        """Derive the logical workload from concrete kernel arguments."""
        ...

    def compile(self) -> None:
        """Compile this specialization through the native TileLang entry point."""
        if self.compiled is not None:
            return
        from nano_omni.core.runtime import tilelang_compat

        tilelang_compat.install()
        import tilelang

        parameters = self.workload.model_dump()
        for name, value in self.arguments.dynamic_parameters():
            dtype = "float32" if type(value) is float else "int32"
            parameters[name] = tilelang.language.dynamic(name, dtype)
        function = self.program.get_tir(**parameters, **self.config.model_dump())
        self.compiled = tilelang_compat.compile(function)

    def submit(self) -> None:
        """Submit the already compiled invocation on the active CUDA stream."""
        from nano_omni.core.runtime import execution

        assert self.compiled is not None, "kernel must be compiled before submission"
        values = self.arguments.values(include_absent=True)
        placeholder = next(value for value in values if value is not None)
        dynamic_values = tuple(
            value for _, value in self.arguments.dynamic_parameters()
        )
        execution.launch_tilelang(
            self.compiled,
            *dynamic_values,
            *(placeholder if value is None else value for value in values),
        )

    def map_arguments(self, function: Callable[[TensorDesc], TensorDesc]) -> Self:
        """Return the same invocation with transformed tensor descriptors."""
        result = copy.copy(self)
        result.arguments = self.arguments.map(function)
        return result
