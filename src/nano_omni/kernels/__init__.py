"""Discover concrete Kernel implementations and their declared contracts."""

import importlib
import inspect
from pathlib import Path
from typing import Any, get_args, get_origin

from nano_omni.core.kernel import Config, Kernel, Workload


def classes() -> dict[str, type[Kernel[Any, Any, Any]]]:
    """Import every implementation module and index concrete Kernels by name."""
    root = Path(__file__).parent
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        module = ".".join((__name__, *path.relative_to(root).with_suffix("").parts))
        importlib.import_module(module)

    pending = list(Kernel.__subclasses__())
    result: dict[str, type[Kernel[Any, Any, Any]]] = {}
    while pending:
        kernel = pending.pop()
        pending.extend(kernel.__subclasses__())
        if not kernel.name or inspect.isabstract(kernel):
            continue
        assert kernel.name not in result, f"duplicate Kernel name: {kernel.name}"
        result[kernel.name] = kernel
    return result


def contract(
    kernel: type[Kernel[Any, Any, Any]],
) -> tuple[type[Workload], type[Config]]:
    """Return the concrete Workload and Config declared by a Kernel subclass."""
    def resolve(value: object, bindings: dict[object, object]) -> object:
        while value in bindings:
            value = bindings[value]
        return value

    def visit(
        owner: type[object], bindings: dict[object, object]
    ) -> tuple[type[Workload], type[Config]] | None:
        for base in owner.__dict__.get("__orig_bases__", ()):
            origin = get_origin(base)
            arguments = tuple(resolve(value, bindings) for value in get_args(base))
            if not isinstance(origin, type) or not issubclass(origin, Kernel):
                continue
            if origin is Kernel and len(arguments) == 3:
                _, workload_type, config_type = arguments
                if (
                    isinstance(workload_type, type)
                    and issubclass(workload_type, Workload)
                    and isinstance(config_type, type)
                    and issubclass(config_type, Config)
                ):
                    return workload_type, config_type
            nested_bindings = dict(bindings)
            nested_bindings.update(zip(origin.__parameters__, arguments, strict=True))
            result = visit(origin, nested_bindings)
            if result is not None:
                return result
        return None

    result = visit(kernel, {})
    if result is not None:
        return result
    raise TypeError(f"{kernel.__name__} must declare concrete Kernel generic types")
