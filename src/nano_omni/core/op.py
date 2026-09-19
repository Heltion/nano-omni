"""Top-level operators bind model activations and emit kernel calls directly."""

from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Self

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.tensor import DType, TensorDesc


class Op[WeightsT](abc.ABC):
    """Bind input activation IDs/slices, optional weights, then outputs with to().

    prepare() lowers these bindings into kernel calls without executing them.
    """

    def __init__(
        self,
        *inputs: TensorDesc,
        outputs: tuple[tuple[DType, tuple[int, ...]], ...],
    ) -> None:
        self.inputs = inputs
        self.output_specs = outputs
        self.outputs: tuple[TensorDesc, ...] | None = None
        self.weights: WeightsT | None = None

    def with_weights(self, weights: WeightsT) -> Self:
        """Attach operator-specific weights before output bindings are finalized."""
        if self.outputs is not None:
            raise ValueError("operator already finalized")
        self.weights = weights
        return self

    def to(self, *outputs: int | tuple[int, int] | TensorDesc) -> Self:
        """Finalize output activation bindings once; return this same operator."""
        assert self.outputs is None, "operator already finalized"
        assert len(outputs) == len(self.output_specs), "operator output count mismatch"
        bound: list[TensorDesc] = []
        for output, (dtype, shape) in zip(outputs, self.output_specs, strict=True):
            tensor = (
                output
                if isinstance(output, TensorDesc)
                else TensorDesc.activation(output, dtype, shape)
            )
            assert tensor.dtype == dtype, "operator output dtype mismatch"
            assert tensor.shape == shape, "operator output shape mismatch"
            bound.append(tensor)
        self.outputs = tuple(bound)
        return self

    @property
    def bound_outputs(self) -> tuple[TensorDesc, ...]:
        """Return finalized output bindings for operator lowering."""
        assert self.outputs is not None, "operator output bindings are required"
        return self.outputs

    @property
    def bound_weights(self) -> WeightsT:
        """Return required weights; weight-optional operators inspect weights directly."""
        assert self.weights is not None, "operator weight bindings are required"
        return self.weights

    def prepare(self) -> tuple[tuple[Kernel, ...], int]:
        """Lower finalized bindings into calls with a fresh local scratch layout."""
        assert self.outputs is not None, "operator output bindings are required"
        scratch = ScratchLayout()
        calls = tuple(self.kernels(scratch))
        return calls, scratch.num_bytes

    @abc.abstractmethod
    def kernels(self, scratch: ScratchLayout) -> Iterable[Kernel]:
        """Yield calls in execution order, reserving scratch at local byte offsets.

        The caller resolves activation and weight references later.
        """
        raise NotImplementedError
