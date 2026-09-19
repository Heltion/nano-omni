"""Generate shared cosine and sine planes from multi-axis positions."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class H3RopeWorkload(Workload):
    """Position matrix and inverse-frequency dimensions."""

    tokens: int = Field(gt=0)
    axes: int = Field(gt=0)
    frequencies: int = Field(gt=0)


@dataclasses.dataclass(frozen=True, slots=True)
class H3RopeArguments(Arguments):
    """Positions, inverse frequencies, and cosine/sine output planes."""

    positions: TensorDesc
    inverse_frequencies: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one program for different token counts."""
        return (("tokens", self.positions.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def h3_rope(tokens, axes, frequencies, tile_elements=256, threads=256):
    """Build the shared multi-axis rotary table."""
    import tilelang.language as T

    dynamic_tokens = T.dynamic("tokens")
    columns = axes * frequencies
    elements = dynamic_tokens * columns

    @T.prim_func
    def main(
        tokens: dynamic_tokens,
        positions: T.Tensor([dynamic_tokens, axes], T.float32),
        inverse_frequencies: T.Tensor([frequencies], T.float32),
        output: T.Tensor([2, dynamic_tokens, columns], T.float32),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                if index < elements:
                    token = index // columns
                    column = index % columns
                    axis = column // frequencies
                    frequency = column % frequencies
                    angle = positions[token, axis] * inverse_frequencies[frequency]
                    output[0, token, column] = T.cos(angle)
                    output[1, token, column] = T.sin(angle)

    return main


class H3RopeKernel(ElementwiseKernel[H3RopeArguments, H3RopeWorkload]):
    """Generate the cosine and sine planes shared by H3 attention blocks."""

    name = "h3_rope"
    program = h3_rope
    launch = (256, 256)

    @classmethod
    def make_arguments(cls, workload: H3RopeWorkload) -> H3RopeArguments:
        """Describe positions, frequencies, and the two output planes."""
        return H3RopeArguments(
            positions=TensorDesc.empty(DType.F32, (workload.tokens, workload.axes)),
            inverse_frequencies=TensorDesc.empty(DType.F32, (workload.frequencies,)),
            output=TensorDesc.empty(
                DType.F32,
                (2, workload.tokens, workload.axes * workload.frequencies),
            ),
        )

    @classmethod
    def make_workload(cls, arguments: H3RopeArguments) -> H3RopeWorkload:
        """Validate tensor shapes and recover the rotary dimensions."""
        assert len(arguments.positions.shape) == 2
        assert len(arguments.inverse_frequencies.shape) == 1
        tokens, axes = arguments.positions.shape
        (frequencies,) = arguments.inverse_frequencies.shape
        assert arguments.output.shape == (2, tokens, axes * frequencies)
        assert all(
            tensor.dtype == DType.F32
            for tensor in (
                arguments.positions,
                arguments.inverse_frequencies,
                arguments.output,
            )
        )
        return H3RopeWorkload(tokens=tokens, axes=axes, frequencies=frequencies)

    @classmethod
    def ref_program(cls, arguments: H3RopeArguments) -> None:
        """Generate both planes directly with Torch."""
        import torch

        angles = (
            arguments.positions.as_torch()[:, :, None]
            * arguments.inverse_frequencies.as_torch()
        ).flatten(1)
        arguments.output.as_torch().copy_(torch.stack((angles.cos(), angles.sin())))
