"""Combine F32 latent estimates with coefficients selected for the current step."""

import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class ResMultistepWorkload(Workload):
    """Number of contiguous elements read from each estimate and written to output."""

    elements: int


@dataclasses.dataclass(frozen=True, slots=True)
class ResMultistepArguments(Arguments):
    """F32 latent/clean/old_denoised/output with equal contiguous element counts.

    The host selects coefficients[5] for the timestep: latent, clean, and old
    estimate weights, followed by an output scale and bias. The F32 output is
    (latent * c[0] + clean * c[1] + old_denoised * c[2]) * c[3] + c[4].
    """

    latent: TensorDesc
    clean: TensorDesc
    old_denoised: TensorDesc
    output: TensorDesc
    coefficients: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def res_multistep(elements, tile_elements=256, threads=128):
    import tilelang.language as T

    @T.prim_func
    def main(
        latent: T.Tensor([elements], T.float32),
        clean: T.Tensor([elements], T.float32),
        old_denoised: T.Tensor([elements], T.float32),
        output: T.Tensor([elements], T.float32),
        coefficients: T.Tensor([5], T.float32),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for offset in T.Parallel(tile_elements):
                index = block * tile_elements + offset
                if index < elements:
                    updated = (
                        latent[index] * coefficients[0]
                        + clean[index] * coefficients[1]
                        + old_denoised[index] * coefficients[2]
                    )
                    output[index] = updated * coefficients[3] + coefficients[4]

    return main.with_attr(
        "global_symbol", f"res_multistep_{elements}_{tile_elements}_{threads}"
    )


class ResMultistepKernel(
    ElementwiseKernel[ResMultistepArguments, ResMultistepWorkload]
):
    name = "res_multistep"
    program = res_multistep
    launch = (256, 128)

    @classmethod
    def make_arguments(cls, workload: ResMultistepWorkload) -> ResMultistepArguments:
        """Describe the three estimates, output, and five coefficients."""
        return ResMultistepArguments(
            latent=TensorDesc.empty(DType.F32, (workload.elements,)),
            clean=TensorDesc.empty(DType.F32, (workload.elements,)),
            old_denoised=TensorDesc.empty(DType.F32, (workload.elements,)),
            output=TensorDesc.empty(DType.F32, (workload.elements,)),
            coefficients=TensorDesc.empty(DType.F32, (5,)),
        )

    @classmethod
    def make_workload(cls, arguments: ResMultistepArguments) -> ResMultistepWorkload:
        """Recover the flat size and validate estimates and coefficients."""
        values = (
            arguments.latent,
            arguments.clean,
            arguments.old_denoised,
            arguments.output,
        )
        assert all(value.dtype == DType.F32 for value in values)
        assert all(value.shape == arguments.latent.shape for value in values)
        assert arguments.coefficients.dtype == DType.F32
        assert arguments.coefficients.shape == (5,)
        return ResMultistepWorkload(elements=math.prod(arguments.latent.shape))

    @classmethod
    def ref_program(cls, arguments: ResMultistepArguments) -> None:
        """Apply the five scheduler coefficients into output."""
        coefficients = arguments.coefficients.as_torch()
        updated = (
            arguments.latent.as_torch() * coefficients[0]
            + arguments.clean.as_torch() * coefficients[1]
            + arguments.old_denoised.as_torch() * coefficients[2]
        )
        arguments.output.as_torch().copy_(updated * coefficients[3] + coefficients[4])
