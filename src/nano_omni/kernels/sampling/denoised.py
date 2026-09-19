"""Convert the model prediction to a clean F32 latent at the current noise level."""

import dataclasses
import math

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class DenoisedWorkload(Workload):
    """Number of contiguous elements read from each input and written to output."""

    elements: int


@dataclasses.dataclass(frozen=True, slots=True)
class DenoisedArguments(Arguments):
    """F32 latent/model/output with equal element counts, plus F32 sigma[1].

    Sigma stores the current timestep's noise level. Each model element is
    rounded to BF16 before computing output = latent - model * sigma[0].
    """

    latent: TensorDesc
    model: TensorDesc
    output: TensorDesc
    sigma: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def denoised(elements, tile_elements=256, threads=128):
    import tilelang.language as T

    @T.prim_func
    def main(
        latent: T.Tensor([elements], T.float32),
        model: T.Tensor([elements], T.float32),
        output: T.Tensor([elements], T.float32),
        sigma: T.Tensor([1], T.float32),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for offset in T.Parallel(tile_elements):
                linear = block * tile_elements + offset
                if linear < elements:
                    prediction = T.cast(
                        T.cast(model[linear], T.bfloat16),
                        T.float32,
                    )
                    output[linear] = latent[linear] - prediction * sigma[0]

    return main.with_attr(
        "global_symbol", f"denoised_{elements}_{tile_elements}_{threads}"
    )


class DenoisedKernel(ElementwiseKernel[DenoisedArguments, DenoisedWorkload]):
    name = "denoised"
    program = denoised
    launch = (256, 128)

    @classmethod
    def make_arguments(cls, workload: DenoisedWorkload) -> DenoisedArguments:
        """Describe flat F32 latent, prediction, output, and scalar sigma."""
        return DenoisedArguments(
            latent=TensorDesc.empty(DType.F32, (workload.elements,)),
            model=TensorDesc.empty(DType.F32, (workload.elements,)),
            output=TensorDesc.empty(DType.F32, (workload.elements,)),
            sigma=TensorDesc.empty(DType.F32, (1,)),
        )

    @classmethod
    def make_workload(cls, arguments: DenoisedArguments) -> DenoisedWorkload:
        """Recover the flat size and validate latent, prediction, and sigma."""
        values = (arguments.latent, arguments.model, arguments.output)
        assert all(value.dtype == DType.F32 for value in values)
        assert all(value.shape == arguments.latent.shape for value in values)
        assert arguments.sigma.dtype == DType.F32
        assert arguments.sigma.shape == (1,)
        return DenoisedWorkload(elements=math.prod(arguments.latent.shape))

    @classmethod
    def ref_program(cls, arguments: DenoisedArguments) -> None:
        """Apply BF16-rounded prediction subtraction into output."""
        import torch

        latent = arguments.latent.as_torch()
        prediction = arguments.model.as_torch().to(torch.bfloat16).float()
        arguments.output.as_torch().copy_(
            latent - prediction * arguments.sigma.as_torch()[0]
        )
