"""Emit H3 sampling calls after host coefficients are bound to activations."""

from nano_omni.core.kernel import Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.op import Op
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.sampling.denoised import DenoisedArguments, DenoisedKernel
from nano_omni.kernels.sampling.res_multistep import (
    ResMultistepArguments,
    ResMultistepKernel,
)


class Combine(Op[None]):
    def __init__(
        self,
        latent: int | tuple[int, int],
        clean: int | tuple[int, int],
        old: int | tuple[int, int],
        *,
        shape: tuple[int, ...],
        scales: tuple[float, float, float] | tuple[int, int],
    ) -> None:
        inputs = tuple(
            TensorDesc.activation(position, DType.F32, shape)
            for position in (latent, clean, old)
        )
        super().__init__(*inputs, outputs=((DType.F32, shape),))
        # The model packs the three coefficients plus [1, 0] into five F32 values,
        # then replaces this tuple with the activation slice before lowering.
        self.scales = scales

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        coefficients = self.scales
        assert len(coefficients) == 2, "sampling coefficients must be bound"
        latent, clean, old = self.inputs
        (output,) = self.bound_outputs
        return (
            ResMultistepKernel(
                ResMultistepArguments(
                    latent,
                    clean,
                    old,
                    output,
                    TensorDesc.activation(coefficients, DType.F32, (5,)),
                )
            ),
        )


class Denoise(Op[None]):
    def __init__(
        self,
        latent: int | tuple[int, int],
        model: int | tuple[int, int],
        *,
        shape: tuple[int, ...],
        sigma: int | tuple[int, int],
    ) -> None:
        self.sigma = (sigma, 0) if isinstance(sigma, int) else sigma
        inputs = tuple(
            TensorDesc.activation(position, DType.F32, shape)
            for position in (latent, model)
        )
        super().__init__(*inputs, outputs=((DType.F32, shape),))

    def kernels(self, scratch: ScratchLayout) -> tuple[Kernel]:
        latent, model = self.inputs
        (output,) = self.bound_outputs
        return (
            DenoisedKernel(
                DenoisedArguments(
                    latent,
                    model,
                    output,
                    TensorDesc.activation(self.sigma, DType.F32, (1,)),
                )
            ),
        )
