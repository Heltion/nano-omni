"""Map token coordinates to interleaved multimodal rotary angles."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel

HEIGHT_FREQUENCIES = 20
WIDTH_FREQUENCIES = 20


class QwenMropeWorkload(Workload):
    """Token, coordinate-axis, and rotary-frequency dimensions."""

    num_tokens: int
    axes: int
    frequencies: int


@dataclasses.dataclass(frozen=True, slots=True)
class QwenMropeArguments(Arguments):
    """Coordinates, inverse frequencies, and generated angle table."""

    positions: TensorDesc
    inverse_frequencies: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.positions.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def qwen_mrope(num_tokens, axes, frequencies, tile_elements=1024, threads=256):
    """Build multimodal coordinate selection and angle generation."""
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    num_elements = dynamic_num_tokens * frequencies

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        positions: T.Tensor([dynamic_num_tokens, axes], T.float32),
        inverse_frequencies: T.Tensor([frequencies], T.float32),
        output: T.Tensor([dynamic_num_tokens, frequencies], T.float32),
    ):
        with T.Kernel(T.ceildiv(num_elements, tile_elements), threads=threads) as block:
            for local in T.Parallel(tile_elements):
                offset = block * tile_elements + local
                if offset < num_elements:
                    token = offset // frequencies
                    frequency = offset % frequencies
                    axis = T.if_then_else(
                        frequency % 3 == 1 and frequency < HEIGHT_FREQUENCIES * 3,
                        1,
                        T.if_then_else(
                            frequency % 3 == 2 and frequency < WIDTH_FREQUENCIES * 3,
                            2,
                            0,
                        ),
                    )
                    output[token, frequency] = (
                        positions[token, axis] * inverse_frequencies[frequency]
                    )

    return main.with_attr(
        "global_symbol", f"qwen_mrope_{axes}_{frequencies}_{tile_elements}_{threads}"
    )


class QwenMropeKernel(ElementwiseKernel[QwenMropeArguments, QwenMropeWorkload]):
    """Generate Qwen multimodal rotary angles."""

    name = "qwen_mrope"
    program = qwen_mrope
    launch = (1024, 256)

    @classmethod
    def make_arguments(cls, workload: QwenMropeWorkload) -> QwenMropeArguments:
        """Describe coordinates, frequencies, and output angles."""
        return QwenMropeArguments(
            positions=TensorDesc.empty(DType.F32, (workload.num_tokens, workload.axes)),
            inverse_frequencies=TensorDesc.empty(DType.F32, (workload.frequencies,)),
            output=TensorDesc.empty(
                DType.F32, (workload.num_tokens, workload.frequencies)
            ),
        )

    @classmethod
    def make_workload(cls, arguments: QwenMropeArguments) -> QwenMropeWorkload:
        """Validate tensor shapes and recover the specialization."""
        assert len(arguments.positions.shape) == 2
        assert len(arguments.inverse_frequencies.shape) == 1
        num_tokens, axes = arguments.positions.shape
        (frequencies,) = arguments.inverse_frequencies.shape
        assert axes == 3
        assert arguments.output.shape == (num_tokens, frequencies)
        assert all(
            tensor.dtype == DType.F32
            for tensor in (
                arguments.positions,
                arguments.inverse_frequencies,
                arguments.output,
            )
        )
        return QwenMropeWorkload(
            num_tokens=num_tokens, axes=axes, frequencies=frequencies
        )

    @classmethod
    def ref_program(cls, arguments: QwenMropeArguments) -> None:
        """Select each frequency's coordinate and multiply it directly."""
        import torch

        frequencies = arguments.inverse_frequencies.shape[0]
        axes = torch.zeros(
            frequencies, dtype=torch.long, device=arguments.output.as_torch().device
        )
        axes[1 : HEIGHT_FREQUENCIES * 3 : 3] = 1
        axes[2 : WIDTH_FREQUENCIES * 3 : 3] = 2
        arguments.output.as_torch().copy_(
            arguments.positions.as_torch()[:, axes]
            * arguments.inverse_frequencies.as_torch()
        )
