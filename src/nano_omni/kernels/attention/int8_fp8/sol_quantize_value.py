"""Quantize Sol values into the FP8 layout used by exact attention."""

import dataclasses
from typing import Literal

from pydantic import Field

from nano_omni.core.kernel import Arguments, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.int8_fp8.quantize_value import (
    int8_fp8_quantize_value,
)
from nano_omni.kernels.attention.int8_fp8.value_scale import Int8Fp8ThreadConfig


class SolInt8Fp8ValueWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    permute: Literal[False] = False


@dataclasses.dataclass(frozen=True, slots=True)
class SolInt8Fp8ValueArguments(Arguments):
    source: TensorDesc
    scale: TensorDesc
    output: TensorDesc
    heads: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.source.shape[0]),)


class SolInt8Fp8ValueKernel(
    Kernel[SolInt8Fp8ValueArguments, SolInt8Fp8ValueWorkload, Int8Fp8ThreadConfig]
):
    """Quantize V in source order for sparse block loads."""

    name = "sol_int8_fp8_quantize_value"
    program = int8_fp8_quantize_value

    @classmethod
    def make_arguments(
        cls, workload: SolInt8Fp8ValueWorkload
    ) -> SolInt8Fp8ValueArguments:
        num_padded_tokens = -(-workload.num_tokens // 64) * 64
        return SolInt8Fp8ValueArguments(
            TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.heads * 128)),
            TensorDesc.empty(DType.F32, (workload.heads, 128)),
            TensorDesc.empty(DType.FP8_E4M3, (workload.heads, 128, num_padded_tokens)),
            workload.heads,
        )

    @classmethod
    def make_config(cls, workload: SolInt8Fp8ValueWorkload) -> Int8Fp8ThreadConfig:
        del workload
        return Int8Fp8ThreadConfig(threads=256)

    @classmethod
    def make_workload(
        cls, arguments: SolInt8Fp8ValueArguments
    ) -> SolInt8Fp8ValueWorkload:
        num_tokens, columns = arguments.source.shape
        assert arguments.source.dtype == DType.BF16
        assert columns == arguments.heads * 128
        assert arguments.scale.dtype == DType.F32
        assert arguments.scale.shape == (arguments.heads, 128)
        assert arguments.output.dtype == DType.FP8_E4M3
        assert arguments.output.shape == (
            arguments.heads,
            128,
            -(-num_tokens // 64) * 64,
        )
        return SolInt8Fp8ValueWorkload(num_tokens=num_tokens, heads=arguments.heads)

    @classmethod
    def ref_program(cls, arguments: SolInt8Fp8ValueArguments) -> None:
        import torch

        source = arguments.source.as_torch()
        scale = arguments.scale.as_torch()
        output = arguments.output.as_torch()
        num_padded_tokens = output.shape[2]
        values = torch.zeros(
            num_padded_tokens,
            arguments.heads,
            128,
            dtype=torch.float32,
            device=source.device,
        )
        values[: source.shape[0]] = source.reshape(-1, arguments.heads, 128)
        values /= scale
        output.copy_(values.permute(1, 2, 0))
