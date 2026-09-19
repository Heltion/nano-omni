"""Fused INT8 projection, LoRA update, and gated residual addition."""

import dataclasses
from typing import TYPE_CHECKING

import tilelang

from nano_omni.core.kernel import Arguments, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.base import MatmulConfig

if TYPE_CHECKING:
    from tilelang.language.eager import PrimFunc


class MatmulInt8LoraResidualGateWorkload(Workload):
    """Token count and matrix dimensions for the fused projection."""

    num_tokens: int
    columns: int
    k: int


@dataclasses.dataclass(frozen=True, slots=True)
class MatmulInt8LoraResidualGateArguments(Arguments):
    input: TensorDesc
    weight: TensorDesc
    lora: TensorDesc
    output: TensorDesc
    weight_scale: TensorDesc
    input_scale: TensorDesc
    hidden: TensorDesc
    gate: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one program for different token counts."""
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def matmul_int8_lora_residual_gate(
    num_tokens: int,
    columns: int,
    k: int,
    tile_tokens: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    threads: int = 256,
    stages: int = 2,
) -> "PrimFunc":
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor((dynamic_num_tokens, k), T.int8),
        weight: T.Tensor((columns, k), T.int8),
        lora: T.Tensor((dynamic_num_tokens, columns), T.bfloat16),
        output: T.Tensor((dynamic_num_tokens, columns), T.bfloat16),
        weight_scale: T.Tensor((columns, 1), T.float32),
        input_scale: T.Tensor((dynamic_num_tokens,), T.float32),
        hidden: T.Tensor((dynamic_num_tokens, columns), T.bfloat16),
        gate: T.Tensor((columns,), T.bfloat16),
    ):
        T.annotate_pass_configs({"tl.disable_warp_specialized": False})
        with T.Kernel(
            T.ceildiv(columns, tile_n),
            T.ceildiv(dynamic_num_tokens, tile_tokens),
            threads=threads,
        ) as (column, row):
            T.use_swizzle(panel_size=8, enable=True)
            a = T.alloc_shared((tile_tokens, tile_k), T.int8)
            b = T.alloc_shared((tile_n, tile_k), T.int8)
            accumulator = T.alloc_fragment((tile_tokens, tile_n), T.int32)
            result = T.alloc_fragment((tile_tokens, tile_n), T.float32)
            output_values = T.alloc_fragment((tile_tokens, tile_n), T.bfloat16)
            T.clear(accumulator)
            for inner in T.Pipelined(T.ceildiv(k, tile_k), num_stages=stages):
                T.copy(input[row * tile_tokens, inner * tile_k], a)
                T.copy(weight[column * tile_n, inner * tile_k], b)
                T.gemm(
                    a, b, accumulator, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                )
            for i, j in T.Parallel(tile_tokens, tile_n):
                r, c = row * tile_tokens + i, column * tile_n + j
                result[i, j] = (
                    accumulator[i, j].astype(T.float32)
                    * input_scale[r]
                    * weight_scale[c, 0]
                )
                result[i, j] += lora[r, c]
                # Match the BF16 update buffer and BF16 arithmetic in segment_gate.
                output_values[i, j] = (
                    hidden[r, c] + T.cast(result[i, j], T.bfloat16) * gate[c]
                )
            T.copy(output_values, output[row * tile_tokens, column * tile_n])

    return main.with_attr(
        "global_symbol",
        f"matmul_int8_lora_residual_gate_{columns}_{k}_{tile_tokens}_{tile_n}_{tile_k}_{threads}_{stages}",
    )


class MatmulInt8LoraResidualGateKernel(
    Kernel[
        MatmulInt8LoraResidualGateArguments,
        MatmulInt8LoraResidualGateWorkload,
        MatmulConfig,
    ]
):
    """Bind the fused INT8 projection to its arguments and reference."""

    name = "matmul_int8_lora_residual_gate"
    program = matmul_int8_lora_residual_gate

    @classmethod
    def make_arguments(
        cls, workload: MatmulInt8LoraResidualGateWorkload
    ) -> MatmulInt8LoraResidualGateArguments:
        """Describe fused projection inputs and its BF16 output."""
        num_tokens, n, k = workload.num_tokens, workload.columns, workload.k
        return MatmulInt8LoraResidualGateArguments(
            input=TensorDesc.empty(DType.I8, (num_tokens, k)),
            weight=TensorDesc.empty(DType.I8, (n, k)),
            lora=TensorDesc.empty(DType.BF16, (num_tokens, n)),
            output=TensorDesc.empty(DType.BF16, (num_tokens, n)),
            weight_scale=TensorDesc.empty(DType.F32, (n, 1)),
            input_scale=TensorDesc.empty(DType.F32, (num_tokens,)),
            hidden=TensorDesc.empty(DType.BF16, (num_tokens, n)),
            gate=TensorDesc.empty(DType.BF16, (n,)),
        )

    @classmethod
    def make_workload(
        cls, arguments: MatmulInt8LoraResidualGateArguments
    ) -> MatmulInt8LoraResidualGateWorkload:
        """Recover the specialization and validate every tensor contract."""
        assert len(arguments.input.shape) == len(arguments.weight.shape) == 2
        num_tokens, k = arguments.input.shape
        columns, weight_k = arguments.weight.shape
        assert weight_k == k
        assert arguments.input.dtype == arguments.weight.dtype == DType.I8
        assert arguments.weight_scale.dtype == DType.F32
        assert arguments.weight_scale.shape == (columns, 1)
        assert arguments.input_scale.dtype == DType.F32
        assert arguments.input_scale.shape == (num_tokens,)
        for tensor in (arguments.lora, arguments.output, arguments.hidden):
            assert tensor.dtype == DType.BF16
            assert tensor.shape == (num_tokens, columns)
        assert arguments.gate.dtype == DType.BF16
        assert arguments.gate.shape == (columns,)
        return MatmulInt8LoraResidualGateWorkload(
            num_tokens=num_tokens,
            columns=columns,
            k=k,
        )

    @classmethod
    def tops(cls, arguments: MatmulInt8LoraResidualGateArguments) -> Tops:
        """Count the INT8 MMA operations in the projection."""
        workload = cls.make_workload(arguments)
        return {
            MmaType.I8I8I32: 2 * workload.num_tokens * workload.columns * workload.k
        }

    @classmethod
    def ref_program(cls, arguments: MatmulInt8LoraResidualGateArguments) -> None:
        """Evaluate the INT8 projection and preserve the fused BF16 boundary."""
        import torch

        input = arguments.input.as_torch()
        weight = arguments.weight.as_torch()
        lora = arguments.lora.as_torch()
        output = arguments.output.as_torch()
        weight_scale = arguments.weight_scale.as_torch()
        input_scale = arguments.input_scale.as_torch()
        hidden = arguments.hidden.as_torch()
        gate = arguments.gate.as_torch()
        rows = input.shape[0]
        if rows <= 16:
            # torch._int_mm rejects small M even though the TileLang kernel does not.
            input = torch.nn.functional.pad(input, (0, 0, 0, 17 - rows))
        projected = torch._int_mm(input, weight.T)[:rows].float()
        projected *= input_scale[:, None]
        projected *= weight_scale.T
        projected += lora.float()
        output.copy_(torch.addcmul(hidden, projected.bfloat16(), gate))

    @classmethod
    def make_config(cls, workload: MatmulInt8LoraResidualGateWorkload) -> MatmulConfig:
        """Select the measured configuration for this implementation."""
        del workload
        return MatmulConfig(
            tile_tokens=128,
            tile_n=128,
            tile_k=128,
            threads=256,
            stages=2,
        )
