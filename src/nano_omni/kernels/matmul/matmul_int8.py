"""INT8 tensor-core matrix multiplication with a scaled fused epilogue."""

import dataclasses
from typing import Literal

import tilelang

from nano_omni.core.kernel import Arguments, MmaType
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.base import MatmulConfig, MatmulKernel, MatmulWorkload


class MatmulInt8Workload(MatmulWorkload):
    """Matrix dimensions and fused epilogue semantics."""

    use_bias: bool
    use_residual: bool
    use_multiply: bool
    use_silu: bool


class MatmulInt8Config(MatmulConfig):
    """Tensor-core tiles and scheduling choices for the INT8 program."""

    warp_policy: Literal["full_row", "full_col", "square"] = "full_row"
    warp_specialization: bool = False
    swizzle: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class MatmulInt8Arguments(Arguments):
    """INT8 A[num_tokens,k] and B[n,k] accumulate into INT32 before F32 dequantization.

    Input scales are F32 [num_tokens]; weight scales are F32 [n,1]. Bias[n] and
    residual/multiply/output[num_tokens,n] use BF16 after the scaled epilogue.
    """

    input: TensorDesc
    weight: TensorDesc
    bias: TensorDesc | None
    residual: TensorDesc | None
    multiply: TensorDesc | None
    output: TensorDesc
    weight_scale: TensorDesc
    input_scale: TensorDesc
    use_silu: bool

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one program for different matrix row counts."""
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def matmul_int8(
    num_tokens,
    n,
    k,
    use_bias,
    use_residual,
    use_multiply,
    use_silu,
    tile_tokens=128,
    tile_n=128,
    tile_k=128,
    threads=256,
    stages=2,
    warp_policy="full_row",
    warp_specialization=True,
    swizzle=8,
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")

    policy = {
        "full_row": T.GemmWarpPolicy.FullRow,
        "full_col": T.GemmWarpPolicy.FullCol,
        "square": T.GemmWarpPolicy.Square,
    }[warp_policy]

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor([dynamic_num_tokens, k], T.int8),
        weight: T.Tensor([n, k], T.int8),
        bias: T.Tensor([n], T.bfloat16),
        residual: T.Tensor([dynamic_num_tokens, n], T.bfloat16),
        multiply: T.Tensor([dynamic_num_tokens, n], T.bfloat16),
        output: T.Tensor([dynamic_num_tokens, n], T.bfloat16),
        weight_scale: T.Tensor([n, 1], T.float32),
        input_scale: T.Tensor([dynamic_num_tokens], T.float32),
    ):
        T.annotate_pass_configs(
            {"tl.disable_warp_specialized": not warp_specialization}
        )
        with T.Kernel(
            T.ceildiv(n, tile_n),
            T.ceildiv(dynamic_num_tokens, tile_tokens),
            threads=threads,
        ) as (column, row):
            T.use_swizzle(panel_size=swizzle, enable=swizzle > 0)
            input_shared = T.alloc_shared([tile_tokens, tile_k], T.int8)
            weight_shared = T.alloc_shared([tile_n, tile_k], T.int8)
            accumulator = T.alloc_fragment([tile_tokens, tile_n], T.int32)
            result = T.alloc_fragment([tile_tokens, tile_n], T.float32)
            T.clear(accumulator)
            for inner in T.Pipelined(T.ceildiv(k, tile_k), num_stages=stages):
                T.copy(input[row * tile_tokens, inner * tile_k], input_shared)
                T.copy(weight[column * tile_n, inner * tile_k], weight_shared)
                T.gemm(
                    input_shared,
                    weight_shared,
                    accumulator,
                    transpose_B=True,
                    policy=policy,
                )
            for i, j in T.Parallel(tile_tokens, tile_n):
                result[i, j] = (
                    accumulator[i, j].astype(T.float32)
                    * input_scale[row * tile_tokens + i]
                    * weight_scale[column * tile_n + j, 0]
                )
                if use_bias:
                    result[i, j] += bias[column * tile_n + j]
                if use_residual:
                    result[i, j] += residual[row * tile_tokens + i, column * tile_n + j]
                if use_silu:
                    result[i, j] /= 1.0 + T.exp(-result[i, j])
                if use_multiply:
                    result[i, j] *= multiply[row * tile_tokens + i, column * tile_n + j]
            T.copy(result, output[row * tile_tokens, column * tile_n])

    return main.with_attr(
        "global_symbol",
        f"matmul_int8_{n}_{k}_{tile_tokens}_{tile_n}_{tile_k}_{threads}_{stages}_{int(use_bias)}{int(use_residual)}{int(use_multiply)}{int(use_silu)}_{warp_policy}_{int(warp_specialization)}_{swizzle}",
    )


class MatmulInt8Kernel(
    MatmulKernel[MatmulInt8Arguments, MatmulInt8Workload, MatmulInt8Config]
):
    """Bind the INT8 MMA program to its arguments and reference semantics."""

    name = "matmul_int8"
    program = matmul_int8
    mma_type = MmaType.I8I8I32

    @classmethod
    def make_arguments(cls, workload: MatmulInt8Workload) -> MatmulInt8Arguments:
        """Describe quantized operands, optional epilogue tensors, and output."""
        output_shape = (workload.num_tokens, workload.n)
        return MatmulInt8Arguments(
            input=TensorDesc.empty(DType.I8, (workload.num_tokens, workload.k)),
            weight=TensorDesc.empty(DType.I8, (workload.n, workload.k)),
            bias=(
                TensorDesc.empty(DType.BF16, (workload.n,))
                if workload.use_bias
                else None
            ),
            residual=(
                TensorDesc.empty(DType.BF16, output_shape)
                if workload.use_residual
                else None
            ),
            multiply=(
                TensorDesc.empty(DType.BF16, output_shape)
                if workload.use_multiply
                else None
            ),
            output=TensorDesc.empty(DType.BF16, output_shape),
            weight_scale=TensorDesc.empty(DType.F32, (workload.n, 1)),
            input_scale=TensorDesc.empty(DType.F32, (workload.num_tokens,)),
            use_silu=workload.use_silu,
        )

    @classmethod
    def make_workload(cls, arguments: MatmulInt8Arguments) -> MatmulInt8Workload:
        """Recover the specialization and validate every tensor contract."""
        input, weight = arguments.input, arguments.weight
        assert len(input.shape) == len(weight.shape) == 2
        num_tokens, k = input.shape
        n, weight_k = weight.shape
        assert weight_k == k
        assert input.dtype == weight.dtype == DType.I8
        assert arguments.output.dtype == DType.BF16
        assert arguments.output.shape == (num_tokens, n)
        assert arguments.weight_scale.dtype == DType.F32
        assert arguments.weight_scale.shape == (n, 1)
        assert arguments.input_scale.dtype == DType.F32
        assert arguments.input_scale.shape == (num_tokens,)
        if arguments.bias is not None:
            assert arguments.bias.dtype == DType.BF16
            assert arguments.bias.shape == (n,)
        for operand in (arguments.residual, arguments.multiply):
            if operand is not None:
                assert operand.dtype == DType.BF16
                assert operand.shape == (num_tokens, n)
        return MatmulInt8Workload(
            num_tokens=num_tokens,
            n=n,
            k=k,
            use_bias=arguments.bias is not None,
            use_residual=arguments.residual is not None,
            use_multiply=arguments.multiply is not None,
            use_silu=arguments.use_silu,
        )

    @classmethod
    def ref_program(cls, arguments: MatmulInt8Arguments) -> None:
        """Evaluate INT8 GEMM and its optional fused FP32 epilogue."""
        import torch

        input = arguments.input.as_torch()
        weight = arguments.weight.as_torch()
        output = arguments.output.as_torch()
        weight_scale = arguments.weight_scale.as_torch()
        input_scale = arguments.input_scale.as_torch()
        bias = arguments.bias.as_torch() if arguments.bias is not None else None
        residual = (
            arguments.residual.as_torch() if arguments.residual is not None else None
        )
        multiply = (
            arguments.multiply.as_torch() if arguments.multiply is not None else None
        )
        rows = input.shape[0]
        if rows <= 16:
            # torch._int_mm rejects small M even though the TileLang kernel does not.
            input = torch.nn.functional.pad(input, (0, 0, 0, 17 - rows))
        value = torch._int_mm(input, weight.T)[:rows].float()
        value *= weight_scale.T
        value *= input_scale[:, None]
        if bias is not None:
            value += bias.float()
        if residual is not None:
            value += residual.float()
        if arguments.use_silu:
            value = torch.nn.functional.silu(value)
        if multiply is not None:
            value *= multiply.float()
        output.copy_(value)

    @classmethod
    def make_config(cls, workload: MatmulInt8Workload) -> MatmulInt8Config:
        """Select the measured production schedule for this workload."""
        if (
            workload.num_tokens in (60778, 73923)
            and (workload.n, workload.k) in ((1792, 5376), (5376, 7168))
            and not workload.use_bias
            and workload.use_residual
            and not workload.use_multiply
            and not workload.use_silu
        ):
            return MatmulInt8Config(
                tile_tokens=128,
                tile_n=256,
                tile_k=64,
                threads=512,
                stages=2,
                warp_specialization=True,
                swizzle=8,
            )
        if (workload.num_tokens, workload.n, workload.k) == (4096, 14336, 5376):
            return MatmulInt8Config(
                tile_tokens=128,
                tile_n=256,
                tile_k=64,
                threads=512,
                stages=2,
                warp_specialization=True,
                swizzle=8,
            )
        return MatmulInt8Config(
            tile_tokens=128,
            tile_n=128,
            tile_k=128,
            threads=256,
            stages=3,
            warp_specialization=True,
            swizzle=8,
        )
