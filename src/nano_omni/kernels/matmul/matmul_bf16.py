"""BF16 tensor-core matrix multiplication with an optional fused epilogue."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, MmaType
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.base import MatmulConfig, MatmulKernel, MatmulWorkload


class MatmulBf16Workload(MatmulWorkload):
    """Matrix dimensions and fused epilogue semantics."""

    use_bias: bool
    use_residual: bool
    use_multiply: bool
    use_silu: bool
    output_scale: float = Field(default=1.0, allow_inf_nan=False)


@dataclasses.dataclass(frozen=True, slots=True)
class MatmulBf16Arguments(Arguments):
    """BF16 A[num_tokens,k], weight[n,k], bias[n], residual/multiply/output[num_tokens,n].

    Accumulate FP32; apply scale, bias, residual, SiLU, multiply in that order."""

    input: TensorDesc
    weight: TensorDesc
    bias: TensorDesc | None
    residual: TensorDesc | None
    multiply: TensorDesc | None
    output: TensorDesc
    output_scale: float | int
    use_silu: bool

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one program for different matrix row counts."""
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def matmul_bf16(
    num_tokens,
    n,
    k,
    use_bias,
    use_residual,
    use_multiply,
    use_silu,
    output_scale,
    tile_tokens=32,
    tile_n=64,
    tile_k=32,
    threads=128,
    stages=2,
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor([dynamic_num_tokens, k], T.bfloat16),
        weight: T.Tensor([n, k], T.bfloat16),
        bias: T.Tensor([n], T.bfloat16),
        residual: T.Tensor([dynamic_num_tokens, n], T.bfloat16),
        multiply: T.Tensor([dynamic_num_tokens, n], T.bfloat16),
        output: T.Tensor([dynamic_num_tokens, n], T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(n, tile_n),
            T.ceildiv(dynamic_num_tokens, tile_tokens),
            threads=threads,
        ) as (column, row):
            input_shared = T.alloc_shared([tile_tokens, tile_k], T.bfloat16)
            weight_shared = T.alloc_shared([tile_n, tile_k], T.bfloat16)
            accumulator = T.alloc_fragment([tile_tokens, tile_n], T.float32)
            T.clear(accumulator)
            for inner_block in T.Pipelined(T.ceildiv(k, tile_k), num_stages=stages):
                T.copy(input[row * tile_tokens, inner_block * tile_k], input_shared)
                T.copy(weight[column * tile_n, inner_block * tile_k], weight_shared)
                T.gemm(
                    input_shared,
                    weight_shared,
                    accumulator,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for i, j in T.Parallel(tile_tokens, tile_n):
                accumulator[i, j] *= output_scale
                if use_bias:
                    accumulator[i, j] += bias[column * tile_n + j]
                if use_residual:
                    accumulator[i, j] += residual[
                        row * tile_tokens + i, column * tile_n + j
                    ]
                if use_silu:
                    accumulator[i, j] /= 1.0 + T.exp(-accumulator[i, j])
                if use_multiply:
                    accumulator[i, j] *= multiply[
                        row * tile_tokens + i, column * tile_n + j
                    ]
            T.copy(accumulator, output[row * tile_tokens, column * tile_n])

    return main.with_attr(
        "global_symbol",
        "matmul_bf16_"
        f"{n}_{k}_{int(use_bias)}_{int(use_residual)}_"
        f"{int(use_multiply)}_{int(use_silu)}_{tile_tokens}_{tile_n}_"
        f"{tile_k}_{threads}_{stages}",
    )


class MatmulBf16Kernel(
    MatmulKernel[MatmulBf16Arguments, MatmulBf16Workload, MatmulConfig]
):
    """Bind the BF16 MMA program to its arguments and reference semantics."""

    name = "matmul_bf16"
    program = matmul_bf16
    mma_type = MmaType.BF16BF16F32

    @classmethod
    def make_arguments(cls, workload: MatmulBf16Workload) -> MatmulBf16Arguments:
        """Describe the runtime tensors required by one logical workload."""
        output_shape = (workload.num_tokens, workload.n)
        return MatmulBf16Arguments(
            input=TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.k)),
            weight=TensorDesc.empty(DType.BF16, (workload.n, workload.k)),
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
            output_scale=workload.output_scale,
            use_silu=workload.use_silu,
        )

    @classmethod
    def make_workload(cls, arguments: MatmulBf16Arguments) -> MatmulBf16Workload:
        """Recover the specialization and validate every tensor contract."""
        input, weight = arguments.input, arguments.weight
        assert len(input.shape) == len(weight.shape) == 2
        num_tokens, k = input.shape
        n, weight_k = weight.shape
        assert weight_k == k
        assert input.dtype == weight.dtype == DType.BF16
        assert arguments.output.dtype == DType.BF16
        assert arguments.output.shape == (num_tokens, n)
        if arguments.bias is not None:
            assert arguments.bias.dtype == DType.BF16
            assert arguments.bias.shape == (n,)
        for operand in (arguments.residual, arguments.multiply):
            if operand is not None:
                assert operand.dtype == DType.BF16
                assert operand.shape == (num_tokens, n)
        return MatmulBf16Workload(
            num_tokens=num_tokens,
            n=n,
            k=k,
            use_bias=arguments.bias is not None,
            use_residual=arguments.residual is not None,
            use_multiply=arguments.multiply is not None,
            use_silu=arguments.use_silu,
            output_scale=arguments.output_scale,
        )

    @classmethod
    def ref_program(cls, arguments: MatmulBf16Arguments) -> None:
        """Evaluate the kernel semantics into the addressed output tensor."""
        import torch

        input_tensor = arguments.input.as_torch()
        weight = arguments.weight.as_torch()
        bias = arguments.bias.as_torch() if arguments.bias is not None else None
        residual = (
            arguments.residual.as_torch() if arguments.residual is not None else None
        )
        multiply = (
            arguments.multiply.as_torch() if arguments.multiply is not None else None
        )
        output = arguments.output.as_torch()
        value = input_tensor.float() @ weight.float().T
        value *= arguments.output_scale
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
    def make_config(cls, workload: MatmulBf16Workload) -> MatmulConfig:
        """Choose the measured tile family from the output width and row count."""
        tile_tokens, tile_n, threads = 128, 128, 256
        if workload.num_tokens <= 106:
            if workload.n <= 2752:
                tile_tokens, tile_n, threads = 32, 64, 128
            else:
                tile_tokens, threads = 64, 128
        elif workload.n <= 256:
            if workload.num_tokens <= 588:
                tile_tokens, tile_n, threads = 32, 64, 128
            elif workload.num_tokens <= 2342:
                tile_tokens, threads = 64, 128
        elif (
            workload.n == 384
            and workload.k == 5376
            and (workload.num_tokens <= 537 or 1935 <= workload.num_tokens < 4000)
        ):
            tile_tokens, threads = 64, 128
        return MatmulConfig(
            tile_tokens=tile_tokens,
            tile_n=tile_n,
            tile_k=32,
            threads=threads,
            stages=2,
        )
