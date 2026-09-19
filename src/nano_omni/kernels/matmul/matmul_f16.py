"""FP16 tensor-core matrix multiplication with optional fused epilogue."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, MmaType
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.base import MatmulConfig, MatmulKernel, MatmulWorkload


class MatmulF16Workload(MatmulWorkload):
    """Logical matrix dimensions, input format and enabled epilogue operations."""

    use_bias: bool
    input_dtype: DType = DType.F32
    output_dtype: DType = DType.BF16
    use_column_scale: bool = False
    use_residual: bool = False
    use_multiply: bool = False
    use_silu: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class MatmulF16Arguments(Arguments):
    """Matrices and optional epilogue operands for one invocation."""

    input: TensorDesc
    matrix: TensorDesc
    bias: TensorDesc | None
    column_scale: TensorDesc | None
    residual: TensorDesc | None
    multiply: TensorDesc | None
    output: TensorDesc
    use_silu: bool

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one program for different matrix row counts."""
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def matmul_f16(
    num_tokens,
    n,
    k,
    use_bias,
    input_dtype,
    output_dtype,
    use_column_scale,
    use_residual,
    use_multiply,
    use_silu,
    tile_tokens=32,
    tile_n=64,
    tile_k=32,
    threads=128,
    stages=2,
):
    """Build A[num_tokens,k] @ B[n,k].T and its fused epilogue."""
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    source_dtype = (
        T.float32
        if input_dtype == DType.F32
        else T.float16
        if input_dtype == DType.F16
        else T.bfloat16
    )
    destination_dtype = T.float16 if output_dtype == DType.F16 else T.bfloat16
    warp_specialized = (n, k, use_multiply, use_silu) == (8192, 2048, True, False) or (
        n,
        k,
        use_multiply,
        use_silu,
    ) == (6144, 2048, False, False)

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor([dynamic_num_tokens, k], source_dtype),
        matrix: T.Tensor([n, k], T.float16),
        bias: T.Tensor([n], T.float16),
        column_scale: T.Tensor([n], T.float16),
        residual: T.Tensor([dynamic_num_tokens, n], destination_dtype),
        multiply: T.Tensor([dynamic_num_tokens, n], destination_dtype),
        output: T.Tensor([dynamic_num_tokens, n], destination_dtype),
    ):
        T.annotate_pass_configs({"tl.disable_warp_specialized": not warp_specialized})
        with T.Kernel(
            T.ceildiv(n, tile_n),
            T.ceildiv(dynamic_num_tokens, tile_tokens),
            threads=threads,
        ) as (column, row):
            input_shared = T.alloc_shared([tile_tokens, tile_k], T.float16)
            matrix_shared = T.alloc_shared([tile_n, tile_k], T.float16)
            accumulator = T.alloc_fragment([tile_tokens, tile_n], T.float32)
            T.clear(accumulator)
            for inner in T.Pipelined(T.ceildiv(k, tile_k), num_stages=stages):
                T.copy(
                    input[
                        row * tile_tokens : (row + 1) * tile_tokens,
                        inner * tile_k : (inner + 1) * tile_k,
                    ],
                    input_shared,
                )
                T.copy(
                    matrix[
                        column * tile_n : (column + 1) * tile_n,
                        inner * tile_k : (inner + 1) * tile_k,
                    ],
                    matrix_shared,
                )
                T.gemm(input_shared, matrix_shared, accumulator, transpose_B=True)
            for i, j in T.Parallel(tile_tokens, tile_n):
                logical_row = row * tile_tokens + i
                logical_column = column * tile_n + j
                if logical_row < dynamic_num_tokens and logical_column < n:
                    if use_bias:
                        accumulator[i, j] += bias[logical_column]
                    if use_column_scale:
                        accumulator[i, j] *= column_scale[logical_column]
                    if use_residual:
                        accumulator[i, j] += residual[logical_row, logical_column]
                    if use_silu:
                        accumulator[i, j] /= 1.0 + T.exp(-accumulator[i, j])
                    if use_multiply:
                        accumulator[i, j] *= multiply[logical_row, logical_column]
                    if input_dtype == DType.F32:
                        output[logical_row, logical_column] = T.cast(
                            accumulator[i, j], T.float16
                        )
                    else:
                        output[logical_row, logical_column] = accumulator[i, j]

    suffix = "_".join(
        str(value)
        for value in (
            n,
            k,
            use_bias,
            input_dtype,
            output_dtype,
            use_column_scale,
            use_residual,
            use_multiply,
            use_silu,
        )
    )
    return main.with_attr(
        "global_symbol",
        f"matmul_f16_{suffix}_{tile_tokens}_{tile_n}_{tile_k}_{threads}_{stages}"
        f"{'_ws' if warp_specialized else ''}",
    )


class MatmulF16Kernel(
    MatmulKernel[MatmulF16Arguments, MatmulF16Workload, MatmulConfig]
):
    """Bind the FP16 MMA program to its logical arguments and reference."""

    name = "matmul_f16"
    program = matmul_f16
    mma_type = MmaType.F16F16F32

    @classmethod
    def make_arguments(cls, workload: MatmulF16Workload) -> MatmulF16Arguments:
        """Describe all tensors in TileLang ABI order."""
        return MatmulF16Arguments(
            input=TensorDesc.empty(
                workload.input_dtype, (workload.num_tokens, workload.k)
            ),
            matrix=TensorDesc.empty(DType.F16, (workload.n, workload.k)),
            bias=(
                TensorDesc.empty(DType.F16, (workload.n,))
                if workload.use_bias
                else None
            ),
            column_scale=(
                TensorDesc.empty(DType.F16, (workload.n,))
                if workload.use_column_scale
                else None
            ),
            residual=(
                TensorDesc.empty(
                    workload.output_dtype, (workload.num_tokens, workload.n)
                )
                if workload.use_residual
                else None
            ),
            multiply=(
                TensorDesc.empty(
                    workload.output_dtype, (workload.num_tokens, workload.n)
                )
                if workload.use_multiply
                else None
            ),
            output=TensorDesc.empty(
                workload.output_dtype, (workload.num_tokens, workload.n)
            ),
            use_silu=workload.use_silu,
        )

    @classmethod
    def make_workload(cls, arguments: MatmulF16Arguments) -> MatmulF16Workload:
        """Recover the specialization and validate every tensor contract."""
        assert len(arguments.input.shape) == len(arguments.matrix.shape) == 2
        num_tokens, k = arguments.input.shape
        n, matrix_k = arguments.matrix.shape
        assert matrix_k == k
        assert arguments.input.dtype in (DType.F32, DType.F16, DType.BF16)
        assert arguments.matrix.dtype == DType.F16
        assert arguments.output.dtype in (DType.F16, DType.BF16)
        assert arguments.output.shape == (num_tokens, n)
        if arguments.bias is not None:
            assert arguments.bias.dtype == DType.F16
            assert arguments.bias.shape == (n,)
        if arguments.column_scale is not None:
            assert arguments.column_scale.dtype == DType.F16
            assert arguments.column_scale.shape == (n,)
        for operand in (arguments.residual, arguments.multiply):
            if operand is not None:
                assert operand.dtype == arguments.output.dtype
                assert operand.shape == (num_tokens, n)
        return MatmulF16Workload(
            num_tokens=num_tokens,
            n=n,
            k=k,
            use_bias=arguments.bias is not None,
            input_dtype=arguments.input.dtype,
            output_dtype=arguments.output.dtype,
            use_column_scale=arguments.column_scale is not None,
            use_residual=arguments.residual is not None,
            use_multiply=arguments.multiply is not None,
            use_silu=arguments.use_silu,
        )

    @classmethod
    def make_config(cls, workload: MatmulF16Workload) -> MatmulConfig:
        """Select the measured configuration for this implementation."""
        if (
            workload.input_dtype == DType.F16
            and workload.output_dtype == DType.F16
            and workload.use_bias
            and workload.use_column_scale
            and workload.use_residual
            and not workload.use_multiply
            and not workload.use_silu
            and workload.n == 2048
            and workload.k in (2048, 8192)
        ):
            return MatmulConfig(
                tile_tokens=64,
                tile_n=128,
                tile_k=64,
                threads=128,
                stages=2,
            )
        if (
            workload.input_dtype == DType.F16
            and workload.output_dtype == DType.F16
            and workload.use_bias
            and not workload.use_column_scale
            and not workload.use_residual
            and not workload.use_multiply
            and not workload.use_silu
            and (workload.n, workload.k) == (6144, 2048)
        ):
            return MatmulConfig(
                tile_tokens=128,
                tile_n=256,
                tile_k=64,
                threads=512,
                stages=2,
            )
        if (
            workload.input_dtype == DType.F16
            and workload.output_dtype == DType.F16
            and workload.use_bias
            and not workload.use_column_scale
            and not workload.use_residual
            and not workload.use_multiply
            and (workload.n, workload.k) == (8192, 2048)
        ):
            return MatmulConfig(
                tile_tokens=128,
                tile_n=256,
                tile_k=64,
                threads=512,
                stages=2,
            )
        return MatmulConfig(
            tile_tokens=128,
            tile_n=128,
            tile_k=64 if workload.k in (2048, 8192) else 32,
            threads=256,
            stages=2,
        )

    @classmethod
    def ref_program(cls, arguments: MatmulF16Arguments) -> None:
        """Evaluate the matrix product and fused epilogue directly with Torch."""
        import torch

        value = arguments.input.as_torch().half().float() @ (
            arguments.matrix.as_torch().float().T
        )
        if arguments.bias is not None:
            value += arguments.bias.as_torch().float()
        if arguments.column_scale is not None:
            value *= arguments.column_scale.as_torch().float()
        if arguments.residual is not None:
            value += arguments.residual.as_torch().float()
        if arguments.use_silu:
            value = torch.nn.functional.silu(value)
        if arguments.multiply is not None:
            value *= arguments.multiply.as_torch().float()
        if arguments.input.dtype == DType.F32:
            value = value.half()
        arguments.output.as_torch().copy_(value)
