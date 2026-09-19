"""TF32 tensor-core matrix multiplication with optional bias."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, MmaType
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.base import MatmulConfig, MatmulKernel, MatmulWorkload


class MatmulF32Workload(MatmulWorkload):
    """Matrix dimensions, bias presence, and output storage dtype."""

    use_bias: bool
    output_bf16: bool


@dataclasses.dataclass(frozen=True, slots=True)
class MatmulF32Arguments(Arguments):
    """F32 A[num_tokens,k], matrix[n,k] and optional bias[n]; TF32 MMA, FP32 sums.

    Output[num_tokens,n] is F32 unless output_bf16 selects BF16 storage."""

    input: TensorDesc
    matrix: TensorDesc
    bias: TensorDesc | None
    output: TensorDesc
    output_bf16: bool

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one program for different matrix row counts."""
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def matmul_f32(
    num_tokens,
    n,
    k,
    use_bias,
    output_bf16,
    tile_tokens=32,
    tile_n=64,
    tile_k=32,
    threads=128,
    stages=2,
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    output_dtype = T.bfloat16 if output_bf16 else T.float32

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor([dynamic_num_tokens, k], T.float32),
        matrix: T.Tensor([n, k], T.float32),
        bias: T.Tensor([n], T.float32),
        output: T.Tensor([dynamic_num_tokens, n], output_dtype),
    ):
        with T.Kernel(
            T.ceildiv(n, tile_n),
            T.ceildiv(dynamic_num_tokens, tile_tokens),
            threads=threads,
        ) as (column, row):
            input_shared = T.alloc_shared([tile_tokens, tile_k], T.float32)
            matrix_shared = T.alloc_shared([tile_n, tile_k], T.float32)
            accumulator = T.alloc_fragment([tile_tokens, tile_n], T.float32)
            T.fill(accumulator, 0)
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
                T.gemm(
                    input_shared,
                    matrix_shared,
                    accumulator,
                    transpose_B=True,
                )
            for i, j in T.Parallel(tile_tokens, tile_n):
                logical_column = column * tile_n + j
                if use_bias and logical_column < n:
                    accumulator[i, j] += bias[logical_column]
            T.copy(
                accumulator,
                output[
                    row * tile_tokens : (row + 1) * tile_tokens,
                    column * tile_n : (column + 1) * tile_n,
                ],
            )

    return main.with_attr(
        "global_symbol",
        "matmul_f32_"
        f"{n}_{k}_{int(use_bias)}_{int(output_bf16)}_"
        f"{tile_tokens}_{tile_n}_{tile_k}_"
        f"{threads}_{stages}",
    )


class MatmulF32Kernel(
    MatmulKernel[MatmulF32Arguments, MatmulF32Workload, MatmulConfig]
):
    """Bind the TF32 MMA program to its arguments and reference semantics."""

    name = "matmul_f32"
    program = matmul_f32
    mma_type = MmaType.TF32TF32F32

    @classmethod
    def make_arguments(cls, workload: MatmulF32Workload) -> MatmulF32Arguments:
        """Describe the runtime tensors required by one logical workload."""
        return MatmulF32Arguments(
            input=TensorDesc.empty(DType.F32, (workload.num_tokens, workload.k)),
            matrix=TensorDesc.empty(DType.F32, (workload.n, workload.k)),
            bias=(
                TensorDesc.empty(DType.F32, (workload.n,))
                if workload.use_bias
                else None
            ),
            output=TensorDesc.empty(
                DType.BF16 if workload.output_bf16 else DType.F32,
                (workload.num_tokens, workload.n),
            ),
            output_bf16=workload.output_bf16,
        )

    @classmethod
    def make_workload(cls, arguments: MatmulF32Arguments) -> MatmulF32Workload:
        """Recover the specialization and validate every tensor contract."""
        assert len(arguments.input.shape) == len(arguments.matrix.shape) == 2
        num_tokens, k = arguments.input.shape
        n, matrix_k = arguments.matrix.shape
        assert matrix_k == k
        assert arguments.input.dtype == arguments.matrix.dtype == DType.F32
        output_dtype = DType.BF16 if arguments.output_bf16 else DType.F32
        assert arguments.output.dtype == output_dtype
        assert arguments.output.shape == (num_tokens, n)
        if arguments.bias is not None:
            assert arguments.bias.dtype == DType.F32
            assert arguments.bias.shape == (n,)
        return MatmulF32Workload(
            num_tokens=num_tokens,
            n=n,
            k=k,
            use_bias=arguments.bias is not None,
            output_bf16=arguments.output_bf16,
        )

    @classmethod
    def ref_program(cls, arguments: MatmulF32Arguments) -> None:
        """Evaluate the kernel semantics into the addressed output tensor."""
        input_tensor = arguments.input.as_torch()
        matrix = arguments.matrix.as_torch()
        value = input_tensor @ matrix.T
        if arguments.bias is not None:
            value += arguments.bias.as_torch()
        arguments.output.as_torch().copy_(value)

    @classmethod
    def make_config(cls, workload: MatmulF32Workload) -> MatmulConfig:
        """Select a tile that preserves enough blocks for narrow products."""
        narrow = workload.n <= 64 or (
            workload.k >= 1024 and 512 <= workload.num_tokens < 1024
        )
        if narrow:
            return MatmulConfig(
                tile_tokens=64,
                tile_n=128,
                tile_k=32,
                threads=128,
                stages=2,
            )
        return MatmulConfig(
            tile_tokens=128,
            tile_n=128,
            tile_k=32,
            threads=256,
            stages=2,
        )
