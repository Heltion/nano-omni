import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, MmaType
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.base import MatmulConfig, MatmulKernel, MatmulWorkload


class MatmulFp4Workload(MatmulWorkload):
    k: int = Field(gt=0, multiple_of=64)
    use_bias: bool
    use_residual: bool
    use_multiply: bool
    use_silu: bool
    use_input_scale: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class MatmulFp4Arguments(Arguments):
    """Packed E2M1 A[num_tokens,k/2] and B[n,k/2] bytes; k counts logical values.

    Block scales pack four UE4M3 values per uint32 in canonical 128-row groups.
    Global weight/input scales are F32 [1]; absent input_scale means one.
    Bias[n], residual/multiply/output[num_tokens,n] use BF16.
    """

    input: TensorDesc
    matrix: TensorDesc
    input_scales: TensorDesc
    matrix_scales: TensorDesc
    bias: TensorDesc | None
    residual: TensorDesc | None
    multiply: TensorDesc | None
    output: TensorDesc
    weight_scale: TensorDesc
    input_scale: TensorDesc | None
    use_silu: bool

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def matmul_fp4(
    num_tokens,
    n,
    k,
    use_bias,
    use_residual,
    use_multiply,
    use_silu,
    tile_tokens=128,
    tile_n=128,
    tile_k=64,
    threads=256,
    stages=2,
    use_input_scale=True,
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    rows, columns, inner = dynamic_num_tokens, n, k
    block_m, block_n, block_k = tile_tokens, tile_n, tile_k
    has_bias, has_residual = use_bias, use_residual
    scale_words = block_k // 64
    inner_blocks = inner // block_k
    padded_columns = -(-columns // 128) * 128
    padded_rows = -(-rows // 128) * 128

    # The automatic warp-specialized pipeline does not protect these scale
    # buffers from reuse. Keep operand and scale loading in one thread group.
    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        packed_input: T.Tensor([rows, inner // 2], T.uint8),
        packed_matrix: T.Tensor([columns, inner // 2], T.uint8),
        input_scales: T.Tensor([padded_rows * inner_blocks, scale_words], T.uint32),
        matrix_scales: T.Tensor([padded_columns * inner_blocks, scale_words], T.uint32),
        bias: T.Tensor([columns], T.bfloat16),
        residual: T.Tensor([rows, columns], T.bfloat16),
        multiply: T.Tensor([rows, columns], T.bfloat16),
        output: T.Tensor([rows, columns], T.bfloat16),
        weight_scale: T.Tensor([1], T.float32),
        input_scale: T.Tensor([1], T.float32),
    ):
        T.annotate_pass_configs({"tl.disable_warp_specialized": True})
        input = T.view(packed_input, shape=[rows, inner], dtype=T.float4_e2m1fn)
        matrix = T.view(packed_matrix, shape=[columns, inner], dtype=T.float4_e2m1fn)
        with T.Kernel(
            T.ceildiv(columns, block_n),
            T.ceildiv(rows, block_m),
            threads=threads,
        ) as (column, row):
            input_shared = T.alloc_shared([block_m, block_k], T.float4_e2m1fn)
            matrix_shared = T.alloc_shared([block_n, block_k], T.float4_e2m1fn)
            input_scale_shared = T.alloc_shared([block_m, scale_words], T.uint32)
            matrix_scale_shared = T.alloc_shared([block_n, scale_words], T.uint32)
            accumulator = T.alloc_fragment([block_m, block_n], T.float32)
            scale = T.alloc_var(T.float32)
            T.clear(accumulator)
            for block in T.Pipelined(inner_blocks, num_stages=stages):
                T.copy(input[row * block_m, block * block_k], input_shared)
                T.copy(matrix[column * block_n, block * block_k], matrix_shared)
                for i, word in T.Parallel(block_m, scale_words):
                    if block_m == 128:
                        input_scale_shared[i, word] = input_scales[
                            (row * inner_blocks + block) * block_m + i, word
                        ]
                    else:
                        # Global scales use canonical 128-row groups; the
                        # MMA shared-memory package uses this tile's rows.
                        local_word = i * scale_words + word
                        packed_row = local_word % block_m
                        logical_row = (
                            row * block_m
                            + packed_row // (block_m // 32)
                            + (packed_row % (block_m // 32)) * 32
                        )
                        group_k = block * scale_words + local_word // block_m
                        global_word = (
                            ((logical_row // 128) * (inner // 64) + group_k) * 128
                            + (logical_row % 32) * 4
                            + (logical_row % 128) // 32
                        )
                        input_scale_shared[i, word] = input_scales[
                            global_word // scale_words, global_word % scale_words
                        ]
                for i, word in T.Parallel(block_n, scale_words):
                    if block_n == 128:
                        matrix_scale_shared[i, word] = matrix_scales[
                            (column * inner_blocks + block) * block_n + i, word
                        ]
                    else:
                        # Global scales use canonical 128-row groups; the
                        # MMA shared-memory package uses this tile's rows.
                        local_word = i * scale_words + word
                        packed_row = local_word % block_n
                        logical_row = (
                            column * block_n
                            + packed_row // (block_n // 32)
                            + (packed_row % (block_n // 32)) * 32
                        )
                        group_k = block * scale_words + local_word // block_n
                        global_word = (
                            ((logical_row // 128) * (inner // 64) + group_k) * 128
                            + (logical_row % 32) * 4
                            + (logical_row % 128) // 32
                        )
                        matrix_scale_shared[i, word] = matrix_scales[
                            global_word // scale_words, global_word % scale_words
                        ]
                T.mma_gemm_blockscaled(
                    input_shared,
                    matrix_shared,
                    accumulator,
                    input_scale_shared,
                    matrix_scale_shared,
                    transpose_B=True,
                    clear_accum=False,
                    k_start=block * block_k,
                    sf_a_granularity_k=16,
                    sf_b_granularity_k=16,
                    sf_layout="blockscaled_chunk_kmajor",
                )
            scale = weight_scale[0] * (input_scale[0] if use_input_scale else 1.0)
            for i, j in T.Parallel(block_m, block_n):
                accumulator[i, j] *= scale
                if has_bias:
                    accumulator[i, j] += bias[column * block_n + j]
                if has_residual:
                    accumulator[i, j] += residual[
                        row * block_m + i, column * block_n + j
                    ]
                if use_silu:
                    accumulator[i, j] /= 1.0 + T.exp(-accumulator[i, j])
                if use_multiply:
                    accumulator[i, j] *= multiply[
                        row * block_m + i, column * block_n + j
                    ]
            T.copy(accumulator, output[row * block_m, column * block_n])

    return main.with_attr(
        "global_symbol",
        "matmul_fp4_"
        f"{columns}_{inner}_{int(use_bias)}_{int(use_residual)}_"
        f"{int(use_multiply)}_{int(use_silu)}_{int(use_input_scale)}_"
        f"{block_m}_{block_n}_{block_k}_{threads}_{stages}",
    )


class MatmulFp4Kernel(
    MatmulKernel[MatmulFp4Arguments, MatmulFp4Workload, MatmulConfig]
):
    name = "matmul_fp4"
    program = matmul_fp4
    mma_type = MmaType.F4F4F32

    @classmethod
    def make_arguments(cls, workload: MatmulFp4Workload) -> MatmulFp4Arguments:
        """Describe packed FP4 operands, blocked scales, and the fused epilogue."""
        output_shape = (workload.num_tokens, workload.n)

        def operand(rows: int) -> tuple[TensorDesc, TensorDesc]:
            padded_rows = -(-rows // 128) * 128
            return (
                TensorDesc.empty(DType.FP4, (rows, workload.k // 2)),
                TensorDesc.empty(
                    DType.FP8_UE4M3,
                    (padded_rows, workload.k // 16),
                ),
            )

        input, input_scales = operand(workload.num_tokens)
        matrix, matrix_scales = operand(workload.n)
        return MatmulFp4Arguments(
            input=input,
            matrix=matrix,
            input_scales=input_scales,
            matrix_scales=matrix_scales,
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
            weight_scale=TensorDesc.empty(DType.F32, ()),
            input_scale=(
                TensorDesc.empty(DType.F32, ()) if workload.use_input_scale else None
            ),
            use_silu=workload.use_silu,
        )

    @classmethod
    def make_workload(cls, arguments: MatmulFp4Arguments) -> MatmulFp4Workload:
        """Recover the specialization and validate packed operands and scales."""
        input, weight = arguments.input, arguments.matrix
        assert len(input.shape) == len(weight.shape) == 2
        num_tokens, packed_k = input.shape
        n, weight_packed_k = weight.shape
        assert weight_packed_k == packed_k
        k = packed_k * 2
        assert not k % 64
        assert input.dtype == weight.dtype == DType.FP4
        assert arguments.input_scales.dtype == DType.FP8_UE4M3
        assert arguments.matrix_scales.dtype == DType.FP8_UE4M3
        scale_columns = k // 16
        assert arguments.input_scales.shape == (
            -(-num_tokens // 128) * 128,
            scale_columns,
        )
        assert arguments.matrix_scales.shape == (-(-n // 128) * 128, scale_columns)
        assert arguments.weight_scale.dtype == DType.F32
        assert arguments.weight_scale.shape == ()
        if arguments.input_scale is not None:
            assert arguments.input_scale.dtype == DType.F32
            assert arguments.input_scale.shape == ()
        assert arguments.output.dtype == DType.BF16
        assert arguments.output.shape == (num_tokens, n)
        if arguments.bias is not None:
            assert arguments.bias.dtype == DType.BF16
            assert arguments.bias.shape == (n,)
        for operand in (arguments.residual, arguments.multiply):
            if operand is not None:
                assert operand.dtype == DType.BF16
                assert operand.shape == (num_tokens, n)
        return MatmulFp4Workload(
            num_tokens=num_tokens,
            n=n,
            k=k,
            use_bias=arguments.bias is not None,
            use_residual=arguments.residual is not None,
            use_multiply=arguments.multiply is not None,
            use_silu=arguments.use_silu,
            use_input_scale=arguments.input_scale is not None,
        )

    @classmethod
    def ref_program(cls, arguments: MatmulFp4Arguments) -> None:
        """Decode packed E2M1 operands and evaluate the fused FP32 epilogue."""
        import torch

        workload = cls.make_workload(arguments)

        def decode(packed: TensorDesc, storage: TensorDesc) -> torch.Tensor:
            values = packed.as_torch()
            rows, packed_columns = values.shape
            columns = packed_columns * 2
            padded_rows = -(-rows // 128) * 128
            scales = (
                storage.as_torch()
                .view(torch.uint8)
                .view(torch.float8_e4m3fn)
                .float()
                .reshape(padded_rows // 128, columns // 64, 32, 4, 4)
                .permute(0, 3, 2, 1, 4)
                .reshape(padded_rows, columns // 16)[:rows]
            )
            codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(1).long()
            magnitudes = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device
            )
            decoded = magnitudes[codes & 7] * torch.where(codes < 8, 1, -1)
            return decoded * scales.repeat_interleave(16, dim=-1)

        input = decode(arguments.input, arguments.input_scales)
        weight = decode(arguments.matrix, arguments.matrix_scales)
        output = arguments.output.as_torch()
        factor = arguments.weight_scale.as_torch()
        if arguments.input_scale is not None:
            factor = factor * arguments.input_scale.as_torch()
        bias = arguments.bias.as_torch() if arguments.bias is not None else None
        residual = (
            arguments.residual.as_torch() if arguments.residual is not None else None
        )
        multiply = (
            arguments.multiply.as_torch() if arguments.multiply is not None else None
        )
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            result = input @ weight.T * factor
            if bias is not None:
                result += bias.float()
            if residual is not None:
                result += residual.float()
            if workload.use_silu:
                result = torch.nn.functional.silu(result)
            if multiply is not None:
                result *= multiply.float()
            output.copy_(result)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    @classmethod
    def make_config(cls, workload: MatmulFp4Workload) -> MatmulConfig:
        """Select the measured production configuration for the workload."""
        if workload.k == 8192 or workload.k % 128 != 0:
            assert workload.k % 64 == 0, (
                "FP4 reduction dimension must be divisible by 64"
            )
            return MatmulConfig(
                tile_tokens=64, tile_n=128, tile_k=64, threads=128, stages=2
            )
        stages = (
            2
            if (
                workload.num_tokens == 7040
                and workload.k == 5120
                and (workload.n == 8192 or workload.use_multiply)
            )
            else 3
        )
        return MatmulConfig(
            tile_tokens=128, tile_n=128, tile_k=128, threads=256, stages=stages
        )
