"""FP8 tensor-core matrix multiplication with an optional fused epilogue."""

import dataclasses
from typing import Literal

import tilelang

from nano_omni.core.kernel import Arguments, MmaType
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.base import MatmulConfig, MatmulKernel, MatmulWorkload


class MatmulFp8Workload(MatmulWorkload):
    use_bias: bool
    use_residual: bool
    use_multiply: bool
    use_silu: bool
    dynamic_input_scale: bool


class MatmulFp8Config(MatmulConfig):
    warp_policy: Literal["full_row", "full_col", "square"] = "full_row"
    warp_specialization: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class MatmulFp8Arguments(Arguments):
    """E4M3 operands, fused epilogue tensors, and F32 scalar tensors."""

    input: TensorDesc
    weight: TensorDesc
    bias: TensorDesc | None
    residual: TensorDesc | None
    multiply: TensorDesc | None
    output: TensorDesc
    weight_scale: TensorDesc
    dynamic_input_scale: TensorDesc | None
    static_input_scale: TensorDesc
    use_silu: bool

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def matmul_fp8(
    num_tokens,
    n,
    k,
    use_bias,
    use_residual,
    use_multiply,
    use_silu,
    dynamic_input_scale,
    tile_tokens=32,
    tile_n=128,
    tile_k=64,
    threads=256,
    stages=2,
    warp_policy="full_row",
    warp_specialization=False,
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
        input: T.Tensor([dynamic_num_tokens, k], T.float8_e4m3fn),
        weight: T.Tensor([n, k], T.float8_e4m3fn),
        bias: T.Tensor([n], T.bfloat16),
        residual: T.Tensor([dynamic_num_tokens, n], T.bfloat16),
        multiply: T.Tensor([dynamic_num_tokens, n], T.bfloat16),
        output: T.Tensor([dynamic_num_tokens, n], T.bfloat16),
        weight_scale: T.Tensor([1], T.float32),
        input_scale: T.Tensor([1], T.float32),
        static_input_scale: T.Tensor([1], T.float32),
    ):
        T.annotate_pass_configs(
            {"tl.disable_warp_specialized": not warp_specialization}
        )
        with T.Kernel(
            T.ceildiv(n, tile_n),
            T.ceildiv(dynamic_num_tokens, tile_tokens),
            threads=threads,
        ) as (column, row):
            input_shared = T.alloc_shared([tile_tokens, tile_k], T.float8_e4m3fn)
            weight_shared = T.alloc_shared([tile_n, tile_k], T.float8_e4m3fn)
            accumulator = T.alloc_fragment([tile_tokens, tile_n], T.float32)
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
            factor = weight_scale[0] * (
                input_scale[0] if dynamic_input_scale else static_input_scale[0]
            )
            for i, j in T.Parallel(tile_tokens, tile_n):
                accumulator[i, j] *= factor
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
        "matmul_fp8_"
        + "_".join(
            str(value)
            for value in (
                n,
                k,
                use_bias,
                use_residual,
                use_multiply,
                use_silu,
                dynamic_input_scale,
                tile_tokens,
                tile_n,
                tile_k,
                threads,
                stages,
                warp_policy,
                warp_specialization,
            )
        ),
    )


class MatmulFp8Kernel(
    MatmulKernel[MatmulFp8Arguments, MatmulFp8Workload, MatmulFp8Config]
):
    name = "matmul_fp8"
    program = matmul_fp8
    mma_type = MmaType.F8F8F32

    @classmethod
    def make_arguments(cls, workload: MatmulFp8Workload) -> MatmulFp8Arguments:
        output_shape = (workload.num_tokens, workload.n)
        scalar = TensorDesc.empty(DType.F32, (1,))
        return MatmulFp8Arguments(
            input=TensorDesc.empty(DType.FP8_E4M3, (workload.num_tokens, workload.k)),
            weight=TensorDesc.empty(DType.FP8_E4M3, (workload.n, workload.k)),
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
            weight_scale=scalar,
            dynamic_input_scale=(
                TensorDesc.empty(DType.F32, (1,))
                if workload.dynamic_input_scale
                else None
            ),
            static_input_scale=TensorDesc.empty(DType.F32, (1,)),
            use_silu=workload.use_silu,
        )

    @classmethod
    def make_workload(cls, arguments: MatmulFp8Arguments) -> MatmulFp8Workload:
        num_tokens, k = arguments.input.shape
        n, weight_k = arguments.weight.shape
        assert weight_k == k
        assert arguments.input.dtype == arguments.weight.dtype == DType.FP8_E4M3
        assert arguments.output.dtype == DType.BF16
        assert arguments.output.shape == (num_tokens, n)
        for scale in (
            arguments.weight_scale,
            arguments.static_input_scale,
            arguments.dynamic_input_scale,
        ):
            if scale is not None:
                assert scale.dtype == DType.F32 and scale.shape == (1,)
        return MatmulFp8Workload(
            num_tokens=num_tokens,
            n=n,
            k=k,
            use_bias=arguments.bias is not None,
            use_residual=arguments.residual is not None,
            use_multiply=arguments.multiply is not None,
            use_silu=arguments.use_silu,
            dynamic_input_scale=arguments.dynamic_input_scale is not None,
        )

    @classmethod
    def make_config(cls, workload: MatmulFp8Workload) -> MatmulFp8Config:
        del workload
        assert False, "matmul_fp8 has no production configuration"

    @classmethod
    def ref_program(cls, arguments: MatmulFp8Arguments) -> None:
        value = (
            arguments.input.as_torch().float() @ arguments.weight.as_torch().float().T
        )
        scale = arguments.weight_scale.as_torch()
        scale = scale * (
            arguments.dynamic_input_scale.as_torch()
            if arguments.dynamic_input_scale is not None
            else arguments.static_input_scale.as_torch()
        )
        value *= scale
        if arguments.bias is not None:
            value += arguments.bias.as_torch().float()
        if arguments.residual is not None:
            value += arguments.residual.as_torch().float()
        if arguments.use_silu:
            value *= value.sigmoid()
        if arguments.multiply is not None:
            value *= arguments.multiply.as_torch().float()
        arguments.output.as_torch().copy_(value)
