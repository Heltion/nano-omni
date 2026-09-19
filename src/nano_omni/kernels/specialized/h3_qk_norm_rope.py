"""Fused H3 Q/K normalization and split-half RoPE."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Kernel
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.specialized.h3_head_norm_rope import (
    H3HeadNormRopeArguments,
    H3HeadNormRopeConfig,
    H3HeadNormRopeKernel,
    H3HeadNormRopeWorkload,
)


@dataclasses.dataclass(frozen=True, slots=True)
class H3QkNormRopeArguments(Arguments):
    """Q/K pair normalized and rotated by one fused launch."""

    query: TensorDesc
    key: TensorDesc
    query_weight: TensorDesc
    key_weight: TensorDesc
    cosines: TensorDesc
    sines: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("tokens", self.query.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def h3_qk_norm_rope(
    tokens: int, heads: int, use_rope: bool, threads: int = 128
):
    """Normalize and rotate Q/K together so both reuse each RoPE table load."""
    import tilelang.language as T

    assert use_rope
    dynamic_tokens = T.dynamic("tokens")
    tile_tokens = 4
    layout = T.Fragment(
        (tile_tokens, 128),
        forward_thread_fn=lambda i, j: (i % 4) * 32 + j % 32,
        forward_index_fn=lambda i, j: (i // 4) * 4 + j // 32,
    )

    @T.prim_func
    def main(
        tokens: dynamic_tokens,
        query: T.Tensor((dynamic_tokens, heads, 128), T.bfloat16),
        key: T.Tensor((dynamic_tokens, heads, 128), T.bfloat16),
        query_weight: T.Tensor((128,), T.bfloat16),
        key_weight: T.Tensor((128,), T.bfloat16),
        cosines: T.Tensor((dynamic_tokens, 48), T.float32),
        sines: T.Tensor((dynamic_tokens, 48), T.float32),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(T.ceildiv(tokens, tile_tokens), heads, threads=threads) as (
            block,
            head,
        ):
            query_value = T.alloc_fragment((tile_tokens, 128), T.float32)
            key_value = T.alloc_fragment((tile_tokens, 128), T.float32)
            query_squared = T.alloc_fragment((tile_tokens, 128), T.float32)
            key_squared = T.alloc_fragment((tile_tokens, 128), T.float32)
            query_sum = T.alloc_fragment((tile_tokens,), T.float32)
            key_sum = T.alloc_fragment((tile_tokens,), T.float32)
            query_norm = T.alloc_shared((tile_tokens, 128), T.float32)
            key_norm = T.alloc_shared((tile_tokens, 128), T.float32)
            query_output = T.alloc_fragment((tile_tokens, 128), T.bfloat16)
            key_output = T.alloc_fragment((tile_tokens, 128), T.bfloat16)
            T.annotate_layout(
                {
                    query_value: layout,
                    key_value: layout,
                    query_squared: layout,
                    key_squared: layout,
                    query_output: layout,
                    key_output: layout,
                }
            )
            begin = block * tile_tokens
            T.copy(query[begin : begin + tile_tokens, head, 0:128], query_value)
            T.copy(key[begin : begin + tile_tokens, head, 0:128], key_value)
            for i, j in T.Parallel(tile_tokens, 128):
                query_squared[i, j] = query_value[i, j] * query_value[i, j]
                key_squared[i, j] = key_value[i, j] * key_value[i, j]
            T.reduce_sum(query_squared, query_sum, dim=1)
            T.reduce_sum(key_squared, key_sum, dim=1)
            for i, j in T.Parallel(tile_tokens, 128):
                query_norm[i, j] = (
                    query_value[i, j]
                    * T.rsqrt(query_sum[i] / 128 + 1e-5)
                    * query_weight[j]
                )
                key_norm[i, j] = (
                    key_value[i, j]
                    * T.rsqrt(key_sum[i] / 128 + 1e-5)
                    * key_weight[j]
                )
            for i, j in T.Parallel(tile_tokens, 128, coalesced_width=1):
                token = begin + i
                if token < tokens:
                    if j < 48:
                        query_output[i, j] = (
                            query_norm[i, j] * cosines[token, j]
                            - query_norm[i, j + 48] * sines[token, j]
                        )
                        key_output[i, j] = (
                            key_norm[i, j] * cosines[token, j]
                            - key_norm[i, j + 48] * sines[token, j]
                        )
                    elif j < 96:
                        query_output[i, j] = (
                            query_norm[i, j - 48] * sines[token, j - 48]
                            + query_norm[i, j] * cosines[token, j - 48]
                        )
                        key_output[i, j] = (
                            key_norm[i, j - 48] * sines[token, j - 48]
                            + key_norm[i, j] * cosines[token, j - 48]
                        )
                    else:
                        query_output[i, j] = query_norm[i, j]
                        key_output[i, j] = key_norm[i, j]
                else:
                    query_output[i, j] = 0
                    key_output[i, j] = 0
            T.copy(query_output, query[begin : begin + tile_tokens, head, 0:128])
            T.copy(key_output, key[begin : begin + tile_tokens, head, 0:128])

    return main.with_attr("global_symbol", f"h3_qk_norm_rope_{heads}_{threads}")


class H3QkNormRopeKernel(
    Kernel[H3QkNormRopeArguments, H3HeadNormRopeWorkload, H3HeadNormRopeConfig]
):
    """Fused in-place Q/K normalization and split-half RoPE."""

    name = "h3_qk_norm_rope"
    program = h3_qk_norm_rope

    @classmethod
    def make_arguments(cls, workload: H3HeadNormRopeWorkload) -> H3QkNormRopeArguments:
        assert workload.use_rope
        shape = (workload.tokens, workload.heads, 128)
        return H3QkNormRopeArguments(
            query=TensorDesc.empty(DType.BF16, shape),
            key=TensorDesc.empty(DType.BF16, shape),
            query_weight=TensorDesc.empty(DType.BF16, (128,)),
            key_weight=TensorDesc.empty(DType.BF16, (128,)),
            cosines=TensorDesc.empty(DType.F32, (workload.tokens, 48)),
            sines=TensorDesc.empty(DType.F32, (workload.tokens, 48)),
        )

    @classmethod
    def make_config(cls, workload: H3HeadNormRopeWorkload) -> H3HeadNormRopeConfig:
        del workload
        return H3HeadNormRopeConfig()

    @classmethod
    def make_workload(cls, arguments: H3QkNormRopeArguments) -> H3HeadNormRopeWorkload:
        assert arguments.query.dtype == arguments.key.dtype == DType.BF16
        assert arguments.query.shape == arguments.key.shape
        assert len(arguments.query.shape) == 3 and arguments.query.shape[2] == 128
        assert arguments.query_weight.dtype == arguments.key_weight.dtype == DType.BF16
        assert arguments.query_weight.shape == arguments.key_weight.shape == (128,)
        assert arguments.cosines.dtype == arguments.sines.dtype == DType.F32
        assert arguments.cosines.shape == arguments.sines.shape == (
            arguments.query.shape[0],
            48,
        )
        return H3HeadNormRopeWorkload(
            tokens=arguments.query.shape[0],
            heads=arguments.query.shape[1],
            use_rope=True,
        )

    @classmethod
    def ref_program(cls, arguments: H3QkNormRopeArguments) -> None:
        for value, weight in (
            (arguments.query, arguments.query_weight),
            (arguments.key, arguments.key_weight),
        ):
            H3HeadNormRopeKernel.ref_program(
                H3HeadNormRopeArguments(
                    value,
                    weight,
                    arguments.cosines,
                    arguments.sines,
                    value,
                )
            )
