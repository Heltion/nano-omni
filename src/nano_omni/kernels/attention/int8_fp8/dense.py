"""Dense d128 attention with INT8 QK and FP8 PV."""

import dataclasses
import math
from typing import Literal

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc


# Uniform FP8 values satisfy the FP16 partial-accumulation range of this kernel.
@tilelang.jit(
    out_idx=[],
    execution_backend="nvrtc",
    pass_configs={"tl.disable_vectorize_256": True},
    compile_flags=["--use_fast_math"],
)
def dense_int8_fp8_attention(
    num_kv_tokens,
    heads,
    num_query_tokens,
    tile_query_tokens=128,
    tile_kv_tokens=64,
    threads=128,
):
    import tilelang.language as T

    if (
        tile_kv_tokens != 64
        or threads not in (64, 128, 256)
        or tile_query_tokens not in (64, 128, 256)
    ):
        raise ValueError(
            "dense INT8/FP8 requires KV tile 64 and supported Q tiles/thread counts"
        )
    if tile_query_tokens // (threads // 32) not in (16, 32):
        raise ValueError("Each warp must own 16 or 32 Q rows within one Q scale group")
    num_query_tokens = num_kv_tokens if num_query_tokens is None else num_query_tokens
    qrows = (num_query_tokens + 31) // 32 * 32
    krows = (num_kv_tokens + 63) // 64 * 64
    warp_rows = tile_query_tokens // (threads // 32)
    warp_m = warp_rows // 16
    scale = 128**-0.5 * 1.4426950408889634
    # |V_fp8| <= 2.25: 64 * 448 * 2.25 = 64512 stays below FP16 overflow.
    # FP16 partials are converted to FP32 after every 64-token PV tile.
    probability_scale = 448 * 64 / tile_kv_tokens
    probability_shift = math.log2(probability_scale)

    @T.prim_func
    def main(
        num_kv_tokens: num_kv_tokens,
        num_query_tokens: num_query_tokens,
        query: T.Tensor((heads, qrows, 128), T.int8),
        key: T.Tensor((heads, krows, 128), T.int8),
        value: T.Tensor((heads, 128, krows), T.float8_e4m3fn),
        qs: T.Tensor((heads, qrows // 32), T.float32),
        ks: T.Tensor((heads, krows // 64), T.float32),
        vs: T.Tensor((heads, 128), T.float32),
        output: T.Tensor((num_query_tokens, heads * 128), T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(num_query_tokens, tile_query_tokens), heads, threads=threads
        ) as (
            block,
            head,
        ):
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            q = T.alloc_local((warp_m, 4), T.uint32)
            q_stage = T.alloc_shared((tile_query_tokens, 128), T.int8)
            q_scale = T.alloc_local((1,), T.float32)
            score_scale = T.alloc_local((1,), T.float32)
            b = T.alloc_local((2,), T.uint32)
            bv = T.alloc_local((tile_kv_tokens // 32, 2), T.uint32)
            # Each lane owns two columns in each of two rows of an m16n8 tile.
            # Alias the finished INT32 score storage as FP32 to bound live registers.
            si = T.alloc_local((warp_m, tile_kv_tokens // 8, 4), T.int32)
            sf = T.view(si, (warp_m, tile_kv_tokens // 8, 4), T.float32)
            pf = T.alloc_local((warp_m, tile_kv_tokens // 8, 4), T.float8_e4m3fn)
            pairs = T.view(pf, (warp_m, tile_kv_tokens // 8, 2), T.uint16)
            p = T.alloc_local((warp_m, tile_kv_tokens // 32, 4), T.uint32)
            partial = T.alloc_local((4,), T.float16)
            result = T.alloc_local((warp_m, 16, 4), T.float32)
            maximum = T.alloc_local((warp_m, 2), T.float32)
            next_max = T.alloc_local((warp_m, 2), T.float32)
            denominator = T.alloc_local((warp_m, 2), T.float32)
            alpha = T.alloc_local((warp_m, 2), T.float32)
            k = T.alloc_shared((tile_kv_tokens, 128), T.int8)
            v = T.alloc_shared((128, tile_kv_tokens), T.float8_e4m3fn)
            T.annotate_layout(
                {
                    q_stage: T.Layout(
                        (tile_query_tokens, 128),
                        lambda i, j: i * 128 + ((j // 16) ^ (i % 8)) * 16 + j % 16,
                    ),
                    k: T.Layout(
                        (tile_kv_tokens, 128),
                        lambda i, j: i * 128 + ((j // 16) ^ (i % 8)) * 16 + j % 16,
                    ),
                    v: T.Layout(
                        (128, tile_kv_tokens),
                        lambda i, j: (
                            i * tile_kv_tokens
                            + ((j // 16) ^ (i % (tile_kv_tokens // 16))) * 16
                            + j % 16
                        ),
                    ),
                }
            )
            T.copy(
                query[
                    head, block * tile_query_tokens : (block + 1) * tile_query_tokens, :
                ],
                q_stage,
            )
            q_scale[0] = qs[
                head,
                T.min(block * tile_query_tokens + warp * warp_rows, qrows - 1) // 32,
            ]
            T.clear(result)
            T.clear(denominator)
            T.fill(maximum, -1e30)
            # Single K/V buffers alternate: V loads overlap QK, next K overlaps PV.
            # Every async_copy commits one group; wait_group(1) completes the older.
            T.async_copy(key[head, 0:tile_kv_tokens, :], k)
            T.async_copy(value[head, :, 0:tile_kv_tokens], v)
            for kb in T.serial(T.ceildiv(num_kv_tokens, tile_kv_tokens)):
                T.ptx_wait_group(1)
                T.sync_threads()
                T.clear(si)
                for inner in T.unroll(4):
                    for m in T.unroll(warp_m):
                        T.ptx_ldmatrix(
                            False,
                            4,
                            T.address_of(
                                q_stage[
                                    warp * warp_rows + m * 16 + lane % 16,
                                    inner * 32 + lane // 16 * 16,
                                ]
                            ),
                            T.access_ptr(q[m, 0], "w", extent=4),
                        )
                    for n in T.unroll(tile_kv_tokens // 8):
                        T.ptx_ldmatrix(
                            False,
                            2,
                            T.address_of(
                                k[n * 8 + lane % 8, inner * 32 + lane % 16 // 8 * 16]
                            ),
                            T.access_ptr(b, "w"),
                        )
                        for m in T.unroll(warp_m):
                            T.ptx_mma(
                                "int32",
                                "m16n8k32",
                                "row",
                                "col",
                                "int8",
                                "int8",
                                "int32",
                                q.data,
                                m * 4,
                                b.data,
                                0,
                                si.data,
                                (m * (tile_kv_tokens // 8) + n) * 4,
                                False,
                            )
                score_scale[0] = (
                    q_scale[0] * ks[head, kb * tile_kv_tokens // 64] * scale
                )
                T.fill(next_max, -1e30)
                for m in T.unroll(warp_m):
                    for n in T.unroll(tile_kv_tokens // 8):
                        for item in T.unroll(4):
                            sf[m, n, item] = (
                                T.cast(si[m, n, item], T.float32) * score_scale[0]
                            )
                if kb == T.ceildiv(num_kv_tokens, tile_kv_tokens) - 1:
                    for m in T.unroll(warp_m):
                        for n in T.unroll(tile_kv_tokens // 8):
                            for item in T.unroll(4):
                                if (
                                    kb * tile_kv_tokens
                                    + n * 8
                                    + lane % 4 * 2
                                    + item % 2
                                    >= num_kv_tokens
                                ):
                                    sf[m, n, item] = -1e30
                for m in T.unroll(warp_m):
                    for n in T.unroll(tile_kv_tokens // 8):
                        for item in T.unroll(4):
                            next_max[m, item // 2] = T.max(
                                next_max[m, item // 2], sf[m, n, item]
                            )
                for m in T.unroll(warp_m):
                    for r in T.unroll(2):
                        next_max[m, r] = T.max(
                            next_max[m, r], T.shfl_xor(next_max[m, r], 1)
                        )
                        next_max[m, r] = T.max(
                            next_max[m, r], T.shfl_xor(next_max[m, r], 2)
                        )
                        next_max[m, r] = T.max(next_max[m, r], maximum[m, r])
                        alpha[m, r] = T.exp2(maximum[m, r] - next_max[m, r])
                        denominator[m, r] *= alpha[m, r]
                        maximum[m, r] = next_max[m, r]
                for m in T.unroll(warp_m):
                    for n in T.unroll(tile_kv_tokens // 8):
                        for item in T.unroll(4):
                            sf[m, n, item] = T.exp2(
                                sf[m, n, item]
                                - (maximum[m, item // 2] - probability_shift)
                            )
                            denominator[m, item // 2] += sf[m, n, item]
                # Quantize the whole fragment, then join adjacent FP8 pairs.
                # V's preparer applies the matching token permutation once.
                T.copy(sf, pf)
                for m in T.unroll(warp_m):
                    for inner in T.unroll(tile_kv_tokens // 32):
                        for item in T.unroll(4):
                            base = inner * 4 + item // 2 * 2
                            packed = T.cast(pairs[m, base, item % 2], T.uint32) | (
                                T.cast(pairs[m, base + 1, item % 2], T.uint32) << 16
                            )
                            p[m, inner, item] = packed
                T.sync_threads()
                if kb + 1 < T.ceildiv(num_kv_tokens, tile_kv_tokens):
                    T.async_copy(
                        key[
                            head,
                            (kb + 1) * tile_kv_tokens : (kb + 2) * tile_kv_tokens,
                            :,
                        ],
                        k,
                    )
                    T.ptx_wait_group(1)
                else:
                    T.ptx_wait_group(0)
                T.sync_threads()
                for n in T.unroll(16):
                    for inner in T.unroll(tile_kv_tokens // 32):
                        T.ptx_ldmatrix(
                            False,
                            2,
                            T.address_of(
                                v[n * 8 + lane % 8, inner * 32 + lane % 16 // 8 * 16]
                            ),
                            T.access_ptr(bv[inner, 0], "w", extent=2),
                        )
                    for m in T.unroll(warp_m):
                        T.clear(partial)
                        for inner in T.unroll(tile_kv_tokens // 32):
                            T.ptx_mma(
                                "float16",
                                "m16n8k32",
                                "row",
                                "col",
                                "e4m3",
                                "e4m3",
                                "fp16",
                                p.data,
                                (m * (tile_kv_tokens // 32) + inner) * 4,
                                bv.data,
                                inner * 2,
                                partial.data,
                                0,
                                False,
                            )
                        for item in T.unroll(4):
                            result[m, n, item] = result[m, n, item] * alpha[
                                m, item // 2
                            ] + T.cast(partial[item], T.float32)
                T.sync_threads()
                if kb + 1 < T.ceildiv(num_kv_tokens, tile_kv_tokens):
                    T.async_copy(
                        value[
                            head,
                            :,
                            (kb + 1) * tile_kv_tokens : (kb + 2) * tile_kv_tokens,
                        ],
                        v,
                    )
            for m in T.unroll(warp_m):
                for r in T.unroll(2):
                    denominator[m, r] += T.shfl_xor(denominator[m, r], 1)
                    denominator[m, r] += T.shfl_xor(denominator[m, r], 2)
            for m in T.unroll(warp_m):
                for n in T.unroll(16):
                    for item in T.unroll(4):
                        row = (
                            block * tile_query_tokens
                            + warp * warp_rows
                            + m * 16
                            + lane // 4
                            + item // 2 * 8
                        )
                        col = n * 8 + lane % 4 * 2 + item % 2
                        if row < num_query_tokens:
                            output[row, head * 128 + col] = (
                                result[m, n, item]
                                * vs[head, col]
                                / denominator[m, item // 2]
                            )

    return main.with_attr(
        "global_symbol",
        f"dense_int8_fp8_{heads}_{tile_query_tokens}_{tile_kv_tokens}_{threads}",
    )


class DenseInt8Fp8AttentionWorkload(Workload):
    num_kv_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    num_query_tokens: int = Field(gt=0)


class DenseInt8Fp8AttentionConfig(Config):
    tile_query_tokens: Literal[64, 128, 256] = 128
    tile_kv_tokens: Literal[64] = 64
    threads: Literal[64, 128, 256] = 128


@dataclasses.dataclass(frozen=True, slots=True)
class DenseInt8Fp8AttentionArguments(Arguments):
    query: TensorDesc
    key: TensorDesc
    value: TensorDesc
    query_scale: TensorDesc
    key_scale: TensorDesc
    value_scale: TensorDesc
    output: TensorDesc
    num_kv_tokens: int
    heads: int
    num_query_tokens: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (
            ("num_kv_tokens", self.num_kv_tokens),
            ("num_query_tokens", self.num_query_tokens),
        )


class DenseInt8Fp8AttentionKernel(
    Kernel[
        DenseInt8Fp8AttentionArguments,
        DenseInt8Fp8AttentionWorkload,
        DenseInt8Fp8AttentionConfig,
    ]
):
    name = "dense_int8_fp8_d128"
    program = dense_int8_fp8_attention

    @classmethod
    def make_arguments(
        cls, workload: DenseInt8Fp8AttentionWorkload
    ) -> DenseInt8Fp8AttentionArguments:
        """Describe padded quantized Q/K/V storage, scales, and BF16 output."""
        query_rows = -(-workload.num_query_tokens // 32) * 32
        key_rows = -(-workload.num_kv_tokens // 64) * 64
        return DenseInt8Fp8AttentionArguments(
            query=TensorDesc.empty(DType.I8, (workload.heads, query_rows, 128)),
            key=TensorDesc.empty(DType.I8, (workload.heads, key_rows, 128)),
            value=TensorDesc.empty(DType.FP8_E4M3, (workload.heads, 128, key_rows)),
            query_scale=TensorDesc.empty(DType.F32, (workload.heads, query_rows // 32)),
            key_scale=TensorDesc.empty(DType.F32, (workload.heads, key_rows // 64)),
            value_scale=TensorDesc.empty(DType.F32, (workload.heads, 128)),
            output=TensorDesc.empty(
                DType.BF16, (workload.num_query_tokens, workload.heads * 128)
            ),
            num_kv_tokens=workload.num_kv_tokens,
            heads=workload.heads,
            num_query_tokens=workload.num_query_tokens,
        )

    @classmethod
    def tops(cls, arguments: DenseInt8Fp8AttentionArguments) -> Tops:
        count = (
            2
            * arguments.heads
            * arguments.num_query_tokens
            * arguments.num_kv_tokens
            * 128
        )
        return {MmaType.I8I8I32: count, MmaType.F8F8F16: count}

    @classmethod
    def ref_program(cls, arguments: DenseInt8Fp8AttentionArguments) -> None:
        """Dequantize Q/K/V and evaluate each independent attention head."""
        import torch

        workload = cls.make_workload(arguments)
        query_data = arguments.query.as_torch()
        key_data = arguments.key.as_torch()
        value_data = arguments.value.as_torch()
        query_scales = arguments.query_scale.as_torch()
        key_scales = arguments.key_scale.as_torch()
        value_scales = arguments.value_scale.as_torch()
        output = arguments.output.as_torch().view(
            workload.num_query_tokens, workload.heads, 128
        )
        key_rows = key_data.shape[1]
        index = torch.arange(key_rows, device=key_data.device)
        logical = (
            index // 16 * 16 + index % 16 // 4 * 2 + index % 2 + index % 4 // 2 * 8
        )
        inverse = torch.argsort(logical)[: workload.num_kv_tokens]
        # Process one head and query tile at a time so the reference does not
        # materialize a [heads, num_query_tokens, num_kv_tokens] score tensor.
        query_tile = 256
        for head in range(workload.heads):
            key = (
                key_data[head].float().reshape(-1, 64, 128)
                * key_scales[head, :, None, None]
            ).reshape(-1, 128)[: workload.num_kv_tokens]
            value = (
                value_data[head, :, inverse].float().transpose(0, 1)
                * value_scales[head]
            )
            for start in range(0, workload.num_query_tokens, query_tile):
                end = min(start + query_tile, workload.num_query_tokens)
                padded_end = -(-end // 32) * 32
                query = (
                    query_data[head, start:padded_end].float().reshape(-1, 32, 128)
                    * query_scales[head, start // 32 : padded_end // 32, None, None]
                ).reshape(-1, 128)[: end - start]
                output[start:end, head] = (
                    torch.nn.functional.scaled_dot_product_attention(
                        query[None, None], key[None, None], value[None, None]
                    )[0, 0]
                )

    @classmethod
    def make_config(
        cls, workload: DenseInt8Fp8AttentionWorkload
    ) -> DenseInt8Fp8AttentionConfig:
        """Return the current production launch configuration."""
        if (
            workload.num_query_tokens < workload.num_kv_tokens
            and workload.num_query_tokens >= 2048
        ):
            return DenseInt8Fp8AttentionConfig(
                tile_query_tokens=128, tile_kv_tokens=64, threads=128
            )
        return DenseInt8Fp8AttentionConfig(
            tile_query_tokens=64, tile_kv_tokens=64, threads=64
        )

    @classmethod
    def make_workload(
        cls, arguments: DenseInt8Fp8AttentionArguments
    ) -> DenseInt8Fp8AttentionWorkload:
        """Recover the workload and validate every quantized attention layout."""
        query_rows = -(-arguments.num_query_tokens // 32) * 32
        key_rows = -(-arguments.num_kv_tokens // 64) * 64
        assert arguments.query.dtype == arguments.key.dtype == DType.I8
        assert arguments.query.shape == (arguments.heads, query_rows, 128)
        assert arguments.key.shape == (arguments.heads, key_rows, 128)
        assert arguments.value.dtype == DType.FP8_E4M3
        assert arguments.value.shape == (arguments.heads, 128, key_rows)
        assert arguments.query_scale.dtype == arguments.key_scale.dtype == DType.F32
        assert arguments.query_scale.shape == (arguments.heads, query_rows // 32)
        assert arguments.key_scale.shape == (arguments.heads, key_rows // 64)
        assert arguments.value_scale.dtype == DType.F32
        assert arguments.value_scale.shape == (arguments.heads, 128)
        assert arguments.output.dtype == DType.BF16
        assert arguments.output.shape == (
            arguments.num_query_tokens,
            arguments.heads * 128,
        )
        return DenseInt8Fp8AttentionWorkload(
            num_kv_tokens=arguments.num_kv_tokens,
            heads=arguments.heads,
            num_query_tokens=arguments.num_query_tokens,
        )
