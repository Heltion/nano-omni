"""Sol-selected INT8 QK and FP8 PV tiles with FP32 online state."""

import math

import tilelang


# Uniform FP8 values satisfy the FP16 partial-accumulation range of this kernel.
@tilelang.jit(
    out_idx=[],
    execution_backend="nvrtc",
    pass_configs={"tl.disable_vectorize_256": True},
    compile_flags=["--use_fast_math"],
)
def sol_int8_fp8_exact(
    num_tokens,
    heads,
    total_heads,
    head_offset=0,
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
            "Sol INT8/FP8 requires KV tile 64 and supported Q tiles/thread counts"
        )
    if tile_query_tokens // (threads // 32) not in (16, 32):
        raise ValueError("Each warp must own 16 or 32 Q rows within one Q scale group")
    qrows = (num_tokens + 31) // 32 * 32
    krows = (num_tokens + 63) // 64 * 64
    warp_rows = tile_query_tokens // (threads // 32)
    warp_m = warp_rows // 16
    scale = 128**-0.5 * 1.4426950408889634
    # |V_fp8| <= 2.25: 64 * 448 * 2.25 = 64512 stays below FP16 overflow.
    # FP16 partials are converted to FP32 after every 64-token PV tile.
    probability_scale = 448 * 64 / tile_kv_tokens
    probability_shift = math.log2(probability_scale)

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        head_offset: head_offset,
        query: T.Tensor((heads, qrows, 128), T.int8),
        key: T.Tensor((heads, krows, 128), T.int8),
        value: T.Tensor((heads, 128, krows), T.float8_e4m3fn),
        qs: T.Tensor((heads, qrows // 32), T.float32),
        ks: T.Tensor((heads, krows // 64), T.float32),
        vs: T.Tensor((heads, 128), T.float32),
        selected: T.Tensor(
            (T.ceildiv(num_tokens, 64), heads, T.ceildiv(num_tokens, 64)), T.uint8
        ),
        output: T.Tensor((num_tokens, total_heads, 128), T.bfloat16),
        state: T.Tensor((num_tokens, total_heads, 2), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(num_tokens, tile_query_tokens), heads, threads=threads
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
            for kb in T.serial(T.ceildiv(num_tokens, tile_kv_tokens)):
                if selected[block, head, kb] != 0:
                    T.copy(
                        key[head, kb * tile_kv_tokens : (kb + 1) * tile_kv_tokens, :], k
                    )
                    T.copy(
                        value[head, :, kb * tile_kv_tokens : (kb + 1) * tile_kv_tokens],
                        v,
                    )
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
                                    k[
                                        n * 8 + lane % 8,
                                        inner * 32 + lane % 16 // 8 * 16,
                                    ]
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
                    if kb == T.ceildiv(num_tokens, tile_kv_tokens) - 1:
                        for m in T.unroll(warp_m):
                            for n in T.unroll(tile_kv_tokens // 8):
                                for item in T.unroll(4):
                                    if (
                                        kb * tile_kv_tokens
                                        + n * 8
                                        + lane % 4 * 2
                                        + item % 2
                                        >= num_tokens
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
                    for n in T.unroll(16):
                        for inner in T.unroll(tile_kv_tokens // 32):
                            T.ptx_ldmatrix(
                                False,
                                2,
                                T.address_of(
                                    v[
                                        n * 8 + lane % 8,
                                        inner * 32 + lane % 16 // 8 * 16,
                                    ]
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
                        if row < num_tokens:
                            output[row, head + head_offset, col] = (
                                result[m, n, item] * vs[head, col] / probability_scale
                            )
                            if n == 0 and item % 2 == 0 and lane % 4 == 0:
                                # Shared Sol state stores the unscaled QK maximum.
                                # Finalize applies the attention scale when it
                                # combines exact and pooled blocks.
                                state[row, head + head_offset, 0] = (
                                    maximum[m, item // 2] / scale
                                )
                                state[row, head + head_offset, 1] = (
                                    denominator[m, item // 2] / probability_scale
                                )

    return main.with_attr(
        "global_symbol",
        f"sol_int8_fp8_exact_{heads}_{total_heads}_{tile_query_tokens}_{tile_kv_tokens}_{threads}",
    )
