"""Sol-selected NVFP4 block-scaled attention with FP32 online softmax."""

import dataclasses
import functools
import math
from typing import Literal

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc


class SolNvfp4Workload(Workload):
    num_query_tokens: int = Field(gt=0)
    num_kv_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    total_heads: int = Field(gt=0)


class SolNvfp4Config(Config):
    tile_query_tokens: Literal[64] = 64
    tile_kv_tokens: Literal[64] = 64
    threads: Literal[128] = 128


@functools.cache
def _expected_exact_pairs(
    num_query_tokens: int, num_kv_tokens: int, num_protected_blocks: int, tau: float
) -> int:
    """Estimate selected token pairs for the public 64-token routing tiles."""
    num_query_blocks = -(-num_query_tokens // 64)
    num_kv_blocks = -(-num_kv_tokens // 64)
    probability = 0.5 * math.erfc(tau / math.sqrt(2))
    exact_pairs = 0.0
    for query_block in range(num_query_blocks):
        query_length = min(64, num_query_tokens - query_block * 64)
        for key_block in range(num_kv_blocks):
            key_length = min(64, num_kv_tokens - key_block * 64)
            mandatory = key_block < num_protected_blocks or (
                query_block - 1 <= key_block <= query_block + 1
            )
            selected_probability = 1.0 if mandatory else probability
            exact_pairs += query_length * key_length * selected_probability
    return round(exact_pairs)


@dataclasses.dataclass(frozen=True, slots=True)
class SolNvfp4Arguments(Arguments):
    query: TensorDesc
    key: TensorDesc
    value: TensorDesc
    query_scales: TensorDesc
    key_scales: TensorDesc
    value_scales: TensorDesc
    selected: TensorDesc
    output: TensorDesc
    state: TensorDesc
    num_query_tokens: int
    num_kv_tokens: int
    head_offset: int
    tau: float
    num_protected_blocks: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (
            ("num_query_tokens", self.num_query_tokens),
            ("num_kv_tokens", self.num_kv_tokens),
            ("head_offset", self.head_offset),
        )


@tilelang.jit(
    out_idx=[],
    execution_backend="nvrtc",
    pass_configs={
        "tl.disable_warp_specialized": True,
        "tl.disable_buffer_init_check": True,
    },
)
def sol_nvfp4_attention(
    num_query_tokens: int,
    num_kv_tokens: int,
    heads: int,
    total_heads: int,
    head_offset: int,
    tile_query_tokens: int = 64,
    tile_kv_tokens: int = 128,
    threads: int = 128,
):
    import tilelang.language as T

    if (
        tile_query_tokens not in (64, 128)
        or tile_kv_tokens not in (64, 128)
        or threads != 2 * tile_query_tokens
    ):
        raise ValueError("unsupported NVFP4 execution configuration")
    query_rows = -(-num_query_tokens // 128) * 128
    key_rows = -(-num_kv_tokens // 128) * 128
    # T.view validates storage in bits; long sequences can exceed int32.
    storage_heads = T.int64(heads)
    log2_scale = 1.4426950408889634 / (128**0.5)
    probability_scale = 6 * 448

    def qk_scale_tile(scales, first_token, rows):
        # Packed scale format: [128-row block, K64 half, row % 32, row // 32].
        stripe = first_token % 128 // 32
        return scales[first_token // 128, 0:2, 0:32, stripe : stripe + rows // 32]

    def head_view(buffer, head):
        # Leading-head tensors are contiguous within each head.
        elements = 1
        for size in buffer.shape[1:]:
            elements *= size
        return T.Buffer(
            list(buffer.shape[1:]),
            buffer.dtype,
            data=buffer.data,
            elem_offset=head * elements,
        )

    def build():
        @T.macro
        def attend_block(key_block, initialize, block, lane, inputs, registers):
            # Literal initialize=True/False specializes the online-softmax start.
            # Each expansion owns staging buffers so only the loop is double-buffered.
            key_words, value_words, key_scale_words, value_scale_words = inputs
            (
                scores,
                group_max,
                next_maximum,
                maximum,
                denominator,
                rescale,
                result,
                q,
                qs,
                k,
                ks,
                v,
                vs,
                p,
                ps,
                ps_bytes,
            ) = registers
            key_tile = T.alloc_shared([tile_kv_tokens // 32, 4, 4, 2, 16], T.uint32)
            value_tile = T.alloc_shared([128, tile_kv_tokens // 8], T.uint32)
            key_scale_tile = T.alloc_shared([2, 32, tile_kv_tokens // 32], T.uint32)
            value_scale_tile = T.alloc_shared([tile_kv_tokens // 64, 32, 4], T.uint32)
            T.annotate_layout(
                {
                    key_scale_tile: T.Layout(
                        [2, 32, tile_kv_tokens // 32],
                        lambda inner, low, high: (
                            inner * tile_kv_tokens + high * 32 + low
                        ),
                    ),
                    value_scale_tile: T.Layout(
                        [tile_kv_tokens // 64, 32, 4],
                        lambda inner, low, high: inner * 128 + high * 32 + low,
                    ),
                    key_tile: T.Layout(
                        [tile_kv_tokens // 32, 4, 4, 2, 16],
                        lambda group, high, low, pair, word: (
                            (group * 32 + low * 8 + high * 2 + pair) * 20 + word
                        ),
                    ),
                    value_tile: T.Layout(
                        [128, tile_kv_tokens // 8],
                        lambda row, word: row * (tile_kv_tokens // 8 + 4) + word,
                    ),
                }
            )
            # K/V are shared across query warps; Q and P never round-trip here.
            T.copy(
                key_words[key_block * (tile_kv_tokens // 32), 0, 0, 0, 0],
                key_tile,
                disable_tma=True,
            )
            T.copy(
                value_words[0, key_block * (tile_kv_tokens // 8)],
                value_tile,
                disable_tma=True,
            )
            T.copy(
                qk_scale_tile(
                    key_scale_words, key_block * tile_kv_tokens, tile_kv_tokens
                ),
                key_scale_tile,
                disable_tma=True,
            )
            T.copy(
                value_scale_words[key_block * (tile_kv_tokens // 64), 0, 0],
                value_scale_tile,
                disable_tma=True,
            )
            T.clear(scores)
            for group in T.unroll(tile_kv_tokens // 32):
                # Each quad preloads four different B scale words.
                # The four MMAs select lanes 0, 1, 2, 3 in turn.
                key_scale_matrix = T.make_tensor_from_addr(
                    T.access_ptr(key_scale_tile[0, 0, group], "r"),
                    [2, 32],
                    T.uint32,
                    strides=[tile_kv_tokens, 1],
                    storage_scope="shared",
                )
                T.copy(key_scale_matrix, ks)
                for pair in T.unroll(4):
                    # The view fixes this tile's base address before copy
                    # lowering. Its stride includes the shared row padding.
                    key_matrix = T.make_tensor_from_addr(
                        T.access_ptr(key_tile[group, 0, pair, 0, 0], "r"),
                        [8, 32],
                        T.uint16,
                        strides=[40, 1],
                        storage_scope="shared",
                    )
                    T.copy(key_matrix, k)
                    for inner in T.unroll(2):
                        # PTX arguments below are flat register offsets:
                        # A advances four uint32 words per K=64 half,
                        # B advances four uint16 units (two packed words).
                        T.ptx_mma_block_scale(
                            "float32",
                            "m16n8k64",
                            "row",
                            "col",
                            "mxf4nvf4",
                            4,
                            "e2m1",
                            "e2m1",
                            "ue4m3",
                            q.data,
                            inner * 4,
                            k.data,
                            inner * 4,
                            scores.data,
                            (group * 4 + pair) * 4,
                            T.access_ptr(qs, "r", extent=1),
                            T.access_ptr(ks[inner, 0], "r", extent=1),
                            scale_a_thread_id=inner,
                            scale_b_thread_id=pair,
                        )
            T.fill(next_maximum, -T.infinity(T.float32))
            for group in T.unroll(tile_kv_tokens // 32):
                for row in T.unroll(2):
                    group_max[row, group] = -T.infinity(T.float32)
                    for col in T.unroll(8):
                        key_column = (
                            key_block * tile_kv_tokens + group * 32 + lane % 4 * 8 + col
                        )
                        if num_kv_tokens % tile_kv_tokens != 0:
                            scores[row, group, col] = T.if_then_else(
                                key_column < num_kv_tokens,
                                scores[row, group, col],
                                -T.infinity(T.float32),
                            )
                        group_max[row, group] = T.max(
                            group_max[row, group], scores[row, group, col]
                        )
                    group_max[row, group] = T.max(
                        group_max[row, group],
                        T.shfl_xor(group_max[row, group], 1),
                    )
                    next_maximum[row] = T.max(next_maximum[row], group_max[row, group])
            for row in T.unroll(2):
                next_maximum[row] = T.max(
                    next_maximum[row], T.shfl_xor(next_maximum[row], 2)
                )
                # The first block establishes the online-softmax state.
                if initialize:
                    maximum[row] = next_maximum[row]
                    denominator[row] = 0.0
                else:
                    next_maximum[row] = T.max(maximum[row], next_maximum[row])
                    rescale[row] = T.exp2(
                        (maximum[row] - next_maximum[row]) * log2_scale
                    )
                    maximum[row] = next_maximum[row]
                    denominator[row] *= rescale[row]
            for group in T.unroll(tile_kv_tokens // 32):
                for row in T.unroll(2):
                    group_max[row, group] = T.exp2(
                        group_max[row, group] * log2_scale
                        - (maximum[row] * log2_scale - 11.392317422778762)
                        - 2.584962500721156
                    )
                    for col in T.unroll(8):
                        scores[row, group, col] = T.exp2(
                            scores[row, group, col] * log2_scale
                            - (maximum[row] * log2_scale - 11.392317422778762)
                        )
                        denominator[row] += scores[row, group, col]
            # Four consecutive groups of 16 form one scale word. Fetch
            # both lane-pair scales while preserving the selected query row.
            for inner in T.unroll(tile_kv_tokens // 64):
                for group in T.unroll(4):
                    scale = T.if_then_else(
                        lane % 2 == 0,
                        group_max[0, inner * 2 + group // 2],
                        group_max[1, inner * 2 + group // 2],
                    )
                    ps_bytes[inner, group] = T.shfl_sync(
                        scale, lane // 4 * 4 + group % 2 * 2 + lane % 2
                    )
            for group in T.unroll(tile_kv_tokens // 32):
                for row in T.unroll(2):
                    group_max[row, group] = T.if_then_else(
                        group_max[row, group] == 0.0,
                        0.0,
                        T.fast_rcp(group_max[row, group]),
                    )
                    for col in T.unroll(8):
                        scores[row, group, col] *= group_max[row, group]
            T.copy(scores, p)
            for column_group in T.unroll(4):
                value_scale_matrix = T.make_tensor_from_addr(
                    T.access_ptr(value_scale_tile[0, 0, column_group], "r"),
                    [tile_kv_tokens // 64, 32],
                    T.uint32,
                    strides=[128, 1],
                    storage_scope="shared",
                )
                T.copy(value_scale_matrix, vs)
                for part in T.unroll(4):
                    column = column_group * 4 + part
                    if not initialize:
                        for row in T.unroll(2):
                            for item in T.unroll(2):
                                result[row, column * 2 + item] *= rescale[row]
                    value_matrix = T.make_tensor_from_addr(
                        T.access_ptr(value_tile[column * 8, 0], "r"),
                        [8, tile_kv_tokens // 4],
                        T.uint16,
                        strides=[tile_kv_tokens // 4 + 8, 1],
                        storage_scope="shared",
                    )
                    T.copy(value_matrix, v)
                    for inner in T.unroll(tile_kv_tokens // 64):
                        T.ptx_mma_block_scale(
                            "float32",
                            "m16n8k64",
                            "row",
                            "col",
                            "mxf4nvf4",
                            4,
                            "e2m1",
                            "e2m1",
                            "ue4m3",
                            p.data,
                            inner * 32,
                            v.data,
                            inner * 4,
                            result.data,
                            column * 4,
                            T.access_ptr(ps[inner], "r"),
                            T.access_ptr(vs[inner, 0], "r", extent=1),
                            scale_b_thread_id=part,
                        )

        @T.prim_func
        def main(
            num_query_tokens: num_query_tokens,
            num_kv_tokens: num_kv_tokens,
            head_offset: head_offset,
            query: T.Tensor([storage_heads, query_rows, 128], T.float4_e2m1fn),
            key: T.Tensor([storage_heads, key_rows, 128], T.float4_e2m1fn),
            value: T.Tensor([storage_heads, 128, key_rows], T.float4_e2m1fn),
            query_scales: T.Tensor([storage_heads, query_rows * 2], T.uint32),
            key_scales: T.Tensor([storage_heads, key_rows * 2], T.uint32),
            value_scales: T.Tensor([storage_heads, key_rows * 2], T.uint32),
            selected: T.Tensor(
                [
                    T.ceildiv(num_query_tokens, 64),
                    storage_heads,
                    T.ceildiv(num_kv_tokens, 64),
                ],
                T.uint8,
            ),
            output: T.Tensor([num_query_tokens, total_heads * 128], T.bfloat16),
            state: T.Tensor([num_query_tokens, total_heads, 2], T.float32),
        ):
            query_words_all = T.view(query, [storage_heads, query_rows, 16], T.uint32)
            key_words_all = T.view(
                key, [storage_heads, key_rows // 32, 4, 4, 2, 16], T.uint32
            )
            value_words_all = T.view(
                value, [storage_heads, 128, key_rows // 8], T.uint32
            )
            query_scale_words_all = T.view(
                query_scales, [storage_heads, query_rows // 128, 2, 32, 4], T.uint32
            )
            key_scale_words_all = T.view(
                key_scales, [storage_heads, key_rows // 128, 2, 32, 4], T.uint32
            )
            value_scale_words_all = T.view(
                value_scales, [storage_heads, key_rows // 64, 32, 4], T.uint32
            )
            with T.Kernel(query_rows // tile_query_tokens, heads, threads=threads) as (
                block,
                head,
            ):
                query_words = head_view(query_words_all, head)
                key_words = head_view(key_words_all, head)
                value_words = head_view(value_words_all, head)
                query_scale_words = head_view(query_scale_words_all, head)
                key_scale_words = head_view(key_scale_words_all, head)
                value_scale_words = head_view(value_scale_words_all, head)
                # Output interleaves heads within each token row.
                output_head = T.Buffer(
                    [num_query_tokens, 128],
                    T.bfloat16,
                    data=output.data,
                    elem_offset=(head + head_offset) * 128,
                    strides=[total_heads * 128, 1],
                )
                thread = T.get_thread_binding()
                lane = thread % 32
                # One warp computes 16 Q rows. A four-lane group shares two
                # rows: query_row and query_row + 8, with columns split by lane.
                query_row = block * tile_query_tokens + thread // 32 * 16 + lane // 4
                # m16n8k64: A has four packed words per lane, B has two.
                # Each lane owns two query rows (eight rows apart).
                # Each word packs eight FP4 channels. The fragment layout assigns
                # each logical (token, word) to the lane and A-register slot
                # consumed by m16n8k64, without shared-memory staging.
                q = T.alloc_fragment([tile_query_tokens, 16], T.uint32)
                qs = T.alloc_fragment([2, 32, tile_query_tokens // 32], T.uint32)
                # ldmatrix operates on 16-bit storage units (four packed FP4s).
                k = T.alloc_fragment([8, 32], T.uint16)
                v = T.alloc_fragment([8, tile_kv_tokens // 4], T.uint16)
                ks = T.alloc_fragment([2, 32], T.uint32)

                vs = T.alloc_fragment([tile_kv_tokens // 64, 32], T.uint32)
                # Each thread owns two Q rows and eight elements of each
                # 16-key quantization group; its adjacent lane owns the other eight.
                # Layouts below translate these coordinates into MMA register order.
                scores = T.alloc_local([2, tile_kv_tokens // 32, 8], T.float32)
                group_max = T.alloc_local([2, tile_kv_tokens // 32], T.float32)
                # The following length-two arrays track the lane's two Q rows.
                # maximum is shared logically by the four lanes of a row group;
                # denominator holds a partial sum until the final shuffle reduction.
                maximum = T.alloc_local([2], T.float32)
                next_maximum = T.alloc_local([2], T.float32)
                denominator = T.alloc_local([2], T.float32)
                rescale = T.alloc_local([2], T.float32)
                p = T.alloc_local([2, tile_kv_tokens // 32, 8], T.float4_e2m1fn)
                ps_bytes = T.alloc_local([tile_kv_tokens // 64, 4], T.float8_e4m3fn)
                ps = T.view(ps_bytes, [tile_kv_tokens // 64], T.uint32)
                # Each thread contributes 32 output channels for each of its two rows.
                result = T.alloc_local([2, 32], T.float32)
                # Packed-word coordinates, lane ownership and register order.
                # K token within each 32-row group is high * 8 + low * 2 + pair.
                # Scale row within each 128-row group is high * 32 + low.
                T.annotate_layout(
                    {
                        k: T.Fragment(
                            [8, 32],
                            replicate=threads // 32,
                            forward_thread_fn=lambda row, col, warp: (
                                warp * 32 + row * 4 + col // 2 % 4
                            ),
                            forward_index_fn=lambda row, col: col // 8 * 2 + col % 2,
                        ),
                        v: T.Fragment(
                            [8, tile_kv_tokens // 4],
                            replicate=threads // 32,
                            forward_thread_fn=lambda row, col, warp: (
                                warp * 32 + row * 4 + col // 2 % 4
                            ),
                            forward_index_fn=lambda row, col: col // 8 * 2 + col % 2,
                        ),
                        ks: T.Fragment(
                            [2, 32],
                            replicate=threads // 32,
                            forward_thread_fn=lambda inner, row, warp: (
                                warp * 32 + row // 8 * 8 + row % 2 * 4 + row % 8 // 2
                            ),
                            forward_index_fn=lambda inner, row: inner,
                        ),
                        vs: T.Fragment(
                            [tile_kv_tokens // 64, 32],
                            replicate=threads // 32,
                            forward_thread_fn=lambda inner, row, warp: (
                                warp * 32 + row % 8 * 4 + row // 8
                            ),
                            forward_index_fn=lambda inner, row: inner,
                        ),
                        # Logical coordinates: row, quantization group, element.
                        scores: T.Layout(
                            [2, tile_kv_tokens // 32, 8],
                            lambda row, group, col: (
                                group * 16 + col // 2 * 4 + row * 2 + col % 2
                            ),
                        ),
                        group_max: T.Layout(
                            [2, tile_kv_tokens // 32],
                            lambda row, group: group * 2 + row,
                        ),
                        p: T.Layout(
                            [2, tile_kv_tokens // 32, 8],
                            lambda row, group, col: (
                                group // 2 * 32 + (group % 2 * 2 + row) * 8 + col
                            ),
                        ),
                        result: T.Layout(
                            [2, 32], lambda row, col: col // 2 * 4 + row * 2 + col % 2
                        ),
                        q: T.Fragment(
                            [tile_query_tokens, 16],
                            forward_thread_fn=lambda row, word: (
                                row // 16 * 32 + row % 8 * 4 + word % 4
                            ),
                            forward_index_fn=lambda row, word: (
                                word // 8 * 4 + word % 8 // 4 * 2 + row % 16 // 8
                            ),
                        ),
                        qs: T.Fragment(
                            [2, 32, tile_query_tokens // 32],
                            forward_thread_fn=lambda inner, low, high: (
                                (high * 32 + low) // 16 * 32
                                + low % 8 * 4
                                + inner * 2
                                + low % 16 // 8
                            ),
                            forward_index_fn=lambda inner, low, high: 0,
                        ),
                    }
                )

                # The block-scaled MMA intrinsic accumulates into C.
                T.clear(result)
                T.copy(query_words[block * tile_query_tokens, 0], q)
                T.copy(
                    qk_scale_tile(
                        query_scale_words, block * tile_query_tokens, tile_query_tokens
                    ),
                    qs,
                )

                inputs = (key_words, value_words, key_scale_words, value_scale_words)
                registers = (
                    scores,
                    group_max,
                    next_maximum,
                    maximum,
                    denominator,
                    rescale,
                    result,
                    q,
                    qs,
                    k,
                    ks,
                    v,
                    vs,
                    p,
                    ps,
                    ps_bytes,
                )
                # Block zero is always selected by the routing preprocessor and
                # establishes the online-softmax state. Remaining blocks are
                # submitted only when their BF16 routing decision is set.
                attend_block(
                    0,
                    True,
                    block,
                    lane,
                    inputs,
                    registers,
                )
                for key_block in T.serial(1, T.ceildiv(num_kv_tokens, tile_kv_tokens)):
                    route_block = key_block * (tile_kv_tokens // 64)
                    if selected[
                        block * (tile_query_tokens // 64), head, route_block
                    ] != 0 or (
                        tile_kv_tokens == 128
                        and route_block + 1 < T.ceildiv(num_kv_tokens, 64)
                        and selected[
                            block * (tile_query_tokens // 64), head, route_block + 1
                        ]
                        != 0
                    ):
                        attend_block(
                            key_block,
                            False,
                            block,
                            lane,
                            inputs,
                            registers,
                        )
                for row in T.unroll(2):
                    # Combine the four lanes' partial softmax denominators, then
                    # map each lane's two output columns back to the dense tensor.
                    denominator[row] += T.shfl_xor(denominator[row], 1)
                    denominator[row] += T.shfl_xor(denominator[row], 2)
                    if query_row + row * 8 < num_query_tokens:
                        for column in T.unroll(16):
                            for item in T.vectorized(2):
                                output_head[
                                    query_row + row * 8,
                                    column * 8 + lane % 4 * 2 + item,
                                ] = result[row, column * 2 + item] / probability_scale
                        if lane % 4 == 0:
                            state[query_row + row * 8, head + head_offset, 0] = maximum[
                                row
                            ]
                            state[query_row + row * 8, head + head_offset, 1] = (
                                denominator[row] / probability_scale
                            )

        return main.with_attr(
            "global_symbol",
            f"sol_nvfp4_{heads}_{total_heads}_"
            f"{tile_query_tokens}_{tile_kv_tokens}_{threads}",
        )

    return build()


class SolNvfp4Kernel(Kernel[SolNvfp4Arguments, SolNvfp4Workload, SolNvfp4Config]):
    name = "sol_nvfp4_attention_d128"
    program = sol_nvfp4_attention
    reference_rtol = 0.03
    reference_atol = 0.03

    @classmethod
    def make_arguments(cls, workload: SolNvfp4Workload) -> SolNvfp4Arguments:
        query_rows = -(-workload.num_query_tokens // 128) * 128
        key_rows = -(-workload.num_kv_tokens // 128) * 128
        return SolNvfp4Arguments(
            TensorDesc.empty(DType.U8, (workload.heads, query_rows, 64)),
            TensorDesc.empty(DType.U8, (workload.heads, key_rows, 64)),
            TensorDesc.empty(DType.U8, (workload.heads, 128, key_rows // 2)),
            TensorDesc.empty(DType.FP8_UE4M3, (workload.heads, query_rows, 8)),
            TensorDesc.empty(DType.FP8_UE4M3, (workload.heads, key_rows, 8)),
            TensorDesc.empty(DType.FP8_UE4M3, (workload.heads, 128, key_rows // 16)),
            TensorDesc.empty(
                DType.U8,
                (
                    -(-workload.num_query_tokens // 64),
                    workload.heads,
                    -(-workload.num_kv_tokens // 64),
                ),
            ),
            TensorDesc.empty(
                DType.BF16, (workload.num_query_tokens, workload.total_heads, 128)
            ),
            TensorDesc.empty(
                DType.F32, (workload.num_query_tokens, workload.total_heads, 2)
            ),
            workload.num_query_tokens,
            workload.num_kv_tokens,
            0,
            1.0,
            1,
        )

    @classmethod
    def make_config(cls, workload: SolNvfp4Workload) -> SolNvfp4Config:
        del workload
        return SolNvfp4Config()

    @classmethod
    def make_workload(cls, arguments: SolNvfp4Arguments) -> SolNvfp4Workload:
        heads, query_rows, half_dim = arguments.query.shape
        key_heads, key_rows, key_half_dim = arguments.key.shape
        assert arguments.query.dtype == arguments.key.dtype == DType.U8
        assert heads == key_heads and half_dim == key_half_dim == 64
        expected_query_rows = -(-arguments.num_query_tokens // 128) * 128
        expected_key_rows = -(-arguments.num_kv_tokens // 128) * 128
        assert query_rows == expected_query_rows and key_rows == expected_key_rows
        assert arguments.value.dtype == DType.U8
        assert arguments.value.shape == (heads, 128, key_rows // 2)
        assert arguments.query_scales.dtype == DType.FP8_UE4M3
        assert arguments.key_scales.dtype == DType.FP8_UE4M3
        assert arguments.value_scales.dtype == DType.FP8_UE4M3
        assert arguments.query_scales.shape == (heads, query_rows, 8)
        assert arguments.key_scales.shape == (heads, key_rows, 8)
        assert arguments.value_scales.shape == (heads, 128, key_rows // 16)
        assert arguments.selected.dtype == DType.U8
        assert arguments.selected.shape == (
            -(-arguments.num_query_tokens // 64),
            heads,
            -(-arguments.num_kv_tokens // 64),
        )
        num_query_tokens, total_heads, dim = arguments.output.shape
        assert arguments.output.dtype == DType.BF16 and dim == 128
        assert arguments.state.dtype == DType.F32
        assert arguments.state.shape == (num_query_tokens, total_heads, 2)
        assert num_query_tokens == arguments.num_query_tokens
        assert arguments.num_kv_tokens > 0
        assert 0 <= arguments.head_offset <= total_heads - heads
        assert arguments.tau == 1.0
        assert 1 <= arguments.num_protected_blocks <= -(-arguments.num_kv_tokens // 64)
        return SolNvfp4Workload(
            num_query_tokens=num_query_tokens,
            num_kv_tokens=arguments.num_kv_tokens,
            heads=heads,
            total_heads=total_heads,
        )

    @classmethod
    def tops(cls, arguments: SolNvfp4Arguments) -> Tops:
        workload = cls.make_workload(arguments)
        exact_pairs = _expected_exact_pairs(
            workload.num_query_tokens,
            workload.num_kv_tokens,
            arguments.num_protected_blocks,
            arguments.tau,
        )
        operations = 4 * workload.heads * 128 * exact_pairs
        return {MmaType.F4F4F32: operations}

    @classmethod
    def ref_program(cls, arguments: SolNvfp4Arguments) -> None:
        """Evaluate the selected NVFP4 blocks and expose online-softmax state."""
        import torch

        workload = cls.make_workload(arguments)

        def dequantize(packed, scales):
            heads, rows, half_columns = packed.shape
            columns = half_columns * 2
            logical_scales = (
                scales.float()
                .reshape(heads, rows // 128, columns // 64, 32, 4, 4)
                .permute(0, 1, 4, 3, 2, 5)
                .reshape(heads, rows, columns // 16)
            )
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(
                heads, rows, columns
            )
            magnitudes = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6],
                device=packed.device,
                dtype=torch.float32,
            )
            values = magnitudes[(codes & 7).long()] * torch.where(codes < 8, 1, -1)
            return values * logical_scales.repeat_interleave(16, dim=-1)

        query = dequantize(
            arguments.query.as_torch(), arguments.query_scales.as_torch()
        )[:, : workload.num_query_tokens]
        key = dequantize(arguments.key.as_torch(), arguments.key_scales.as_torch())[
            :, : workload.num_kv_tokens
        ]
        value = dequantize(
            arguments.value.as_torch(), arguments.value_scales.as_torch()
        ).transpose(1, 2)[:, : workload.num_kv_tokens]
        selected = arguments.selected.as_torch().bool()
        output = arguments.output.as_torch()
        state = arguments.state.as_torch()
        log2_scale = 128**-0.5 * math.log2(math.e)
        probability_scale = 6 * 448
        magnitudes = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6],
            device=query.device,
            dtype=torch.float64,
        )

        def quantize_probability(probability):
            """Round 16-value groups to the NVFP4 values consumed by PV MMA."""
            rows, columns = probability.shape
            padded = -(-columns // 16) * 16
            values = torch.nn.functional.pad(probability, (0, padded - columns))
            groups = values.reshape(rows, padded // 16, 16)
            block_scale = (groups.abs().amax(-1) / 6).clamp(max=448)
            encoded_scale = block_scale.to(torch.float8_e4m3fn)
            inverse = torch.where(block_scale == 0, 0, block_scale.reciprocal())
            normalized = (groups * inverse[..., None]).double()
            distances = (normalized.abs()[..., None] - magnitudes).abs()
            nearest = distances == distances.amin(-1, keepdim=True)
            indices = torch.arange(8, device=query.device)
            priority = indices + (indices % 2) * 16
            codes = torch.where(nearest, priority, 100).argmin(-1)
            quantized = magnitudes[codes] * torch.where(normalized.signbit(), -1, 1)
            restored = quantized.float() * encoded_scale.float()[..., None]
            return restored.reshape(rows, padded)[:, :columns]

        num_query_blocks = -(-workload.num_query_tokens // 64)
        num_kv_blocks = -(-workload.num_kv_tokens // 64)
        for query_block in range(num_query_blocks):
            start = query_block * 64
            stop = min(start + 64, workload.num_query_tokens)
            for head in range(workload.heads):
                maximum = torch.full((stop - start,), -torch.inf, device=query.device)
                denominator = torch.zeros_like(maximum)
                numerator = torch.zeros(
                    (stop - start, 128), device=query.device, dtype=torch.float32
                )
                for key_block in range(num_kv_blocks):
                    if key_block and not selected[query_block, head, key_block]:
                        continue
                    key_start = key_block * 64
                    key_stop = min(key_start + 64, workload.num_kv_tokens)
                    scores = query[head, start:stop] @ key[head, key_start:key_stop].T
                    next_maximum = torch.maximum(maximum, scores.amax(1))
                    correction = torch.exp2((maximum - next_maximum) * log2_scale)
                    probability = (
                        torch.exp2((scores - next_maximum[:, None]) * log2_scale)
                        * probability_scale
                    )
                    numerator = numerator * correction[:, None] + (
                        quantize_probability(probability)
                        @ value[head, key_start:key_stop]
                    )
                    denominator = denominator * correction + probability.sum(1)
                    maximum = next_maximum
                target = head + arguments.head_offset
                output[start:stop, target].copy_(numerator / probability_scale)
                state[start:stop, target, 0].copy_(maximum)
                state[start:stop, target, 1].copy_(denominator / probability_scale)
