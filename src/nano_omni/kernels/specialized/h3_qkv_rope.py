"""Normalize and rotate packed QKV projections."""

import dataclasses
from typing import Self

import tilelang
from pydantic import Field, model_validator

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class H3QkvRopeWorkload(Workload):
    """Normalize each head and rotate an even split-half prefix within its width."""

    tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    dim: int = Field(gt=0)
    rope_dim: int = Field(ge=2, multiple_of=2)
    use_norm_weight: bool
    use_rope: bool
    qkv_interleaved: bool = False
    dtype: DType = DType.BF16

    @model_validator(mode="after")
    def validate_head_dimensions(self) -> Self:
        if self.rope_dim > self.dim:
            raise ValueError("Rotary dimensions cannot exceed the head width")
        return self


class H3QkvRopeConfig(Config):
    tile_tokens: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class H3QkvRopeArguments(Arguments):
    """FP16/BF16 QKV [tokens, 3*heads*dim] to [3, tokens, heads, dim].

    Enabled normalization needs matching weights [dim]; enabled rotation needs FP32
    cosine/sine planes [tokens, rope_dim/2]. Disabled operands may be omitted.
    """

    qkv: TensorDesc
    q_weight: TensorDesc | None
    k_weight: TensorDesc | None
    cosines: TensorDesc | None
    sines: TensorDesc | None
    output: TensorDesc
    heads: int
    dim: int
    rope_dim: int
    use_norm_weight: bool
    use_rope: bool
    qkv_interleaved: bool = False

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one program for different token counts."""
        return (("tokens", self.qkv.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def h3_qkv_rope(
    tokens,
    heads,
    dim,
    rope_dim,
    use_norm_weight,
    use_rope,
    qkv_interleaved,
    dtype,
    tile_tokens=1,
    threads=128,
):
    import tilelang.language as T

    dynamic_tokens = T.dynamic("tokens")
    width = heads * dim
    half_rope = rope_dim // 2
    tail_dim = dim - rope_dim
    norm_dim = 1 << (dim - 1).bit_length()
    head_stride = dim * 3 if qkv_interleaved else dim
    key_head_offset = dim if qkv_interleaved else width
    value_head_offset = dim * 2 if qkv_interleaved else width * 2
    storage_type = T.float16 if dtype == DType.F16 else T.bfloat16

    @T.prim_func
    def main(
        tokens: dynamic_tokens,
        qkv: T.Tensor([dynamic_tokens, width * 3], storage_type),
        q_weight: T.Tensor([dim], storage_type),
        k_weight: T.Tensor([dim], storage_type),
        cosines: T.Tensor([dynamic_tokens, half_rope], T.float32),
        sines: T.Tensor([dynamic_tokens, half_rope], T.float32),
        output: T.Tensor(
            [3, dynamic_tokens, heads, dim],
            storage_type,
        ),
    ):
        with T.Kernel(
            T.ceildiv(tokens, tile_tokens),
            heads,
            threads=threads,
        ) as (token_block, head_block):
            query_first = T.alloc_fragment([tile_tokens, half_rope], T.float32)
            query_second = T.alloc_fragment([tile_tokens, half_rope], T.float32)
            query_tail = T.alloc_fragment([tile_tokens, tail_dim], T.float32)
            key_first = T.alloc_fragment([tile_tokens, half_rope], T.float32)
            key_second = T.alloc_fragment([tile_tokens, half_rope], T.float32)
            key_tail = T.alloc_fragment([tile_tokens, tail_dim], T.float32)
            query_squares = T.alloc_fragment([tile_tokens, norm_dim], T.float32)
            key_squares = T.alloc_fragment([tile_tokens, norm_dim], T.float32)
            query_total = T.alloc_fragment([tile_tokens], T.float32)
            key_total = T.alloc_fragment([tile_tokens], T.float32)
            for i, j in T.Parallel(tile_tokens, norm_dim):
                token = token_block * tile_tokens + i
                column = head_block * head_stride + j
                query_value = T.if_then_else(
                    token < tokens and j < dim,
                    qkv[token, column],
                    0.0,
                )
                key_value = T.if_then_else(
                    token < tokens and j < dim,
                    qkv[token, key_head_offset + column],
                    0.0,
                )
                query_squares[i, j] = query_value * query_value
                key_squares[i, j] = key_value * key_value
            T.reduce_sum(query_squares, query_total, dim=1)
            T.reduce_sum(key_squares, key_total, dim=1)
            for i, j in T.Parallel(tile_tokens, half_rope):
                token = token_block * tile_tokens + i
                column = head_block * head_stride + j
                query_first[i, j] = (
                    qkv[token, column]
                    * T.rsqrt(query_total[i] / dim + 1e-5)
                    * T.if_then_else(use_norm_weight, q_weight[j], 1.0)
                )
                key_first[i, j] = (
                    qkv[token, key_head_offset + column]
                    * T.rsqrt(key_total[i] / dim + 1e-5)
                    * T.if_then_else(use_norm_weight, k_weight[j], 1.0)
                )
            for i, j in T.Parallel(tile_tokens, half_rope):
                token = token_block * tile_tokens + i
                dimension = j + half_rope
                column = head_block * head_stride + dimension
                query_second[i, j] = (
                    qkv[token, column]
                    * T.rsqrt(query_total[i] / dim + 1e-5)
                    * T.if_then_else(use_norm_weight, q_weight[dimension], 1.0)
                )
                key_second[i, j] = (
                    qkv[token, key_head_offset + column]
                    * T.rsqrt(key_total[i] / dim + 1e-5)
                    * T.if_then_else(use_norm_weight, k_weight[dimension], 1.0)
                )
            for i, j in T.Parallel(tile_tokens, tail_dim):
                token = token_block * tile_tokens + i
                dimension = j + rope_dim
                column = head_block * head_stride + dimension
                query_tail[i, j] = (
                    qkv[token, column]
                    * T.rsqrt(query_total[i] / dim + 1e-5)
                    * T.if_then_else(use_norm_weight, q_weight[dimension], 1.0)
                )
                key_tail[i, j] = (
                    qkv[token, key_head_offset + column]
                    * T.rsqrt(key_total[i] / dim + 1e-5)
                    * T.if_then_else(use_norm_weight, k_weight[dimension], 1.0)
                )
            for i, j in T.Parallel(tile_tokens, half_rope):
                token = token_block * tile_tokens + i
                if token < tokens:
                    cosine = T.if_then_else(use_rope, cosines[token, j], 1.0)
                    sine = T.if_then_else(use_rope, sines[token, j], 0.0)
                    output[0, token, head_block, j] = (
                        query_first[i, j] * cosine - query_second[i, j] * sine
                    )
                    output[0, token, head_block, j + half_rope] = (
                        query_first[i, j] * sine + query_second[i, j] * cosine
                    )
                    output[1, token, head_block, j] = (
                        key_first[i, j] * cosine - key_second[i, j] * sine
                    )
                    output[1, token, head_block, j + half_rope] = (
                        key_first[i, j] * sine + key_second[i, j] * cosine
                    )
            for i, j in T.Parallel(tile_tokens, tail_dim):
                token = token_block * tile_tokens + i
                if token < tokens:
                    output[0, token, head_block, j + rope_dim] = query_tail[i, j]
                    output[1, token, head_block, j + rope_dim] = key_tail[i, j]
            for i, j in T.Parallel(tile_tokens, dim):
                token = token_block * tile_tokens + i
                if token < tokens:
                    output[2, token, head_block, j] = qkv[
                        token, value_head_offset + head_block * head_stride + j
                    ]

    return main.with_attr(
        "global_symbol",
        f"h3_qkv_rope_{heads}_{dim}_{rope_dim}_{int(use_norm_weight)}_{int(use_rope)}_{int(qkv_interleaved)}_{dtype.value}_{tile_tokens}_{threads}",
    )


class H3QkvRopeKernel(Kernel[H3QkvRopeArguments, H3QkvRopeWorkload, H3QkvRopeConfig]):
    name = "h3_qkv_rope"
    program = h3_qkv_rope

    @classmethod
    def make_arguments(cls, workload: H3QkvRopeWorkload) -> H3QkvRopeArguments:
        """Describe packed QKV, optional normalization/rotation data, and output."""
        width = workload.heads * workload.dim
        return H3QkvRopeArguments(
            qkv=TensorDesc.empty(workload.dtype, (workload.tokens, 3 * width)),
            q_weight=(
                TensorDesc.empty(workload.dtype, (workload.dim,))
                if workload.use_norm_weight
                else None
            ),
            k_weight=(
                TensorDesc.empty(workload.dtype, (workload.dim,))
                if workload.use_norm_weight
                else None
            ),
            cosines=(
                TensorDesc.empty(DType.F32, (workload.tokens, workload.rope_dim // 2))
                if workload.use_rope
                else None
            ),
            sines=(
                TensorDesc.empty(DType.F32, (workload.tokens, workload.rope_dim // 2))
                if workload.use_rope
                else None
            ),
            output=TensorDesc.empty(
                workload.dtype,
                (3, workload.tokens, workload.heads, workload.dim),
            ),
            heads=workload.heads,
            dim=workload.dim,
            rope_dim=workload.rope_dim,
            use_norm_weight=workload.use_norm_weight,
            use_rope=workload.use_rope,
            qkv_interleaved=workload.qkv_interleaved,
        )

    @classmethod
    def make_workload(cls, arguments: H3QkvRopeArguments) -> H3QkvRopeWorkload:
        assert len(arguments.qkv.shape) == 2, (
            "QKV must be a matrix of packed head projections"
        )
        workload = H3QkvRopeWorkload(
            tokens=arguments.qkv.shape[0],
            heads=arguments.heads,
            dim=arguments.dim,
            rope_dim=arguments.rope_dim,
            use_norm_weight=arguments.use_norm_weight,
            use_rope=arguments.use_rope,
            qkv_interleaved=arguments.qkv_interleaved,
            dtype=arguments.qkv.dtype,
        )
        assert arguments.qkv.shape[1] == 3 * workload.heads * workload.dim and (
            arguments.output.shape == (3, workload.tokens, workload.heads, workload.dim)
        ), "QKV and output shapes must match the head dimensions"
        assert arguments.qkv.dtype == arguments.output.dtype
        assert arguments.qkv.dtype in (DType.F16, DType.BF16)
        assert not workload.use_norm_weight or not any(
            weight is None or weight.shape != (workload.dim,)
            for weight in (arguments.q_weight, arguments.k_weight)
        ), "Enabled normalization requires two head-width weights"
        assert not workload.use_rope or not any(
            plane is None or plane.shape != (workload.tokens, workload.rope_dim // 2)
            for plane in (arguments.cosines, arguments.sines)
        ), "Enabled RoPE requires matching cosine and sine planes"
        return workload

    @classmethod
    def make_config(cls, workload: H3QkvRopeWorkload) -> H3QkvRopeConfig:
        """Match the thread count to the rotary half-width."""
        return H3QkvRopeConfig(
            threads=32 if workload.rope_dim <= 64 else 64,
            tile_tokens=1,
        )

    @classmethod
    def ref_program(cls, arguments: H3QkvRopeArguments) -> None:
        """Normalize Q/K, apply split-half RoPE, and copy V unchanged."""
        import torch

        workload = cls.make_workload(arguments)
        qkv = arguments.qkv.as_torch()
        if workload.qkv_interleaved:
            values = qkv.view(workload.tokens, workload.heads, 3, workload.dim).permute(
                2, 0, 1, 3
            )
        else:
            values = qkv.view(workload.tokens, 3, workload.heads, workload.dim).permute(
                1, 0, 2, 3
            )
        output = arguments.output.as_torch()
        normalized = values[:2].float()
        normalized *= torch.rsqrt(normalized.square().mean(-1, keepdim=True) + 1e-5)
        if workload.use_norm_weight:
            assert arguments.q_weight is not None and arguments.k_weight is not None
            weights = torch.stack(
                (arguments.q_weight.as_torch(), arguments.k_weight.as_torch())
            ).float()
            normalized *= weights[:, None, None, :]
        if workload.use_rope:
            assert arguments.cosines is not None and arguments.sines is not None
            half = workload.rope_dim // 2
            first = normalized[..., :half]
            second = normalized[..., half : workload.rope_dim]
            cosine = arguments.cosines.as_torch().float()[None, :, None, :]
            sine = arguments.sines.as_torch().float()[None, :, None, :]
            output[:2, ..., :half] = first * cosine - second * sine
            output[:2, ..., half : workload.rope_dim] = first * sine + second * cosine
            output[:2, ..., workload.rope_dim :] = normalized[..., workload.rope_dim :]
        else:
            output[:2] = normalized
        output[2] = values[2]
