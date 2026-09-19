"""Tiled H3 head normalization and split-half RoPE with stable reduction order."""

import dataclasses
from typing import Literal

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class H3HeadNormRopeWorkload(Workload):
    tokens: int
    heads: int
    use_rope: bool


class H3HeadNormRopeConfig(Config):
    threads: Literal[128] = 128


@dataclasses.dataclass(frozen=True, slots=True)
class H3HeadNormRopeArguments(Arguments):
    input: TensorDesc
    weight: TensorDesc
    cosines: TensorDesc | None
    sines: TensorDesc | None
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Reuse one program for different token counts."""
        return (("tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def h3_head_norm_rope(tokens: int, heads: int, use_rope: bool, threads: int = 128):
    import tilelang.language as T

    dynamic_tokens = T.dynamic("tokens")
    # Neighboring pairs and the 64-thread sum preserve the established BF16 boundary.
    layout = T.Fragment(
        (32, 128),
        forward_thread_fn=lambda i, j: (i % 2) * 64 + j // 2,
        forward_index_fn=lambda i, j: (i // 2) * 2 + j % 2,
    )

    rotated_layout = T.Fragment(
        (32, 128),
        forward_thread_fn=lambda i, j: (i % 4) * 32 + j % 32,
        forward_index_fn=lambda i, j: (i // 4) * 4 + j // 32,
    )

    @T.prim_func
    def main(
        tokens: dynamic_tokens,
        source: T.Tensor((dynamic_tokens, heads, 128), T.bfloat16),
        weight: T.Tensor((128,), T.bfloat16),
        cosines: T.Tensor((dynamic_tokens, 48), T.float32),
        sines: T.Tensor((dynamic_tokens, 48), T.float32),
        output: T.Tensor((dynamic_tokens, heads, 128), T.bfloat16),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(T.ceildiv(tokens, 32), heads, threads=threads) as (block, head):
            x = T.alloc_fragment((32, 128), T.float32)
            squared = T.alloc_fragment((32, 128), T.float32)
            total = T.alloc_fragment((32,), T.float32)
            normed = T.alloc_shared((32, 128), T.float32)
            rotated = T.alloc_fragment((32, 128), T.bfloat16)
            T.annotate_layout({x: layout, squared: layout, rotated: rotated_layout})
            T.copy(source[block * 32 : block * 32 + 32, head, 0:128], x)
            for i, j in T.Parallel(32, 128):
                squared[i, j] = x[i, j] * x[i, j]
            T.reduce_sum(squared, total, dim=1)
            for i, j in T.Parallel(32, 128):
                normed[i, j] = x[i, j] * T.rsqrt(total[i] / 128 + 1e-5) * weight[j]
            for i, j in T.Parallel(32, 128, coalesced_width=1):
                token = block * 32 + i
                if token < tokens:
                    if use_rope and j < 48:
                        rotated[i, j] = (
                            normed[i, j] * cosines[token, j]
                            - normed[i, j + 48] * sines[token, j]
                        )
                    elif use_rope and j < 96:
                        rotated[i, j] = (
                            normed[i, j - 48] * sines[token, j - 48]
                            + normed[i, j] * cosines[token, j - 48]
                        )
                    else:
                        rotated[i, j] = normed[i, j]
                else:
                    rotated[i, j] = 0
            T.copy(rotated, output[block * 32 : block * 32 + 32, head, 0:128])

    return main.with_attr(
        "global_symbol", f"h3_head_norm_rope_{heads}_{int(use_rope)}_{threads}"
    )


class H3HeadNormRopeKernel(
    Kernel[H3HeadNormRopeArguments, H3HeadNormRopeWorkload, H3HeadNormRopeConfig]
):
    name = "h3_head_norm_rope"
    program = h3_head_norm_rope

    @classmethod
    def make_arguments(
        cls, workload: H3HeadNormRopeWorkload
    ) -> H3HeadNormRopeArguments:
        """Describe 128-wide heads, optional rotary planes, and output."""
        shape = (workload.tokens, workload.heads, 128)
        return H3HeadNormRopeArguments(
            input=TensorDesc.empty(DType.BF16, shape),
            weight=TensorDesc.empty(DType.BF16, (128,)),
            cosines=(
                TensorDesc.empty(DType.F32, (workload.tokens, 48))
                if workload.use_rope
                else None
            ),
            sines=(
                TensorDesc.empty(DType.F32, (workload.tokens, 48))
                if workload.use_rope
                else None
            ),
            output=TensorDesc.empty(DType.BF16, shape),
        )

    @classmethod
    def make_config(
        cls, workload: H3HeadNormRopeWorkload
    ) -> H3HeadNormRopeConfig:
        del workload
        return H3HeadNormRopeConfig()

    @classmethod
    def make_workload(
        cls, arguments: H3HeadNormRopeArguments
    ) -> H3HeadNormRopeWorkload:
        assert arguments.input.dtype == arguments.output.dtype == DType.BF16
        assert arguments.input.shape == arguments.output.shape
        assert len(arguments.input.shape) == 3
        assert arguments.input.shape[2] == 128
        assert arguments.weight.dtype == DType.BF16
        assert arguments.weight.shape == (128,)
        assert (arguments.cosines is None) == (arguments.sines is None)
        if arguments.cosines is not None:
            assert arguments.sines is not None
            assert arguments.cosines.dtype == arguments.sines.dtype == DType.F32
            assert (
                arguments.cosines.shape
                == arguments.sines.shape
                == (
                    arguments.input.shape[0],
                    48,
                )
            )
        return H3HeadNormRopeWorkload(
            tokens=arguments.input.shape[0],
            heads=arguments.input.shape[1],
            use_rope=arguments.cosines is not None,
        )

    @classmethod
    def ref_program(cls, arguments: H3HeadNormRopeArguments) -> None:
        """Normalize each head, apply optional split-half RoPE, and store BF16."""
        import torch

        value = arguments.input.as_torch().float()
        inverse = torch.rsqrt(
            torch.linalg.vector_norm(value, dim=-1).square_() / 128 + 1e-5
        )
        value.mul_(inverse[..., None]).mul_(arguments.weight.as_torch().float())
        output = arguments.output.as_torch()
        use_rope = arguments.cosines is not None
        assert (arguments.sines is not None) == use_rope, (
            "cosine and sine planes must be present together"
        )
        if not use_rope:
            output.copy_(value)
            return
        assert arguments.cosines is not None and arguments.sines is not None
        cosine = arguments.cosines.as_torch()[:, None]
        sine = arguments.sines.as_torch()[:, None]
        first, second = value[..., :48], value[..., 48:96]
        output[..., :48].copy_(first * cosine - second * sine)
        output[..., 48:96].copy_(first * sine + second * cosine)
        output[..., 96:].copy_(value[..., 96:])
