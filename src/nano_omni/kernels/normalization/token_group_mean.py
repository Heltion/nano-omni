"""Column means over token groups, with explicit zero-padding semantics."""

import dataclasses

import tilelang
from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class TokenGroupMeanWorkload(Workload):
    """Consecutive groups along the token axis and the final-group divisor."""

    num_tokens: int = Field(gt=0)
    columns: int = Field(gt=0)
    tokens_per_block: int = Field(gt=0)
    include_padding: bool = False


class TokenGroupMeanConfig(Config):
    """Feature columns per tile and CUDA threads per token-group block."""

    tile_columns: int = Field(gt=0)
    threads: int = Field(gt=0)


@dataclasses.dataclass(frozen=True, slots=True)
class TokenGroupMeanArguments(Arguments):
    """Contiguous BF16 input [tokens, columns] and F32 per-group means.

    Output is [ceildiv(tokens, group_tokens), columns]. Every output element
    sums its group's valid tokens in F32; input beyond tokens is never read.
    The final divisor is group_tokens when include_padding is true, otherwise
    the number of valid tokens. Padding contributes zero to the sum.
    """

    input: TensorDesc
    output: TensorDesc
    tokens_per_block: int
    include_padding: bool

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (
            ("num_tokens", self.input.shape[0]),
            ("tokens_per_block", self.tokens_per_block),
        )


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def token_group_mean(
    num_tokens,
    columns,
    tokens_per_block,
    include_padding,
    tile_columns=32,
    threads=128,
):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    dynamic_tokens_per_block = T.dynamic("tokens_per_block")
    num_blocks = T.ceildiv(dynamic_num_tokens, dynamic_tokens_per_block)

    # Current NVRTC codegen takes the address of a temporary float2 when
    # broadcasting an F32 scalar into a 256-bit vector. Keep 128-bit vectors.
    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        tokens_per_block: dynamic_tokens_per_block,
        input: T.Tensor([dynamic_num_tokens, columns], T.bfloat16),
        output: T.Tensor([num_blocks, columns], T.float32),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(
            num_blocks, T.ceildiv(columns, tile_columns), threads=threads
        ) as (group, column_block):
            values = T.alloc_fragment([128, tile_columns], T.float32)
            partial = T.alloc_fragment([tile_columns], T.float32)
            total = T.alloc_fragment([tile_columns], T.float32)
            T.clear(total)
            for block in T.serial(T.ceildiv(dynamic_tokens_per_block, 128)):
                for row, column in T.Parallel(128, tile_columns):
                    local_row = block * 128 + row
                    token = group * dynamic_tokens_per_block + local_row
                    channel = column_block * tile_columns + column
                    values[row, column] = T.if_then_else(
                        (local_row < dynamic_tokens_per_block)
                        & (token < dynamic_num_tokens)
                        & (channel < columns),
                        input[token, channel],
                        0.0,
                    )
                T.reduce_sum(values, partial, dim=0)
                for column in T.Parallel(tile_columns):
                    total[column] += partial[column]
            for column in T.Parallel(tile_columns):
                channel = column_block * tile_columns + column
                if channel < columns:
                    if include_padding:
                        output[group, channel] = total[column] * (
                            1.0 / dynamic_tokens_per_block
                        )
                    else:
                        output[group, channel] = total[column] * (
                            1.0
                            / T.min(
                                dynamic_tokens_per_block,
                                dynamic_num_tokens - group * dynamic_tokens_per_block,
                            )
                        )

    return main.with_attr(
        "global_symbol",
        f"token_group_mean_{columns}_{include_padding}_{threads}_{tile_columns}",
    )


class TokenGroupMeanKernel(
    Kernel[TokenGroupMeanArguments, TokenGroupMeanWorkload, TokenGroupMeanConfig]
):
    name = "token_group_mean"
    program = token_group_mean

    @classmethod
    def make_arguments(
        cls, workload: TokenGroupMeanWorkload
    ) -> TokenGroupMeanArguments:
        """Describe grouped BF16 rows and their FP32 column means."""
        num_blocks = -(-workload.num_tokens // workload.tokens_per_block)
        return TokenGroupMeanArguments(
            input=TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.columns)),
            output=TensorDesc.empty(DType.F32, (num_blocks, workload.columns)),
            tokens_per_block=workload.tokens_per_block,
            include_padding=workload.include_padding,
        )

    @classmethod
    def make_workload(
        cls, arguments: TokenGroupMeanArguments
    ) -> TokenGroupMeanWorkload:
        """Recover the workload and validate grouped mean storage."""
        assert len(arguments.input.shape) == 2
        num_tokens, columns = arguments.input.shape
        assert arguments.tokens_per_block > 0
        num_blocks = -(-num_tokens // arguments.tokens_per_block)
        assert arguments.input.dtype == DType.BF16
        assert arguments.output.dtype == DType.F32
        assert arguments.output.shape == (num_blocks, columns)
        return TokenGroupMeanWorkload(
            num_tokens=num_tokens,
            columns=columns,
            tokens_per_block=arguments.tokens_per_block,
            include_padding=arguments.include_padding,
        )

    @classmethod
    def ref_program(cls, arguments: TokenGroupMeanArguments) -> None:
        """Compute each logical token-group mean in FP32."""
        import torch

        workload = cls.make_workload(arguments)
        values = arguments.input.as_torch().float()
        num_blocks = arguments.output.shape[0]
        padded_tokens = num_blocks * workload.tokens_per_block
        if padded_tokens != workload.num_tokens:
            values = torch.nn.functional.pad(
                values, (0, 0, 0, padded_tokens - workload.num_tokens)
            )
        totals = values.view(
            num_blocks, workload.tokens_per_block, workload.columns
        ).sum(1)
        if workload.include_padding:
            divisors = workload.tokens_per_block
        else:
            starts = (
                torch.arange(num_blocks, device=values.device)
                * workload.tokens_per_block
            )
            divisors = (workload.num_tokens - starts).clamp(
                max=workload.tokens_per_block
            )[:, None]
        arguments.output.as_torch().copy_(totals / divisors)

    @classmethod
    def make_config(cls, workload: TokenGroupMeanWorkload) -> TokenGroupMeanConfig:
        """Select the production configuration for this workload."""
        del workload
        return TokenGroupMeanConfig(threads=256, tile_columns=64)
