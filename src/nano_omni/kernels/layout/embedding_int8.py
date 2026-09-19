"""Gather and dequantize signed INT8 embedding rows."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class EmbeddingInt8Workload(Workload):
    """Embedding table and padded output dimensions."""

    num_tokens: int
    num_padded_tokens: int
    vocabulary: int
    hidden: int


class EmbeddingInt8Config(Config):
    """Tile dimensions and CUDA thread count."""

    tile_tokens: int
    tile_hidden: int
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class EmbeddingInt8Arguments(Arguments):
    """Token IDs, quantized table, row scales, and padded output."""

    token_ids: TensorDesc
    table: TensorDesc
    scales: TensorDesc
    output: TensorDesc

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        """Pass effective and padded token counts without specializing them."""
        return (
            ("num_tokens", self.token_ids.shape[0]),
            ("num_padded_tokens", self.output.shape[0]),
        )


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def embedding_int8(
    num_tokens,
    num_padded_tokens,
    vocabulary,
    hidden,
    tile_tokens=1,
    tile_hidden=512,
    threads=128,
):
    """Build the tiled embedding gather and row-wise dequantization."""
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    dynamic_num_padded_tokens = T.dynamic("num_padded_tokens")

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        num_padded_tokens: dynamic_num_padded_tokens,
        token_ids: T.Tensor([dynamic_num_tokens], T.uint32),
        table: T.Tensor([vocabulary, hidden], T.int8),
        scales: T.Tensor([vocabulary, 1], T.float32),
        output: T.Tensor([dynamic_num_padded_tokens, hidden], T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(dynamic_num_padded_tokens, tile_tokens),
            T.ceildiv(hidden, tile_hidden),
            threads=threads,
        ) as (token_block, hidden_block):
            for i, j in T.Parallel(tile_tokens, tile_hidden):
                row = token_block * tile_tokens + i
                column = hidden_block * tile_hidden + j
                if row < dynamic_num_tokens and column < hidden:
                    token = token_ids[row]
                    output[row, column] = table[token, column] * scales[token, 0]
                elif row < dynamic_num_padded_tokens and column < hidden:
                    output[row, column] = 0

    return main.with_attr(
        "global_symbol",
        f"embedding_int8_{vocabulary}_{hidden}_{tile_tokens}_{tile_hidden}_{threads}",
    )


class EmbeddingInt8Kernel(
    Kernel[EmbeddingInt8Arguments, EmbeddingInt8Workload, EmbeddingInt8Config]
):
    """Gather INT8 rows, apply their scales, and emit BF16 embeddings."""

    name = "embedding_int8"
    program = embedding_int8

    @classmethod
    def make_arguments(cls, workload: EmbeddingInt8Workload) -> EmbeddingInt8Arguments:
        """Describe the tensors required by the workload."""
        return EmbeddingInt8Arguments(
            token_ids=TensorDesc.empty(DType.U32, (workload.num_tokens,)),
            table=TensorDesc.empty(DType.I8, (workload.vocabulary, workload.hidden)),
            scales=TensorDesc.empty(DType.F32, (workload.vocabulary, 1)),
            output=TensorDesc.empty(
                DType.BF16, (workload.num_padded_tokens, workload.hidden)
            ),
        )

    @classmethod
    def make_workload(cls, arguments: EmbeddingInt8Arguments) -> EmbeddingInt8Workload:
        """Validate the tensor contract and recover its logical dimensions."""
        assert arguments.token_ids.dtype == DType.U32
        assert len(arguments.token_ids.shape) == 1
        assert arguments.table.dtype == DType.I8
        assert len(arguments.table.shape) == 2
        assert arguments.scales.dtype == DType.F32
        assert arguments.scales.shape == (arguments.table.shape[0], 1)
        assert arguments.output.dtype == DType.BF16
        assert len(arguments.output.shape) == 2
        assert arguments.output.shape[1] == arguments.table.shape[1]
        assert arguments.output.shape[0] >= arguments.token_ids.shape[0]
        return EmbeddingInt8Workload(
            num_tokens=arguments.token_ids.shape[0],
            num_padded_tokens=arguments.output.shape[0],
            vocabulary=arguments.table.shape[0],
            hidden=arguments.table.shape[1],
        )

    @classmethod
    def make_config(
        cls, workload: EmbeddingInt8Workload
    ) -> EmbeddingInt8Config:
        """Select the measured production configuration."""
        del workload
        return EmbeddingInt8Config(threads=128, tile_hidden=512, tile_tokens=1)

    @classmethod
    def ref_program(cls, arguments: EmbeddingInt8Arguments) -> None:
        """Evaluate the gather and dequantization directly with Torch."""
        token_ids = arguments.token_ids.as_torch().long()
        output = arguments.output.as_torch()
        output[: token_ids.numel()].copy_(
            arguments.table.as_torch()[token_ids].float()
            * arguments.scales.as_torch()[token_ids]
        )
        output[token_ids.numel() :].zero_()
