"""BF16-routed Sol attention with INT8 QK and FP8 PV exact num_blocks."""

import dataclasses
from typing import Literal

from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, MmaType, Tops, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.attention.sol_int8_fp8_exact import sol_int8_fp8_exact


def _mandatory_pairs(num_tokens: int, num_protected_blocks: int) -> int:
    """Count protected and three-block-neighborhood token pairs in O(1)."""
    num_blocks = -(-num_tokens // 64)
    protected = min(num_protected_blocks, num_blocks)
    protected_tokens = min(protected * 64, num_tokens)

    # Count ordered block pairs (query, key) in the local three-block band,
    # excluding protected key num_blocks which were counted above.
    key_blocks = num_blocks - protected
    pair_blocks = key_blocks
    pair_blocks += num_blocks - max(protected, 1)
    pair_blocks += max(num_blocks - 1 - protected, 0)
    local_pairs = pair_blocks * 64 * 64

    # Every block is full except the final one. Correct the constant-size set
    # of local pairs that touches that partial block.
    final = num_blocks - 1
    final_length = num_tokens - final * 64
    affected = {
        (query, key)
        for query in range(max(0, final - 1), num_blocks)
        for key in range(max(protected, query - 1), min(num_blocks, query + 2))
        if query == final or key == final
    }
    for query, key in affected:
        query_length = final_length if query == final else 64
        key_length = final_length if key == final else 64
        local_pairs += query_length * key_length - 64 * 64
    return protected_tokens * num_tokens + local_pairs


class SolInt8Fp8AttentionWorkload(Workload):
    num_tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    total_heads: int = Field(gt=0)


class SolInt8Fp8AttentionConfig(Config):
    tile_query_tokens: Literal[64] = 64
    tile_kv_tokens: Literal[64] = 64
    threads: Literal[64, 128] = 64


@dataclasses.dataclass(frozen=True, slots=True)
class SolInt8Fp8AttentionArguments(Arguments):
    query: TensorDesc
    key: TensorDesc
    value: TensorDesc
    query_scale: TensorDesc
    key_scale: TensorDesc
    value_scale: TensorDesc
    selected: TensorDesc
    output: TensorDesc
    state: TensorDesc
    num_tokens: int
    head_offset: int
    tau: float
    num_protected_blocks: int

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.num_tokens), ("head_offset", self.head_offset))


class SolInt8Fp8AttentionKernel(
    Kernel[
        SolInt8Fp8AttentionArguments,
        SolInt8Fp8AttentionWorkload,
        SolInt8Fp8AttentionConfig,
    ]
):
    """Compute exact selected num_blocks and expose online-softmax state."""

    name = "sol_int8_fp8_attention_d128"
    program = sol_int8_fp8_exact

    @classmethod
    def make_arguments(
        cls, workload: SolInt8Fp8AttentionWorkload
    ) -> SolInt8Fp8AttentionArguments:
        num_blocks = -(-workload.num_tokens // 64)
        qrows = -(-workload.num_tokens // 32) * 32
        krows = num_blocks * 64
        return SolInt8Fp8AttentionArguments(
            TensorDesc.empty(DType.I8, (workload.heads, qrows, 128)),
            TensorDesc.empty(DType.I8, (workload.heads, krows, 128)),
            TensorDesc.empty(DType.FP8_E4M3, (workload.heads, 128, krows)),
            TensorDesc.empty(DType.F32, (workload.heads, qrows // 32)),
            TensorDesc.empty(DType.F32, (workload.heads, num_blocks)),
            TensorDesc.empty(DType.F32, (workload.heads, 128)),
            TensorDesc.empty(DType.U8, (num_blocks, workload.heads, num_blocks)),
            TensorDesc.empty(
                DType.BF16, (workload.num_tokens, workload.total_heads, 128)
            ),
            TensorDesc.empty(DType.F32, (workload.num_tokens, workload.total_heads, 2)),
            workload.num_tokens,
            0,
            1.0,
            0,
        )

    @classmethod
    def make_config(
        cls, workload: SolInt8Fp8AttentionWorkload
    ) -> SolInt8Fp8AttentionConfig:
        del workload
        return SolInt8Fp8AttentionConfig()

    @classmethod
    def make_workload(
        cls, arguments: SolInt8Fp8AttentionArguments
    ) -> SolInt8Fp8AttentionWorkload:
        heads, qrows, dim = arguments.query.shape
        num_blocks = -(-arguments.num_tokens // 64)
        expected_qrows = -(-arguments.num_tokens // 32) * 32
        krows = num_blocks * 64
        assert arguments.query.dtype == arguments.key.dtype == DType.I8 and dim == 128
        assert qrows == expected_qrows
        assert arguments.key.shape == (heads, krows, 128)
        assert arguments.value.dtype == DType.FP8_E4M3
        assert arguments.value.shape == (heads, 128, krows)
        assert arguments.query_scale.dtype == DType.F32
        assert arguments.key_scale.dtype == DType.F32
        assert arguments.value_scale.dtype == DType.F32
        assert arguments.query_scale.shape == (heads, qrows // 32)
        assert arguments.key_scale.shape == (heads, num_blocks)
        assert arguments.value_scale.shape == (heads, 128)
        assert arguments.selected.dtype == DType.U8
        assert arguments.selected.shape == (num_blocks, heads, num_blocks)
        assert arguments.output.dtype == DType.BF16
        assert arguments.state.dtype == DType.F32
        num_tokens, total_heads, output_dim = arguments.output.shape
        assert num_tokens == arguments.num_tokens and output_dim == 128
        assert arguments.state.shape == (num_tokens, total_heads, 2)
        assert 0 <= arguments.head_offset <= total_heads - heads
        assert arguments.tau == 1.0
        assert 0 <= arguments.num_protected_blocks <= num_blocks
        return SolInt8Fp8AttentionWorkload(
            num_tokens=num_tokens,
            heads=heads,
            total_heads=total_heads,
        )

    @classmethod
    def tops(cls, arguments: SolInt8Fp8AttentionArguments) -> Tops:
        workload = cls.make_workload(arguments)
        probability = 0.15865525393145707
        mandatory = _mandatory_pairs(
            workload.num_tokens, arguments.num_protected_blocks
        )
        exact_pairs = mandatory + probability * (workload.num_tokens**2 - mandatory)
        operations = round(2 * workload.heads * 128 * exact_pairs)
        return {MmaType.I8I8I32: operations, MmaType.F8F8F16: operations}

    @classmethod
    def ref_program(cls, arguments: SolInt8Fp8AttentionArguments) -> None:
        """Evaluate selected exact num_blocks and their unnormalized softmax state."""
        import torch

        workload = cls.make_workload(arguments)
        query = (
            arguments.query.as_torch().float().reshape(workload.heads, -1, 32, 128)
            * arguments.query_scale.as_torch()[:, :, None, None]
        ).reshape(workload.heads, -1, 128)[:, : workload.num_tokens]
        key = (
            arguments.key.as_torch().float().reshape(workload.heads, -1, 64, 128)
            * arguments.key_scale.as_torch()[:, :, None, None]
        ).reshape(workload.heads, -1, 128)[:, : workload.num_tokens]
        num_padded_tokens = arguments.value.shape[2]
        index = torch.arange(num_padded_tokens, device="cuda")
        logical = (
            index // 16 * 16 + index % 16 // 4 * 2 + index % 2 + index % 4 // 2 * 8
        )
        inverse = torch.argsort(logical)[: workload.num_tokens]
        value = arguments.value.as_torch()[:, :, inverse].float().transpose(1, 2)
        value_scale = arguments.value_scale.as_torch()
        selected = arguments.selected.as_torch().bool()
        output = arguments.output.as_torch()
        state = arguments.state.as_torch()
        scale = 128**-0.5
        for block in range(-(-workload.num_tokens // 64)):
            start, stop = block * 64, min((block + 1) * 64, workload.num_tokens)
            for head in range(workload.heads):
                maximum = torch.full(
                    (stop - start,), -torch.inf, device="cuda", dtype=torch.float32
                )
                denominator = torch.zeros_like(maximum)
                numerator = torch.zeros(
                    (stop - start, 128), device="cuda", dtype=torch.float32
                )
                for key_block in torch.nonzero(selected[block, head]).flatten():
                    key_start = int(key_block) * 64
                    key_stop = min(key_start + 64, workload.num_tokens)
                    # Keep the public state maximum in raw QK units.  Scaling
                    # belongs to the softmax update and to the shared finalize.
                    scores = query[head, start:stop] @ key[head, key_start:key_stop].T
                    next_maximum = torch.maximum(maximum, scores.amax(1))
                    correction = torch.exp((maximum - next_maximum) * scale)
                    probability = torch.exp((scores - next_maximum[:, None]) * scale)
                    probability_fp8 = (probability * 448).to(torch.float8_e4m3fn)
                    partial = (
                        (probability_fp8.float() @ value[head, key_start:key_stop])
                        .half()
                        .float()
                    )
                    numerator = numerator * correction[:, None] + (
                        partial * value_scale[head] / 448
                    )
                    denominator = denominator * correction + probability.sum(1)
                    maximum = next_maximum
                target = head + arguments.head_offset
                output[start:stop, target].copy_(numerator)
                state[start:stop, target, 0].copy_(maximum)
                state[start:stop, target, 1].copy_(denominator)
