"""Joint checks for the state contract shared by Sol attention stages."""

import dataclasses
import math
import unittest

from pydantic import ValidationError

from nano_omni.core.kernel import MmaType
from nano_omni.core.tensor import DType, TensorDesc, TensorKind
from nano_omni.kernels.attention.causal_attention import (
    CausalAttentionKernel,
    CausalAttentionWorkload,
)
from nano_omni.kernels.attention.dense_nvfp4 import (
    DenseNvfp4Kernel,
    DenseNvfp4Workload,
)
from nano_omni.kernels.attention.int8_fp8.sol import (
    SolInt8Fp8AttentionArguments,
    SolInt8Fp8AttentionKernel,
    SolInt8Fp8AttentionWorkload,
)
from nano_omni.kernels.attention.nvfp4.sol_finalize import (
    SolNvfp4FinalizeArguments,
    SolNvfp4FinalizeKernel,
)
from nano_omni.kernels.attention.nvfp4.sol_select import (
    SolNvfp4SelectArguments,
    SolNvfp4SelectKernel,
)
from nano_omni.kernels.attention.self_attention import (
    SelfAttentionArguments,
    SelfAttentionKernel,
)
from nano_omni.kernels.attention.sol_nvfp4 import (
    SolNvfp4Arguments,
    SolNvfp4Config,
    SolNvfp4Kernel,
    SolNvfp4Workload,
    _expected_exact_pairs,
)


class SolAttentionStateTest(unittest.TestCase):
    def test_attention_tops_use_causal_and_public_sol_tiles(self) -> None:
        causal = CausalAttentionKernel.make_arguments(
            CausalAttentionWorkload(
                num_padded_tokens=3,
                num_tokens=3,
                heads=2,
                kv_heads=2,
                dim=64,
                dtype=DType.BF16,
            )
        )
        self.assertEqual(
            CausalAttentionKernel.tops(causal),
            {MmaType.BF16BF16F32: 4 * 2 * 6 * 64},
        )

        probability = 0.5 * math.erfc(1 / math.sqrt(2))
        lengths = (64, 64, 2)
        expected = 0.0
        for query_block, query_length in enumerate(lengths):
            for key_block, key_length in enumerate(lengths):
                mandatory = key_block == 0 or abs(query_block - key_block) <= 1
                expected += (
                    query_length * key_length * (1.0 if mandatory else probability)
                )
        self.assertEqual(_expected_exact_pairs(130, 130, 1, 1.0), round(expected))

    def test_sol_selector_reference_includes_zero_variance_epsilon(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        query_value = torch.ones((256, 1, 128), dtype=torch.bfloat16, device="cuda")
        pooled_value = torch.zeros((1, 4, 128), dtype=torch.bfloat16, device="cuda")
        pooled_value[:, 2].fill_(0.00004)  # route is below sqrt(1e-6)
        pooled_value[:, 3].fill_(0.0002)  # route is above sqrt(1e-6)
        stats_value = torch.zeros((1, 2, 128), dtype=torch.float32, device="cuda")
        output_value = torch.empty((4, 1, 4), dtype=torch.uint8, device="cuda")

        def desc(dtype: DType, value: torch.Tensor) -> TensorDesc:
            return TensorDesc(
                dtype,
                tuple(value.shape),
                TensorKind.DEVICE,
                (value.data_ptr(), 0),
            )

        SolNvfp4SelectKernel.ref_program(
            SolNvfp4SelectArguments(
                desc(DType.BF16, query_value),
                desc(DType.BF16, pooled_value),
                desc(DType.F32, stats_value),
                desc(DType.U8, output_value),
                1.0,
                0,
            )
        )
        self.assertEqual(int(output_value[0, 0, 2]), 0)
        self.assertEqual(int(output_value[0, 0, 3]), 1)

    def test_fixed_int8_attention_reference_uses_integer_qk(self) -> None:
        import torch
        from torch.nn import functional

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        query_value = torch.zeros((2, 64), dtype=torch.bfloat16, device="cuda")
        key_value = torch.zeros_like(query_value)
        value_value = (
            torch.arange(128, dtype=torch.float32, device="cuda")
            .reshape(2, 64)
            .to(torch.bfloat16)
        )
        query_value[0, 0:4] = torch.tensor(
            [0.09, -0.09, 10.0, -10.0], dtype=torch.bfloat16, device="cuda"
        )
        query_value[1, 0:4] = torch.tensor(
            [0.11, -0.11, 7.9, -7.9], dtype=torch.bfloat16, device="cuda"
        )
        key_value.copy_(query_value.flip(0))
        output_value = torch.empty_like(query_value)

        def desc(value: torch.Tensor) -> TensorDesc:
            return TensorDesc(
                DType.BF16,
                tuple(value.shape),
                TensorKind.DEVICE,
                (value.data_ptr(), 0),
            )

        arguments = SelfAttentionArguments(
            desc(query_value),
            desc(key_value),
            desc(value_value),
            desc(output_value),
            1,
            64,
            False,
            True,
        )
        SelfAttentionKernel.ref_program(arguments)

        quantized_query = (query_value * 16).clamp(-127, 127).to(torch.int8).to(
            torch.bfloat16
        ) / 16
        quantized_key = (key_value * 16).clamp(-127, 127).to(torch.int8).to(
            torch.bfloat16
        ) / 16
        expected = functional.scaled_dot_product_attention(
            quantized_query.view(2, 1, 64).transpose(0, 1),
            quantized_key.view(2, 1, 64).transpose(0, 1),
            value_value.view(2, 1, 64).transpose(0, 1),
        ).transpose(0, 1)
        torch.testing.assert_close(output_value.view(2, 1, 64), expected)

    def test_sol_nvfp4_public_config_matches_64_token_routing(self) -> None:
        for values in (
            {"tile_query_tokens": 128, "tile_kv_tokens": 64, "threads": 256},
            {"tile_query_tokens": 64, "tile_kv_tokens": 128, "threads": 128},
        ):
            with self.assertRaises(ValidationError):
                SolNvfp4Config(**values)

        arguments = SolNvfp4Kernel.make_arguments(
            SolNvfp4Workload(
                num_query_tokens=65,
                num_kv_tokens=65,
                heads=1,
                total_heads=1,
            )
        )
        with self.assertRaises(AssertionError):
            SolNvfp4Kernel.make_workload(
                dataclasses.replace(arguments, num_protected_blocks=0)
            )

    def test_padded_capacity_is_part_of_nvfp4_abi(self) -> None:
        for kernel, workload in (
            (
                DenseNvfp4Kernel,
                DenseNvfp4Workload(
                    num_query_tokens=65, num_kv_tokens=65, heads=1, total_heads=1
                ),
            ),
            (
                SolNvfp4Kernel,
                SolNvfp4Workload(
                    num_query_tokens=65, num_kv_tokens=65, heads=1, total_heads=1
                ),
            ),
        ):
            arguments = kernel.make_arguments(workload)
            oversized_query = dataclasses.replace(arguments.query, shape=(1, 256, 64))
            oversized_query_scales = dataclasses.replace(
                arguments.query_scales, shape=(1, 256, 8)
            )
            oversized = dataclasses.replace(
                arguments,
                query=oversized_query,
                query_scales=oversized_query_scales,
            )
            with self.assertRaises(AssertionError):
                kernel.make_workload(oversized)

    def test_int8_sol_rejects_oversized_query_and_non_f32_scales(self) -> None:
        arguments = SolInt8Fp8AttentionKernel.make_arguments(
            SolInt8Fp8AttentionWorkload(num_tokens=65, heads=1, total_heads=1)
        )
        oversized = dataclasses.replace(
            arguments,
            query=dataclasses.replace(arguments.query, shape=(1, 128, 128)),
            query_scale=dataclasses.replace(arguments.query_scale, shape=(1, 4)),
        )
        with self.assertRaises(AssertionError):
            SolInt8Fp8AttentionKernel.make_workload(oversized)

        wrong_scale_dtype = dataclasses.replace(
            arguments,
            query_scale=dataclasses.replace(arguments.query_scale, dtype=DType.F16),
        )
        with self.assertRaises(AssertionError):
            SolInt8Fp8AttentionKernel.make_workload(wrong_scale_dtype)

    def test_int8_exact_and_finalize_share_raw_qk_maximum(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        device = "cuda"
        num_tokens = 70
        num_blocks = 2
        heads = total_heads = 1
        owners: list[torch.Tensor] = []

        def tensor(dtype: DType, value: torch.Tensor) -> TensorDesc:
            value = value.contiguous()
            owners.append(value)
            return TensorDesc(
                dtype, tuple(value.shape), TensorKind.DEVICE, (value.data_ptr(), 0)
            )

        query = tensor(
            DType.I8,
            torch.ones((heads, 96, 128), dtype=torch.int8, device=device),
        )
        key = tensor(
            DType.I8,
            torch.ones((heads, 128, 128), dtype=torch.int8, device=device),
        )
        value = tensor(
            DType.FP8_E4M3,
            torch.ones((heads, 128, 128), dtype=torch.float8_e4m3fn, device=device),
        )
        query_scale = tensor(DType.F32, torch.ones((heads, 3), device=device))
        key_scale = tensor(DType.F32, torch.ones((heads, 2), device=device))
        value_scale = tensor(DType.F32, torch.ones((heads, 128), device=device))
        selected_value = torch.zeros(
            (num_blocks, heads, num_blocks), dtype=torch.uint8, device=device
        )
        selected_value[0, 0, 0] = 1
        selected_value[1, 0, 1] = 1
        selected = tensor(DType.U8, selected_value)
        exact_value = torch.empty(
            (num_tokens, total_heads, 128), dtype=torch.bfloat16, device=device
        )
        state_value = torch.empty(
            (num_tokens, total_heads, 2), dtype=torch.float32, device=device
        )
        exact = tensor(DType.BF16, exact_value)
        state = tensor(DType.F32, state_value)

        exact_arguments = SolInt8Fp8AttentionArguments(
            query,
            key,
            value,
            query_scale,
            key_scale,
            value_scale,
            selected,
            exact,
            state,
            num_tokens,
            0,
            1.0,
            0,
        )
        SolInt8Fp8AttentionKernel.ref_program(exact_arguments)

        # Both a full exact block and the six-token tail have raw QK maximum 128.
        torch.testing.assert_close(
            state_value[:, 0, 0], torch.full((num_tokens,), 128.0, device=device)
        )

        pooled_value = torch.zeros(
            (3, heads, num_blocks, 128), dtype=torch.bfloat16, device=device
        )
        pooled_value[1].fill_(127 / 128)
        pooled_value[2, :, 0].fill_(128)
        pooled_value[2, :, 1].fill_(12)
        pooled = tensor(DType.BF16, pooled_value)
        finalize_query_value = torch.ones(
            (num_tokens, heads, 128), dtype=torch.bfloat16, device=device
        )
        finalize_query = tensor(DType.BF16, finalize_query_value)
        output_value = torch.empty_like(exact_value)
        output = tensor(DType.BF16, output_value)

        expected = torch.empty_like(output_value)
        scale = 128**-0.5
        for query_block, (start, stop) in enumerate(((0, 64), (64, 70))):
            approximate_block = 1 - query_block
            approximate_length = 6 if approximate_block == 1 else 64
            score = float(
                finalize_query_value[start, 0].float()
                @ pooled_value[1, 0, approximate_block].float()
            )
            maximum = torch.maximum(
                state_value[start:stop, 0, 0],
                torch.full((stop - start,), score, device=device),
            )
            exact_correction = torch.exp2(
                (state_value[start:stop, 0, 0] - maximum) * scale * math.log2(math.e)
            )
            approximate_probability = torch.exp2(
                (torch.full_like(maximum, score) - maximum) * scale * math.log2(math.e)
            )
            numerator = exact_value[start:stop, 0].float() * exact_correction[:, None]
            numerator += (
                approximate_probability[:, None]
                * pooled_value[2, 0, approximate_block].float()
            )
            denominator = state_value[start:stop, 0, 1] * exact_correction
            denominator += approximate_probability * approximate_length
            expected[start:stop, 0] = (numerator / denominator[:, None]).to(
                torch.bfloat16
            )

        SolNvfp4FinalizeKernel.ref_program(
            SolNvfp4FinalizeArguments(
                finalize_query,
                pooled,
                selected,
                exact,
                state,
                output,
                0,
            )
        )
        torch.testing.assert_close(output_value, expected, rtol=0.02, atol=0.02)

    def test_nvfp4_exact_and_finalize_share_probability_scale(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        device = "cuda"
        num_tokens = 130
        num_blocks = 3
        heads = total_heads = 1
        owners: list[torch.Tensor] = []

        def tensor(dtype: DType, value: torch.Tensor) -> TensorDesc:
            value = value.contiguous()
            owners.append(value)
            return TensorDesc(
                dtype, tuple(value.shape), TensorKind.DEVICE, (value.data_ptr(), 0)
            )

        # FP4 code 2 represents +1; both nibbles therefore decode to one.
        packed = torch.full((heads, 256, 64), 0x22, dtype=torch.uint8, device=device)
        packed_value = torch.full(
            (heads, 128, 128), 0x22, dtype=torch.uint8, device=device
        )
        scales = torch.ones((heads, 256, 8), dtype=torch.float8_e4m3fn, device=device)
        value_scales = torch.ones(
            (heads, 128, 16), dtype=torch.float8_e4m3fn, device=device
        )
        selected_value = torch.zeros(
            (num_blocks, heads, num_blocks), dtype=torch.uint8, device=device
        )
        selected_value[:, :, 0] = 1
        exact_value = torch.empty(
            (num_tokens, total_heads, 128), dtype=torch.bfloat16, device=device
        )
        state_value = torch.empty(
            (num_tokens, total_heads, 2), dtype=torch.float32, device=device
        )
        selected = tensor(DType.U8, selected_value)
        exact = tensor(DType.BF16, exact_value)
        state = tensor(DType.F32, state_value)
        SolNvfp4Kernel.ref_program(
            SolNvfp4Arguments(
                tensor(DType.U8, packed),
                tensor(DType.U8, packed.clone()),
                tensor(DType.U8, packed_value),
                tensor(DType.FP8_UE4M3, scales),
                tensor(DType.FP8_UE4M3, scales.clone()),
                tensor(DType.FP8_UE4M3, value_scales),
                selected,
                exact,
                state,
                num_tokens,
                num_tokens,
                0,
                1.0,
                1,
            )
        )

        # The exact stage publishes ordinary softmax numerator and denominator,
        # rather than its internal 6 * 448 probability encoding.
        torch.testing.assert_close(
            state_value[:, 0, 1], torch.full((num_tokens,), 64.0, device=device)
        )
        torch.testing.assert_close(
            exact_value[:, 0].float(),
            torch.full((num_tokens, 128), 64.0, device=device),
        )

        query_value = torch.ones(
            (num_tokens, heads, 128), dtype=torch.bfloat16, device=device
        )
        pooled_value = torch.zeros(
            (3, heads, num_blocks, 128), dtype=torch.bfloat16, device=device
        )
        pooled_value[1].fill_(1)
        lengths = (64, 64, 2)
        for block, length in enumerate(lengths):
            pooled_value[2, :, block].fill_(length * 2)
        output_value = torch.empty_like(exact_value)
        SolNvfp4FinalizeKernel.ref_program(
            SolNvfp4FinalizeArguments(
                tensor(DType.BF16, query_value),
                tensor(DType.BF16, pooled_value),
                selected,
                exact,
                state,
                tensor(DType.BF16, output_value),
                0,
            )
        )

        # Equal logits give every token equal probability: block zero has V=1,
        # while both pooled approximate blocks have V=2.
        expected = (64 + (64 + 2) * 2) / num_tokens
        torch.testing.assert_close(
            output_value.float(),
            torch.full_like(output_value.float(), expected),
            rtol=0.02,
            atol=0.02,
        )


if __name__ == "__main__":
    unittest.main()
