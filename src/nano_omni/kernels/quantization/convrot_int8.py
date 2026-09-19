"""ConvRot and rowwise INT8 quantization without a global BF16 intermediate."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class ConvRotInt8Workload(Workload):
    num_tokens: int
    columns: int
    group: int


class ConvRotInt8Config(Config):
    threads: int = 256


@dataclasses.dataclass(frozen=True, slots=True)
class ConvRotInt8Arguments(Arguments):
    """BF16 input and INT8 output [rows, columns], plus F32 scales [rows].

    Rotation acts on each group, but quantization uses one scale for the whole
    row after BF16 rounding: max(absmax(row) / 127, 1e-30). Only local scratch
    pads columns to a power of two; global writes are rows * columns output
    bytes and rows * 4 scale bytes.
    """

    input: TensorDesc
    output: TensorDesc
    scales: TensorDesc
    group: int = 256

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def convrot_int8(num_tokens, columns, group, threads=256):
    import tilelang.language as T

    dynamic_num_tokens = T.dynamic("num_tokens")
    width = columns
    stages = (group.bit_length() - 1) // 2

    @T.prim_func
    def main(
        num_tokens: dynamic_num_tokens,
        input: T.Tensor((dynamic_num_tokens, columns), T.bfloat16),
        output: T.Tensor((dynamic_num_tokens, columns), T.int8),
        scales: T.Tensor((dynamic_num_tokens,), T.float32),
    ):
        with T.Kernel(dynamic_num_tokens, threads=threads) as row:
            values = T.alloc_shared((width,), T.float32)
            updated = T.alloc_fragment((width,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            for i in T.Parallel(width):
                values[i] = T.if_then_else(i < columns, input[row, i], 0.0)
            # Each group rotates independently; only the final scale spans the row.
            for stage in T.unroll(stages):
                for i in T.Parallel(width):
                    stride = 1 << (2 * stage)
                    base = i // (4 * stride) * (4 * stride) + i % stride
                    digit = i // stride % 4
                    updated[i] = (
                        values[base]
                        + values[base + stride]
                        + values[base + 2 * stride]
                        + values[base + 3 * stride]
                        - 2.0 * values[base + (3 - digit) * stride]
                    )
                if stage + 1 < stages:
                    T.copy(updated, values)
            for i in T.Parallel(width):
                # Preserve the unfused BF16 rounding boundary before finding scale.
                updated[i] = T.cast(
                    T.cast(updated[i] / (group**0.5), T.bfloat16), T.float32
                )
            T.reduce_absmax(updated, maximum)
            scales[row] = T.max(maximum[0] / 127.0, 1e-30)
            for i in T.Parallel(width):
                if i < columns:
                    output[row, i] = T.cast(
                        T.max(
                            -127.0,
                            T.min(
                                127.0,
                                T.nearbyint(
                                    updated[i] / T.max(maximum[0] / 127.0, 1e-30)
                                ),
                            ),
                        ),
                        T.int8,
                    )

    return main.with_attr("global_symbol", f"convrot_int8_{columns}_{group}_{threads}")


class ConvRotInt8Kernel(
    Kernel[ConvRotInt8Arguments, ConvRotInt8Workload, ConvRotInt8Config]
):
    name = "convrot_int8"
    program = convrot_int8

    @classmethod
    def make_arguments(cls, workload: ConvRotInt8Workload) -> ConvRotInt8Arguments:
        """Describe the BF16 input, rotated INT8 output, and row scales."""
        return ConvRotInt8Arguments(
            input=TensorDesc.empty(DType.BF16, (workload.num_tokens, workload.columns)),
            output=TensorDesc.empty(DType.I8, (workload.num_tokens, workload.columns)),
            scales=TensorDesc.empty(DType.F32, (workload.num_tokens,)),
            group=workload.group,
        )

    @classmethod
    def make_workload(cls, arguments: ConvRotInt8Arguments) -> ConvRotInt8Workload:
        num_tokens, columns = arguments.input.shape
        group = arguments.group
        assert (
            group >= 4
            and not group & (group - 1)
            and not (group.bit_length() - 1) % 2
            and not columns % group
        ), "ConvRot requires a power-of-four group dividing columns"
        return ConvRotInt8Workload(num_tokens=num_tokens, columns=columns, group=group)

    @classmethod
    def ref_program(cls, arguments: ConvRotInt8Arguments) -> None:
        """Apply the group rotation, BF16 boundary, and rowwise INT8 scaling."""
        import torch

        input = arguments.input.as_torch()
        output = arguments.output.as_torch()
        scales = arguments.scales.as_torch()
        group = arguments.group
        base = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float32,
            device=input.device,
        )
        rotation = base
        while rotation.shape[0] < group:
            rotation = torch.kron(rotation, base)
        rotation /= group**0.5

        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            values = (
                (input.float().reshape(-1, group) @ rotation)
                .bfloat16()
                .float()
                .reshape_as(input)
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        scale = (values.abs().amax(dim=1) / 127).clamp_min(1e-30)
        output.copy_((values / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
        scales.copy_(scale)

    @classmethod
    def make_config(cls, workload: ConvRotInt8Workload) -> ConvRotInt8Config:
        if workload.columns == 14336:
            threads = 512
        elif workload.columns == 5376 and workload.num_tokens <= 32:
            threads = 256
        else:
            threads = 128
        return ConvRotInt8Config(threads=threads)
