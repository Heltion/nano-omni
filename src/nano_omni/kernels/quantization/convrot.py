"""Regular Hadamard rotation used by ConvRot checkpoint weights."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload
from nano_omni.core.tensor import DType, TensorDesc


class ConvRotWorkload(Workload):
    num_tokens: int
    columns: int
    group: int


class ConvRotConfig(Config):
    threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class ConvRotArguments(Arguments):
    """BF16 input/output [num_tokens, columns], with independent groups along each row.

    Each group is normalized by sqrt(group); no scale buffer is produced.
    Output occupies exactly num_tokens * columns * 2 bytes, without padded columns.
    """

    input: TensorDesc
    output: TensorDesc
    group: int = 256

    def dynamic_parameters(self) -> tuple[tuple[str, int], ...]:
        return (("num_tokens", self.input.shape[0]),)


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def convrot(num_tokens, columns, group, threads=128):
    import tilelang.language as T

    stages = (group.bit_length() - 1) // 2

    @T.prim_func
    def main(
        num_tokens: num_tokens,
        input: T.Tensor((num_tokens, columns), T.bfloat16),
        output: T.Tensor((num_tokens, columns), T.bfloat16),
    ):
        with T.Kernel(num_tokens * (columns // group), threads=threads) as index:
            row = index // (columns // group)
            block = index % (columns // group)
            values = T.alloc_shared((group,), T.float32)
            updated = T.alloc_shared((group,), T.float32)
            for i in T.Parallel(group):
                values[i] = input[row, block * group + i]
            # H4 has a single negative entry in each row, at column 3-row.
            # Apply its Kronecker factors, then normalize once by sqrt(group).
            for stage in T.unroll(stages):
                for i in T.Parallel(group):
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
                T.copy(updated, values)
            for i in T.Parallel(group):
                output[row, block * group + i] = values[i] / (group**0.5)

    return main.with_attr("global_symbol", f"convrot_{columns}_{group}_{threads}")


class ConvRotKernel(Kernel[ConvRotArguments, ConvRotWorkload, ConvRotConfig]):
    name = "convrot"
    program = convrot

    @classmethod
    def make_arguments(cls, workload: ConvRotWorkload) -> ConvRotArguments:
        shape = (workload.num_tokens, workload.columns)
        return ConvRotArguments(
            TensorDesc.empty(DType.BF16, shape),
            TensorDesc.empty(DType.BF16, shape),
            workload.group,
        )

    @classmethod
    def make_workload(cls, arguments: ConvRotArguments) -> ConvRotWorkload:
        num_tokens, columns = arguments.input.shape
        group = arguments.group
        assert arguments.output.shape == arguments.input.shape
        assert arguments.input.dtype == arguments.output.dtype == DType.BF16
        assert (
            group >= 4 and not group & (group - 1) and not (group.bit_length() - 1) % 2
        ), "ConvRot group must be a power of four"
        assert not columns % group, "ConvRot group must divide the feature dimension"
        return ConvRotWorkload(num_tokens=num_tokens, columns=columns, group=group)

    @classmethod
    def make_config(cls, workload: ConvRotWorkload) -> ConvRotConfig:
        del workload
        assert False, "convrot has no production configuration"

    @classmethod
    def ref_program(cls, arguments: ConvRotArguments) -> None:
        import torch

        source = arguments.input.as_torch()
        base = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float32,
            device=source.device,
        )
        rotation = base
        while rotation.shape[0] < arguments.group:
            rotation = torch.kron(rotation, base)
        rotation /= arguments.group**0.5
        result = source.float().reshape(-1, arguments.group) @ rotation
        arguments.output.as_torch().copy_(result.reshape_as(source))
