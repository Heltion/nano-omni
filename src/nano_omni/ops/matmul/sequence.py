"""Direct kernel emission and shared preparation of one unchanged input."""

from __future__ import annotations

import dataclasses

from nano_omni.core.kernel import Arguments, Kernel
from nano_omni.core.layout import ScratchLayout
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.matmul.matmul_bf16 import MatmulBf16Arguments, MatmulBf16Kernel
from nano_omni.kernels.matmul.matmul_fp8 import MatmulFp8Arguments, MatmulFp8Kernel
from nano_omni.kernels.matmul.matmul_int8 import MatmulInt8Arguments, MatmulInt8Kernel
from nano_omni.kernels.quantization.convrot import ConvRotArguments, ConvRotKernel
from nano_omni.kernels.quantization.convrot_int8 import (
    ConvRotInt8Arguments,
    ConvRotInt8Kernel,
)
from nano_omni.kernels.quantization.fp8_scale import Fp8ScaleArguments, Fp8ScaleKernel
from nano_omni.kernels.quantization.int8_rowwise import (
    Int8RowwiseArguments,
    Int8RowwiseKernel,
)
from nano_omni.kernels.quantization.quantize_fp8 import (
    QuantizeFp8Arguments,
    QuantizeFp8Kernel,
)


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedInput:
    """Reuse preparation only while this input remains unchanged in one sequence.

    Sharing is explicit: callers retain this object across sibling projections.
    Quantization keys include its mode and parameters; LoRA keys are full views.
    Prepared buffers remain owned by the sequence's append-only scratch storage.
    """

    sequence: KernelSequence
    input: TensorDesc
    _quantized: dict[
        tuple[DType, TensorDesc | None, int], tuple[TensorDesc, TensorDesc | None]
    ] = dataclasses.field(default_factory=dict, init=False, repr=False)
    _compressed: dict[TensorDesc, TensorDesc] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )

    def compress(self, name: str, weight: TensorDesc) -> TensorDesc:
        if weight not in self._compressed:
            output = self.sequence.temporary(
                f"{name}.down", DType.BF16, (self.input.shape[0], weight.shape[0])
            )
            self.sequence.emit(
                MatmulBf16Kernel,
                MatmulBf16Arguments(
                    self.input, weight, None, None, None, output, 1.0, False
                ),
            )
            self._compressed[weight] = output
        return self._compressed[weight]

    def quantize(
        self,
        name: str,
        dtype: DType,
        input_scale: TensorDesc | None,
        rotation_group: int,
    ) -> tuple[TensorDesc, TensorDesc | None]:
        key = (
            dtype,
            input_scale if dtype == DType.FP8_E4M3 else None,
            rotation_group if dtype == DType.I8 else 0,
        )
        if key not in self._quantized:
            sequence, input = self.sequence, self.input
            quantized = sequence.temporary(f"{name}.quantized", dtype, input.shape)
            scales = None
            if dtype == DType.FP8_E4M3:
                if input_scale is None:
                    scales = sequence.temporary(f"{name}.scale", DType.F32, (1,))
                    sequence.emit(
                        Fp8ScaleKernel, Fp8ScaleArguments(input, None, scales)
                    )
                static_scale = input_scale
                if static_scale is None:
                    assert scales is not None
                    static_scale = scales
                sequence.emit(
                    QuantizeFp8Kernel,
                    QuantizeFp8Arguments(
                        input,
                        scales,
                        quantized,
                        static_scale,
                    ),
                )
            elif dtype == DType.I8:
                scales = sequence.temporary(
                    f"{name}.scales", DType.F32, (input.shape[0],)
                )
                if rotation_group and input.shape[1] <= 16384:
                    sequence.emit(
                        ConvRotInt8Kernel,
                        ConvRotInt8Arguments(input, quantized, scales, rotation_group),
                    )
                else:
                    rotated = input
                    if rotation_group:
                        rotated = sequence.temporary(
                            f"{name}.rotated", DType.BF16, input.shape
                        )
                        sequence.emit(
                            ConvRotKernel,
                            ConvRotArguments(input, rotated, rotation_group),
                        )
                    sequence.emit(
                        Int8RowwiseKernel,
                        Int8RowwiseArguments(rotated, quantized, scales),
                    )
            else:
                raise ValueError(f"unsupported quantized input dtype: {dtype}")
            self._quantized[key] = quantized, scales
        return self._quantized[key]


class KernelSequence:
    """An op owns its named temporary regions and ordered kernel calls."""

    def __init__(self, scratch: ScratchLayout) -> None:
        self.scratch = scratch
        self.calls: list[Kernel] = []

    def temporary(self, name: str, dtype: DType, shape: tuple[int, ...]) -> TensorDesc:
        return self.scratch.reserve(name, dtype, shape)

    def emit(self, kernel: type[Kernel], arguments: Arguments) -> None:
        self.calls.append(kernel(arguments))

    def prepare(self, input: TensorDesc) -> PreparedInput:
        return PreparedInput(self, input)

    def linear(
        self,
        name: str,
        input: TensorDesc | PreparedInput,
        weight: TensorDesc,
        output: TensorDesc,
        *,
        weight_scale: float | TensorDesc = 1.0,
        input_scale: TensorDesc | None = None,
        bias: TensorDesc | None = None,
        residual: TensorDesc | None = None,
        multiply: TensorDesc | None = None,
        use_silu: bool = False,
        loras: tuple[tuple[TensorDesc, TensorDesc, float], ...] = (),
        rotation_group: int = 0,
    ) -> None:
        """Emit a projection; INT8 weights require an explicit weight-scale buffer."""
        prepared = input if isinstance(input, PreparedInput) else self.prepare(input)
        if prepared.sequence is not self:
            raise ValueError("prepared input belongs to a different kernel sequence")
        source = prepared.input
        if source.dtype != DType.BF16 or source.shape[1] != weight.shape[1]:
            raise ValueError("linear expects BF16 input with matching inner dimension")
        if (
            output.shape != (source.shape[0], weight.shape[0])
            or output.dtype != DType.BF16
        ):
            raise ValueError("linear output shape or dtype mismatch")
        int8_weight_scale = None
        if weight.dtype == DType.I8:
            if not isinstance(
                weight_scale,
                TensorDesc,
            ):
                raise TypeError("INT8 linear requires a weight_scale buffer")
            int8_weight_scale = weight_scale
        # Only A(input) is shared; every B update preserves residual order and scale.
        for index, (down, up, strength) in enumerate(loras):
            compressed = prepared.compress(f"{name}.lora{index}", down)
            update = self.temporary(
                f"{name}.lora{index}.up", DType.BF16, output.shape
            )
            self.emit(
                MatmulBf16Kernel,
                MatmulBf16Arguments(
                    compressed, up, None, residual, None, update, strength, False
                ),
            )
            residual = update
        if weight.dtype == DType.BF16:
            self.emit(
                MatmulBf16Kernel,
                MatmulBf16Arguments(
                    source, weight, bias, residual, multiply, output, 1.0, use_silu
                ),
            )
        elif weight.dtype == DType.FP8_E4M3:
            assert isinstance(weight_scale, TensorDesc), (
                "FP8 linear requires a weight-scale tensor"
            )
            quantized, scale = prepared.quantize(name, DType.FP8_E4M3, input_scale, 0)
            self.emit(
                MatmulFp8Kernel,
                MatmulFp8Arguments(
                    quantized,
                    weight,
                    bias,
                    residual,
                    multiply,
                    output,
                    weight_scale,
                    scale,
                    weight_scale if input_scale is None else input_scale,
                    use_silu,
                ),
            )
        elif weight.dtype == DType.I8:
            quantized, scales = prepared.quantize(name, DType.I8, None, rotation_group)
            assert scales is not None and int8_weight_scale is not None
            self.emit(
                MatmulInt8Kernel,
                MatmulInt8Arguments(
                    quantized,
                    weight,
                    bias,
                    residual,
                    multiply,
                    output,
                    int8_weight_scale,
                    scales,
                    use_silu,
                ),
            )
        else:
            raise ValueError(f"unsupported linear weight dtype: {weight.dtype}")
