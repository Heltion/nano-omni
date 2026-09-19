"""Activation, model scalar, and operator scratch byte layouts."""

import dataclasses

from nano_omni.core.tensor import DType, TensorDesc, TensorKind


def aligned(size: int, alignment: int) -> int:
    """Round a nonnegative byte size up to a positive power-of-two alignment."""
    assert size >= 0, "size must be nonnegative"
    assert alignment > 0 and not alignment & (alignment - 1), (
        "alignment must be a positive power of two"
    )
    return (size + alignment - 1) // alignment * alignment


class ActivationLayout:
    """Collect model activation slot requirements without allocating memory."""

    def __init__(self) -> None:
        self.requirements: dict[int, tuple[int, int]] = {}

    def require(
        self,
        activation: int | tuple[int, int],
        num_bytes: int,
        alignment: int = 256,
    ) -> None:
        identity, offset = (
            (activation, 0) if isinstance(activation, int) else activation
        )
        aligned(num_bytes, alignment)
        assert offset % alignment == 0, (
            "activation slice does not satisfy required alignment"
        )
        old_size, old_alignment = self.requirements.get(identity, (0, 1))
        self.requirements[identity] = (
            max(old_size, offset + num_bytes),
            max(old_alignment, alignment),
        )

    def layout(self) -> tuple[dict[int, int], int]:
        offsets: dict[int, int] = {}
        cursor = 0
        for identity, (size, alignment) in self.requirements.items():
            cursor = aligned(cursor, alignment)
            offsets[identity] = cursor
            cursor += size
        return offsets, aligned(cursor, 256)


@dataclasses.dataclass(frozen=True, slots=True)
class ScalarRegion:
    data: bytes
    tensor: TensorDesc


class ScalarLayout:
    """Pack immutable model-built scalar data into one aligned device arena."""

    def __init__(self) -> None:
        self.regions: list[ScalarRegion] = []
        self._values: dict[tuple[DType, tuple[int, ...], bytes], TensorDesc] = {}
        self.num_bytes = 0

    def reserve(
        self,
        dtype: DType,
        shape: tuple[int, ...],
        data: bytes,
    ) -> TensorDesc:
        """Intern one immutable value in the 16-byte-aligned model arena."""
        key = (dtype, shape, data)
        existing = self._values.get(key)
        if existing is not None:
            return existing
        offset = aligned(self.num_bytes, 16)
        tensor = TensorDesc(dtype, shape, TensorKind.SCALAR, (0, offset))
        assert len(data) == tensor.num_bytes, "scalar data size does not match shape"
        self.regions.append(ScalarRegion(data, tensor))
        self._values[key] = tensor
        self.num_bytes = offset + tensor.num_bytes
        return tensor


class ScratchLayout:
    """Append named tensor regions for one operator and return their descriptors."""

    def __init__(self) -> None:
        self.regions: set[str] = set()
        self.num_bytes = 0

    def reserve(
        self,
        name: str,
        dtype: DType,
        shape: tuple[int, ...],
        alignment: int = 256,
    ) -> TensorDesc:
        assert name not in self.regions, f"scratch region already declared: {name}"
        offset = aligned(self.num_bytes, alignment)
        tensor = TensorDesc.scratch(offset, dtype, shape)
        aligned(tensor.num_bytes, alignment)
        self.regions.add(name)
        self.num_bytes = offset + tensor.num_bytes
        return tensor
