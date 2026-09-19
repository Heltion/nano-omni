"""Named checkpoint slices used by explicit kernel weight references."""

import json
import struct

from nano_omni.core.model import ModelMetadata
from nano_omni.core.tensor import TensorDesc, TensorKind


class WeightRegistry:
    """Assign local IDs to checkpoint byte ranges; reference() reads no tensor data."""

    def __init__(self, metadata: ModelMetadata) -> None:
        self.metadata = metadata
        self.entries: list[TensorDesc] = []
        self.identities: dict[tuple[tuple[int, int], int], int] = {}

    def reference(
        self, name: str, *, file_index: int = 0, part: int = 0, parts: int = 1
    ) -> TensorDesc:
        """Return a symbolic, zero-offset view of a checkpoint row partition.

        part is zero-based; a split must divide the first dimension evenly.
        IDs follow first-reference order and are reused for the same file range.
        """
        metadata = self.metadata.files[file_index].weights[name]
        shape = metadata.shape
        if parts <= 0 or not 0 <= part < parts:
            raise ValueError("weight part is outside the requested partition")
        if parts != 1 and (not shape or shape[0] % parts):
            raise ValueError("weight rows must divide into the requested parts")
        size = metadata.num_bytes // parts
        if parts != 1:
            shape = (shape[0] // parts, *shape[1:])
        source_file, source_offset = metadata.info
        position = (source_file, source_offset + part * size)
        key = (position, size)
        if key not in self.identities:
            self.identities[key] = len(self.entries)
            self.entries.append(
                TensorDesc(metadata.dtype, shape, TensorKind.FILE, position)
            )
        return TensorDesc(
            metadata.dtype, shape, TensorKind.WEIGHT, (self.identities[key], 0)
        )

    def scalar(self, name: str, *, file_index: int = 0) -> float:
        """Read one four-byte checkpoint entry as little-endian float32 on the CPU."""
        file = self.metadata.files[file_index]
        metadata = file.weights[name]
        if metadata.num_bytes != 4:
            raise ValueError("expected a float32 checkpoint scalar")
        _, byte_offset = metadata.info
        with file.path.open("rb") as source:
            source.seek(byte_offset)
            return struct.unpack("<f", source.read(4))[0]

    def quantization(self, prefix: str) -> dict[str, object]:
        """Read JSON from the base checkpoint's <prefix>.comfy_quant entry on the CPU."""
        file = self.metadata.files[0]
        metadata = file.weights[prefix + ".comfy_quant"]
        with file.path.open("rb") as source:
            source.seek(metadata.info[1])
            return json.loads(source.read(metadata.num_bytes))
