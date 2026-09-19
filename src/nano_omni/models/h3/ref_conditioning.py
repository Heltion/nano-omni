"""Present references in request order, then append the generation prompt."""

import dataclasses
from collections import Counter
from typing import Literal

import numpy
from tokenizers import Tokenizer

from nano_omni.models.h3 import conditioning
from nano_omni.models.h3.conditioning import H3Presentation, H3VisionInput


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceItem:
    """One media label; video visions each represent a timestamped frame pair."""

    kind: Literal["image", "audio", "video"]
    visions: tuple[H3VisionInput, ...] = ()
    timestamps: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "video" and len(self.timestamps) != len(self.visions):
            raise ValueError("each video frame pair requires a timestamp")
        if self.kind == "image" and len(self.visions) != 1:
            raise ValueError("an image reference requires one vision input")
        if self.kind == "audio" and (self.visions or self.timestamps):
            raise ValueError("audio contributes a label to the text presentation")


def build_presentation(
    tokenizer: Tokenizer, prompt: str, items: tuple[ReferenceItem, ...]
) -> H3Presentation:
    """Keep audio labels in text; scatter vision embeddings only into image pads."""
    tokens: list[int] = []
    scatter: list[tuple[int, int]] = []
    modalities: list[tuple[int, int]] = []
    grids: list[tuple[int, int, int]] = []
    counters: Counter[str] = Counter()
    labels = {"image": "Picture", "audio": "Audio", "video": "Video"}
    for item in items:
        counters[item.kind] += 1
        tokens.extend(
            tokenizer.encode(
                f"<{labels[item.kind]} {counters[item.kind]}>: ",
                add_special_tokens=False,
            ).ids
        )
        for index, vision in enumerate(item.visions):
            if item.kind == "video":
                tokens.extend(
                    tokenizer.encode(
                        f"<{item.timestamps[index]:.1f} seconds>",
                        add_special_tokens=False,
                    ).ids
                )
            start, count = len(tokens), vision.merged_tokens
            tokens.append(conditioning.VISION_START)
            tokens.extend([conditioning.IMAGE_PAD] * count)
            tokens.append(conditioning.VISION_END)
            scatter.append((start + 1, count))
            modalities.append((start, count + 2))
            grids.append((1, vision.grid_height, vision.grid_width))
    tokens.extend(tokenizer.encode(prompt, add_special_tokens=False).ids)
    return H3Presentation(
        numpy.asarray(tokens, dtype=numpy.uint32),
        tuple(scatter),
        tuple(modalities),
        conditioning._mrope_positions(len(tokens), tuple(scatter), tuple(grids)),
        1.0 / (5_000_000.0 ** (numpy.arange(0, 128, 2, dtype=numpy.float32) / 128)),
    )
