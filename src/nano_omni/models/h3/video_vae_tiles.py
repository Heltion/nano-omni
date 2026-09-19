"""Shared spatial tiling geometry for video VAE encoding and decoding."""

from itertools import accumulate


def spatial_tiles(
    output_size: int, tile_size: int = 256
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return starts in latent pixels and overlaps in decoded pixels.

    Overlap grows in 16-pixel increments,
    distributed from the first boundary in the original round-robin order.
    """
    assert tile_size % 16 == 0 and tile_size >= 64
    if output_size <= tile_size:
        return (0,), ()
    step = tile_size - 64
    count = (output_size - 64 + step - 1) // step
    increments = (step * count + 64 - output_size) // 16
    shared, extra = divmod(increments, count - 1)
    overlaps = tuple(64 + 16 * (shared + (i < extra)) for i in range(count - 1))
    starts = (0, *accumulate((tile_size - overlap) // 16 for overlap in overlaps))
    return starts, overlaps
