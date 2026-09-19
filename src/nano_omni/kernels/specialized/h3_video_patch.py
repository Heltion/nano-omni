"""Pack contiguous F32 video into spatial 2x2 patches, or restore its axes."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class H3VideoPatchWorkload(Workload):
    """Channel/frame/spatial extents in elements; full inversion needs even H/W."""

    channels: int
    frames: int
    height: int
    width: int
    reverse: bool


@dataclasses.dataclass(frozen=True, slots=True)
class H3VideoPatchArguments(Arguments):
    """F32 [C, F, H, W] <-> [F * (H // 2) * (W // 2), 4 * C].

    A patch row follows frame, patch-y, patch-x order; its column is
    4 * channel + 2 * (y % 2) + x % 2. reverse swaps input/output layouts.
    With even H/W both views span 4 * C * F * H * W bytes. Slice base addresses
    belong to the caller; no padding or byte-offset adjustment is performed here.
    """

    input: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def h3_video_patch(
    channels, frames, height, width, reverse, tile_elements=256, threads=128
):
    import tilelang.language as T

    patch_height = height // 2
    patch_width = width // 2
    rows = frames * patch_height * patch_width
    columns = channels * 4
    elements = rows * columns
    video_shape = [channels, frames, height, width]
    row_shape = [rows, columns]
    input_shape = row_shape if reverse else video_shape
    output_shape = video_shape if reverse else row_shape

    @T.prim_func
    def main(
        input: T.Tensor(input_shape, T.float32),
        output: T.Tensor(output_shape, T.float32),
    ):
        with T.Kernel(T.ceildiv(elements, tile_elements), threads=threads) as block:
            for offset in T.Parallel(tile_elements):
                linear = block * tile_elements + offset
                if linear < elements:
                    if reverse:
                        channel = linear // (frames * height * width)
                        remaining = linear % (frames * height * width)
                        frame = remaining // (height * width)
                        spatial = remaining % (height * width)
                        y = spatial // width
                        x = spatial % width
                        row = (
                            frame * patch_height * patch_width
                            + (y // 2) * patch_width
                            + x // 2
                        )
                        column = channel * 4 + (y % 2) * 2 + x % 2
                        output[channel, frame, y, x] = input[row, column]
                    else:
                        row = linear // columns
                        column = linear % columns
                        frame = row // (patch_height * patch_width)
                        spatial = row % (patch_height * patch_width)
                        patch_row = spatial // patch_width
                        patch_column = spatial % patch_width
                        channel = column // 4
                        pixel = column % 4
                        y = patch_row * 2 + pixel // 2
                        x = patch_column * 2 + pixel % 2
                        output[row, column] = input[channel, frame, y, x]

    return main.with_attr(
        "global_symbol",
        f"h3_video_patch_{channels}_{frames}_{height}_{width}_{int(reverse)}_{tile_elements}_{threads}",
    )


class H3VideoPatchKernel(
    ElementwiseKernel[H3VideoPatchArguments, H3VideoPatchWorkload]
):
    name = "h3_video_patch"
    program = h3_video_patch
    launch = (1024, 512)

    @classmethod
    def make_arguments(cls, workload: H3VideoPatchWorkload) -> H3VideoPatchArguments:
        """Describe video and patch-matrix layouts in the requested direction."""
        video_shape = (
            workload.channels,
            workload.frames,
            workload.height,
            workload.width,
        )
        row_shape = (
            workload.frames * workload.height * workload.width // 4,
            workload.channels * 4,
        )
        input_shape, output_shape = (
            (row_shape, video_shape) if workload.reverse else (video_shape, row_shape)
        )
        return H3VideoPatchArguments(
            input=TensorDesc.empty(DType.F32, input_shape),
            output=TensorDesc.empty(DType.F32, output_shape),
        )

    @classmethod
    def make_workload(cls, arguments: H3VideoPatchArguments) -> H3VideoPatchWorkload:
        reverse = len(arguments.input.shape) == 2
        video_shape = arguments.output.shape if reverse else arguments.input.shape
        row_shape = arguments.input.shape if reverse else arguments.output.shape
        assert len(video_shape) == 4 and len(row_shape) == 2
        channels, frames, height, width = video_shape
        assert height % 2 == 0 and width % 2 == 0
        assert row_shape == (frames * height * width // 4, channels * 4)
        assert arguments.input.dtype == arguments.output.dtype == DType.F32
        return H3VideoPatchWorkload(
            channels=channels,
            frames=frames,
            height=height,
            width=width,
            reverse=reverse,
        )

    @classmethod
    def ref_program(cls, arguments: H3VideoPatchArguments) -> None:
        """Pack or restore spatial 2x2 patches without changing values."""
        workload = cls.make_workload(arguments)
        source = arguments.input.as_torch()
        output = arguments.output.as_torch()
        if workload.reverse:
            output.copy_(
                source.view(
                    workload.frames,
                    workload.height // 2,
                    workload.width // 2,
                    workload.channels,
                    2,
                    2,
                )
                .permute(3, 0, 1, 4, 2, 5)
                .reshape(output.shape)
            )
        else:
            output.copy_(
                source.view(
                    workload.channels,
                    workload.frames,
                    workload.height // 2,
                    2,
                    workload.width // 2,
                    2,
                )
                .permute(1, 2, 4, 0, 3, 5)
                .reshape(output.shape)
            )
