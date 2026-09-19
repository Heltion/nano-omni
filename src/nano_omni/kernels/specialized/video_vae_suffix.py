"""Append register tokens and one zero token to decoder patch tokens."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class VideoVaeSuffixWorkload(Workload):
    """Patch/register row counts, feature width, and register dtype."""

    patch_tokens: int
    register_tokens: int
    columns: int
    register_dtype: DType
    dtype: DType = DType.BF16


@dataclasses.dataclass(frozen=True, slots=True)
class VideoVaeSuffixArguments(Arguments):
    """Patch rows, register rows, and zero-terminated output rows."""

    patches: TensorDesc
    registers: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def video_vae_suffix(
    patch_tokens,
    register_tokens,
    columns,
    register_dtype,
    dtype,
    tile_elements=256,
    threads=128,
):
    import tilelang.language as T

    register_type = T.float16 if register_dtype == DType.F16 else T.bfloat16
    storage_type = T.float16 if dtype == DType.F16 else T.bfloat16
    output_tokens = patch_tokens + register_tokens + 1
    patch_elements = patch_tokens * columns
    register_elements = register_tokens * columns
    output_elements = output_tokens * columns

    @T.prim_func
    def main(
        patches: T.Tensor((patch_tokens, columns), storage_type),
        registers: T.Tensor((register_tokens, columns), register_type),
        output: T.Tensor((output_tokens, columns), storage_type),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(
            T.ceildiv(output_elements, tile_elements), threads=threads
        ) as block:
            for local in T.Parallel(tile_elements):
                index = block * tile_elements + local
                row = index // columns
                column = index % columns
                if index < patch_elements:
                    output[row, column] = patches[row, column]
                elif index < patch_elements + register_elements:
                    register_index = index - patch_elements
                    output[row, column] = registers[
                        register_index // columns, register_index % columns
                    ]
                elif index < output_elements:
                    output[row, column] = 0.0

    return main.with_attr(
        "global_symbol",
        f"video_vae_suffix_{patch_tokens}_{register_tokens}_{columns}_{register_dtype.value}_{dtype.value}_{tile_elements}_{threads}",
    )


class VideoVaeSuffixKernel(
    ElementwiseKernel[VideoVaeSuffixArguments, VideoVaeSuffixWorkload]
):
    name = "video_vae_suffix"
    program = video_vae_suffix
    launch = (4096, 256)

    @classmethod
    def make_arguments(
        cls, workload: VideoVaeSuffixWorkload
    ) -> VideoVaeSuffixArguments:
        """Describe patch rows, register rows, and the zero-terminated output."""
        return VideoVaeSuffixArguments(
            patches=TensorDesc.empty(
                workload.dtype, (workload.patch_tokens, workload.columns)
            ),
            registers=TensorDesc.empty(
                workload.register_dtype,
                (workload.register_tokens, workload.columns),
            ),
            output=TensorDesc.empty(
                workload.dtype,
                (
                    workload.patch_tokens + workload.register_tokens + 1,
                    workload.columns,
                ),
            ),
        )

    @classmethod
    def make_workload(
        cls, arguments: VideoVaeSuffixArguments
    ) -> VideoVaeSuffixWorkload:
        assert len(arguments.patches.shape) == len(arguments.registers.shape) == 2
        assert arguments.patches.shape[1] == arguments.registers.shape[1]
        assert arguments.patches.dtype == arguments.output.dtype
        assert arguments.patches.dtype in (DType.F16, DType.BF16)
        workload = VideoVaeSuffixWorkload(
            patch_tokens=arguments.patches.shape[0],
            register_tokens=arguments.registers.shape[0],
            columns=arguments.patches.shape[1],
            register_dtype=arguments.registers.dtype,
            dtype=arguments.patches.dtype,
        )
        assert arguments.output.shape == (
            workload.patch_tokens + workload.register_tokens + 1,
            workload.columns,
        )
        return workload

    @classmethod
    def ref_program(cls, arguments: VideoVaeSuffixArguments) -> None:
        """Append converted register rows and one all-zero row."""
        output = arguments.output.as_torch()
        patch_tokens = arguments.patches.shape[0]
        register_tokens = arguments.registers.shape[0]
        output[:patch_tokens].copy_(arguments.patches.as_torch())
        output[patch_tokens : patch_tokens + register_tokens].copy_(
            arguments.registers.as_torch()
        )
        output[patch_tokens + register_tokens].zero_()
