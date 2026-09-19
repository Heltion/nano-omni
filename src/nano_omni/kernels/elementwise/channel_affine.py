"""Broadcast per-channel affine weights over channel-first values."""

import dataclasses

import tilelang

from nano_omni.core.kernel import Arguments, Workload
from nano_omni.core.tensor import DType, TensorDesc
from nano_omni.kernels.elementwise.base import ElementwiseKernel


class ChannelAffineWorkload(Workload):
    """Channel matrix shape and input/weight storage types."""

    channels: int
    spatial: int
    input_dtype: DType
    weight_dtype: DType
    output_dtype: DType = DType.BF16


@dataclasses.dataclass(frozen=True, slots=True)
class ChannelAffineArguments(Arguments):
    """Input matrix, per-channel mean/std, and reduced-precision output matrix."""

    input: TensorDesc
    mean: TensorDesc
    std: TensorDesc
    output: TensorDesc


@tilelang.jit(out_idx=[], execution_backend="nvrtc")
def channel_affine(
    channels, spatial, input_dtype, weight_dtype, output_dtype, tile_elements=1024, threads=256
):
    import tilelang.language as T

    elements = channels * spatial
    input_type = T.float32 if input_dtype == DType.F32 else T.bfloat16
    weight_type = T.float16 if weight_dtype == DType.F16 else T.bfloat16
    output_type = T.float16 if output_dtype == DType.F16 else T.bfloat16

    @T.prim_func
    def main(
        input: T.Tensor((elements,), input_type),
        mean: T.Tensor((channels,), weight_type),
        std: T.Tensor((channels,), weight_type),
        output: T.Tensor((elements,), output_type),
    ):
        T.annotate_pass_configs({"tl.disable_vectorize_256": True})
        with T.Kernel(T.ceildiv(spatial, tile_elements), channels, threads=threads) as (
            block,
            channel,
        ):
            for local in T.Parallel(tile_elements):
                spatial_index = block * tile_elements + local
                if spatial_index < spatial:
                    index = channel * spatial + spatial_index
                    output[index] = T.cast(input[index], T.float32) * T.cast(
                        std[channel], T.float32
                    ) + T.cast(mean[channel], T.float32)

    return main.with_attr(
        "global_symbol",
        f"channel_affine_{channels}_{spatial}_{input_dtype.value}_{weight_dtype.value}_{output_dtype.value}_{tile_elements}_{threads}",
    )


class ChannelAffineKernel(
    ElementwiseKernel[ChannelAffineArguments, ChannelAffineWorkload]
):
    name = "channel_affine"
    program = channel_affine
    launch = (1024, 256)

    @classmethod
    def make_arguments(cls, workload: ChannelAffineWorkload) -> ChannelAffineArguments:
        """Describe the flattened channel-by-spatial matrices."""
        matrix = (workload.channels, workload.spatial)
        return ChannelAffineArguments(
            input=TensorDesc.empty(workload.input_dtype, matrix),
            mean=TensorDesc.empty(workload.weight_dtype, (workload.channels,)),
            std=TensorDesc.empty(workload.weight_dtype, (workload.channels,)),
            output=TensorDesc.empty(workload.output_dtype, matrix),
        )

    @classmethod
    def make_workload(cls, arguments: ChannelAffineArguments) -> ChannelAffineWorkload:
        assert len(arguments.input.shape) == 2
        workload = ChannelAffineWorkload(
            channels=arguments.input.shape[0],
            spatial=arguments.input.shape[1],
            input_dtype=arguments.input.dtype,
            weight_dtype=arguments.mean.dtype,
            output_dtype=arguments.output.dtype,
        )
        assert arguments.mean.shape == arguments.std.shape == (workload.channels,)
        assert arguments.mean.dtype == arguments.std.dtype
        assert arguments.output.shape == arguments.input.shape
        assert arguments.output.dtype in (DType.F16, DType.BF16)
        return workload

    @classmethod
    def ref_program(cls, arguments: ChannelAffineArguments) -> None:
        """Apply the per-channel scale and bias before storing the output."""
        arguments.output.as_torch().copy_(
            arguments.input.as_torch().float()
            * arguments.std.as_torch().float()[:, None]
            + arguments.mean.as_torch().float()[:, None]
        )
