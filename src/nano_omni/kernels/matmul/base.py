"""Shared matrix dimensions and launch configuration, independent of precision."""

from typing import ClassVar

from pydantic import Field

from nano_omni.core.kernel import Arguments, Config, Kernel, MmaType, Tops, Workload


class MatmulWorkload(Workload):
    """Logical A[num_tokens, k] @ B[n, k].T dimensions; exclude tile padding."""

    num_tokens: int = Field(gt=0)
    n: int = Field(gt=0)
    k: int = Field(gt=0)


class MatmulConfig(Config):
    """Compile-time tile extents, thread count and pipeline depth."""

    tile_tokens: int = Field(gt=0)
    tile_n: int = Field(gt=0)
    tile_k: int = Field(gt=0)
    threads: int = Field(ge=32, le=1024, multiple_of=32)
    stages: int = Field(ge=0)


class MatmulKernel[
    ArgumentsT: Arguments,
    WorkloadT: MatmulWorkload,
    ConfigT: MatmulConfig,
](Kernel[ArgumentsT, WorkloadT, ConfigT]):
    """Classify both MMA operands by their arithmetic format, not storage dtype."""

    mma_type: ClassVar[MmaType]

    @classmethod
    def tops(cls, arguments: ArgumentsT) -> Tops:
        """Count logical 2*num_tokens*n*k operations; omit padding and epilogue work."""
        workload = cls.make_workload(arguments)
        return {cls.mma_type: 2 * workload.num_tokens * workload.n * workload.k}
