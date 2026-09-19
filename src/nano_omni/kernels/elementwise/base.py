"""Fixed launches shared by independently masked elementwise kernels."""

from typing import ClassVar

from nano_omni.core.kernel import Arguments, Config, Kernel, Workload


class ElementwiseConfig(Config):
    """Element count per masked tile and CUDA threads per block."""

    tile_elements: int
    threads: int


class ElementwiseKernel[
    ArgumentsT: Arguments,
    WorkloadT: Workload,
](Kernel[ArgumentsT, WorkloadT, ElementwiseConfig]):
    """Build fixed launches for independent, tail-masked elements."""

    # Elements per tile and CUDA threads, declared by each concrete kernel.
    launch: ClassVar[tuple[int, int]]

    @classmethod
    def make_config(cls, workload: WorkloadT) -> ElementwiseConfig:
        """Build the fixed launch declared by the concrete kernel."""
        del workload
        elements, threads = cls.launch
        return ElementwiseConfig(tile_elements=elements, threads=threads)
