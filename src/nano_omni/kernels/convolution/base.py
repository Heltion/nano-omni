"""Launch configuration shared by implicit-GEMM convolutions."""

from nano_omni.core.kernel import Config


class ConvolutionConfig(Config):
    tile_m: int
    tile_n: int
    tile_k: int
    threads: int
    stages: int
