"""Configuration fields shared by dense attention kernels."""

from nano_omni.core.kernel import Config


class AttentionConfig(Config):
    """Query/KV tile sizes, pipeline stages, and CUDA threads per block."""

    tile_query_tokens: int
    tile_kv_tokens: int
    stages: int
    threads: int
