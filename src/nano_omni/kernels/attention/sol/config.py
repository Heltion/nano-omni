"""Configuration shared by Sol preparation kernels."""

from typing import Literal

from nano_omni.core.kernel import Config


class SolPrepareConfig(Config):
    threads: Literal[64, 128, 256] = 128
