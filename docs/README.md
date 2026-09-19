# Documentation

Start with the setup and benchmarking guides. The core pages describe the
implementation's planning, storage, compilation, and runtime contracts.

| Topic | English | 中文 |
| --- | --- | --- |
| Getting started | [English](en/getting-started.md) | [中文](zh/getting-started.md) |
| Benchmarking | [English](en/benchmarking.md) | [中文](zh/benchmarking.md) |
| Licensing and assets | [English](en/licensing.md) | [中文](zh/licensing.md) |
| Code style | [English](en/code-style.md) | [中文](zh/code-style.md) |
| Core overview | [English](en/core/README.md) | [中文](zh/core/README.md) |
| Core contracts | [English](en/core/contracts.md) | [中文](zh/core/contracts.md) |
| Core storage | [English](en/core/storage.md) | [中文](zh/core/storage.md) |
| Core planning | [English](en/core/planning.md) | [中文](zh/core/planning.md) |
| Core compilation | [English](en/core/compilation.md) | [中文](zh/core/compilation.md) |
| Core runtime | [English](en/core/runtime.md) | [中文](zh/core/runtime.md) |
| Scripts | [English](en/scripts.md) | [中文](zh/scripts.md) |

English and Chinese pages share relative paths and link to each other. The
documentation hook checks counterpart links, paired updates, and Python source
inventories. Each file under `src/nano_omni/core/` appears once across the core
pages, in the same order in both languages. The scripts guide inventories
non-experimental Python files under `scripts/`; hooks and tests live separately
under `hooks/` and `tests/`.
