# Kernel compilation and binding

[中文](../../zh/core/compilation.md)

[`runtime/compilation.py`](../../../src/nano_omni/core/runtime/compilation.py) compiles unique Kernel specializations and binds their CUDA launch metadata. The model action list carries Kernel instances directly.

## Specialization identity

Compilation uses `(concrete Kernel type, workload, config)` as its identity. `workload` contains dimensions and compile-time branches; `config` is the measured production configuration selected by the Kernel. Repeated layers with the same identity share one compiled program.

The compilation helper maps every tensor leaf to a zero-address `DEVICE TensorDesc` with the same dtype and shape. Compilation needs the ABI and specialization only; it neither allocates model workspace nor reads model tensor contents.

## Preparation

`prepare(commands, label=...)` collects Kernel instances, maps their descriptors to compilation placeholders, and retains the first instance of each identity. It calls `Kernel.compile()` once per identity, populating TileLang's normal disk and process caches. It does not call `Kernel.submit()`.

## Binding

`bind_kernel_modules(model, runtime)` makes the runtime's CUDA context current and uses its specialization-to-program map. The map lives for the complete pipeline, so equal specializations share one compiled program across model boundaries as well as within one model. Each model Kernel receives both that compiled reference and the selected Config.

Binding then maps tensor leaves to zero-address descriptors and calls `Kernel.submit()` under the binding mode. [`runtime/tilelang_compat.py`](../../../src/nano_omni/core/runtime/tilelang_compat.py) and the execution layer intercept CUDA launch and tensor-map setup during this pass, so the generated launcher initializes required attributes without running model computation. The real cached launcher remains available for execution.

## Execution boundary

During normal execution, addressing replaces logical descriptors with actual device addresses. `Kernel.submit()` expands `Arguments.values(include_absent=True)`, substitutes an existing descriptor for each unused optional ABI slot, and calls `execution.launch_tilelang()` on the active compute stream.

Compilation and binding operate on placeholder descriptors. Workspace allocation, checkpoint file positions, and runtime buffer ownership remain in the execution layer. Timing labels separately record request collection, compilation, argument binding, and CUDA attribute binding.
