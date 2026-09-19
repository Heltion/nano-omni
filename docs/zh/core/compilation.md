# Kernel 编译与绑定

[English](../../en/core/compilation.md)

[`runtime/compilation.py`](../../../src/nano_omni/core/runtime/compilation.py) 编译去重后的 Kernel specialization，并绑定 CUDA launch 元数据。Model action 列表直接携带 Kernel 实例。

## Specialization 标识

编译使用 `(具体 Kernel 类型, workload, config)` 作为标识。`workload` 包含问题规模和编译期分支；`config` 是该 workload 选中的生产配置。具有相同标识的重复层共享一个 compiled program。

编译辅助函数将每个 tensor 叶节点映射为 dtype、shape 相同且地址为零的 `DEVICE TensorDesc`。编译只需要 ABI 与 specialization；它不申请模型 workspace，也不读取模型 tensor 内容。

## 准备

`prepare(commands, label=...)` 收集 Kernel 实例，将描述符替换成编译占位符，并为每个标识保留第一个实例。它对每个标识调用一次 `Kernel.compile()`，直接编译该 workload 与 config 对应的 TIR。

准备阶段保留 TileLang 正常的磁盘与进程缓存，不调用 `Kernel.submit()`。

## 绑定

`bind_kernel_modules(model, runtime)` 将 runtime 的 CUDA context 设为当前 context，并使用其中从 specialization 标识到 compiled program 的映射。该映射覆盖整个 pipeline，因此相同 specialization 在 Model 内部和不同 Model 之间都只保留一个 compiled program。每个 Model Kernel 都取得这个共享引用和获胜 Config。

随后，绑定阶段将 tensor 叶节点映射为零地址描述符，并在 binding 模式下调用 `Kernel.submit()`。[`runtime/tilelang_compat.py`](../../../src/nano_omni/core/runtime/tilelang_compat.py) 与执行层在这一步拦截 CUDA launch 和 tensor-map 设置，使生成的 launcher 初始化所需属性而不运行模型计算。真实缓存 launcher 仍供执行阶段使用。

## 执行边界

正常执行时，addressing 将逻辑描述符替换为实际设备地址。`Kernel.submit()` 展开 `Arguments.values(include_absent=True)`，为未使用的可选 ABI 槽位代入调用中已有的描述符，并在当前 compute stream 上调用 `execution.launch_tilelang()`。

编译与绑定只操作占位描述符。Workspace 分配、checkpoint 文件位置和 runtime buffer 所有权仍属于执行层。计时标签分别记录请求收集、编译、参数绑定和 CUDA 属性绑定。
