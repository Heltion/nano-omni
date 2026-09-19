# Tensor 存储与布局

[English](../../en/core/storage.md) · [Core 总览](README.md)

Core 从 model 构建到 runtime 执行始终使用同一个不可变
`TensorDesc`。它保存 `dtype`、`shape`、存储 `kind` 和 `info`，
只描述一个 view。底层文件、host 内存或 device allocation 的所有者
不存在 descriptor 中。

## 位置与大小

`EMPTY` descriptor 的 `info=()`。其余当前 kind 使用
`(identifier_or_base, byte_offset)`：Address 之前，
第一项标识 file、input、output、activation slot、weight 或 workspace；
address 之后，第一项是 host 或 CUDA 基地址。第二项始终是字节偏移。

`num_bytes` 等于 `shape` 各维乘积与 `DType.itemsize` 的乘积。Packed FP4
使用 `DType.FP4`，descriptor 的 shape 表示 kernel 实际消费的 packed
字节存储。

`data_ptr()` 只能用于 addressing 产生的 `DEVICE` descriptor，它将字节
偏移加到 device 基地址。`device` 返回 TileLang launcher 需要的共享
CUDA device view；项目固定使用 device 0。
`as_torch()` 将同一段已寻址内存暴露为零拷贝、连续的 Torch view。
参考程序与数值验证统一使用这一个转换入口。

## Tensor kind

| Kind | 含义 |
| --- | --- |
| `EMPTY` | 只声明 dtype 与 shape，尚未分配存储。 |
| `FILE` | 索引 checkpoint 文件中的字节范围。 |
| `INPUT`、`OUTPUT` | 调用方所有的 runtime buffer。 |
| `ACTIVATION` | 放入 workspace 前的 model activation slot。 |
| `SCRATCH` | 放入 workspace 前的 op 局部临时区。 |
| `WEIGHT` | 常驻规划前的已注册 checkpoint tensor。 |
| `WORKSPACE` | Model workspace 中已规划的字节范围。 |
| `PINNED` | Pinned host staging pool 中已规划的字节范围。 |
| `HOST`、`DEVICE` | 已绑定实际 host 或 CUDA 地址的 view。 |

Planning 使用 `dataclasses.replace` 更新 `kind` 和 `info`，同时保持
dtype、shape 与字节数。各阶段不会再用另一种 tensor reference 包装它。

## Activation 与 scratch 布局

[`core/layout.py`](../../../src/nano_omni/core/layout.py) 包含两个逻辑布局器。

`ActivationLayout.require(position, num_bytes, alignment)` 合并同一 activation slot
的多次需求。切片位置以 `byte_offset + num_bytes` 计入该 slot 的范围。
`layout()` 按首次声明顺序放置 slot，保留每个 slot 的最大对齐，
返回各 slot 偏移和 activation 总范围。

`ScratchLayout.reserve(name, dtype, shape, alignment)` 追加一个唯一命名的临时
区并返回它的 `SCRATCH TensorDesc`。每个 op 都使用新的 scratch layout。
Model planner 只保留所有 op 中最大的 scratch 范围，因此顺序执行的 op
复用同一段 workspace。

这些 API 的大小和偏移均以字节为单位。对齐、命名唯一性和边界等
内部不变量在 descriptor 创建或绑定时使用 assert 校验。

## Checkpoint weight

`core/model.py` 将 safetensors header 读取为 `FILE TensorDesc`。File index
选择 model 有序文件列表中的一项，info 的字节偏移
偏移直接指向 tensor payload。Header 字节范围必须等于
`TensorDesc.num_bytes`。

[`core/weights.py`](../../../src/nano_omni/core/weights.py) 按首次引用顺序分配稳定
weight ID。`WeightRegistry.reference()` 返回 `WEIGHT TensorDesc`；同一文件范围
的重复引用共享同一 ID。按行分片时保留 dtype 与后续 shape，并调整
首维和字节偏移。

Registry 只描述 weight，不读取 payload。Planner 随后选择常驻 workspace
范围，并为其余 weight 排定传输。Addressing 最后将计划位置替换为
实际 host 或 device 指针，详见[规划](planning.md)和[运行时](runtime.md)。

## 资源边界

Pipeline 和 Model 也直接传递 `TensorDesc`；底层 allocation 的容量和 active 状态保存在
allocator 中。
[`core/runtime/synchronization.py`](../../../src/nano_omni/core/runtime/synchronization.py)
选择 CUDA 依赖机制，不改变 tensor 的所有权。

Windows 资源边界由
[`core/platform/__init__.py`](../../../src/nano_omni/core/platform/__init__.py)、
[`core/platform/memory.py`](../../../src/nano_omni/core/platform/memory.py) 和
[`core/platform/process.py`](../../../src/nano_omni/core/platform/process.py) 实现。
这些模块限制进程 working set 并跟踪子进程；该限制与 CUDA workspace
及 pinned staging 布局相互独立。
