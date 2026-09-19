# 层级契约

[English](../../en/core/contracts.md) · [Core 总览](README.md)

公共契约将依赖形状的计算依次交给 Kernel、Op、Model 和 Pipeline。每层只负责一种决策：Kernel 定义一次调用，Op 绑定语义输入并展开调用，Model 排列操作并规划内存，Pipeline 协调整次生成。

[`core/__init__.py`](../../../src/nano_omni/core/__init__.py) 定义四层推理结构的包边界。

## Kernel

[`core/kernel.py`](../../../src/nano_omni/core/kernel.py) 定义 `Arguments`、`Workload`、`Config`，以及唯一的调用实体 `Kernel`。

一个 Kernel 实例包含四个值：

| 字段 | 含义 |
| --- | --- |
| `arguments` | 调用 tensor，以及 stride、padding、epsilon、融合开关等语义值 |
| `workload` | 从 arguments 校验并复制的不可变问题规模与编译期分支 |
| `config` | 根据 workload 选择的 Config |
| `compiled` | 已准备的 TileLang 特化程序 |

具体 Kernel 提供稳定的 `name`，将 TileLang callable 直接保存在 `program`，并实现：

| 方法 | 职责 |
| --- | --- |
| `make_arguments(workload)` | 为独立测量构造未寻址描述符 |
| `make_workload(arguments)` | 验证参数并恢复 specialization |
| `make_config(workload)` | 为 workload 选择 Config |
| `ref_program(arguments)` | 将直接数学 reference 写入已寻址参数 |
| `tops(arguments)` | 按完整 `MmaType` 估算逻辑 MMA 工作量；基类返回空字典 |

编译期语义值同时保存在 `Arguments` 和 `Workload` 中。Op 只构造 `Arguments`；`make_workload()` 校验并复制 specialization 值。这样 `ref_program(arguments)` 能独立表达数学语义，而编译与缓存 identity 使用不可变 Workload。

对于带 stride 的切片，`Arguments` 与 `Workload` 都保存逻辑尺寸和 strides；`TensorDesc` 只描述偏移指针实际可访问的字节跨度。例如 `volume_to_tokens` 将 CFHW tile 表达成一维 span，再通过两边重复保存的 strides 恢复 CFHW 索引。通用分配边界因此可以精确检查，也不需要新增切片 wrapper 或按原点生成 specialization。

`Arguments.map()` 在替换全部 `TensorDesc` 时重建同一种参数结构。`Arguments.values(include_absent=True)` 按字段顺序展开 tensor ABI 叶节点，并保留 `None` tensor 槽位。`Arguments.dynamic_parameters()` 单独提供具名的运行时 scalar。Kernel 基类统一用这些操作完成寻址、编译占位和提交。

`Kernel.compile()` 将 `workload.model_dump()` 和 `config.model_dump()` 展开给 `program.get_tir()`，再编译该 TIR。`Kernel.submit()` 只把已经编译的调用提交到当前 CUDA stream。

## TensorDesc

[`core/tensor.py`](../../../src/nano_omni/core/tensor.py) 提供所有层共用的唯一 tensor 描述符。它包含 `dtype`、`shape`、`TensorKind` 和变长 `info` tuple。`EMPTY` 描述测量形状；`ACTIVATION`、`SCRATCH`、`SCALAR`、`WEIGHT` 描述模型逻辑区域；`INPUT`、`OUTPUT`、`WORKSPACE` 描述 runtime 绑定；`HOST` 与 `DEVICE` 携带已寻址内存。

`TensorDesc.view()` 在保留存储身份的同时改变 shape 和字节偏移。`num_bytes` 用于检查范围，`data_ptr()` 解析设备指针，`as_torch()` 为 reference program 创建零拷贝 CUDA view。

`FP8_E4M3` 表示有符号 E4M3 张量值，`FP8_UE4M3` 表示 NVFP4 block scale。两者都占一个字节，但数值语义不同。safetensors 没有 UE4M3 标签，因此 H3 权重绑定依据 FP4 projection 合约将 block scale 重新解释为 `FP8_UE4M3`。

独立测量会保留描述符的对象身份：重复使用同一个描述符表示有意别名。因此，即使 dtype 和 shape 相同，互相独立的输入与输出也必须分别构造描述符。

## Op

[`core/op.py`](../../../src/nano_omni/core/op.py) 将模型激活和权重绑定到操作。构造函数接收输入描述符，`with_weights()` 附加权重结构，`to()` 完成输出描述符绑定。具体 `kernels(scratch)` 返回有序 Kernel 实例，并从传入的 `ScratchLayout` 登记自己的临时区域。

`Op.prepare()` 检查绑定是否完整，为该 Op 新建局部 scratch layout，取得 Kernel 实例，并返回这些实例和 scratch 范围。这里不编译，也不执行。

激活切片直接表示为 `TensorDesc.info=(identity, byte_offset)`。Model 因此可以切分 token 或写入多个输出，而无需增加切片包装类型。

## Model

[`core/model.py`](../../../src/nano_omni/core/model.py) 管理 checkpoint 元数据和可执行 action 列表。`FileMetadata` 将 safetensors header 读取为 `FILE TensorDesc`，`ModelMetadata` 保留文件顺序。

`PlannedModel` 实现：

| 方法 | 结果 |
| --- | --- |
| `plan(metadata, spec, total_memory, workspace_limit)` | Model 配置、有序 action 和 workspace 字节数 |
| `run(args)` | Model 特定的执行结果 |

`plan()` 登记 activation ID，构造 Op，绑定 registry 权重，再调用 `planning.operators.commands()`。规划器布局 activation、共享 scratch、不可变 scalar 数据和权重区，并插入传输依赖。最终 action 列表直接包含 Kernel 实例、复制与同步操作。

`kernel_inventory()` 使用同一套 plan，按执行顺序返回 Kernel 实例。测量代码需要 launch 数量时，再按具体类型和 workload 聚合这些实例。Kernel 没有第二种 inventory record 或 command wrapper。

`PlannedModel.prepare()` 规划 staging，并返回 `ModelPreparation`。Pipeline warmup 单独调用 `runtime.compilation.prepare()` 编译 Kernel。`execute()` 使用当前输入、输出、checkpoint 路径和 workspace 解析保存的提交计划，然后执行。

## Pipeline

[`core/pipeline.py`](../../../src/nano_omni/core/pipeline.py) 定义生成生命周期。具体 Pipeline 实现 `prepare()`、`execute()`、`workload()` 和 `model_requests()`。

`run()` 应用配置的 working-set 限制，准备缺失的本地素材并执行生成。`warmup()` 准备素材，取得每个不同的 Model request，汇总其 Kernel inventory，并在整个 pipeline 内将每种 specialization 只编译一次；它不规划权重 staging、不提交 Kernel，也不生成媒体。正常执行期间，不同 pipeline 阶段的模型准备可以重叠。

## 指标

`Kernel.tops(arguments)` 按 `F4F4F32`、`F8F8F16`、`F8F8F32` 等完整 MMA 签名返回估算的逻辑运算量。动态 Sol mask 在规划阶段不可读取，因此使用配置对应的统计选择概率。`PlannedModel.tops` 对有序 Kernel 实例求和。Pipeline 报告使用正常端到端区间计算逻辑指标；来自 NSys 的 kernel% 和 gpu% 使用 profiler 影响后的区间。
