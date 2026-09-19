# 测量方法

[English](../en/benchmarking.md)

## 工作量与计时

`perf/` 中各配置指定输入、尺寸、seed、采样参数和输出路径. 主对比两边均使用
dense BF16 DiT attention, 不代表所有模型阶段及权重均为 BF16. 请求帧数和
内部补齐帧数分别记录.

发布耗时使用每个配置的一次完整进程运行. 维护者在测试机器上的实测运行间差异
不超过 0.1%, 因此这套测量流程不要求重复运行. 加速比为 ComfyUI 耗时除以
Nano-Omni 耗时; 小于一表示 Nano-Omni 更慢.

两个普通入口从启动到退出测量子进程墙钟时间, 包含初始化、模型加载、规划、
生成、解码、输出和清理. Nano-Omni 缺失资产时, 下载也在此计时范围内.
比较生成耗时前应准备好资产. 启动前的温度检查不计时. 这些是端到端耗时,
不是单 attention 或单采样阶段耗时.

入口使用新的子进程及已有操作系统/编译缓存. `warmup=0` 表示不额外预热,
不会清空缓存. Nano-Omni 的 `--warmup 1` 准备资产并编译 kernel, 不执行一次
预热生成. ComfyUI 的 warmup 执行完整但不计入样本的生成.
普通 benchmark 使用 `--warmup 0 --repetitions 1`.

## ComfyUI 执行方式

基线使用仓库中的 ComfyUI 子模块, 通过 `PromptExecutor` 执行一次完整 API
prompt. Attention 使用默认分派器, 实际实现及运行环境记录在报告中.
运行元数据还记录动态 VRAM、fast-disk、H3 按模态生成的噪声、固定页内存状态
及主机工作集限制. 入口将配置的 seed、LoRA strength 和 sampling shift 传给对应节点,
在测量前拒绝不支持的 attention 或 LoRA 选择.

两种实现使用配置的主机内存预算, 各自管理内存. 相同预算不代表分配和搬运策略
相同. 解析后的配置、workload、环境及后端字段描述实际执行的工作量.

## 报告与指标

`nano.json`、`comfy.json` 包含普通耗时、解析配置、workload 及运行环境.
每次运行具有随机 `run_id`; 质量报告使用对应 Nano-Omni 运行的 ID.
首页从这些报告、`nano-nsys.json` 和 `error.json` 生成, 过滤未成功报告、非法耗时以及工作量不匹配
的对比. `repetitions=1` 时, `mean_seconds` 就是该次耗时.

`nano-nsys.json` 记录独立分析运行. Kernel% 为 CUDA kernel 区间并集除以
capture 时长; GPU% 还包含捕获的复制、memset 及视频引擎工作.
`gpu_complete` 标记覆盖是否完整. 分析耗时与普通运行耗时分开使用.

`Tensor%` (JSON 中的 `tensor_percent`) 按以下公式估计:

```text
100 * sum(逻辑 MMA 运算量 / 名义 dense 每秒运算量) / 普通墙钟秒数
```

名义速率来自 `perf/hardware.json`; Sol 对动态选块使用统计估计.
此指标不是实测 Tensor Core 利用率, 也不能与另一次分析运行的 Kernel% 或 GPU%
相减来计算 CPU 时间.

FL2VA attention 表格将解码 RGB 帧与 seed-0 Nano-Omni dense BF16 输出比较.
PSNR 使用全部 RGB 元素的均方误差; SSIM 使用 11x11、sigma 1.5 的高斯窗口;
SSIM 对帧取平均. 这些指标衡量输出一致性,
不是绝对生成质量、音频质量或与 ComfyUI 的等质量证明. BF16 行是参考与自身
比较. 对编码视频的比较也包含压缩差异.

## 运行完整集合

准备依赖、模型资产和 Nsight Systems 后, 从仓库根目录运行:

```powershell
.\scripts\measure.ps1
```

每个主配置分别执行一次 Nano-Omni、一次 ComfyUI, 然后执行一次 Nano-Omni
NSys 分析. 其他 attention 配置各执行一次普通 Nano-Omni 和一次 NSys 分析.
脚本遇到首条失败命令即停止, 在配置旁写入结果, 最后重新生成首页.
结果文件会被覆盖; 本地实验请指定独立输出路径.
单独运行命令见[脚本文档](scripts.md).
