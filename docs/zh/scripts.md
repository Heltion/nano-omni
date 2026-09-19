# 脚本

[English](../en/scripts.md)

安装见[入门](getting-started.md), 计时与指标定义见[测量方法](benchmarking.md).

## 运行

配置中的文件路径相对于仓库根目录, 保存时统一使用 `/`. 绝对路径、带盘符的相对路径、
用户主目录路径以及 `..` 跳转均不接受, 自定义 LoRA 路径也遵守此规则.

| 脚本 | 用途 |
| --- | --- |
| [`scripts/measure.ps1`](../../scripts/measure.ps1) | 执行测量集合并重新生成首页. |
| [`scripts/run/config.py`](../../scripts/run/config.py) | 加载和验证共用 YAML 配置. |
| [`scripts/run/nano-omni.py`](../../scripts/run/nano-omni.py) | 运行 Nano-Omni 并写入普通或 NSys 测量 JSON. |
| [`scripts/run/comfy-ui.py`](../../scripts/run/comfy-ui.py) | 执行完整 ComfyUI API prompt 并写入测量 JSON. |
| [`scripts/run/common.py`](../../scripts/run/common.py) | 提供共用的配置、进程、计时、视频指标和分析工具. |

从仓库根目录执行命令. 两个入口均接受 `--config` 和 `--output` 等配置覆盖
参数. YAML 内路径相对于仓库根目录. 普通运行默认
`--warmup 0 --repetitions 1`.

```powershell
uv run scripts/run/nano-omni.py --config perf/h3-fl2va-official-dense-bf16-4step/config.yaml --output outputs/example.mp4
```

ComfyUI 入口需要子模块与 experiment 依赖组:

```powershell
git submodule update --init --recursive
uv run --group experiment scripts/run/comfy-ui.py --config perf/h3-fl2va-official-dense-bf16-4step/config.yaml
```

普通样本测量一个新子进程直到退出的时间. Nano-Omni 的 `--warmup 1` 只编译
选中的 kernel, 不进行预热生成; ComfyUI warmup 执行完整但不计入样本的生成.
两者都不清空操作系统或编译缓存. 可选的 `--repetitions`、`--append` 用于本地
诊断, 发布 benchmark 每个配置只需一次普通运行. Append 要求已有完整报告的
配置、workload、环境和后端一致, 不支持 NSys.

Nano-Omni 的 `--hardware` 为 `tensor_percent` 提供名义吞吐规格.
`--nsys` 分析一个 Nano-Omni 子进程并写入独立的 `-nsys.json`, 要求
`--repetitions 1`, 且 PATH 中存在 Nsight Systems. 编译预热在 NSys 外执行.
ComfyUI 报告记录实际选择的 attention 后端.

报告包含随机 run ID、解析配置、耗时和运行环境. 新报告先标记为 `incomplete`,
防止失败运行留下仍有效的旧成功报告. ComfyUI 接受当前子进程输出, 并传播
monitor 线程错误. Nano-Omni 质量报告共享运行 ID; 配置 `quality_reference`
后记录解码视频的 PSNR 和 SSIM. `memory_samples` 保存逐次内存
记录, `memory` 保存最后一次记录.

运行完整集合:

```powershell
.\scripts\measure.ps1
```

每个主配置执行一次 Nano-Omni 和一次 ComfyUI, 再执行一次独立的 Nano-Omni
NSys 分析. 其他 attention 配置各执行一次普通 Nano-Omni 和一次 NSys 分析.
命令串行运行, 失败即停止, 覆盖 `perf/` 下各配置旁的结果.
仅重新生成首页使用 `python hooks/readme.py`.

[`scripts/experiment/`](../../scripts/experiment/) 除 `.gitignore` 外均被忽略,
用于本地临时实验.

## 调优

| 脚本 | 用途 |
| --- | --- |
| [`scripts/tune.py`](../../scripts/tune.py) | 检查并测量指定 kernel workload 和配置. |
| [`scripts/update.py`](../../scripts/update.py) | 更新模型清单或 kernel 测量. |

Tune 接受 `--kernel`、JSON `--workload` 和完整 JSON `--config`.
先对指定形状的随机输入执行 reference 检查, 再测量候选.
`--record perf/<instance>/tuning.csv` 按 kernel、workload 和配置插入或替换候选.
`--trials` 可选重复 kernel 测量; `--dynamic` 覆盖支持的运行时标量,
例如 `num_protected_blocks`. 实际标量值属于候选身份.
Kernel 调优和端到端 benchmark 计时是两种独立操作.

Update 使用 `--instance` 选择 `kernels.csv`:

| 子命令 | 操作 |
| --- | --- |
| `model` | 重建指定模型的清单, 保留匹配测量和其他模型, 新测量初始化为零. |
| `kernel` | 根据 model、kernel 和 workload 定位一行, 使用生产 `make_config(workload)` 测量. |
| `all-kernels` | 测量 latency 为零的行. |

CSV 列为 `model`、`kernel`、`percent`、`latency`、`launch`、`mem%`、`ncu%`、
`tops`、`workload` 和 `config`. Latency 为每次 launch 的毫秒数, 贡献比例包含
launch 次数. 每次更新重新计算比例并按贡献排序.

Kernel 清单和调优 CSV 属于本地产物, 不随基准结果发布. 测量清单中的行前,
先用 `scripts/update.py model` 生成清单. `perf/` 只跟踪 README 对应的配置、
输入、来源说明与结果文件.
