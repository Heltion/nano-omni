# 入门

[English](../en/getting-started.md)

## 环境要求

Nano-Omni 支持 MiniMax H3 文本音视频生成 T2VA、首尾帧条件生成 FL2VA、
参考音视频条件生成 REF2VA. 需要 NVIDIA GPU、匹配的驱动和 CUDA 开发环境、
Python 3.12 或更新版本、Git 和 uv. Kernel 通过 NVRTC 编译, 不提供 CPU
推理后端. 测试环境使用 Windows、RTX 5060 Ti、CUDA 13.0 和 TileLang 0.1.14.
具体运行版本记录在各次测量 JSON 中.

## 安装与运行

在 PowerShell 中执行:

```powershell
git clone https://github.com/Heltion/nano-omni.git
cd nano-omni
uv sync --locked
uv run scripts/run/nano-omni.py --help
uv run scripts/run/nano-omni.py --config perf/h3-fl2va-official-dense-bf16-4step/config.yaml --output outputs/example.mp4
```

从仓库根目录运行. YAML 内路径相对于仓库根目录, 而非 YAML 所在目录.
`--output` 指定输出位置; 普通运行写入 MP4 和同名 JSON 报告. 示例选择
dense BF16 DiT attention, 不代表整个模型均为 BF16. 首页表格列出了其他
attention 配置.

缺失模型文件通过 ModelScope 自动下载到 `models/`, 已有文件直接复用.
配置的来源为 `Comfy-Org/MiniMax-H3` 和 `Qwen/Qwen2.5-7B-Instruct`.
自定义 LoRA 路径必须已经存在. FL2VA 示例使用:

```text
models/
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
  vae/minimax_h3_video_vae_fp16.safetensors
  vae/minimax_h3_audio_vae_fp32.safetensors
  loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
  tokenizers/minimax_h3/{merges.txt,tokenizer.json,tokenizer_config.json,vocab.json}
```

为模型资产预留磁盘空间, 为固定页缓冲区和激活预留内存. 首次运行可能包含下载
和编译. `--warmup 1` 在独立子进程中准备资产并编译选中的 kernel, 然后执行
计时生成, 不是只下载的命令.

## 配置

| 任务 | 配置 |
| --- | --- |
| T2VA | `perf/h3-t2va-official-dense-bf16-4step/config.yaml` |
| FL2VA | `perf/h3-fl2va-official-dense-bf16-4step/config.yaml` |
| REF2VA | `perf/h3-ref2va-0.4mp-dense-bf16-4step/config.yaml` |

`prompt` 和 `prompt_file` 必须且只能提供一个. FL2VA 接受首帧及可选尾帧,
不能只提供尾帧. REF2VA 需要同时提供参考视频和参考音频. 宽高至少为 256,
且为 32 的倍数. FL2VA 的命名 LoRA `turbo_4step` 要求 1344x768 和四步;
`turbo_8step` 要求八步.

`attention` 选择 `dense` 或 `sol`; `attention_precision` 选择 `bf16`、
`int8_fp8` 或 `nvfp4`. 可用 `--attention-precision bf16` 覆盖精度配置.
内存参数包括 `working_set_gib`、`pinned_gib` 和 `workspace_gib`.
入口参数见[脚本文档](scripts.md).

## ComfyUI 与分析

ComfyUI 基线需要子模块及 experiment 依赖组:

```powershell
git submodule update --init --recursive
uv run --group experiment scripts/run/comfy-ui.py --config perf/h3-fl2va-official-dense-bf16-4step/config.yaml
```

基线通过 ComfyUI 的 `PromptExecutor` 执行完整 API prompt.
Nano-Omni 分析运行添加 `--nsys`, 并要求 PATH 中有 Nsight Systems.
普通生成不需要 Nsight Systems. 计时边界和指标定义见[测量方法](benchmarking.md).

## 开发

[核心文档](core/README.md) 介绍规划与运行时契约. 修改生成首页应编辑
`hooks/readme.py`, 然后执行 `python hooks/readme.py`. 中英文文档同步维护.
在项目环境中使用 `uv run python -m unittest discover -s tests -v` 运行测试.

失败时查看入口日志, 检查资产、CUDA 兼容性、内存设置和配置. 本地实验使用
`outputs/`; benchmark 命令会覆盖配置旁已跟踪的结果文件.
代码与素材条款见[许可证说明](licensing.md).
