# 许可证与素材

[English](../en/licensing.md)

## 代码与依赖

Nano-Omni 未声明项目许可证, 仓库中没有项目 `LICENSE` 文件.
依赖和模型的许可证不构成对 Nano-Omni 自有代码的授权.

ComfyUI 子模块采用 GNU GPL v3. `pyproject.toml` 列出的 Python 包保留各自的
许可证和声明. 组合分发需要遵守适用的组件许可证.

## 模型与媒体

运行时模型文件下载到 `models/`, 不被 Git 跟踪.

| 来源 | 素材 | 声明的许可证 |
| --- | --- | --- |
| `Comfy-Org/MiniMax-H3` | H3 权重、VAE、embedding 和 Turbo LoRA | MiniMax H3 Community License Agreement |
| `Qwen/Qwen2.5-7B-Instruct` | Tokenizer 文件 | Apache-2.0 |

下载器按文件名选择素材并复用已有本地文件. 具体条款见来源模型卡.

每个 benchmark 目录的 `SOURCES.md` 和 `SOURCES.zh.md` 记录 prompt 与参考
媒体的来源. 注明来源本身不授予再分发权. 参考媒体与生成视频适用相应素材条款
及模型许可证, 与代码许可证分别处理.
