# Licensing and assets

[中文](../zh/licensing.md)

## Code and dependencies

Nano-Omni does not declare a project license; the repository has no project
`LICENSE` file. Dependency and model licenses do not grant a license to
Nano-Omni's own code.

The ComfyUI submodule carries GNU GPL v3. Python packages listed in
`pyproject.toml` retain their respective licenses and notices. Combined
distributions must comply with the applicable component licenses.

## Models and media

Runtime model files download into `models/` and are not tracked in Git.

| Source | Assets | Declared license |
| --- | --- | --- |
| `Comfy-Org/MiniMax-H3` | H3 weights, VAEs, embeddings, and Turbo LoRAs | MiniMax H3 Community License Agreement |
| `Qwen/Qwen2.5-7B-Instruct` | Tokenizer files | Apache-2.0 |

The downloader selects filenames and reuses existing local files. Refer to the
source model cards for the applicable terms.

Each benchmark directory's `SOURCES.md` and `SOURCES.zh.md` identify the prompt
and reference-media sources. Attribution does not itself grant redistribution
rights. Reference media and generated videos remain subject to the relevant
asset terms and model license, separately from the code.
