"""Measure ComfyUI H3 baselines with its default attention backend."""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
import gc
import importlib
import json
import os
import shutil
import sys
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Self, cast

import psutil
import torch
import typer

if TYPE_CHECKING:
    from typing import Protocol

    from comfy.nested_tensor import NestedTensor
    from comfy_api.latest import AudioInput
    from config import RunConfig

    class _ModelManagement(Protocol):
        def unload_all_models(self) -> None: ...

    class _Nodes(Protocol):
        async def init_extra_nodes(
            self, *, init_custom_nodes: bool, init_api_nodes: bool
        ) -> object: ...

    class _FolderPaths(Protocol):
        def get_input_directory(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ComfyParameters:
    """Semantic generation parameters supported by the ComfyUI comparison."""

    lora: str
    video_shift: float
    audio_shift: float


class _WorkflowServer:
    """Minimal PromptExecutor event sink for an in-process workflow run."""

    client_id: str | None = None
    last_node_id: str | None = None

    def send_sync(self, event: str, data: object, sid: str | None = None) -> None:
        del event, data, sid


def _link(node: str, output: int = 0) -> list[str | int]:
    return [node, output]


def fl2va_workflow(
    *,
    prompt: str,
    output_prefix: str,
    width: int,
    height: int,
    seconds: float,
    checkpoint: str,
    lora_name: str | None,
    lora_strength: float,
    steps: int,
    seed: int,
    video_shift: float,
    audio_shift: float,
    first_frame_name: str | None,
    last_frame_name: str | None,
) -> dict[str, dict[str, object]]:
    """Build the complete FL2VA API prompt executed by ComfyUI."""
    frames = max(5, round(seconds * 24))
    frames += (5 - frames % 17) % 17
    image_inputs: dict[str, object] = {}
    if first_frame_name is not None:
        image_inputs["first_frame"] = _link("first_frame")
    if last_frame_name is not None:
        image_inputs["last_frame"] = _link("last_frame")
    workflow: dict[str, dict[str, object]] = {
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
        },
        "video_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "audio_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "conditioning": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": _link("clip"),
                "vae": _link("video_vae"),
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": frames,
                **image_inputs,
            },
        },
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": checkpoint, "weight_dtype": "default"},
        },
        "shift": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": _link("lora" if lora_name is not None else "unet"),
                "shift_video": video_shift,
                "shift_audio": audio_shift,
            },
        },
        "scheduler": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": _link("shift"),
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "guider": {
            "class_type": "BasicGuider",
            "inputs": {"model": _link("shift"), "conditioning": _link("conditioning")},
        },
        "noise": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "sampler": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": _link("noise"),
                "guider": _link("guider"),
                "sampler": _link("sampler"),
                "sigmas": _link("scheduler"),
                "latent_image": _link("conditioning", 1),
            },
        },
        "separate": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {"av_latent": _link("sample")},
        },
        "video_decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": _link("separate"), "vae": _link("video_vae")},
        },
        "audio_decode": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": _link("separate", 1), "vae": _link("audio_vae")},
        },
        "video": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": _link("video_decode"),
                "audio": _link("audio_decode"),
                "fps": 24.0,
                "bit_depth": 8,
                "color_space": "sRGB",
            },
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": _link("video"),
                "filename_prefix": output_prefix,
                "format": "mp4",
                "format.codec": "h264",
            },
        },
    }
    if lora_name is not None:
        workflow["lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": _link("unet"),
                "lora_name": lora_name,
                "strength_model": lora_strength,
            },
        }
    if first_frame_name is not None:
        workflow["first_frame"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_frame_name},
        }
    if last_frame_name is not None:
        workflow["last_frame"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_frame_name},
        }
    return workflow


def ref2va_workflow(
    *,
    prompt: str,
    output_prefix: str,
    width: int,
    height: int,
    seconds: float,
    checkpoint: str,
    lora_name: str,
    lora_strength: float,
    steps: int,
    seed: int,
    video_shift: float,
    audio_shift: float,
    reference_video_name: str,
    reference_audio_name: str,
) -> dict[str, dict[str, object]]:
    """Build the complete REF2VA API prompt executed by ComfyUI."""
    workflow = fl2va_workflow(
        prompt=prompt,
        output_prefix=output_prefix,
        width=width,
        height=height,
        seconds=seconds,
        checkpoint=checkpoint,
        lora_name=lora_name,
        lora_strength=lora_strength,
        steps=steps,
        seed=seed,
        video_shift=video_shift,
        audio_shift=audio_shift,
        first_frame_name=None,
        last_frame_name=None,
    )
    workflow["reference_video"] = {
        "class_type": "LoadVideo",
        "inputs": {"file": reference_video_name},
    }
    workflow["reference_components"] = {
        "class_type": "GetVideoComponents",
        "inputs": {"video": _link("reference_video")},
    }
    workflow["reference_audio"] = {
        "class_type": "LoadAudio",
        "inputs": {"audio": reference_audio_name},
    }
    workflow["conditioning"] = {
        "class_type": "MiniMaxH3ReferenceToVideo",
        "inputs": {
            "clip": _link("clip"),
            "vae": _link("video_vae"),
            "audio_vae": _link("audio_vae"),
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": max(5, round(seconds * 24)),
            "ref_image_size": "match",
            "ref_videos.ref_video_0": _link("reference_components"),
            "ref_video_audios.ref_video_audio_0": _link("reference_components", 1),
            "ref_audios.ref_audio_0": _link("reference_audio"),
        },
    }
    workflow["trim_video"] = {
        "class_type": "ImageFromBatch",
        "inputs": {
            "image": _link("video_decode"),
            "batch_index": 0,
            "length": round(seconds * 24),
        },
    }
    video = workflow["video"]
    assert isinstance(video, dict)
    video_inputs = video["inputs"]
    assert isinstance(video_inputs, dict)
    video_inputs["images"] = _link("trim_video")
    return workflow


def comfy_parameters(config: RunConfig) -> ComfyParameters:
    """Resolve shared generation parameters or reject an inexact baseline."""
    from nano_omni.models import h3

    if config.attention != "dense" or config.attention_precision != "bf16":
        raise ValueError("ComfyUI comparisons support only dense BF16 attention")
    if config.pipeline == "h3_ref2va":
        if config.lora not in (None, "turbo_4step"):
            raise ValueError("ComfyUI REF2VA uses its fixed turbo_4step LoRA")
        if config.steps != 4:
            raise ValueError("ComfyUI REF2VA requires four steps")
        lora = "turbo_4step"
    else:
        lora = config.lora or "none"
        if lora not in ("none", "turbo_4step", "turbo_8step"):
            raise ValueError("ComfyUI comparisons do not support custom LoRA paths")
    return ComfyParameters(
        lora=lora,
        video_shift=config.video_shift or h3.VIDEO_SHIFT,
        audio_shift=config.audio_shift or h3.AUDIO_SHIFT,
    )


def record_comfy_attention() -> None:
    """Report the attention implementation selected by ComfyUI."""
    attention = importlib.import_module("comfy.ldm.modules.attention")
    function = attention.optimized_attention
    print(
        "backend="
        + json.dumps(
            {
                "component": "comfy_default_attention",
                "implementation": f"{function.__module__}.{function.__name__}",
                "scope": "default dispatcher; official per-call selection applies",
            }
        ),
        flush=True,
    )


def measurement_environment() -> dict[str, object]:
    """Record the software and device used by the measured ComfyUI child."""
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "execution_backend": "comfyui_default_dispatcher",
    }


def read_child_metadata(log: Path) -> dict[str, object]:
    """Read structured facts emitted by one measured ComfyUI child."""
    result: dict[str, object] = {}
    for line in log.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        for name in ("backend", "environment"):
            if line.startswith(name + "="):
                result[name] = json.loads(line.removeprefix(name + "="))
    return result


def initialize_comfy(*, disable_pinned_memory: bool = False) -> None:
    sys.path.insert(0, str(COMFY_ROOT))
    arguments = sys.argv
    sys.argv = [
        arguments[0],
        "--disable-metadata",
        "--enable-dynamic-vram",
        "--fast-disk",
    ]
    if disable_pinned_memory:
        sys.argv.append("--disable-pinned-memory")
    try:
        import comfy.options

        comfy.options.enable_args_parsing()
        import comfy.cli_args
        import comfy_aimdo.control

        comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=True)
        import comfy.memory_management
        import comfy.model_management
        import comfy.model_patcher

        headroom = int(comfy.cli_args.args.vram_headroom * 1024**3)
        devices = comfy.model_management.get_all_torch_devices()
        if not comfy_aimdo.control.init_devices(
            (device.index, headroom) for device in devices
        ):
            raise RuntimeError("ComfyUI dynamic VRAM initialization failed")
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
        print(
            f"comfy_runtime=dynamic_vram fast_disk=true pinned_memory={not disable_pinned_memory}",
            flush=True,
        )
    finally:
        sys.argv = arguments


ROOT = Path(__file__).resolve().parents[2]
COMFY_ROOT = ROOT / "ComfyUI"
MODELS = ROOT / "models"
GIB = 1 << 30
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_SET_QUOTA = 0x0100
MEMORY_JOB_HANDLE: int | None = None


class IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_uint64),
        ("write_operation_count", ctypes.c_uint64),
        ("other_operation_count", ctypes.c_uint64),
        ("read_transfer_count", ctypes.c_uint64),
        ("write_transfer_count", ctypes.c_uint64),
        ("other_transfer_count", ctypes.c_uint64),
    ]


class BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", ctypes.wintypes.DWORD),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.wintypes.DWORD),
        ("scheduling_class", ctypes.wintypes.DWORD),
    ]


class ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", BasicLimitInformation),
        ("io_info", IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


def check_win32[T](result: T) -> T:
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return result


def create_process_tree_job(working_set_gib: int) -> None:
    global MEMORY_JOB_HANDLE
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = ctypes.wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = ctypes.wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.wintypes.HANDLE,
    ]
    kernel32.AssignProcessToJobObject.restype = ctypes.wintypes.BOOL
    kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE

    handle = check_win32(kernel32.CreateJobObjectW(None, None))
    limits = ExtendedLimitInformation()
    limits.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    check_win32(
        kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
    )
    check_win32(kernel32.AssignProcessToJobObject(handle, kernel32.GetCurrentProcess()))
    MEMORY_JOB_HANDLE = int(handle)
    print(
        f"memory_limit=tree_rss:{working_set_gib}GiB",
        flush=True,
    )


def trim_process_tree_working_sets() -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        ctypes.wintypes.DWORD,
        ctypes.wintypes.BOOL,
        ctypes.wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    psapi.EmptyWorkingSet.argtypes = [ctypes.wintypes.HANDLE]
    for process in [psutil.Process(), *psutil.Process().children(recursive=True)]:
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_SET_QUOTA, False, process.pid
        )
        if handle:
            psapi.EmptyWorkingSet(handle)
            kernel32.CloseHandle(handle)


def tree_rss() -> int:
    processes = [psutil.Process(), *psutil.Process().children(recursive=True)]
    return sum(process.memory_info().rss for process in processes)


@dataclass(slots=True)
class Monitor:
    working_set_threshold: int
    started: float = field(init=False, default=0.0)
    stop: threading.Event = field(init=False, default_factory=threading.Event)
    thread: threading.Thread | None = field(init=False, default=None)
    peak_tree_rss: int = field(init=False, default=0)
    minimum_host_available: int = field(init=False, default=1 << 62)
    minimum_vram_available: int = field(init=False, default=1 << 62)
    error: Exception | None = field(init=False, default=None)

    def sample(self) -> None:
        memory = psutil.virtual_memory()
        resident = tree_rss()
        if resident >= self.working_set_threshold:
            trim_process_tree_working_sets()
            memory = psutil.virtual_memory()
            resident = tree_rss()
        self.peak_tree_rss = max(self.peak_tree_rss, resident)
        self.minimum_host_available = min(self.minimum_host_available, memory.available)
        if resident >= self.working_set_threshold:
            print(
                f"working_set_stop=tree_rss:{resident / GIB:.2f}GiB",
                flush=True,
            )
            os._exit(137)
        free_vram, _ = torch.cuda.mem_get_info()
        self.minimum_vram_available = min(self.minimum_vram_available, free_vram)

    def watch(self) -> None:
        try:
            while not self.stop.wait(0.1):
                self.sample()
        except Exception as error:  # noqa: BLE001 - propagate from the monitor thread
            self.error = error
            self.stop.set()

    def __enter__(self) -> Self:
        self.started = perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        assert self.thread is not None
        self.thread.join()
        if self.error is not None:
            raise self.error
        self.sample()
        print(
            f"elapsed={perf_counter() - self.started:.3f}s "
            f"peak_tree_rss={self.peak_tree_rss / GIB:.2f}GiB "
            f"min_host_free={self.minimum_host_available / GIB:.2f}GiB "
            f"min_vram_free={self.minimum_vram_available / GIB:.2f}GiB",
            flush=True,
        )


def release_comfy(model_management: _ModelManagement) -> None:
    model_management.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()
    trim_process_tree_working_sets()


def run_h3_workflow(
    *,
    pipeline: str,
    nodes: _Nodes,
    folder_paths: _FolderPaths,
    prompt: str,
    output_prefix: str,
    working_set_gib: int,
    width: int,
    height: int,
    seconds: float,
    lora: str,
    steps: int,
    seed: int,
    lora_strength: float,
    video_shift: float,
    audio_shift: float,
    first_frame: Path | None,
    last_frame: Path | None,
    reference_video: Path | None,
    reference_audio: Path | None,
) -> None:
    """Execute one complete H3 graph through ComfyUI's workflow runtime."""
    import comfy.model_management
    import comfy.nested_tensor
    import comfy.sample
    import execution

    from nano_omni.models.h3 import noise
    from nano_omni.pipelines import h3_fl2va, h3_ref2va

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    input_directory = Path(folder_paths.get_input_directory())
    input_directory.mkdir(parents=True, exist_ok=True)
    copied_inputs: list[Path] = []

    def copy_input(source: Path | None) -> str | None:
        if source is None:
            return None
        target = input_directory / f"nano-omni-{uuid.uuid4().hex}{source.suffix}"
        shutil.copy2(source, target)
        copied_inputs.append(target)
        return target.name

    if pipeline == "h3_ref2va":
        assert reference_video is not None and reference_audio is not None
        graph = ref2va_workflow(
            prompt=prompt,
            output_prefix=output_prefix,
            width=width,
            height=height,
            seconds=seconds,
            checkpoint=h3_ref2va.DIFFUSION.name,
            lora_name=h3_ref2va.LORA.name,
            lora_strength=lora_strength,
            steps=steps,
            seed=seed,
            video_shift=video_shift,
            audio_shift=audio_shift,
            reference_video_name=cast(str, copy_input(reference_video)),
            reference_audio_name=cast(str, copy_input(reference_audio)),
        )
    else:
        lora_name = {
            "none": None,
            "turbo_4step": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "turbo_8step": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        }[lora]
        graph = fl2va_workflow(
            prompt=prompt,
            output_prefix=output_prefix,
            width=width,
            height=height,
            seconds=seconds,
            checkpoint=h3_fl2va.DIFFUSION.name,
            lora_name=lora_name,
            lora_strength=lora_strength,
            steps=steps,
            seed=seed,
            video_shift=video_shift,
            audio_shift=audio_shift,
            first_frame_name=copy_input(first_frame),
            last_frame_name=copy_input(last_frame),
        )
    prompt_id = uuid.uuid4().hex
    valid, error, outputs, node_errors = asyncio.run(
        execution.validate_prompt(prompt_id, graph, ["save"])
    )
    if not valid:
        raise RuntimeError(
            "ComfyUI H3 workflow validation failed: "
            + json.dumps({"error": error, "nodes": node_errors}, default=str)
        )

    def official_noise(
        samples: NestedTensor,
        noise_seed: int,
        noise_inds: Sequence[int] | None = None,
    ) -> NestedTensor:
        if noise_inds is not None:
            raise ValueError("H3 comparison expects one sample without batch indices")
        video, audio = samples.unbind()
        arrays = noise.generate(tuple(video.shape[1:]), audio.shape[-1], noise_seed)
        return comfy.nested_tensor.NestedTensor(
            tuple(torch.from_numpy(array).unsqueeze(0) for array in arrays)
        )

    cache_ram = min(10.0, max(2.0, comfy.model_management.total_ram * 0.10 / 1024.0))
    cache_ram_inactive = min(128.0, comfy.model_management.total_ram / 1024.0)
    executor = execution.PromptExecutor(
        _WorkflowServer(),
        cache_type=execution.CacheType.RAM_PRESSURE,
        cache_args={"lru": 0, "ram": cache_ram, "ram_inactive": cache_ram_inactive},
    )
    prepare_noise = comfy.sample.prepare_noise
    try:
        comfy.sample.prepare_noise = cast("Callable[..., NestedTensor]", official_noise)
        with Monitor(working_set_gib * GIB):
            executor.execute(graph, prompt_id, {}, outputs)
    finally:
        comfy.sample.prepare_noise = prepare_noise
        for path in copied_inputs:
            path.unlink(missing_ok=True)
    if not executor.success:
        raise RuntimeError(
            "ComfyUI H3 workflow execution failed: "
            + json.dumps(executor.status_messages, default=str)
        )


@torch.inference_mode()
def benchmark(
    pipeline: str,
    prompt: str,
    output_prefix: str,
    working_set_gib: int,
    width: int,
    height: int,
    seconds: float,
    lora: str,
    steps: int,
    seed: int = 0,
    lora_strength: float = 1.0,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    first_frame: Path | None = None,
    reference_video: Path | None = None,
    reference_audio: Path | None = None,
    last_frame: Path | None = None,
) -> None:
    sys.path.insert(0, str(COMFY_ROOT))
    import_monitor = Monitor(working_set_gib * GIB)
    import_monitor.__enter__()
    import comfy.model_management
    import comfy_extras.nodes_audio
    import comfy_extras.nodes_lt
    import comfy_extras.nodes_minimax_h3
    import comfy_extras.nodes_video
    import folder_paths
    import nodes

    import_monitor.__exit__()

    record_comfy_attention()
    print("environment=" + json.dumps(measurement_environment()), flush=True)

    destination = Path(output_prefix)
    folder_paths.set_output_directory(str(destination.parent))
    output_prefix = destination.name
    folder_paths.add_model_folder_path(
        "diffusion_models", str(MODELS / "diffusion_models")
    )
    folder_paths.add_model_folder_path("text_encoders", str(MODELS / "text_encoders"))
    folder_paths.add_model_folder_path("vae", str(MODELS / "vae"))
    folder_paths.add_model_folder_path("loras", str(MODELS / "loras"))

    if pipeline in ("h3_fl2va", "h3_ref2va"):
        run_h3_workflow(
            pipeline=pipeline,
            nodes=nodes,
            folder_paths=folder_paths,
            prompt=prompt,
            output_prefix=output_prefix,
            working_set_gib=working_set_gib,
            width=width,
            height=height,
            seconds=seconds,
            lora=lora,
            steps=steps,
            seed=seed,
            lora_strength=lora_strength,
            video_shift=video_shift,
            audio_shift=audio_shift,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_video=reference_video,
            reference_audio=reference_audio,
        )
        return

    with Monitor(working_set_gib * GIB):
        stage_started = perf_counter()
        (clip,) = nodes.CLIPLoader().load_clip(
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax"
        )

        def load_frame(path: Path | None) -> torch.Tensor | None:
            if path is None:
                return None
            import numpy
            from PIL import Image

            with Image.open(path) as source:
                pixels = numpy.asarray(source.convert("RGB")).copy()
            return torch.from_numpy(pixels).to(torch.float32).div_(255).unsqueeze(0)

        image = load_frame(first_frame)
        last_image = load_frame(last_frame)
        if pipeline == "h3_ref2va":
            from comfy_api.latest import InputImpl

            assert reference_video is not None and reference_audio is not None
            components = InputImpl.VideoFromFile(str(reference_video)).get_components()
            voice_waveform, voice_rate = comfy_extras.nodes_audio.load(
                str(reference_audio)
            )
            voice = {
                "waveform": voice_waveform.unsqueeze(0),
                "sample_rate": voice_rate,
            }
            (reference_video_vae,) = nodes.VAELoader().load_vae(
                "minimax_h3_video_vae_fp16.safetensors"
            )
            (reference_audio_vae,) = nodes.VAELoader().load_vae(
                "minimax_h3_audio_vae_fp32.safetensors"
            )
            positive, latent = (
                comfy_extras.nodes_minimax_h3.MiniMaxH3ReferenceToVideo.execute(
                    clip,
                    prompt,
                    width,
                    height,
                    round(seconds * 24),
                    vae=reference_video_vae,
                    audio_vae=reference_audio_vae,
                    ref_videos={"ref_video_0": components.images},
                    ref_video_audios={"ref_video_audio_0": components.audio},
                    ref_audios={"ref_audio_0": voice},
                ).args
            )
            del reference_video_vae, reference_audio_vae
        else:
            frame_vae = None
            if image is not None or last_image is not None:
                (frame_vae,) = nodes.VAELoader().load_vae(
                    "minimax_h3_video_vae_fp16.safetensors"
                )
            positive, latent = (
                comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo.execute(
                    clip,
                    frame_vae,
                    prompt,
                    width,
                    height,
                    round(seconds * 24),
                    first_frame=image,
                    last_frame=last_image,
                ).args
            )
            del frame_vae
        (negative,) = nodes.ConditioningZeroOut().zero_out(positive)
        del clip
        release_comfy(comfy.model_management)
        print(
            "comfy_stage="
            + json.dumps(
                {"name": "text_encoder", "seconds": perf_counter() - stage_started}
            ),
            flush=True,
        )

        stage_started = perf_counter()
        from nano_omni.pipelines import h3_fl2va, h3_ref2va

        checkpoint = (
            h3_ref2va.DIFFUSION if pipeline == "h3_ref2va" else h3_fl2va.DIFFUSION
        )
        print("comfy_checkpoint=" + checkpoint.name, flush=True)
        (model,) = nodes.UNETLoader().load_unet(checkpoint.name, "default")
        lora_name: str | None = None
        if pipeline == "h3_ref2va":
            lora_name = h3_ref2va.LORA.name
            (model,) = nodes.LoraLoaderModelOnly().load_lora_model_only(
                model, lora_name, lora_strength
            )
        elif lora != "none":
            lora_name = {
                "turbo_4step": (
                    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
                ),
                "turbo_8step": (
                    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
                ),
            }[lora]
            (model,) = nodes.LoraLoaderModelOnly().load_lora_model_only(
                model, lora_name, lora_strength
            )
        print(
            "comfy_lora="
            + json.dumps(
                {
                    "file": lora_name if lora != "none" else None,
                    "strength": lora_strength,
                    "video_shift": video_shift,
                    "audio_shift": audio_shift,
                }
            ),
            flush=True,
        )
        (model,) = comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute(
            model, video_shift, audio_shift
        ).args
        torch.cuda.synchronize()
        sampling_started = perf_counter()
        # Match the official H3 modality-specific CPU RNG instead of the
        # generic ComfyUI nested-latent RNG stream. Restore the hook after sampling.
        import comfy.nested_tensor
        import comfy.sample

        from nano_omni.models.h3 import noise

        prepare_noise = comfy.sample.prepare_noise

        def official_noise(
            samples: NestedTensor,
            noise_seed: int,
            noise_inds: Sequence[int] | None = None,
        ) -> NestedTensor:
            if noise_inds is not None:
                raise ValueError(
                    "H3 comparison expects one sample without batch indices"
                )
            video, audio = samples.unbind()
            arrays = noise.generate(tuple(video.shape[1:]), audio.shape[-1], noise_seed)
            return comfy.nested_tensor.NestedTensor(
                tuple(torch.from_numpy(x).unsqueeze(0) for x in arrays)
            )

        comfy.sample.prepare_noise = cast("Callable[..., NestedTensor]", official_noise)
        try:
            (sampled,) = nodes.KSampler().sample(
                model,
                seed,
                steps,
                1.0,
                "res_multistep",
                "simple",
                positive,
                negative,
                latent,
                1.0,
            )
        finally:
            comfy.sample.prepare_noise = prepare_noise
        torch.cuda.synchronize()
        print(
            "comfy_stage="
            + json.dumps(
                {"name": "sampling", "seconds": perf_counter() - sampling_started}
            ),
            flush=True,
        )
        del model, positive, negative, latent
        release_comfy(comfy.model_management)
        print(
            "comfy_stage="
            + json.dumps(
                {
                    "name": "diffusion_with_load",
                    "seconds": perf_counter() - stage_started,
                }
            ),
            flush=True,
        )

        stage_started = perf_counter()
        video_latent, audio_latent = comfy_extras.nodes_lt.LTXVSeparateAVLatent.execute(
            sampled
        ).args
        del sampled
        (video_vae,) = nodes.VAELoader().load_vae(
            "minimax_h3_video_vae_fp16.safetensors"
        )
        (frames,) = nodes.VAEDecode().decode(video_vae, video_latent)
        del video_vae, video_latent
        release_comfy(comfy.model_management)
        print(
            "comfy_stage="
            + json.dumps(
                {"name": "video_vae", "seconds": perf_counter() - stage_started}
            ),
            flush=True,
        )

        stage_started = perf_counter()
        (audio_vae,) = nodes.VAELoader().load_vae(
            "minimax_h3_audio_vae_fp32.safetensors"
        )
        audio: AudioInput
        (audio,) = comfy_extras.nodes_audio.VAEDecodeAudio.execute(
            audio_vae, audio_latent
        ).args
        del audio_vae, audio_latent
        release_comfy(comfy.model_management)
        print(
            "comfy_stage="
            + json.dumps(
                {"name": "audio_vae", "seconds": perf_counter() - stage_started}
            ),
            flush=True,
        )

        stage_started = perf_counter()
        frames = frames[: round(seconds * 24)]
        audio = {
            **audio,
            "waveform": audio["waveform"][..., : round(seconds * audio["sample_rate"])],
        }
        (video,) = comfy_extras.nodes_video.CreateVideo.execute(
            frames, 24.0, audio
        ).args
        comfy_extras.nodes_video.SaveVideo.execute(
            video,
            output_prefix,
            {"format": "mp4", "codec": {"codec": "h264"}},
        )
        print(
            "comfy_stage="
            + json.dumps(
                {"name": "encode_mux", "seconds": perf_counter() - stage_started}
            ),
            flush=True,
        )


CHILD = "NANO_OMNI_COMFY_CHILD"


def workload(config: RunConfig) -> dict[str, object]:
    """Describe dimensions shared by the native and ComfyUI measurements."""
    from nano_omni.models import h3

    frames = h3.frame_count(config.seconds)
    parameters = comfy_parameters(config)
    return {
        "pipeline": config.pipeline,
        "mode": (
            "ref2va"
            if config.pipeline == "h3_ref2va"
            else "fl2va"
            if config.first_frame is not None
            else "t2va"
        ),
        "width": config.width,
        "height": config.height,
        "requested_frames": h3.requested_frame_count(config.seconds),
        "frames": frames,
        "video_latent_frames": h3.video_latent_frames(frames),
        "audio_latent_frames": h3.audio_latent_frames(frames),
        "steps": config.steps,
        "seed": config.seed,
        "lora": parameters.lora,
        "lora_strength": config.lora_strength,
        "video_shift": parameters.video_shift,
        "audio_shift": parameters.audio_shift,
    }


def changed_outputs(output: Path, previous: dict[Path, tuple[int, int]]) -> list[Path]:
    """Return only media created or replaced by the current child run."""
    return sorted(
        (
            path
            for path in output.parent.glob(output.stem + "*" + output.suffix)
            if path not in previous
            or previous[path] != (path.stat().st_mtime_ns, path.stat().st_size)
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )


def child(config: RunConfig, *, disable_pinned_memory: bool = False) -> None:
    """Run one child measurement using the output path set by the parent."""
    initialize_comfy(disable_pinned_memory=disable_pinned_memory)
    assert config.output is not None
    parameters = comfy_parameters(config)
    output = Path(config.output).resolve()
    previous = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in output.parent.glob(output.stem + "*" + output.suffix)
    }
    benchmark(
        config.pipeline,
        config.read_prompt(ROOT),
        str(output.with_suffix("")),
        config.working_set_gib,
        config.width,
        config.height,
        config.seconds,
        parameters.lora,
        config.steps,
        config.seed,
        config.lora_strength,
        parameters.video_shift,
        parameters.audio_shift,
        None if config.first_frame is None else ROOT / config.first_frame,
        None if config.reference_video is None else ROOT / config.reference_video,
        None if config.reference_audio is None else ROOT / config.reference_audio,
        None if config.last_frame is None else ROOT / config.last_frame,
    )
    candidates = changed_outputs(output, previous)
    if not candidates:
        raise FileNotFoundError("ComfyUI did not produce the requested output")
    if candidates[-1] != output:
        candidates[-1].replace(output)


app = typer.Typer(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)


@app.command()
def main(
    context: typer.Context,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    warmup: Annotated[int, typer.Option(min=0)] = 0,
    repetitions: Annotated[int, typer.Option(min=1)] = 1,
    disable_pinned_memory: Annotated[bool, typer.Option()] = False,
    append: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run the matching ComfyUI baseline in measured child processes."""
    from common import (
        check_gpu_temperature,
        invoke,
        merge_measurements,
        overrides,
        resolve,
        timing_statistics,
        write_json,
    )

    try:
        values = overrides(context.args)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    resolved = resolve(config, dict(values))
    if resolved.pipeline not in ("h3_fl2va", "h3_ref2va"):
        raise typer.BadParameter("Unsupported ComfyUI pipeline")
    try:
        comfy_parameters(resolved)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if append and (warmup or repetitions != 1):
        raise typer.BadParameter("--append requires --warmup 0 and --repetitions 1")
    output = Path(resolved.output or ROOT / "outputs/h3.mp4")
    if not output.is_absolute():
        output = ROOT / output
    if os.environ.get(CHILD) == "1":
        child(resolved, disable_pinned_memory=disable_pinned_memory)
        return
    output = output.with_name("comfy" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    report = output.with_suffix(".json")
    previous: dict[str, object] | None = None
    if append:
        loaded = json.loads(report.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict) and loaded.get("status") == "complete"
        previous = loaded
        check_gpu_temperature(resolved.maximum_gpu_temperature_celsius)
    write_json(
        report, {"status": "incomplete", "output": output.name, "run_id": run_id}
    )
    environment = {**os.environ, CHILD: "1", "NANO_OMNI_RUN_ID": run_id}
    arguments = [
        sys.executable,
        str(ROOT / "scripts/run/comfy-ui.py"),
        "--config",
        str(config),
        *context.args,
        "--output",
        Path(os.path.relpath(output, ROOT)).as_posix(),
    ]
    if disable_pinned_memory:
        arguments.append("--disable-pinned-memory")
    for index in range(warmup):
        log = output.with_name(f".{output.stem}-warmup-{index}.log")
        code, _ = invoke(
            arguments,
            log,
            environment=environment,
            maximum_gpu_temperature_celsius=(
                None if append else resolved.maximum_gpu_temperature_celsius
            ),
        )
        if code:
            raise RuntimeError(f"ComfyUI warmup {index} failed; see {log}")
        log.unlink(missing_ok=True)
    samples = []
    child_metadata: dict[str, object] = {}
    log = output.with_suffix(".log")
    for index in range(repetitions):
        code, latency = invoke(
            arguments,
            log,
            environment=environment,
            maximum_gpu_temperature_celsius=(
                None if append else resolved.maximum_gpu_temperature_celsius
            ),
        )
        if code:
            raise RuntimeError(f"ComfyUI run {index} failed; see {log}")
        samples.append(latency)
        current = read_child_metadata(log)
        if index and current != child_metadata:
            raise ValueError("ComfyUI repetitions produced different backend metadata")
        child_metadata = current
    result: dict[str, object] = {
        "status": "complete",
        "output": output.name,
        "run_id": run_id,
        "resolved_config": resolved.model_dump(mode="json"),
        "workload": workload(resolved),
        "baseline": {
            "implementation": "ComfyUI PromptExecutor workflow",
            "adaptations": [
                "dynamic_vram",
                "fast_disk",
                "official_h3_noise",
            ],
            "pinned_memory": not disable_pinned_memory,
            "working_set_gib": resolved.working_set_gib,
        },
        "environment": child_metadata.get("environment"),
        "backend": child_metadata.get("backend"),
        "measurement": {
            "timing_boundary": "parent wall time from child launch through child exit",
            "process": "fresh child per sample",
            "cache_state": "uncontrolled existing OS and backend caches",
            "missing_assets": "downloaded inside the timed child",
        },
        "warmup": warmup,
        "repetitions": repetitions,
        "latency_seconds": samples,
        **timing_statistics(samples),
    }
    if previous is not None:
        result = merge_measurements(previous, result)
    write_json(report, result)


if __name__ == "__main__":
    app()
