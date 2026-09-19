from __future__ import annotations

import ctypes
import dataclasses
import math
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Self

import av
import numpy
from cuda.bindings import driver  # pyrefly: ignore [missing-module-attribute]

from nano_omni.core.runtime import execution, observation
from nano_omni.core.runtime.buffers import PipelineMemory
from nano_omni.core.runtime.execution import CudaRuntime, Stream
from nano_omni.core.tensor import TensorDesc
from nano_omni.models.h3 import FPS
from nano_omni.models.h3.audio_vae_decoder import (
    H3AudioVaeDecoder,
    H3AudioVaeDecoderArgs,
)
from nano_omni.models.h3.conditioning import H3Presentation, H3VisionInput
from nano_omni.models.h3.text_encoder import (
    H3TextEncoder,
    H3TextEncoderArgs,
)
from nano_omni.models.h3.video_vae_decoder import (
    H3VideoVaeDecoder,
    H3VideoVaeDecoderArgs,
)
from nano_omni.models.h3.vision_encoder import (
    H3VisionEncoder,
    H3VisionEncoderArgs,
)


def cuda_total_memory() -> int:
    """Read the selected CUDA device's total capacity in bytes."""
    execution.check_cuda(driver.cuInit(0))
    device = execution.check_cuda(driver.cuDeviceGet(0))
    return int(execution.check_cuda(driver.cuDeviceTotalMem(device)))


def encode_text(
    tokens: numpy.ndarray,
    encoder: H3TextEncoder,
    pipeline: PipelineMemory,
    runtime: CudaRuntime,
    working_set_threshold: int,
    visual: tuple[TensorDesc, TensorDesc, TensorDesc, TensorDesc]
    | None = None,
    presentation: H3Presentation | None = None,
) -> TensorDesc:
    """Upload token/position inputs and return the device text features."""
    with observation.stage("text_encoder", working_set_threshold):
        output = pipeline.empty((len(tokens), 5120), numpy.uint16)
        token_buffer = pipeline.upload(tokens)
        mrope = pipeline.upload(presentation.mrope_positions) if presentation else None
        frequencies = (
            pipeline.upload(presentation.inverse_frequencies) if presentation else None
        )
        encoder.run(
            H3TextEncoderArgs(
                token_buffer,
                output,
                runtime,
                *(visual or (None, None, None, None)),
                mrope,
                frequencies,
            )
        )
        pipeline.release(token_buffer, mrope, frequencies)
    return output


def encode_vision(
    vision: H3VisionInput,
    encoder: H3VisionEncoder,
    pipeline: PipelineMemory,
    runtime: CudaRuntime,
    working_set_threshold: int,
) -> tuple[TensorDesc, TensorDesc, TensorDesc, TensorDesc]:
    """Return final visual features followed by the three deepstack features."""
    with observation.stage("vision_encoder", working_set_threshold):
        rows = vision.merged_tokens
        merged, deepstack_0, deepstack_1, deepstack_2 = (
            pipeline.empty((rows, 5120), numpy.uint16) for _ in range(4)
        )
        outputs = (merged, deepstack_0, deepstack_1, deepstack_2)
        patches, indices, weights, positions, frequencies = (
            pipeline.upload(value)
            for value in (
                vision.patches,
                vision.position_indices,
                vision.position_weights,
                vision.rope_positions,
                vision.inverse_frequencies,
            )
        )
        encoder.run(
            H3VisionEncoderArgs(
                patches,
                indices,
                weights,
                positions,
                frequencies,
                *outputs,
                runtime,
            )
        )
        pipeline.release(patches, indices, weights, positions, frequencies)
    return outputs


def decode_video(
    latent: TensorDesc,
    vae: H3VideoVaeDecoder,
    pipeline: PipelineMemory,
    runtime: CudaRuntime,
    working_set_threshold: int,
    output_frames: int,
) -> list[H264Packet]:
    """Decode device latents to NV12 frames and encode them directly with NVENC."""
    with observation.stage("video_vae", working_set_threshold):
        nv12 = vae.run(H3VideoVaeDecoderArgs(latent, runtime))
        config = vae.config
        assert output_frames <= config.frames
        runtime.synchronize(Stream.COMPUTE)
        with H264Encoder(
            runtime.context, config.width, config.height, FPS, output_frames
        ) as encoder:
            packets = encoder.encode(nv12.data_ptr(), config.pitch, output_frames)
        pipeline.release(latent)
    return packets


def decode_audio(
    latent: TensorDesc,
    vae: H3AudioVaeDecoder,
    pipeline: PipelineMemory,
    runtime: CudaRuntime,
    working_set_threshold: int,
) -> numpy.ndarray:
    """Decode audio latents and download the resulting stereo waveform."""
    spec = vae.spec
    with observation.stage("audio_vae", working_set_threshold):
        output = pipeline.empty(
            (spec.stereo, spec.latent_frames * 800),
            numpy.float32,
        )
        vae.run(H3AudioVaeDecoderArgs(latent, output, runtime))
        waveform = pipeline.download(output)
        pipeline.release(latent, output)
    return waveform


NVENC_API_VERSION = 13 | (1 << 24)
NV_ENC_BUFFER_FORMAT_NV12 = 1
NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR = 1
NV_ENC_INPUT_IMAGE = 0
NV_ENC_PIC_STRUCT_FRAME = 1
NV_ENC_PIC_FLAG_FORCE_IDR = 2
NV_ENC_PIC_FLAG_OUTPUT_SPSPPS = 4
NV_ENC_PIC_FLAG_EOS = 8
NV_ENC_ERR_NEED_MORE_INPUT = 17


def _struct_version(version: int) -> int:
    return NVENC_API_VERSION | (version << 16) | (7 << 28)


class Guid(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_uint8 * 8),
    ]


def _guid(data1: int, data2: int, data3: int, data4: tuple[int, ...]) -> Guid:
    return Guid(data1, data2, data3, (ctypes.c_uint8 * 8)(*data4))


H264_GUID = _guid(
    0x6BC82762, 0x4E63, 0x4CA4, (0xAA, 0x85, 0x1E, 0x50, 0xF3, 0x21, 0xF6, 0xBF)
)
PRESET_P4_GUID = _guid(
    0x90A7B826, 0xDF06, 0x4862, (0xB9, 0xD2, 0xCD, 0x6D, 0x73, 0xA0, 0x86, 0x81)
)

NV_ENC_PARAMS_RC_VBR = 1
NV_ENC_TUNING_INFO_HIGH_QUALITY = 1
NV_ENC_B_FRAMES = 3
NV_ENC_AVERAGE_BIT_RATE = 3_000_000
NV_ENC_MAX_BIT_RATE = 6_000_000
NV_ENC_TARGET_QUALITY = 23


class FunctionList(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("functions", ctypes.c_void_p * 43),
        ("reserved2", ctypes.c_void_p * 275),
    ]


class OpenSession(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("device_type", ctypes.c_uint32),
        ("device", ctypes.c_void_p),
        ("reserved", ctypes.c_void_p),
        ("api_version", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32 * 253),
        ("reserved2", ctypes.c_void_p * 64),
    ]


class InitializeParams(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("encode_guid", Guid),
        ("preset_guid", Guid),
        ("encode_width", ctypes.c_uint32),
        ("encode_height", ctypes.c_uint32),
        ("dar_width", ctypes.c_uint32),
        ("dar_height", ctypes.c_uint32),
        ("frame_rate_num", ctypes.c_uint32),
        ("frame_rate_den", ctypes.c_uint32),
        ("enable_encode_async", ctypes.c_uint32),
        ("enable_ptd", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("priv_data_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
        ("encode_config", ctypes.c_void_p),
        ("max_width", ctypes.c_uint32),
        ("max_height", ctypes.c_uint32),
        ("max_me_hint_counts", ctypes.c_uint32 * 8),
        ("tuning_info", ctypes.c_uint32),
        ("buffer_format", ctypes.c_uint32),
        ("num_state_buffers", ctypes.c_uint32),
        ("output_stats_level", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32 * 284),
        ("reserved2", ctypes.c_void_p * 64),
    ]


class EncodeQp(ctypes.Structure):
    _fields_ = [
        ("inter_p", ctypes.c_uint32),
        ("inter_b", ctypes.c_uint32),
        ("intra", ctypes.c_uint32),
    ]


class RateControlParams(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("constant_qp", EncodeQp),
        ("average_bit_rate", ctypes.c_uint32),
        ("max_bit_rate", ctypes.c_uint32),
        ("vbv_buffer_size", ctypes.c_uint32),
        ("vbv_initial_delay", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("minimum_qp", EncodeQp),
        ("maximum_qp", EncodeQp),
        ("initial_qp", EncodeQp),
        ("temporal_layer_index_mask", ctypes.c_uint32),
        ("temporal_layer_qp", ctypes.c_uint8 * 8),
        ("target_quality", ctypes.c_uint8),
        ("target_quality_lsb", ctypes.c_uint8),
        ("lookahead_depth", ctypes.c_uint16),
        ("low_delay_keyframe_scale", ctypes.c_uint8),
        ("y_dc_qp_index_offset", ctypes.c_int8),
        ("u_dc_qp_index_offset", ctypes.c_int8),
        ("v_dc_qp_index_offset", ctypes.c_int8),
        ("qp_map_mode", ctypes.c_uint32),
        ("multi_pass", ctypes.c_uint32),
        ("alpha_layer_bit_rate_ratio", ctypes.c_uint32),
        ("cb_qp_index_offset", ctypes.c_int8),
        ("cr_qp_index_offset", ctypes.c_int8),
        ("reserved2", ctypes.c_uint16),
        ("lookahead_level", ctypes.c_uint32),
        ("view_bit_rate_ratios", ctypes.c_uint8 * 7),
        ("reserved3", ctypes.c_uint8),
        ("reserved1", ctypes.c_uint32),
    ]


class EncodeConfig(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("profile_guid", Guid),
        ("gop_length", ctypes.c_uint32),
        ("frame_interval_p", ctypes.c_int32),
        ("monochrome", ctypes.c_uint32),
        ("frame_field_mode", ctypes.c_uint32),
        ("motion_vector_precision", ctypes.c_uint32),
        ("rate_control", RateControlParams),
        ("codec", ctypes.c_uint8 * 1792),
        ("reserved", ctypes.c_uint32 * 278),
        ("reserved2", ctypes.c_void_p * 64),
    ]


class PresetConfig(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("config", EncodeConfig),
        ("reserved1", ctypes.c_uint32 * 256),
        ("reserved2", ctypes.c_void_p * 64),
    ]


class RegisterResource(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("resource_type", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pitch", ctypes.c_uint32),
        ("subresource_index", ctypes.c_uint32),
        ("resource", ctypes.c_void_p),
        ("registered", ctypes.c_void_p),
        ("buffer_format", ctypes.c_uint32),
        ("buffer_usage", ctypes.c_uint32),
        ("input_fence", ctypes.c_void_p),
        ("chroma_offset", ctypes.c_uint32 * 2),
        ("chroma_offset_in", ctypes.c_uint32 * 2),
        ("reserved1", ctypes.c_uint32 * 244),
        ("reserved2", ctypes.c_void_p * 61),
    ]


class MapInputResource(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("subresource_index", ctypes.c_uint32),
        ("input_resource", ctypes.c_void_p),
        ("registered", ctypes.c_void_p),
        ("mapped", ctypes.c_void_p),
        ("mapped_format", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32 * 251),
        ("reserved2", ctypes.c_void_p * 63),
    ]


class CreateBitstreamBuffer(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("memory_heap", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("buffer", ctypes.c_void_p),
        ("buffer_pointer", ctypes.c_void_p),
        ("reserved1", ctypes.c_uint32 * 58),
        ("reserved2", ctypes.c_void_p * 64),
    ]


class PictureParams(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("input_width", ctypes.c_uint32),
        ("input_height", ctypes.c_uint32),
        ("input_pitch", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frame_index", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint64),
        ("duration", ctypes.c_uint64),
        ("input_buffer", ctypes.c_void_p),
        ("output_bitstream", ctypes.c_void_p),
        ("completion_event", ctypes.c_void_p),
        ("buffer_format", ctypes.c_uint32),
        ("picture_struct", ctypes.c_uint32),
        ("picture_type", ctypes.c_uint32),
        ("tail", ctypes.c_uint8 * (2840 - 76)),
    ]


class LockBitstream(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("output_bitstream", ctypes.c_void_p),
        ("slice_offsets", ctypes.c_void_p),
        ("frame_index", ctypes.c_uint32),
        ("hardware_status", ctypes.c_uint32),
        ("num_slices", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint64),
        ("duration", ctypes.c_uint64),
        ("data", ctypes.c_void_p),
        ("tail", ctypes.c_uint8 * (1544 - 64)),
    ]


@dataclasses.dataclass(frozen=True, slots=True)
class H264Packet:
    """Encoded bytes with timestamps and duration measured in video frames."""

    data: bytes
    timestamp: int
    duration: int


class NvencError(RuntimeError): ...


@dataclasses.dataclass
class _Frame:
    pointer: int
    registered: int
    bitstream: int
    mapped: int = 0


class H264Encoder:
    """Encode CUDA NV12 surfaces with one seekable GOP per generated clip."""

    def __init__(
        self,
        context: driver.CUcontext,
        width: int,
        height: int,
        fps: int,
        frames: int,
    ) -> None:
        assert width % 2 == 0 and height % 2 == 0
        assert frames > 0
        self.width = width
        self.height = height
        self.library = ctypes.WinDLL("nvEncodeAPI64.dll")
        self.functions = FunctionList()
        self.functions.version = _struct_version(2)
        create = self.library.NvEncodeAPICreateInstance
        create.argtypes = [ctypes.POINTER(FunctionList)]
        create.restype = ctypes.c_int
        self._check(create(ctypes.byref(self.functions)), "create API instance")
        self.encoder = ctypes.c_void_p()
        open_params = OpenSession()
        open_params.version = _struct_version(1)
        open_params.device_type = 1
        open_params.device = int(context)
        open_params.api_version = NVENC_API_VERSION
        self._check(
            self._call(
                29, ctypes.POINTER(OpenSession), ctypes.POINTER(ctypes.c_void_p)
            )(ctypes.byref(open_params), ctypes.byref(self.encoder)),
            "open encode session",
        )
        initialize = InitializeParams()
        initialize.version = _struct_version(7) | (1 << 31)
        initialize.encode_guid = H264_GUID
        initialize.preset_guid = PRESET_P4_GUID
        initialize.encode_width = width
        initialize.encode_height = height
        initialize.dar_width = width
        initialize.dar_height = height
        initialize.frame_rate_num = fps
        initialize.frame_rate_den = 1
        initialize.enable_ptd = 1
        initialize.tuning_info = NV_ENC_TUNING_INFO_HIGH_QUALITY
        preset = PresetConfig()
        preset.version = _struct_version(5) | (1 << 31)
        preset.config.version = _struct_version(9) | (1 << 31)
        self._check(
            self._call(
                39,
                ctypes.c_void_p,
                Guid,
                Guid,
                ctypes.c_uint32,
                ctypes.POINTER(PresetConfig),
            )(
                self.encoder,
                H264_GUID,
                PRESET_P4_GUID,
                NV_ENC_TUNING_INFO_HIGH_QUALITY,
                ctypes.byref(preset),
            ),
            "get encode preset config",
        )
        self.encode_config = preset.config
        self.encode_config.gop_length = frames
        self.encode_config.frame_interval_p = NV_ENC_B_FRAMES + 1
        self.encode_config.rate_control.mode = NV_ENC_PARAMS_RC_VBR
        self.encode_config.rate_control.average_bit_rate = NV_ENC_AVERAGE_BIT_RATE
        self.encode_config.rate_control.max_bit_rate = NV_ENC_MAX_BIT_RATE
        self.encode_config.rate_control.target_quality = NV_ENC_TARGET_QUALITY
        initialize.encode_config = ctypes.addressof(self.encode_config)
        self._invoke(11, initialize)
        bitstream = CreateBitstreamBuffer()
        bitstream.version = _struct_version(1)
        self._invoke(14, bitstream)
        self.bitstream = bitstream.buffer

    def _call(self, index: int, *arguments: type) -> ctypes._CFuncPtr:
        return ctypes.WINFUNCTYPE(ctypes.c_int, *arguments)(
            self.functions.functions[index]
        )

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise NvencError(f"NVENC {operation} failed with status {status}")

    def _invoke(
        self, index: int, parameter: ctypes.Structure | ctypes.c_void_p | int | None
    ) -> None:
        # Session methods take either a structure pointer or an opaque handle.
        pointer = (
            ctypes.byref(parameter)
            if isinstance(parameter, ctypes.Structure)
            else parameter
        )
        status = self._call(index, ctypes.c_void_p, ctypes.c_void_p)(
            self.encoder, pointer
        )
        self._check(status, f"function {index}")

    def encode(self, pointer: int, pitch: int, frames: int) -> list[H264Packet]:
        """Queue frames and reuse outputs only after reordered packets are collected."""
        frame_bytes = pitch * self.height * 3 // 2
        # NVENC requires at least four output buffers plus the B-frame depth.
        depth = min(4 + NV_ENC_B_FRAMES, frames)
        slots = []
        pending = []
        packets = []
        try:
            for index in range(frames):
                address = pointer + index * frame_bytes
                resource = RegisterResource()
                resource.version = _struct_version(5)
                resource.resource_type = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR
                resource.width = self.width
                resource.height = self.height
                resource.pitch = pitch
                resource.resource = address
                resource.buffer_format = NV_ENC_BUFFER_FORMAT_NV12
                resource.buffer_usage = NV_ENC_INPUT_IMAGE
                self._invoke(30, resource)
                slot = _Frame(address, resource.registered, 0)
                slots.append(slot)
                if index == 0:
                    slot.bitstream = self.bitstream
                elif index < depth:
                    output = CreateBitstreamBuffer()
                    output.version = _struct_version(1)
                    self._invoke(14, output)
                    slot.bitstream = output.buffer
                else:
                    slot.bitstream = slots[index % depth].bitstream
            for index in range(frames):
                if len(pending) == depth:
                    packets.append(self._collect(pending.pop(0)))
                slot = slots[index]
                mapped = MapInputResource()
                mapped.version = _struct_version(4)
                mapped.registered = slot.registered
                self._invoke(25, mapped)
                slot.mapped = mapped.mapped
                picture = PictureParams()
                picture.version = _struct_version(7) | 1 << 31
                picture.input_width = self.width
                picture.input_height = self.height
                picture.input_pitch = pitch
                picture.flags = (
                    NV_ENC_PIC_FLAG_FORCE_IDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS
                    if index == 0
                    else 0
                )
                picture.frame_index = index
                picture.timestamp = index
                picture.duration = 1
                picture.input_buffer = slot.mapped
                picture.output_bitstream = slot.bitstream
                picture.buffer_format = mapped.mapped_format
                picture.picture_struct = NV_ENC_PIC_STRUCT_FRAME
                status = self._call(16, ctypes.c_void_p, ctypes.c_void_p)(
                    self.encoder, ctypes.byref(picture)
                )
                if status not in (0, NV_ENC_ERR_NEED_MORE_INPUT):
                    self._check(status, "encode picture")
                pending.append(slot)
            eos = PictureParams()
            eos.version = _struct_version(7) | 1 << 31
            eos.flags = NV_ENC_PIC_FLAG_EOS
            self._invoke(16, eos)
            for slot in pending:
                packets.append(self._collect(slot))
            return packets
        finally:
            for slot in reversed(slots):
                if slot.mapped:
                    self._invoke(26, slot.mapped)
                self._invoke(31, slot.registered)
            for bitstream in dict.fromkeys(
                slot.bitstream
                for slot in slots
                if slot.bitstream and slot.bitstream != self.bitstream
            ):
                self._invoke(15, bitstream)

    def _collect(self, slot: _Frame) -> H264Packet:
        locked = LockBitstream()
        locked.version = _struct_version(2) | 1 << 31
        locked.output_bitstream = slot.bitstream
        self._invoke(17, locked)
        try:
            packet = H264Packet(
                ctypes.string_at(locked.data, locked.size),
                int(locked.timestamp),
                int(locked.duration),
            )
        finally:
            self._invoke(18, slot.bitstream)
        self._invoke(26, slot.mapped)
        slot.mapped = 0
        return packet

    def close(self) -> None:
        """Release the reusable output buffer and the encoder session."""
        if self.encoder:
            self._invoke(15, self.bitstream)
            status = self._call(27, ctypes.c_void_p)(self.encoder)
            self._check(status, "destroy encoder")
            self.encoder = ctypes.c_void_p()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def h264_aac(
    video_packets: list[H264Packet],
    waveform: numpy.ndarray,
    output: Path,
    *,
    fps: int,
    sample_rate: int,
) -> None:
    """Mux H.264 packets and a sample-aligned AAC waveform into an MP4 file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = numpy.clip(waveform, -1.0, 1.0).astype(numpy.float32)
    expected_samples = round(len(video_packets) / fps * sample_rate)
    if samples.shape[1] < expected_samples:
        samples = numpy.pad(samples, ((0, 0), (0, expected_samples - samples.shape[1])))
    else:
        samples = samples[:, :expected_samples]
    # The MP4 edit list must represent both frame and sample boundaries exactly.
    with av.open(
        str(output),
        mode="w",
        options={"movie_timescale": str(math.lcm(fps, sample_rate))},
    ) as container:
        video_stream = container.add_stream("h264", rate=fps)
        video_stream.time_base = Fraction(1, fps)
        audio_stream = container.add_stream("aac", rate=sample_rate)
        audio_stream.layout = "stereo"
        decode_offset = min(
            encoded.timestamp - index for index, encoded in enumerate(video_packets)
        )
        for index, encoded in enumerate(video_packets):
            packet = av.Packet(encoded.data)
            packet.stream = video_stream
            packet.pts = encoded.timestamp
            packet.dts = index + decode_offset
            packet.duration = encoded.duration
            packet.time_base = Fraction(1, fps)
            packet.is_keyframe = encoded.timestamp == 0
            container.mux(packet)
        audio_frame = av.AudioFrame.from_ndarray(
            samples, format="fltp", layout="stereo"
        )
        audio_frame.sample_rate = sample_rate
        # Timestamp the original samples so AAC can emit negative priming PTS
        # and the MP4 muxer can describe the encoder delay without shifting content.
        audio_frame.pts = 0
        audio_frame.time_base = Fraction(1, sample_rate)
        for packet in audio_stream.encode(audio_frame):
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
