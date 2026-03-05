#!/usr/bin/env python3
"""Record three microphone sources and save as separate MP3 files.

Sources:
1) summary_0
2) summary_1
3) keyword

Each stream uses a simple noise gate:
- samples with absolute amplitude below threshold are set to 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyaudio
from pydub import AudioSegment


DEFAULT_RATE = 24000
DEFAULT_CHUNK = 4800
MAX_DURATION_SEC = 180.0


def list_input_devices() -> None:
    pa = pyaudio.PyAudio()
    try:
        print("\nInput devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                print(
                    f"  [{i:2d}] {info.get('name')} "
                    f"(channels={int(info.get('maxInputChannels', 0))}, "
                    f"default_sr={int(info.get('defaultSampleRate', 0))})"
                )
        print()
    finally:
        pa.terminate()


def apply_noise_gate(chunk: bytes, threshold: int) -> bytes:
    if threshold <= 0:
        return chunk
    arr = np.frombuffer(chunk, dtype=np.int16).copy()
    arr[np.abs(arr) < threshold] = 0
    return arr.tobytes()


def write_mp3(raw_pcm: bytes, output_path: Path, sample_rate: int) -> None:
    segment = AudioSegment(
        data=raw_pcm,
        sample_width=2,  # int16
        frame_rate=sample_rate,
        channels=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment.export(output_path, format="mp3")


def record_three_sources(
    summary0_device: int,
    summary1_device: int,
    keyword_device: int,
    duration_sec: float,
    threshold: int,
    sample_rate: int,
    chunk_size: int,
) -> tuple[bytes, bytes, bytes]:
    pa = pyaudio.PyAudio()
    streams: dict[str, pyaudio.Stream] = {}
    buffers: dict[str, list[bytes]] = {
        "summary_0": [],
        "summary_1": [],
        "keyword": [],
    }

    try:
        streams["summary_0"] = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=summary0_device,
            frames_per_buffer=chunk_size,
        )
        streams["summary_1"] = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=summary1_device,
            frames_per_buffer=chunk_size,
        )
        streams["keyword"] = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=keyword_device,
            frames_per_buffer=chunk_size,
        )

        print(
            f"Recording {duration_sec:.1f}s | threshold={threshold} | "
            f"rate={sample_rate} | chunk={chunk_size}"
        )
        print(
            f"summary_0={summary0_device}, summary_1={summary1_device}, keyword={keyword_device}"
        )

        start = time.time()
        while time.time() - start < duration_sec:
            for key, stream in streams.items():
                chunk = stream.read(chunk_size, exception_on_overflow=False)
                buffers[key].append(apply_noise_gate(chunk, threshold))
    finally:
        for stream in streams.values():
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()

    return (
        b"".join(buffers["summary_0"]),
        b"".join(buffers["summary_1"]),
        b"".join(buffers["keyword"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record 3 mics (summary_0, summary_1, keyword) and save MP3 files."
    )
    parser.add_argument("--list", action="store_true", help="List input devices and exit")
    parser.add_argument("--summary0-device", type=int, help="Input device index for summary_0")
    parser.add_argument("--summary1-device", type=int, help="Input device index for summary_1")
    parser.add_argument("--keyword-device", type=int, help="Input device index for keyword")
    parser.add_argument(
        "--duration",
        type=float,
        default=MAX_DURATION_SEC,
        help=f"Recording duration in seconds (max {int(MAX_DURATION_SEC)}s)",
    )
    parser.add_argument("--threshold", type=int, default=300, help="Noise gate threshold (0 disables)")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_RATE, help="Sample rate")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK, help="Chunk size in frames")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("live/logs/audio"),
        help="Output directory for MP3 files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix (e.g., sessionA_)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        list_input_devices()
        return 0

    if args.duration > MAX_DURATION_SEC:
        print(
            f"Requested duration {args.duration:.1f}s exceeds max "
            f"{MAX_DURATION_SEC:.0f}s. Using {MAX_DURATION_SEC:.0f}s."
        )
        args.duration = MAX_DURATION_SEC

    required = [args.summary0_device, args.summary1_device, args.keyword_device]
    if any(v is None for v in required):
        print(
            "Missing required devices. Use --summary0-device, --summary1-device, --keyword-device "
            "or run with --list first."
        )
        return 2

    raw_sum0, raw_sum1, raw_keyword = record_three_sources(
        summary0_device=int(args.summary0_device),
        summary1_device=int(args.summary1_device),
        keyword_device=int(args.keyword_device),
        duration_sec=float(args.duration),
        threshold=int(args.threshold),
        sample_rate=int(args.sample_rate),
        chunk_size=int(args.chunk_size),
    )

    prefix = args.prefix or ""
    out_sum0 = args.out_dir / f"{prefix}summary_0.mp3"
    out_sum1 = args.out_dir / f"{prefix}summary_1.mp3"
    out_keyword = args.out_dir / f"{prefix}keyword.mp3"

    write_mp3(raw_sum0, out_sum0, sample_rate=int(args.sample_rate))
    write_mp3(raw_sum1, out_sum1, sample_rate=int(args.sample_rate))
    write_mp3(raw_keyword, out_keyword, sample_rate=int(args.sample_rate))

    print("Saved:")
    print(f"  - {out_sum0}")
    print(f"  - {out_sum1}")
    print(f"  - {out_keyword}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
