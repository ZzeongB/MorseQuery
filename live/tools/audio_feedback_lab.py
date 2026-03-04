#!/usr/bin/env python3
"""
Audio feedback (earcon) tester for MorseQuery realtime UI.

No third-party dependencies required.

Usage:
  python live/tools/audio_feedback_lab.py list
  python live/tools/audio_feedback_lab.py play confirmed
  python live/tools/audio_feedback_lab.py play loading --seconds 4
  python live/tools/audio_feedback_lab.py play demo
  python live/tools/audio_feedback_lab.py export error --out /tmp/error.wav
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from typing import Iterable, List, Sequence, Tuple

SAMPLE_RATE = 44100

# Tone event: (frequency_hz, duration_ms, volume_0_1, waveform)
Tone = Tuple[float, int, float, str]


def _tone_samples(
    freq: float,
    duration_ms: int,
    volume: float = 0.12,
    waveform: str = "sine",
    sample_rate: int = SAMPLE_RATE,
) -> List[float]:
    total = int(sample_rate * (duration_ms / 1000.0))
    if total <= 0:
        return []

    out: List[float] = []
    for i in range(total):
        t = i / sample_rate
        phase = 2.0 * math.pi * freq * t

        if waveform == "square":
            sample = 1.0 if math.sin(phase) >= 0 else -1.0
        elif waveform == "triangle":
            sample = (2.0 / math.pi) * math.asin(math.sin(phase))
        else:
            sample = math.sin(phase)

        # Short fade-in/out to avoid clicks.
        fade = min(1.0, i / 120.0, (total - i) / 120.0)
        out.append(sample * volume * fade)

    return out


def _silence_samples(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> List[float]:
    total = int(sample_rate * (duration_ms / 1000.0))
    return [0.0] * max(total, 0)


def _sweep_samples(
    start_freq: float,
    end_freq: float,
    duration_ms: int,
    volume: float = 0.05,
    waveform: str = "sine",
    sample_rate: int = SAMPLE_RATE,
) -> List[float]:
    total = int(sample_rate * (duration_ms / 1000.0))
    if total <= 0:
        return []

    out: List[float] = []
    for i in range(total):
        t = i / sample_rate
        progress = i / max(total - 1, 1)
        freq = start_freq + (end_freq - start_freq) * progress
        phase = 2.0 * math.pi * freq * t

        if waveform == "square":
            sample = 1.0 if math.sin(phase) >= 0 else -1.0
        elif waveform == "triangle":
            sample = (2.0 / math.pi) * math.asin(math.sin(phase))
        else:
            sample = math.sin(phase)

        # Smooth Hann-like envelope for softer attack/release.
        env = 0.5 - 0.5 * math.cos(2.0 * math.pi * progress)
        out.append(sample * volume * env)

    return out


def render_sequence(seq: Sequence[Tone], gap_ms: int = 80) -> List[float]:
    samples: List[float] = []
    for idx, (freq, dur, vol, wave_type) in enumerate(seq):
        samples.extend(_tone_samples(freq, dur, vol, wave_type))
        if idx < len(seq) - 1 and gap_ms > 0:
            samples.extend(_silence_samples(gap_ms))
    return samples


def write_wav(path: str, samples: Iterable[float], sample_rate: int = SAMPLE_RATE) -> None:
    frames = bytearray()
    for x in samples:
        clamped = max(-1.0, min(1.0, x))
        frames.extend(struct.pack("<h", int(clamped * 32767)))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


def _find_player() -> List[str] | None:
    # macOS
    if shutil.which("afplay"):
        return ["afplay"]
    # Linux
    if shutil.which("aplay"):
        return ["aplay"]
    if shutil.which("paplay"):
        return ["paplay"]
    return None


def play_wav(path: str) -> bool:
    player = _find_player()
    if not player:
        return False
    try:
        subprocess.run(player + [path], check=True)
        return True
    except Exception:
        return False


def build_named_pattern(name: str, seconds: float = 3.0) -> Tuple[List[float], str]:
    name = name.lower()

    if name == "confirmed":
        # Rising two-tone confirm.
        return render_sequence([(880, 80, 0.10, "sine"), (1174, 80, 0.10, "sine")]), "confirmed"

    if name == "loading_soft":
        # Smooth "breathing" pulse: gentle upward sweep + pause.
        cycle_ms = 700
        cycles = max(1, int((seconds * 1000) / cycle_ms))
        samples: List[float] = []
        for i in range(cycles):
            samples.extend(_sweep_samples(480, 620, 280, 0.045, "sine"))
            # soft tail so pulses are less abrupt
            samples.extend(_tone_samples(620, 80, 0.025, "sine"))
            if i < cycles - 1:
                samples.extend(_silence_samples(cycle_ms - 360))
        return samples, f"loading_soft ({seconds:.1f}s)"

    if name == "loading":
        # Confirmed(880 -> 1174)와 잘 붙는 하모닉 로딩 루프.
        # 밝지만 과하지 않게, 짧은 2음 + 충분한 휴지.
        cycle_ms = 900
        cycles = max(1, int((seconds * 1000) / cycle_ms))
        samples: List[float] = []
        for i in range(cycles):
            samples.extend(_tone_samples(988, 55, 0.040, "sine"))
            samples.extend(_silence_samples(85))
            samples.extend(_tone_samples(1174, 65, 0.042, "sine"))
            if i < cycles - 1:
                samples.extend(_silence_samples(cycle_ms - 205))
        return samples, f"loading_match ({seconds:.1f}s)"

    if name == "loading_clock":
        # Clock ticking feel: "tick-tock" pair every ~1s.
        cycle_ms = 1000
        cycles = max(1, int((seconds * 1000) / cycle_ms))
        samples: List[float] = []
        for i in range(cycles):
            # tick
            samples.extend(_tone_samples(1850, 18, 0.05, "square"))
            samples.extend(_tone_samples(1500, 20, 0.03, "sine"))
            samples.extend(_silence_samples(460))
            # tock
            samples.extend(_tone_samples(1300, 24, 0.055, "square"))
            samples.extend(_tone_samples(980, 24, 0.03, "sine"))
            if i < cycles - 1:
                samples.extend(_silence_samples(454))
        return samples, f"loading_clock ({seconds:.1f}s)"

    if name == "judging":
        # Mid-low triple hit.
        return render_sequence(
            [(520, 50, 0.09, "triangle"), (520, 50, 0.09, "triangle"), (520, 50, 0.09, "triangle")],
            gap_ms=120,
        ), "judging"

    if name == "skipped":
        return render_sequence([(440, 100, 0.09, "sine")], gap_ms=0), "skipped"

    if name == "error":
        # Descending two-tone error.
        return render_sequence([(700, 100, 0.10, "sine"), (420, 140, 0.11, "sine")]), "error"

    if name == "tts_start":
        return render_sequence([(1200, 40, 0.06, "square")], gap_ms=0), "tts_start"

    if name == "tts_end":
        return render_sequence([(500, 40, 0.06, "square")], gap_ms=0), "tts_end"

    if name == "tap":
        return render_sequence([(900, 45, 0.08, "sine")], gap_ms=0), "tap"

    if name == "doubletap":
        return render_sequence([(900, 45, 0.08, "sine"), (900, 45, 0.08, "sine")], gap_ms=120), "doubletap"

    if name == "longpress":
        return render_sequence([(780, 180, 0.09, "triangle")], gap_ms=0), "longpress"

    if name == "demo":
        demo_order = [
            "confirmed",
            "loading",
            "judging",
            "skipped",
            "error",
            "tts_start",
            "tts_end",
            "tap",
            "doubletap",
            "longpress",
        ]
        combined: List[float] = []
        for idx, key in enumerate(demo_order):
            chunk, _ = build_named_pattern(key, seconds=1.5 if key == "loading" else seconds)
            combined.extend(chunk)
            if idx < len(demo_order) - 1:
                combined.extend(_silence_samples(350))
        return combined, "demo"

    raise ValueError(f"Unknown pattern: {name}")


def print_pattern_list() -> None:
    print("Available patterns:")
    print("  confirmed")
    print("  loading")
    print("  loading_clock")
    print("  loading_soft")
    print("  judging")
    print("  skipped")
    print("  error")
    print("  tts_start")
    print("  tts_end")
    print("  tap")
    print("  doubletap")
    print("  longpress")
    print("  demo")


def cmd_play(args: argparse.Namespace) -> int:
    try:
        samples, label = build_named_pattern(args.pattern, seconds=args.seconds)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name

    write_wav(temp_path, samples)
    ok = play_wav(temp_path)
    os.unlink(temp_path)

    if ok:
        print(f"Played: {label}")
        return 0

    print(f"Audio player not found. Try export instead. (pattern={label})", file=sys.stderr)
    return 1


def cmd_export(args: argparse.Namespace) -> int:
    try:
        samples, label = build_named_pattern(args.pattern, seconds=args.seconds)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_wav(out, samples)
    print(f"Exported: {label} -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Earcon test tool for realtime audio-only feedback.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available patterns")
    p_list.set_defaults(func=lambda _args: (print_pattern_list(), 0)[1])

    p_play = sub.add_parser("play", help="Generate and play a pattern")
    p_play.add_argument("pattern", help="Pattern name (or demo)")
    p_play.add_argument("--seconds", type=float, default=3.0, help="Duration for repeating patterns like loading")
    p_play.set_defaults(func=cmd_play)

    p_export = sub.add_parser("export", help="Generate and export a WAV file")
    p_export.add_argument("pattern", help="Pattern name (or demo)")
    p_export.add_argument("--seconds", type=float, default=3.0, help="Duration for repeating patterns like loading")
    p_export.add_argument("--out", required=True, help="Output .wav path")
    p_export.set_defaults(func=cmd_export)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
