"""Multi-microphone audio source selector and mixer.

Usage:
    python multi_mic.py              # List devices
    python multi_mic.py --test 0 3   # Test devices 0 and 3
    python multi_mic.py --stream 0 3 # Stream mixed audio to stdout (PCM16)
"""

import argparse
import sys
import time
from typing import Optional

import numpy as np
import pyaudio

# Audio settings (matching config.py)
AUDIO_RATE = 24000
AUDIO_CHUNK = 4800


def list_devices() -> list[dict]:
    """List all available audio input devices."""
    pa = pyaudio.PyAudio()
    devices = []

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append({
                "index": i,
                "name": info["name"],
                "channels": info["maxInputChannels"],
                "sample_rate": int(info["defaultSampleRate"]),
            })

    pa.terminate()
    return devices


def print_devices():
    """Print available input devices."""
    devices = list_devices()
    print(f"\nFound {len(devices)} input device(s):\n")
    for d in devices:
        print(f"  [{d['index']:2d}] {d['name']}")
        print(f"       channels: {d['channels']}, sample_rate: {d['sample_rate']}")
    print()


class MultiMicStream:
    """Stream and mix audio from multiple microphones."""

    def __init__(self, device_indices: list[int], sample_rate: int = AUDIO_RATE, chunk_size: int = AUDIO_CHUNK):
        self.device_indices = device_indices
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.pa: Optional[pyaudio.PyAudio] = None
        self.streams: list[dict] = []
        self.running = False

    def open(self) -> bool:
        """Open audio streams for all specified devices."""
        self.pa = pyaudio.PyAudio()
        self.streams = []

        for idx in self.device_indices:
            try:
                info = self.pa.get_device_info_by_index(idx)
                stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=idx,
                    frames_per_buffer=self.chunk_size,
                )
                self.streams.append({
                    "stream": stream,
                    "index": idx,
                    "name": info["name"],
                })
                print(f"Opened: [{idx}] {info['name']}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to open device {idx}: {e}", file=sys.stderr)

        if not self.streams:
            print("No streams opened!", file=sys.stderr)
            self.pa.terminate()
            return False

        self.running = True
        return True

    def read_mixed(self) -> Optional[bytes]:
        """Read and mix audio from all streams."""
        if not self.running:
            return None

        chunks = []
        for s in self.streams:
            try:
                data = s["stream"].read(self.chunk_size, exception_on_overflow=False)
                chunks.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                print(f"Error reading from [{s['index']}]: {e}", file=sys.stderr)

        if not chunks:
            return None

        # Mix: average all channels
        if len(chunks) == 1:
            mixed = chunks[0]
        else:
            stacked = np.stack(chunks, axis=0)
            mixed = np.mean(stacked, axis=0).astype(np.int16)

        return mixed.tobytes()

    def close(self):
        """Close all streams."""
        self.running = False
        for s in self.streams:
            try:
                s["stream"].stop_stream()
                s["stream"].close()
            except Exception:
                pass
        if self.pa:
            self.pa.terminate()
        self.streams = []
        print("Streams closed.", file=sys.stderr)


def test_devices(device_indices: list[int], duration: float = 5.0):
    """Test recording from specified devices."""
    print(f"\nTesting devices {device_indices} for {duration} seconds...")

    mic = MultiMicStream(device_indices)
    if not mic.open():
        return

    start_time = time.time()
    chunk_count = 0
    total_bytes = 0

    try:
        while time.time() - start_time < duration:
            data = mic.read_mixed()
            if data:
                chunk_count += 1
                total_bytes += len(data)

                # Show level meter
                audio = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio).mean()
                bars = int(level / 500)
                print(f"\rLevel: {'█' * min(bars, 50):50s} {level:6.0f}", end="", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    mic.close()

    elapsed = time.time() - start_time
    print(f"\n\nRecorded {chunk_count} chunks, {total_bytes} bytes in {elapsed:.1f}s")
    print(f"Rate: {total_bytes / elapsed / 1024:.1f} KB/s")


def stream_to_stdout(device_indices: list[int]):
    """Stream mixed audio to stdout as raw PCM16."""
    print(f"Streaming from devices {device_indices}...", file=sys.stderr)
    print(f"Format: PCM16, {AUDIO_RATE}Hz, mono", file=sys.stderr)

    mic = MultiMicStream(device_indices)
    if not mic.open():
        sys.exit(1)

    try:
        while True:
            data = mic.read_mixed()
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass

    mic.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-microphone audio tool")
    parser.add_argument("--list", "-l", action="store_true", help="List available devices")
    parser.add_argument("--test", "-t", type=int, nargs="+", metavar="IDX", help="Test specified device indices")
    parser.add_argument("--stream", "-s", type=int, nargs="+", metavar="IDX", help="Stream mixed audio to stdout")
    parser.add_argument("--duration", "-d", type=float, default=5.0, help="Test duration in seconds")

    args = parser.parse_args()

    if args.stream:
        stream_to_stdout(args.stream)
    elif args.test:
        test_devices(args.test, args.duration)
    else:
        print_devices()


if __name__ == "__main__":
    main()
