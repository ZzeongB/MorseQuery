#!/usr/bin/env python3
"""
Microphone Activation Level Visualizer

This script visualizes real-time and historical microphone activation levels
to help determine appropriate thresholds for avoiding overlap between
different audio sources.

Usage:
    python mic_level_visualizer.py [--device INDEX] [--duration SECONDS]
"""

import argparse
import collections
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000

# Visualization settings
HISTORY_SECONDS = 30  # How many seconds of history to display
UPDATE_INTERVAL_MS = 50  # How often to update the plot


class MicLevelVisualizer:
    """Real-time microphone level visualization."""

    def __init__(self, device_index: Optional[int] = None, duration: Optional[int] = None):
        self.device_index = device_index
        self.duration = duration
        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream = None

        # Data storage
        history_size = int(HISTORY_SECONDS * RATE / CHUNK)
        self.rms_history = collections.deque(maxlen=history_size)
        self.peak_history = collections.deque(maxlen=history_size)
        self.time_history = collections.deque(maxlen=history_size)

        # Statistics
        self.start_time = 0
        self.max_rms = 0
        self.max_peak = 0

        # Threshold line (adjustable)
        self.threshold = 500

    def list_devices(self) -> list[dict]:
        """List available audio input devices."""
        pa = pyaudio.PyAudio()
        devices = []

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "rate": int(info["defaultSampleRate"]),
                })

        pa.terminate()
        return devices

    def calculate_levels(self, data: bytes) -> tuple[float, float]:
        """Calculate RMS and peak levels from audio data."""
        audio_data = np.frombuffer(data, dtype=np.int16)

        # RMS (Root Mean Square) - average power
        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))

        # Peak - maximum absolute value
        peak = np.max(np.abs(audio_data))

        return float(rms), float(peak)

    def run(self):
        """Run the visualizer."""
        print("=" * 60)
        print("Microphone Activation Level Visualizer")
        print("=" * 60)

        # List devices
        devices = self.list_devices()
        print("\nAvailable input devices:")
        for d in devices:
            marker = " *" if d["index"] == self.device_index else ""
            print(f"  [{d['index']}] {d['name']} ({d['channels']}ch, {d['rate']}Hz){marker}")

        if self.device_index is not None:
            selected = next((d for d in devices if d["index"] == self.device_index), None)
            if selected:
                print(f"\nUsing device: [{selected['index']}] {selected['name']}")
            else:
                print(f"\nWarning: Device index {self.device_index} not found, using default")
                self.device_index = None
        else:
            print("\nUsing default input device")

        print("\nControls:")
        print("  - Close the window to stop")
        print("  - The red dashed line shows the current threshold")
        print("  - Adjust threshold in code or use keyboard:\n")

        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()

        try:
            self.stream = self.pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK,
            )
            print("Audio stream opened successfully.\n")
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.pa.terminate()
            return

        # Setup matplotlib
        plt.ion()  # Interactive mode
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Microphone Activation Level Monitor", fontsize=14)

        # Subplot 1: Real-time RMS level (bar)
        ax_bar = axes[0, 0]
        ax_bar.set_title("Current RMS Level")
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(0, 5000)
        ax_bar.set_ylabel("Level")
        bar = ax_bar.barh(0.5, 0, height=0.3, color='steelblue')
        threshold_line_bar = ax_bar.axvline(x=self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {self.threshold}')
        ax_bar.legend(loc='upper right')
        ax_bar.set_yticks([])

        # Subplot 2: Real-time waveform display
        ax_wave = axes[0, 1]
        ax_wave.set_title("Current Audio Waveform")
        ax_wave.set_ylim(-32768, 32768)
        ax_wave.set_xlabel("Sample")
        ax_wave.set_ylabel("Amplitude")
        wave_line, = ax_wave.plot([], [], 'b-', linewidth=0.5)

        # Subplot 3: RMS history
        ax_rms = axes[1, 0]
        ax_rms.set_title(f"RMS Level History ({HISTORY_SECONDS}s)")
        ax_rms.set_xlabel("Time (s)")
        ax_rms.set_ylabel("RMS Level")
        ax_rms.set_ylim(0, 3000)
        rms_line, = ax_rms.plot([], [], 'b-', linewidth=1, label='RMS')
        threshold_line_rms = ax_rms.axhline(y=self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {self.threshold}')
        ax_rms.legend(loc='upper right')
        ax_rms.grid(True, alpha=0.3)

        # Subplot 4: Peak history
        ax_peak = axes[1, 1]
        ax_peak.set_title(f"Peak Level History ({HISTORY_SECONDS}s)")
        ax_peak.set_xlabel("Time (s)")
        ax_peak.set_ylabel("Peak Level")
        ax_peak.set_ylim(0, 32768)
        peak_line, = ax_peak.plot([], [], 'g-', linewidth=1, label='Peak')
        ax_peak.legend(loc='upper right')
        ax_peak.grid(True, alpha=0.3)

        # Stats text
        stats_text = fig.text(0.02, 0.02, "", fontsize=10, family='monospace',
                              verticalalignment='bottom')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        self.start_time = time.time()
        last_update = 0

        try:
            while plt.fignum_exists(fig.number):
                # Read audio data
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    break

                # Calculate levels
                rms, peak = self.calculate_levels(data)
                current_time = time.time() - self.start_time

                # Update statistics
                self.max_rms = max(self.max_rms, rms)
                self.max_peak = max(self.max_peak, peak)

                # Store history
                self.rms_history.append(rms)
                self.peak_history.append(peak)
                self.time_history.append(current_time)

                # Check duration limit
                if self.duration and current_time >= self.duration:
                    print(f"\nDuration limit ({self.duration}s) reached.")
                    break

                # Update plot at specified interval
                now = time.time()
                if (now - last_update) * 1000 >= UPDATE_INTERVAL_MS:
                    last_update = now

                    # Update bar
                    bar[0].set_width(rms)
                    color = 'red' if rms > self.threshold else 'steelblue'
                    bar[0].set_color(color)

                    # Update waveform
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    wave_line.set_data(range(len(audio_array)), audio_array)
                    ax_wave.set_xlim(0, len(audio_array))

                    # Update history plots
                    if len(self.time_history) > 1:
                        times = list(self.time_history)
                        rms_values = list(self.rms_history)
                        peak_values = list(self.peak_history)

                        rms_line.set_data(times, rms_values)
                        peak_line.set_data(times, peak_values)

                        ax_rms.set_xlim(max(0, current_time - HISTORY_SECONDS), current_time + 1)
                        ax_peak.set_xlim(max(0, current_time - HISTORY_SECONDS), current_time + 1)

                        # Auto-scale Y axis based on data
                        if rms_values:
                            max_rms_display = max(max(rms_values) * 1.2, self.threshold * 1.5)
                            ax_rms.set_ylim(0, max_rms_display)
                        if peak_values:
                            max_peak_display = max(max(peak_values) * 1.2, 5000)
                            ax_peak.set_ylim(0, max_peak_display)

                    # Update stats
                    above_threshold = sum(1 for r in self.rms_history if r > self.threshold)
                    total_samples = len(self.rms_history)
                    pct_above = (above_threshold / total_samples * 100) if total_samples > 0 else 0

                    avg_rms = np.mean(list(self.rms_history)) if self.rms_history else 0

                    stats_text.set_text(
                        f"Current RMS: {rms:7.1f} | Peak: {peak:7.0f} | "
                        f"Max RMS: {self.max_rms:7.1f} | Max Peak: {self.max_peak:7.0f} | "
                        f"Avg RMS: {avg_rms:7.1f} | "
                        f"Above Threshold: {pct_above:5.1f}% | "
                        f"Time: {current_time:.1f}s"
                    )

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                # Small sleep to prevent CPU overload
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.cleanup()

        # Print final statistics
        print("\n" + "=" * 60)
        print("Final Statistics")
        print("=" * 60)
        print(f"  Duration:        {time.time() - self.start_time:.1f} seconds")
        print(f"  Max RMS:         {self.max_rms:.1f}")
        print(f"  Max Peak:        {self.max_peak:.0f}")
        if self.rms_history:
            avg = np.mean(list(self.rms_history))
            std = np.std(list(self.rms_history))
            percentiles = np.percentile(list(self.rms_history), [50, 75, 90, 95, 99])
            print(f"  Average RMS:     {avg:.1f}")
            print(f"  Std Dev RMS:     {std:.1f}")
            print(f"  Percentiles (RMS):")
            print(f"    50th (median): {percentiles[0]:.1f}")
            print(f"    75th:          {percentiles[1]:.1f}")
            print(f"    90th:          {percentiles[2]:.1f}")
            print(f"    95th:          {percentiles[3]:.1f}")
            print(f"    99th:          {percentiles[4]:.1f}")
            print(f"\nRecommended thresholds:")
            print(f"  Conservative (low false positives): {percentiles[4]:.0f}")
            print(f"  Balanced:                           {percentiles[3]:.0f}")
            print(f"  Aggressive (catches more audio):    {percentiles[2]:.0f}")
        print("=" * 60)

    def cleanup(self):
        """Clean up resources."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        if self.pa:
            try:
                self.pa.terminate()
            except Exception:
                pass
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description="Visualize microphone activation levels to determine appropriate thresholds."
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=None,
        help="Audio input device index (use -l to list devices)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "-t", "--duration",
        type=int,
        default=None,
        help="Duration in seconds to run (default: unlimited)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=500,
        help="Initial threshold value for visualization (default: 500)",
    )

    args = parser.parse_args()

    visualizer = MicLevelVisualizer(
        device_index=args.device,
        duration=args.duration,
    )

    if args.list:
        devices = visualizer.list_devices()
        print("Available input devices:")
        for d in devices:
            print(f"  [{d['index']}] {d['name']} ({d['channels']}ch, {d['rate']}Hz)")
        return

    visualizer.threshold = args.threshold
    visualizer.run()


if __name__ == "__main__":
    main()
