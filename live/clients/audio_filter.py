"""Adaptive noise gate for filtering background audio.

This module provides an adaptive noise gate that automatically adjusts
its threshold based on the current noise environment, filtering out
background noise like nearby conversations while allowing primary audio through.
"""

import collections
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from logger import log_print


class GateState(Enum):
    """Current state of the noise gate."""
    CLOSED = "closed"  # Audio is being filtered
    OPEN = "open"      # Audio is passing through
    ATTACK = "attack"  # Transitioning from closed to open
    RELEASE = "release"  # Transitioning from open to closed


@dataclass
class NoiseGateConfig:
    """Configuration for the adaptive noise gate."""

    # Threshold calculation
    noise_percentile: float = 30.0  # Percentile for noise floor (lower = more aggressive)
    margin_multiplier: float = 2.0  # Threshold = noise_floor * margin
    min_threshold: float = 100.0    # Minimum threshold (prevent too quiet)
    max_threshold: float = 5000.0   # Maximum threshold (prevent too loud)

    # Hysteresis (prevents rapid on/off)
    open_threshold_multiplier: float = 1.0   # Multiplier for opening gate
    close_threshold_multiplier: float = 0.7  # Multiplier for closing gate (lower = stays open longer)

    # Timing
    hold_time_ms: float = 200.0     # Keep gate open for this long after signal drops
    attack_time_ms: float = 10.0    # Time to fully open gate
    release_time_ms: float = 100.0  # Time to fully close gate

    # Adaptation
    adaptation_rate: float = 0.1    # How fast to adapt to new noise levels (0-1)
    history_seconds: float = 5.0    # How much history to keep for noise calculation

    # Sample rate info
    sample_rate: int = 24000
    chunk_size: int = 4800


@dataclass
class NoiseGateStats:
    """Statistics from the noise gate."""

    total_chunks: int = 0
    passed_chunks: int = 0
    filtered_chunks: int = 0

    current_rms: float = 0.0
    current_threshold: float = 0.0
    noise_floor: float = 0.0

    state: GateState = GateState.CLOSED
    state_duration_ms: float = 0.0

    # History for analysis
    rms_history: list = field(default_factory=list)
    threshold_history: list = field(default_factory=list)


class AdaptiveNoiseGate:
    """Adaptive noise gate that filters audio based on dynamic threshold.

    The gate automatically tracks the background noise level and adjusts
    its threshold to filter out ambient noise while passing through
    louder (primary) audio.

    Usage:
        gate = AdaptiveNoiseGate()

        # In audio processing loop:
        for chunk in audio_chunks:
            if gate.process(chunk):
                # Audio passed - send to API
                send_to_api(chunk)
            else:
                # Audio filtered - optionally send silence
                pass
    """

    def __init__(
        self,
        config: Optional[NoiseGateConfig] = None,
        session_id: str = "default",
    ):
        self.config = config or NoiseGateConfig()
        self.session_id = session_id

        # Calculate history size based on config
        chunks_per_second = self.config.sample_rate / self.config.chunk_size
        history_size = int(self.config.history_seconds * chunks_per_second)

        # RMS history for noise floor calculation
        self._rms_history = collections.deque(maxlen=history_size)
        self._lock = threading.Lock()

        # Current state
        self._state = GateState.CLOSED
        self._state_start_time = time.time()
        self._last_above_threshold_time = 0.0

        # Smoothed values
        self._smoothed_threshold = self.config.min_threshold
        self._noise_floor = self.config.min_threshold

        # Statistics
        self._stats = NoiseGateStats()

        # Calibration mode
        self._calibrating = False
        self._calibration_samples: list[float] = []

        log_print(
            "INFO",
            "AdaptiveNoiseGate created",
            session_id=session_id,
            config=vars(self.config),
        )

    def calculate_rms(self, chunk: bytes) -> float:
        """Calculate RMS (root mean square) level of audio chunk.

        Args:
            chunk: Raw PCM16 audio bytes

        Returns:
            RMS level as float
        """
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        if len(audio_data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))

    def _update_noise_floor(self, rms: float) -> None:
        """Update noise floor estimate based on recent RMS values."""
        with self._lock:
            self._rms_history.append(rms)

            if len(self._rms_history) < 10:
                # Not enough data yet, use minimum
                return

            # Calculate noise floor as percentile of recent values
            rms_array = np.array(list(self._rms_history))
            new_floor = np.percentile(rms_array, self.config.noise_percentile)

            # Smooth the noise floor update
            self._noise_floor = (
                self._noise_floor * (1 - self.config.adaptation_rate) +
                new_floor * self.config.adaptation_rate
            )

    def _calculate_threshold(self) -> float:
        """Calculate current threshold based on noise floor."""
        threshold = self._noise_floor * self.config.margin_multiplier
        return np.clip(threshold, self.config.min_threshold, self.config.max_threshold)

    def _get_effective_threshold(self) -> tuple[float, float]:
        """Get open and close thresholds (with hysteresis).

        Returns:
            Tuple of (open_threshold, close_threshold)
        """
        base_threshold = self._calculate_threshold()
        open_threshold = base_threshold * self.config.open_threshold_multiplier
        close_threshold = base_threshold * self.config.close_threshold_multiplier
        return open_threshold, close_threshold

    def process(self, chunk: bytes) -> bool:
        """Process audio chunk and determine if it should pass.

        This is the main method to call for each audio chunk.

        Args:
            chunk: Raw PCM16 audio bytes

        Returns:
            True if audio should pass through, False if filtered
        """
        current_time = time.time()
        rms = self.calculate_rms(chunk)

        # Update noise floor (always, for adaptation)
        self._update_noise_floor(rms)

        # Handle calibration mode
        if self._calibrating:
            self._calibration_samples.append(rms)
            return True  # Pass all audio during calibration

        # Get thresholds with hysteresis
        open_threshold, close_threshold = self._get_effective_threshold()

        # State machine for gate
        should_pass = False

        if self._state == GateState.CLOSED:
            if rms >= open_threshold:
                # Signal above threshold - open gate
                self._state = GateState.OPEN
                self._state_start_time = current_time
                self._last_above_threshold_time = current_time
                should_pass = True
            else:
                should_pass = False

        elif self._state == GateState.OPEN:
            if rms >= close_threshold:
                # Signal still above close threshold
                self._last_above_threshold_time = current_time
                should_pass = True
            else:
                # Signal dropped below close threshold
                time_since_above = (current_time - self._last_above_threshold_time) * 1000

                if time_since_above < self.config.hold_time_ms:
                    # Still in hold time - keep open
                    should_pass = True
                else:
                    # Hold time expired - close gate
                    self._state = GateState.CLOSED
                    self._state_start_time = current_time
                    should_pass = False

        # Update statistics
        self._stats.total_chunks += 1
        if should_pass:
            self._stats.passed_chunks += 1
        else:
            self._stats.filtered_chunks += 1

        self._stats.current_rms = rms
        self._stats.current_threshold = open_threshold
        self._stats.noise_floor = self._noise_floor
        self._stats.state = self._state
        self._stats.state_duration_ms = (current_time - self._state_start_time) * 1000

        return should_pass

    def start_calibration(self) -> None:
        """Start calibration mode to measure ambient noise."""
        log_print("INFO", "Starting noise calibration", session_id=self.session_id)
        self._calibrating = True
        self._calibration_samples = []

    def stop_calibration(self) -> dict:
        """Stop calibration and return measured values.

        Returns:
            Dict with calibration results including recommended thresholds
        """
        self._calibrating = False

        if not self._calibration_samples:
            return {"error": "No samples collected"}

        samples = np.array(self._calibration_samples)

        results = {
            "samples_collected": len(samples),
            "duration_sec": len(samples) * self.config.chunk_size / self.config.sample_rate,
            "mean_rms": float(np.mean(samples)),
            "std_rms": float(np.std(samples)),
            "min_rms": float(np.min(samples)),
            "max_rms": float(np.max(samples)),
            "percentiles": {
                "p10": float(np.percentile(samples, 10)),
                "p25": float(np.percentile(samples, 25)),
                "p50": float(np.percentile(samples, 50)),
                "p75": float(np.percentile(samples, 75)),
                "p90": float(np.percentile(samples, 90)),
                "p95": float(np.percentile(samples, 95)),
                "p99": float(np.percentile(samples, 99)),
            },
            "recommended": {
                "conservative": float(np.percentile(samples, 99) * 1.5),
                "balanced": float(np.percentile(samples, 95) * 2.0),
                "aggressive": float(np.percentile(samples, 90) * 2.5),
            },
        }

        log_print(
            "INFO",
            "Calibration complete",
            session_id=self.session_id,
            results=results,
        )

        self._calibration_samples = []
        return results

    def set_threshold_override(self, threshold: Optional[float]) -> None:
        """Set a manual threshold override.

        Args:
            threshold: Fixed threshold value, or None to use adaptive
        """
        if threshold is not None:
            # Override by setting very high adaptation and noise floor
            self._noise_floor = threshold / self.config.margin_multiplier
            log_print(
                "INFO",
                f"Threshold override set: {threshold}",
                session_id=self.session_id,
            )
        else:
            # Reset to adaptive
            self._rms_history.clear()
            self._noise_floor = self.config.min_threshold
            log_print("INFO", "Threshold reset to adaptive", session_id=self.session_id)

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters dynamically.

        Args:
            **kwargs: Config parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                log_print(
                    "DEBUG",
                    f"Config updated: {key}={value}",
                    session_id=self.session_id,
                )

    def get_stats(self) -> NoiseGateStats:
        """Get current statistics."""
        return self._stats

    def get_status(self) -> dict:
        """Get current status as dict (for API/UI)."""
        open_threshold, close_threshold = self._get_effective_threshold()

        return {
            "state": self._state.value,
            "current_rms": round(self._stats.current_rms, 1),
            "noise_floor": round(self._noise_floor, 1),
            "open_threshold": round(open_threshold, 1),
            "close_threshold": round(close_threshold, 1),
            "total_chunks": self._stats.total_chunks,
            "passed_chunks": self._stats.passed_chunks,
            "filtered_chunks": self._stats.filtered_chunks,
            "pass_rate": round(
                self._stats.passed_chunks / max(1, self._stats.total_chunks) * 100, 1
            ),
        }

    def reset(self) -> None:
        """Reset gate state and statistics."""
        with self._lock:
            self._rms_history.clear()

        self._state = GateState.CLOSED
        self._state_start_time = time.time()
        self._noise_floor = self.config.min_threshold
        self._stats = NoiseGateStats()

        log_print("INFO", "NoiseGate reset", session_id=self.session_id)


class DualSourceGate:
    """Noise gate for managing two audio sources to prevent overlap.

    This gate monitors a secondary source (e.g., TTS playback) and
    automatically adjusts filtering on the primary source (e.g., mic)
    to prevent interference.

    Usage:
        gate = DualSourceGate()

        # When TTS starts playing
        gate.set_secondary_active(True)

        # In mic audio loop
        if gate.should_pass_primary(mic_chunk):
            send_to_api(mic_chunk)
    """

    def __init__(
        self,
        primary_config: Optional[NoiseGateConfig] = None,
        session_id: str = "default",
    ):
        self.session_id = session_id

        # Primary source gate (e.g., microphone)
        self.primary_gate = AdaptiveNoiseGate(
            config=primary_config,
            session_id=f"{session_id}_primary",
        )

        # Secondary source state (e.g., TTS/speaker)
        self._secondary_active = False
        self._secondary_start_time = 0.0
        self._secondary_cooldown_ms = 500.0  # Wait after secondary stops

        # Boost threshold when secondary is active
        self._secondary_threshold_boost = 1.5  # Multiply threshold by this

        self._lock = threading.Lock()

        log_print(
            "INFO",
            "DualSourceGate created",
            session_id=session_id,
        )

    def set_secondary_active(self, active: bool) -> None:
        """Set whether secondary source (e.g., TTS) is active.

        Args:
            active: True if secondary source is playing
        """
        with self._lock:
            was_active = self._secondary_active
            self._secondary_active = active

            if active and not was_active:
                self._secondary_start_time = time.time()
                log_print(
                    "DEBUG",
                    "Secondary source activated",
                    session_id=self.session_id,
                )
            elif not active and was_active:
                log_print(
                    "DEBUG",
                    "Secondary source deactivated",
                    session_id=self.session_id,
                )

    def is_secondary_active(self) -> bool:
        """Check if secondary source is currently active (including cooldown)."""
        with self._lock:
            if self._secondary_active:
                return True

            # Check cooldown period
            if self._secondary_start_time > 0:
                elapsed = (time.time() - self._secondary_start_time) * 1000
                if elapsed < self._secondary_cooldown_ms:
                    return True

            return False

    def should_pass_primary(self, chunk: bytes) -> bool:
        """Check if primary audio chunk should pass.

        When secondary is active, threshold is boosted to require
        louder primary audio to pass through.

        Args:
            chunk: Raw PCM16 audio bytes from primary source

        Returns:
            True if audio should pass
        """
        # Temporarily boost threshold if secondary is active
        original_multiplier = self.primary_gate.config.margin_multiplier

        if self.is_secondary_active():
            self.primary_gate.config.margin_multiplier *= self._secondary_threshold_boost

        result = self.primary_gate.process(chunk)

        # Restore original
        self.primary_gate.config.margin_multiplier = original_multiplier

        return result

    def get_status(self) -> dict:
        """Get combined status."""
        status = self.primary_gate.get_status()
        status["secondary_active"] = self.is_secondary_active()
        return status

    def reset(self) -> None:
        """Reset all state."""
        self.primary_gate.reset()
        with self._lock:
            self._secondary_active = False
            self._secondary_start_time = 0.0
