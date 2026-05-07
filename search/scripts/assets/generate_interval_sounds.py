"""
Generate interval boundary sound effects as MP3 files.

Recreates the blocked seek cue from index.js playBlockedSeekCue() function.
"""

import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import os

SAMPLE_RATE = 44100


def generate_oscillator(frequency, duration, wave_type='sine', detune=0, sample_rate=SAMPLE_RATE):
    """Generate oscillator waveform."""
    # Apply detune (cents to frequency multiplier)
    freq = frequency * (2 ** (detune / 1200))
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    if wave_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == 'triangle':
        return 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    elif wave_type == 'sawtooth':
        return 2 * (t * freq - np.floor(0.5 + t * freq))
    else:
        return np.sin(2 * np.pi * freq * t)


def apply_envelope(signal, attack=0.01, release_start=0.01, duration=0.1, sample_rate=SAMPLE_RATE):
    """Apply ADSR-like envelope with linear attack and exponential release."""
    n_samples = len(signal)
    envelope = np.ones(n_samples)

    # Attack phase (linear ramp from 0.0001 to 1.0)
    attack_samples = int(attack * sample_rate)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0.0001, 1.0, attack_samples)

    # Release phase (exponential decay to 0.0001)
    release_start_sample = int(release_start * sample_rate)
    if release_start_sample < n_samples:
        release_samples = n_samples - release_start_sample
        envelope[release_start_sample:] = np.logspace(0, -4, release_samples)

    return signal * envelope


def generate_blocked_seek_cue():
    """
    Generate the blocked seek cue sound from index.js.

    Notes:
        { frequency: 1568, duration: 0.1, delay: 0 }
        { frequency: 2093, duration: 0.16, delay: 0.11 }

    Voices:
        { type: 'triangle', detune: 0, gain: 0.26 }
        { type: 'sawtooth', detune: -8, gain: 0.08 }
        { type: 'sine', detune: 7, gain: 0.06 }
    """
    notes = [
        {'frequency': 1568, 'duration': 0.1, 'delay': 0},
        {'frequency': 2093, 'duration': 0.16, 'delay': 0.11},
    ]
    voices = [
        {'type': 'triangle', 'detune': 0, 'gain': 0.26},
        {'type': 'sawtooth', 'detune': -8, 'gain': 0.08},
        {'type': 'sine', 'detune': 7, 'gain': 0.06},
    ]

    # Calculate total duration
    max_end = max(note['delay'] + note['duration'] for note in notes)
    total_samples = int((max_end + 0.05) * SAMPLE_RATE)  # Add small padding

    mixed = np.zeros(total_samples)

    for note in notes:
        delay_samples = int(note['delay'] * SAMPLE_RATE)
        duration = note['duration']

        for voice in voices:
            # Generate oscillator
            osc = generate_oscillator(
                note['frequency'],
                duration,
                voice['type'],
                voice['detune']
            )

            # Apply envelope (attack=0.01, release starts at 0.01)
            osc = apply_envelope(osc, attack=0.01, release_start=0.01, duration=duration)

            # Apply gain
            osc *= voice['gain']

            # Mix into output
            end_sample = delay_samples + len(osc)
            if end_sample <= total_samples:
                mixed[delay_samples:end_sample] += osc
            else:
                mixed[delay_samples:] += osc[:total_samples - delay_samples]

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val * 0.8

    return mixed


def generate_interval_start_sound():
    """
    Generate a softer 'start' sound - lower pitch, ascending.
    """
    notes = [
        {'frequency': 880, 'duration': 0.08, 'delay': 0},
        {'frequency': 1047, 'duration': 0.1, 'delay': 0.09},
    ]
    voices = [
        {'type': 'sine', 'detune': 0, 'gain': 0.3},
        {'type': 'triangle', 'detune': 5, 'gain': 0.15},
    ]

    max_end = max(note['delay'] + note['duration'] for note in notes)
    total_samples = int((max_end + 0.05) * SAMPLE_RATE)

    mixed = np.zeros(total_samples)

    for note in notes:
        delay_samples = int(note['delay'] * SAMPLE_RATE)
        duration = note['duration']

        for voice in voices:
            osc = generate_oscillator(
                note['frequency'],
                duration,
                voice['type'],
                voice['detune']
            )
            osc = apply_envelope(osc, attack=0.01, release_start=0.01, duration=duration)
            osc *= voice['gain']

            end_sample = delay_samples + len(osc)
            if end_sample <= total_samples:
                mixed[delay_samples:end_sample] += osc
            else:
                mixed[delay_samples:] += osc[:total_samples - delay_samples]

    max_val = np.max(np.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val * 0.8

    return mixed


def save_as_mp3(signal, filename, sample_rate=SAMPLE_RATE):
    """Save signal as MP3 file."""
    # Convert to 16-bit PCM
    signal_int16 = (signal * 32767).astype(np.int16)

    # Save as temporary WAV
    temp_wav = filename.replace('.mp3', '_temp.wav')
    wavfile.write(temp_wav, sample_rate, signal_int16)

    # Convert to MP3 using pydub
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(filename, format='mp3')

    # Remove temporary WAV
    os.remove(temp_wav)
    print(f"Saved: {filename}")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'sounds')
    os.makedirs(output_dir, exist_ok=True)

    # Generate blocked seek cue (interval end boundary)
    blocked_cue = generate_blocked_seek_cue()
    save_as_mp3(blocked_cue, os.path.join(output_dir, 'interval_end.mp3'))

    # Generate interval start sound
    start_sound = generate_interval_start_sound()
    save_as_mp3(start_sound, os.path.join(output_dir, 'interval_start.mp3'))

    print(f"\nSound files saved to: {output_dir}")


if __name__ == '__main__':
    main()
