"""Thin wrapper around sounddevice for blocking playback.

Kept separate from synth.py so the synthesis and model logic can be
unit-tested (and simulated) on machines/containers with no audio
hardware at all, which is where most of this project's test suite runs.
"""

from __future__ import annotations

import numpy as np

from earcal.synth import SAMPLE_RATE


class AudioUnavailableError(RuntimeError):
    pass


def probe_output_device() -> str | None:
    """Return a human-readable description of the default output device,
    or None if no audio output is available."""
    try:
        import sounddevice as sd

        device_info = sd.query_devices(kind="output")
        return f"{device_info['name']} (default output)"
    except Exception:
        return None


def play_blocking(wave: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak > 1.0:
        raise ValueError(f"refusing to play a clipping buffer (peak={peak:.3f})")

    try:
        import sounddevice as sd
    except Exception as exc:
        raise AudioUnavailableError(
            "sounddevice is not available. Install it with "
            "`pip install sounddevice` and make sure PortAudio is on your "
            "system (on Linux: `sudo apt install libportaudio2`)."
        ) from exc

    try:
        sd.play(wave, sample_rate)
        sd.wait()
    except Exception as exc:
        raise AudioUnavailableError(f"could not play audio: {exc}") from exc
