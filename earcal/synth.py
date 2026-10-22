"""Tone synthesis.

We synthesize a fundamental plus a short, fixed harmonic series (not a
real piano sample) so that every band has an *identical* timbre. A real
piano sample's spectral envelope changes a lot with pitch, which would
confound a loudness judgement between two different notes with a timbre
difference. A pure sine is clean but sounds thin/clicky and is a bit
harder to attend to steadily; a few quickly-decaying harmonics make it
sound more like a musical tone (closer to the "piano key" feel the user
asked for) while staying fully controlled.
"""

import numpy as np

SAMPLE_RATE = 44100

# Relative linear amplitudes of fundamental + first 3 overtones (-6 dB/harmonic).
HARMONIC_AMPLITUDES = (1.0, 0.5, 0.25, 0.125)

# RMS amplitude (full-scale float, 1.0 = digital clipping) that corresponds
# to a presented "level" of 0 dB. Kept conservative for ear/speaker safety;
# all level_db values are relative to this.
BASE_RMS = 0.08

# Hard safety clamp on presented level. The adaptive algorithm should never
# need to go outside this range once a session's base volume is set.
MIN_LEVEL_DB = -40.0
MAX_LEVEL_DB = 12.0


def db_to_linear_rms(level_db: float) -> float:
    level_db = max(MIN_LEVEL_DB, min(MAX_LEVEL_DB, level_db))
    return BASE_RMS * 10.0 ** (level_db / 20.0)


def synth_tone(
    freq_hz: float,
    level_db: float,
    duration_s: float = 0.55,
    attack_s: float = 0.02,
    release_s: float = 0.05,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return a mono float32 buffer of a harmonic tone at the given level.

    level_db is relative to BASE_RMS, and the returned buffer's RMS
    amplitude equals db_to_linear_rms(level_db) regardless of how many
    harmonics were usable (harmonics above ~0.9x Nyquist are dropped to
    avoid aliasing/harshness near the top of the range), so "level" means
    the same perceptual energy input at every band.
    """
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate

    nyquist_margin = 0.9 * (sample_rate / 2.0)
    wave = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate(HARMONIC_AMPLITUDES, start=1):
        harmonic_freq = freq_hz * k
        if harmonic_freq > nyquist_margin:
            break
        wave += amp * np.sin(2.0 * np.pi * harmonic_freq * t)

    raw_rms = np.sqrt(np.mean(wave**2))
    if raw_rms > 0:
        wave = wave / raw_rms

    wave *= db_to_linear_rms(level_db)

    envelope = _attack_release_envelope(n, attack_s, release_s, sample_rate)
    wave *= envelope

    peak = np.max(np.abs(wave))
    if peak > 0.98:
        wave *= 0.98 / peak

    return wave.astype(np.float32)


def _attack_release_envelope(
    n: int, attack_s: float, release_s: float, sample_rate: int
) -> np.ndarray:
    attack_n = min(n // 2, int(round(attack_s * sample_rate)))
    release_n = min(n - attack_n, int(round(release_s * sample_rate)))

    env = np.ones(n, dtype=np.float64)
    if attack_n > 0:
        env[:attack_n] = 0.5 - 0.5 * np.cos(np.pi * np.arange(attack_n) / attack_n)
    if release_n > 0:
        ramp = 0.5 + 0.5 * np.cos(np.pi * np.arange(release_n) / release_n)
        env[n - release_n :] = ramp
    return env


def silence(duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(round(duration_s * sample_rate)), dtype=np.float32)
