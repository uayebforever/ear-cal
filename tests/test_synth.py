import numpy as np

from earcal.synth import db_to_linear_rms, synth_tone, SAMPLE_RATE


def test_tone_rms_matches_target_level():
    for level_db in (-10.0, -3.0, 0.0, 6.0):
        wave = synth_tone(1000.0, level_db, duration_s=0.5)
        # exclude the fade in/out edges, where RMS is attenuated by design
        steady = wave[int(0.1 * SAMPLE_RATE) : int(0.4 * SAMPLE_RATE)]
        rms = np.sqrt(np.mean(steady.astype(np.float64) ** 2))
        assert abs(rms - db_to_linear_rms(level_db)) < 0.01


def test_tone_duration_matches_request():
    wave = synth_tone(440.0, 0.0, duration_s=0.3)
    assert wave.shape[0] == int(round(0.3 * SAMPLE_RATE))


def test_no_clicks_at_edges():
    wave = synth_tone(1000.0, 6.0, duration_s=0.5)
    assert abs(wave[0]) < 1e-6
    assert abs(wave[-1]) < 1e-3


def test_high_band_drops_harmonics_above_nyquist():
    # at 12500 Hz with the default 44100 Hz sample rate, only the
    # fundamental fits under the 0.9x-Nyquist margin (2nd harmonic =
    # 25000 Hz > 19845 Hz margin), so it should look like a pure sine.
    wave = synth_tone(12500.0, 0.0, duration_s=0.5)
    steady = wave[int(0.1 * SAMPLE_RATE) : int(0.4 * SAMPLE_RATE)]
    rms = np.sqrt(np.mean(steady.astype(np.float64) ** 2))
    assert abs(rms - db_to_linear_rms(0.0)) < 0.01


def test_level_clamped_to_safety_range():
    from earcal.synth import MIN_LEVEL_DB, MAX_LEVEL_DB

    assert db_to_linear_rms(MAX_LEVEL_DB + 50) == db_to_linear_rms(MAX_LEVEL_DB)
    assert db_to_linear_rms(MIN_LEVEL_DB - 50) == db_to_linear_rms(MIN_LEVEL_DB)
