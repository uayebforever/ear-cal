from earcal.bands import build_bands, reference_band


def test_build_bands_matches_iso_frequencies():
    bands = build_bands()
    assert bands[0].freq_hz == 100
    assert bands[-1].freq_hz == 12500
    assert len(bands) == len(set(b.freq_hz for b in bands))


def test_reference_band_is_1khz():
    bands = build_bands()
    ref = reference_band(bands)
    assert ref.freq_hz == 1000
    assert ref.is_reference


def test_indices_are_contiguous():
    bands = build_bands()
    assert [b.index for b in bands] == list(range(len(bands)))
