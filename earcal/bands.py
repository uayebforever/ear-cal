"""1/3-octave band frequencies (ISO 266 preferred frequencies)."""

from dataclasses import dataclass

# ISO 266 preferred 1/3-octave center frequencies, restricted to a range
# that's practical for consumer speakers/headphones and typical adult
# hearing: below ~100 Hz small drivers roll off hard and loudness judgments
# get confounded with distortion; above ~12.5 kHz many listeners (even
# young ones) have little usable sensitivity, and cheap DACs/speakers roll
# off there too.
ISO_1_3_OCTAVE_HZ = (
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    6300, 8000, 10000, 12500,
)

REFERENCE_HZ = 1000.0

# Rough, illustrative equal-loudness-contour shape (dB needed relative to
# 1 kHz, loosely modeled on ISO 226 at moderate phon levels): ears are
# typically most sensitive around 2-4 kHz and less sensitive at both
# ends. Used only as a reference overlay / sanity check, never as a
# default for a real person's curve.
TYPICAL_SHAPE_HZ_DB = {
    100: 20.0, 125: 14.0, 160: 10.0, 200: 8.0, 250: 6.0, 315: 4.0,
    400: 3.0, 500: 2.0, 630: 1.0, 800: 0.5, 1000: 0.0, 1250: -1.0,
    1600: -3.0, 2000: -5.0, 2500: -7.0, 3150: -8.0, 4000: -7.0,
    5000: -5.0, 6300: -2.0, 8000: 1.0, 10000: 5.0, 12500: 12.0,
}


@dataclass(frozen=True)
class Band:
    index: int
    freq_hz: float

    @property
    def is_reference(self) -> bool:
        return self.freq_hz == REFERENCE_HZ


def build_bands(freqs: tuple[float, ...] = ISO_1_3_OCTAVE_HZ) -> list[Band]:
    return [Band(i, f) for i, f in enumerate(freqs)]


def reference_band(bands: list[Band]) -> Band:
    for b in bands:
        if b.is_reference:
            return b
    raise ValueError("band list has no reference frequency (1000 Hz)")
