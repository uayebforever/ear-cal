"""A virtual listener, used to validate and tune the adaptive algorithm
without spending any real listening time.

This lets us check (and the user can re-check after tweaking constants
in model.py) that the trial-selection/update rules actually converge to
a known ground-truth hearing curve within a reasonable number of trials,
before ever running the real thing on real ears.
"""

from __future__ import annotations

import numpy as np

from earcal.bands import Band, TYPICAL_SHAPE_HZ_DB, build_bands
from earcal.model import HearingModel


class VirtualListener:
    """Judges loudness with a fixed true offset curve plus response noise."""

    def __init__(
        self,
        true_offsets: dict[int, float],
        jnd_db: float = 2.5,
        internal_noise_db: float = 1.2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.true_offsets = true_offsets
        self.jnd_db = jnd_db
        self.internal_noise_db = internal_noise_db
        self.rng = rng or np.random.default_rng()

    def judge(
        self, prev_idx: int, prev_level_db: float, next_idx: int, next_level_db: float
    ) -> str:
        perceived_prev = prev_level_db + self.true_offsets[prev_idx]
        perceived_next = next_level_db + self.true_offsets[next_idx]
        diff = perceived_next - perceived_prev
        diff += self.rng.normal(0.0, self.internal_noise_db)

        half_jnd = self.jnd_db / 2.0
        if diff > half_jnd:
            return "louder"
        if diff < -half_jnd:
            return "quieter"
        return "same"


def random_ground_truth(
    bands: list[Band], rng: np.random.Generator, idiosyncratic_sigma_db: float = 2.0
) -> dict[int, float]:
    """A plausible-looking personal offset curve: the typical shape above
    plus per-band idiosyncratic noise, so each simulated listener is a bit
    different (and not perfectly smooth, like a real ear).
    """
    offsets = {}
    for b in bands:
        shape = TYPICAL_SHAPE_HZ_DB.get(b.freq_hz, 0.0)
        idiosyncratic = rng.normal(0.0, idiosyncratic_sigma_db)
        offsets[b.index] = shape + idiosyncratic
    ref_idx = next(b.index for b in bands if b.is_reference)
    zero = offsets[ref_idx]
    return {idx: v - zero for idx, v in offsets.items()}


def run_simulation(
    bands: list[Band] | None = None,
    true_offsets: dict[int, float] | None = None,
    n_trials: int = 400,
    seed: int = 0,
    jnd_db: float = 2.5,
    internal_noise_db: float = 1.2,
) -> dict:
    bands = bands or build_bands()
    rng = np.random.default_rng(seed)
    true_offsets = true_offsets or random_ground_truth(bands, rng)

    model = HearingModel(bands=bands, rng=rng)
    listener = VirtualListener(true_offsets, jnd_db, internal_noise_db, rng)

    prev_band = next(b for b in bands if b.is_reference)
    prev_level_db = 0.0
    rms_error_history = []

    for _ in range(n_trials):
        next_band = model.select_next_band(prev_band.index)
        next_level_db, delta_db = model.plan_trial(prev_band, prev_level_db, next_band)

        response = listener.judge(prev_band.index, prev_level_db, next_band.index, next_level_db)

        model.observe(
            prev_band.index,
            next_band.index,
            delta_db,
            response,
            prev_level_db=prev_level_db,
            next_level_db=next_level_db,
        )

        est = model.offsets()
        errs = [est[b.index] - true_offsets[b.index] for b in bands]
        rms_error_history.append(float(np.sqrt(np.mean(np.square(errs)))))

        prev_band, prev_level_db = next_band, next_level_db
        prev_level_db = max(-20.0, min(prev_level_db, 6.0))

    return {
        "true_offsets": true_offsets,
        "estimated_offsets": model.offsets(),
        "rms_error_history": rms_error_history,
        "convergence_fraction": model.convergence_fraction(),
        "bands": bands,
    }


def _main() -> None:
    result = run_simulation()
    bands_by_idx = {b.index: b for b in result["bands"]}
    print(f"final RMS error: {result['rms_error_history'][-1]:.2f} dB")
    print(f"convergence fraction: {result['convergence_fraction']:.0%}")
    print(f"{'freq':>8}  {'true':>7}  {'est':>7}  {'err':>6}")
    for idx in sorted(bands_by_idx):
        t = result["true_offsets"][idx]
        e = result["estimated_offsets"][idx]
        freq = bands_by_idx[idx].freq_hz
        print(f"{freq:8.0f}  {t:7.2f}  {e:7.2f}  {e - t:6.2f}")


if __name__ == "__main__":
    _main()
