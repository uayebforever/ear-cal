import numpy as np

from earcal.bands import build_bands
from earcal.model import HearingModel
from earcal.simulate import random_ground_truth, run_simulation


def test_observe_updates_offsets_in_expected_direction():
    bands = build_bands()
    model = HearingModel(bands=bands, rng=np.random.default_rng(0))
    ref_idx = next(b.index for b in bands if b.is_reference)
    other_idx = next(b.index for b in bands if not b.is_reference)

    # delta=0, but the listener reports the "next" tone (other_idx) as
    # louder than the "prev" tone (the reference) at equal presented
    # level: that means other_idx is more sensitive than currently
    # credited, so its offset estimate should increase and the
    # reference's *would* decrease but is pinned at 0.
    model.observe(ref_idx, other_idx, delta_db=0.0, response="louder")
    assert model.states[other_idx].offset_db > 0
    assert model.states[ref_idx].offset_db == 0.0  # reference is pinned

    model2 = HearingModel(bands=bands, rng=np.random.default_rng(0))
    model2.observe(ref_idx, other_idx, delta_db=0.0, response="quieter")
    assert model2.states[other_idx].offset_db < 0


def test_step_shrinks_over_trials():
    bands = build_bands()
    model = HearingModel(bands=bands, rng=np.random.default_rng(0))
    ref_idx = next(b.index for b in bands if b.is_reference)
    other_idx = next(b.index for b in bands if not b.is_reference)
    initial_step = model.states[other_idx].step_db

    for _ in range(50):
        model.observe(ref_idx, other_idx, delta_db=1.0, response="louder")

    assert model.states[other_idx].step_db < initial_step


def test_reference_offset_always_stays_zero():
    bands = build_bands()
    model = HearingModel(bands=bands, rng=np.random.default_rng(0))
    ref_idx = next(b.index for b in bands if b.is_reference)
    other_idx = next(b.index for b in bands if not b.is_reference)

    for response in ("louder", "quieter", "same", "louder", "quieter"):
        model.observe(ref_idx, other_idx, delta_db=2.0, response=response)
        assert model.states[ref_idx].offset_db == 0.0


def test_simulation_converges_close_to_ground_truth():
    bands = build_bands()
    rng = np.random.default_rng(42)
    true_offsets = random_ground_truth(bands, rng)

    result = run_simulation(bands=bands, true_offsets=true_offsets, n_trials=1200, seed=42)

    assert result["rms_error_history"][-1] < 1.5
    assert result["convergence_fraction"] > 0.9


def test_simulation_error_decreases_over_time():
    result = run_simulation(n_trials=1000, seed=7)
    history = result["rms_error_history"]
    early = np.mean(history[:50])
    late = np.mean(history[-50:])
    assert late < early
