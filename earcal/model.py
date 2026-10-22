"""The adaptive loudness-matching model.

Each 1/3-octave band has an estimated dB offset relative to the 1 kHz
reference band: the correction that would need to be added to a tone at
that band to make it sound as loud as the same physical level at 1 kHz.
Bands with a less certain estimate get a bigger "step" (the probe
difference deliberately added on top of the current prediction); the
step shrinks on reversals, like a classic transformed up-down staircase,
which is what gives the "large differences at first, vanishing as it
converges" behavior the whole test is built around.

Trials are a *sequence*: each new tone is only ever compared with the
one immediately before it, exactly as proposed. A trial's probe delta is
derived from the two bands' current uncertainty (with randomized sign
and magnitude jitter) rather than any fixed schedule, so there's no
pattern to anticipate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from earcal.bands import Band, reference_band

RESPONSE_SCORES = {"quieter": -1.0, "same": 0.0, "louder": 1.0}

INITIAL_STEP_DB = 8.0
MIN_STEP_DB = 0.5
STEP_DECAY_ON_REVERSAL = 0.70
STEP_DECAY_PER_TRIAL = 0.985
DISCRIMINATION_SCALE_DB = 2.5  # assumed loudness JND, used only to shape the update
LEARNING_RATE = 1.5
ANCHOR_EVERY_N_TRIALS = 6
REPEAT_BAND_PROB = 0.08


@dataclass
class BandState:
    offset_db: float = 0.0
    step_db: float = INITIAL_STEP_DB
    last_err_sign: float = 0.0
    visits: int = 0

    def converged(self, tol: float = 1.25) -> bool:
        return self.step_db <= MIN_STEP_DB * tol


@dataclass
class TrialRecord:
    trial_num: int
    timestamp: float
    prev_idx: int
    next_idx: int
    prev_level_db: float
    next_level_db: float
    delta_db: float
    response: str


@dataclass
class HearingModel:
    bands: list[Band]
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    states: dict[int, BandState] = field(default_factory=dict)
    history: list[TrialRecord] = field(default_factory=list)
    trial_num: int = 0

    def __post_init__(self) -> None:
        if not self.states:
            self.states = {b.index: BandState() for b in self.bands}
        self._ref_idx = reference_band(self.bands).index

    def select_next_band(self, prev_idx: int | None) -> Band:
        force_anchor = (
            prev_idx is not None
            and self.trial_num > 0
            and self.trial_num % ANCHOR_EVERY_N_TRIALS == 0
            and prev_idx != self._ref_idx
        )
        if force_anchor:
            return self.bands[self._ref_idx]

        candidates = self.bands
        if prev_idx is not None and self.rng.random() > REPEAT_BAND_PROB:
            candidates = [b for b in self.bands if b.index != prev_idx]

        weights = np.array([self.states[b.index].step_db for b in candidates])
        weights = weights / weights.sum()
        choice = self.rng.choice(len(candidates), p=weights)
        return candidates[choice]

    def plan_trial(
        self, prev_band: Band, prev_level_db: float, next_band: Band
    ) -> tuple[float, float]:
        """Return (next_level_db, delta_db) for presenting next_band."""
        prev_state = self.states[prev_band.index]
        next_state = self.states[next_band.index]

        step = (prev_state.step_db + next_state.step_db) / 2.0
        magnitude = step * self.rng.uniform(0.5, 1.5)
        magnitude = max(MIN_STEP_DB * 0.5, min(magnitude, 15.0))
        sign = self.rng.choice([-1.0, 1.0])
        delta_db = sign * magnitude

        predicted_equal_level = (
            prev_level_db + prev_state.offset_db - next_state.offset_db
        )
        next_level_db = predicted_equal_level + delta_db
        return next_level_db, delta_db

    def observe(
        self,
        prev_idx: int,
        next_idx: int,
        delta_db: float,
        response: str,
        prev_level_db: float = 0.0,
        next_level_db: float = 0.0,
    ) -> None:
        if response not in RESPONSE_SCORES:
            raise ValueError(f"unknown response: {response!r}")

        prev_state = self.states[prev_idx]
        next_state = self.states[next_idx]

        y = RESPONSE_SCORES[response]
        expected = max(-1.0, min(1.0, delta_db / DISCRIMINATION_SCALE_DB))
        err = y - expected

        total_step = prev_state.step_db + next_state.step_db
        w_prev = prev_state.step_db / total_step
        w_next = next_state.step_db / total_step

        prev_state.offset_db -= LEARNING_RATE * err * w_prev
        next_state.offset_db += LEARNING_RATE * err * w_next

        for state in (prev_state, next_state):
            if state.last_err_sign != 0.0 and np.sign(err) != state.last_err_sign and err != 0.0:
                state.step_db = max(MIN_STEP_DB, state.step_db * STEP_DECAY_ON_REVERSAL)
            state.step_db = max(MIN_STEP_DB, state.step_db * STEP_DECAY_PER_TRIAL)
            if err != 0.0:
                state.last_err_sign = float(np.sign(err))
            state.visits += 1

        if self._ref_idx == prev_idx:
            prev_state.offset_db = 0.0
        if self._ref_idx == next_idx:
            next_state.offset_db = 0.0

        self.history.append(
            TrialRecord(
                trial_num=self.trial_num,
                timestamp=time.time(),
                prev_idx=prev_idx,
                next_idx=next_idx,
                prev_level_db=prev_level_db,
                next_level_db=next_level_db,
                delta_db=delta_db,
                response=response,
            )
        )
        self.trial_num += 1

    def convergence_fraction(self) -> float:
        converged = sum(1 for s in self.states.values() if s.converged())
        return converged / len(self.states)

    def offsets(self) -> dict[int, float]:
        return {idx: s.offset_db for idx, s in self.states.items()}

    def snapshot(self) -> dict:
        return {
            "trial_num": self.trial_num,
            "states": {
                str(idx): {
                    "offset_db": s.offset_db,
                    "step_db": s.step_db,
                    "last_err_sign": s.last_err_sign,
                    "visits": s.visits,
                }
                for idx, s in self.states.items()
            },
        }

    @classmethod
    def from_snapshot(cls, bands: list[Band], snapshot: dict, rng=None) -> "HearingModel":
        model = cls(bands=bands, rng=rng or np.random.default_rng())
        model.trial_num = snapshot.get("trial_num", 0)
        for idx_str, s in snapshot.get("states", {}).items():
            model.states[int(idx_str)] = BandState(**s)
        return model
