"""Orchestrates a calibration run: picks bands, computes levels, delegates
the actual playback/response-gathering to caller-supplied callables (so
this module stays UI- and audio-backend-agnostic and is easy to test),
and persists everything needed to resume later.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from earcal.bands import Band, build_bands, reference_band
from earcal.model import HearingModel, TrialRecord
from earcal.synth import MAX_LEVEL_DB, MIN_LEVEL_DB

SCHEMA_VERSION = 1

INAUDIBLE_RESPONSE = "inaudible"
REPEAT_RESPONSE = "repeat"

# Pause between two tones played back-to-back for a replay/repeat, so they
# don't blend into what sounds like one continuous note.
REPLAY_GAP_S = 0.25

# dB nudge per keypress in tune mode.
TUNE_STEP_DB = 1.0

PlayTone = Callable[[float, float], None]
AskResponse = Callable[[], str]


@dataclass
class TrialOutcome:
    prev_band: Band
    next_band: Band
    next_level_db: float
    delta_db: float
    response: str
    scored_as: str


class Session:
    def __init__(
        self,
        bands: list[Band],
        model: HearingModel,
        play_tone: PlayTone,
        ask_response: AskResponse,
        prev_band_idx: int | None = None,
        prev_level_db: float = 0.0,
        base_level_db: float = 0.0,
        path: str | None = None,
        created_at: float | None = None,
        inaudible_count: int = 0,
        needs_reference_replay: bool = False,
        tune_adjustments_db: dict[int, float] | None = None,
    ) -> None:
        self.bands = bands
        self.model = model
        self.play_tone = play_tone
        self.ask_response = ask_response
        self.prev_band_idx = prev_band_idx
        self.prev_level_db = prev_level_db
        self.base_level_db = base_level_db
        self.path = path
        self.created_at = created_at or time.time()
        self.inaudible_count = inaudible_count
        self.needs_reference_replay = needs_reference_replay
        self.tune_adjustments_db = dict(tune_adjustments_db) if tune_adjustments_db else {}

    @classmethod
    def new(
        cls,
        play_tone: PlayTone,
        ask_response: AskResponse,
        path: str,
        base_level_db: float = 0.0,
        bands: list[Band] | None = None,
        seed: int | None = None,
    ) -> "Session":
        bands = bands or build_bands()
        model = HearingModel(bands=bands, rng=np.random.default_rng(seed))
        return cls(
            bands=bands,
            model=model,
            play_tone=play_tone,
            ask_response=ask_response,
            base_level_db=base_level_db,
            path=path,
        )

    @classmethod
    def load(cls, path: str, play_tone: PlayTone, ask_response: AskResponse) -> "Session":
        with open(path) as f:
            data = json.load(f)

        bands = build_bands(tuple(data["bands"]))
        rng = np.random.default_rng()
        rng.bit_generator.state = data["rng_state"]
        model = HearingModel.from_snapshot(bands, data["model"], rng=rng)
        model.history = [TrialRecord(**rec) for rec in data["history"]]

        return cls(
            bands=bands,
            model=model,
            play_tone=play_tone,
            ask_response=ask_response,
            prev_band_idx=data["prev_band_idx"],
            prev_level_db=data["prev_level_db"],
            base_level_db=data["base_level_db"],
            path=path,
            created_at=data["created_at"],
            inaudible_count=data.get("inaudible_count", data.get("discarded_count", 0)),
            needs_reference_replay=data.get("needs_reference_replay", False),
            tune_adjustments_db={
                int(idx): db for idx, db in data.get("tune_adjustments_db", {}).items()
            },
        )

    def save(self) -> None:
        if self.path is None:
            return
        data = {
            "version": SCHEMA_VERSION,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "bands": [b.freq_hz for b in self.bands],
            "base_level_db": self.base_level_db,
            "prev_band_idx": self.prev_band_idx,
            "prev_level_db": self.prev_level_db,
            "inaudible_count": self.inaudible_count,
            "needs_reference_replay": self.needs_reference_replay,
            "tune_adjustments_db": {
                str(idx): db for idx, db in self.tune_adjustments_db.items()
            },
            "model": self.model.snapshot(),
            "history": [dataclasses.asdict(r) for r in self.model.history],
            "rng_state": self.model.rng.bit_generator.state,
        }
        dir_name = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".earcal-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def prime(self) -> None:
        """Play a first, reference-level tone to anchor the sequence,
        if one hasn't already been played (e.g. on a fresh session)."""
        if self.prev_band_idx is not None:
            return
        ref = reference_band(self.bands)
        self.play_tone(ref.freq_hz, self.base_level_db)
        self.prev_band_idx = ref.index
        self.prev_level_db = self.base_level_db

    def _pick_candidate(
        self, prev_band: Band, prev_level_db: float
    ) -> tuple[Band, float, float]:
        next_band = self.model.select_next_band(prev_band.index)
        next_level_db, delta_db = self.model.plan_trial(
            prev_band, prev_level_db, next_band, self.base_level_db
        )
        next_level_db = max(MIN_LEVEL_DB, min(MAX_LEVEL_DB, next_level_db))
        return next_band, next_level_db, delta_db

    def run_trial(self) -> TrialOutcome:
        """Play one new tone and compare it to the last one heard.

        'repeat' just replays the current pair again, nothing else
        changes. 'inaudible' ("so quiet I didn't hear it") *does* count
        as data - it's scored as "quieter" than the reference, since
        that's exactly what it means - but the tone itself can't serve
        as a memory anchor for the next comparison, so prev_band/
        prev_level_db stay put and the next trial replays the reference
        before playing its new candidate.
        """
        if self.prev_band_idx is None:
            raise RuntimeError("call prime() before run_trial()")

        prev_band = self.bands[self.prev_band_idx]
        prev_level_db = self.prev_level_db

        if self.needs_reference_replay:
            self.play_tone(prev_band.freq_hz, prev_level_db)
            time.sleep(REPLAY_GAP_S)
            self.needs_reference_replay = False

        next_band, next_level_db, delta_db = self._pick_candidate(prev_band, prev_level_db)
        self.play_tone(next_band.freq_hz, next_level_db)

        while True:
            response = self.ask_response()
            if response != REPEAT_RESPONSE:
                break
            self.play_tone(prev_band.freq_hz, prev_level_db)
            time.sleep(REPLAY_GAP_S)
            self.play_tone(next_band.freq_hz, next_level_db)

        scored_as = "quieter" if response == INAUDIBLE_RESPONSE else response
        self.model.observe(
            self.prev_band_idx,
            next_band.index,
            delta_db,
            scored_as,
            prev_level_db=prev_level_db,
            next_level_db=next_level_db,
        )

        outcome = TrialOutcome(prev_band, next_band, next_level_db, delta_db, response, scored_as)

        if response == INAUDIBLE_RESPONSE:
            self.inaudible_count += 1
            self.needs_reference_replay = True
            # prev_band/prev_level_db deliberately NOT advanced: a tone
            # the user couldn't hear can't anchor the next comparison.
        else:
            self.prev_band_idx = next_band.index
            self.prev_level_db = next_level_db

        self.save()
        return outcome

    def convergence_fraction(self) -> float:
        return self.model.convergence_fraction()

    def uncertainty_db(self) -> float:
        return self.model.uncertainty_db()

    def offsets_by_freq(self) -> dict[float, float]:
        offsets = self.model.offsets()
        return {b.freq_hz: offsets[b.index] for b in self.bands}

    def measured_offset_db(self, band_idx: int) -> float:
        return self.model.offsets()[band_idx]

    def tuned_offset_db(self, band_idx: int) -> float:
        return self.measured_offset_db(band_idx) + self.tune_adjustments_db.get(band_idx, 0.0)

    def adjust_tune(self, band_idx: int, delta_db: float) -> None:
        self.tune_adjustments_db[band_idx] = (
            self.tune_adjustments_db.get(band_idx, 0.0) + delta_db
        )

    def play_band_tuned(self, band_idx: int) -> None:
        """Play a band at its currently tuned (measured + manual adjustment)
        level, for the sighted playback/tune mode - not part of the blind
        calibration trial flow."""
        band = self.bands[band_idx]
        level_db = self.base_level_db + self.tuned_offset_db(band_idx)
        level_db = max(MIN_LEVEL_DB, min(MAX_LEVEL_DB, level_db))
        self.play_tone(band.freq_hz, level_db)
