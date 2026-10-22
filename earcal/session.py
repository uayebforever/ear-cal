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

PlayTone = Callable[[float, float], None]
AskResponse = Callable[[], str]


@dataclass
class TrialOutcome:
    prev_band: Band
    next_band: Band
    next_level_db: float
    delta_db: float
    response: str


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
        discarded_count: int = 0,
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
        self.discarded_count = discarded_count

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
            discarded_count=data.get("discarded_count", 0),
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
            "discarded_count": self.discarded_count,
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

    def run_trial(self) -> TrialOutcome:
        if self.prev_band_idx is None:
            raise RuntimeError("call prime() before run_trial()")

        prev_band = self.bands[self.prev_band_idx]
        next_band = self.model.select_next_band(self.prev_band_idx)
        next_level_db, delta_db = self.model.plan_trial(
            prev_band, self.prev_level_db, next_band
        )
        next_level_db = max(MIN_LEVEL_DB, min(MAX_LEVEL_DB, next_level_db))

        self.play_tone(next_band.freq_hz, next_level_db)
        response = self.ask_response()

        outcome = TrialOutcome(prev_band, next_band, next_level_db, delta_db, response)

        if response == INAUDIBLE_RESPONSE:
            # Discarded: not fed to the model, and the "last sound heard"
            # stays whatever it was before this one, since the user didn't
            # actually perceive it.
            self.discarded_count += 1
            self.save()
            return outcome

        self.model.observe(
            self.prev_band_idx,
            next_band.index,
            delta_db,
            response,
            prev_level_db=self.prev_level_db,
            next_level_db=next_level_db,
        )

        self.prev_band_idx = next_band.index
        self.prev_level_db = next_level_db
        self.save()
        return outcome

    def convergence_fraction(self) -> float:
        return self.model.convergence_fraction()

    def offsets_by_freq(self) -> dict[float, float]:
        offsets = self.model.offsets()
        return {b.freq_hz: offsets[b.index] for b in self.bands}
