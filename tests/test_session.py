import itertools

from earcal.session import REPEAT_RESPONSE, Session


def _fake_player(calls):
    def play(freq_hz, level_db):
        calls.append((freq_hz, level_db))

    return play


def _fake_responder(responses):
    it = itertools.cycle(responses)

    def ask():
        return next(it)

    return ask


def test_prime_plays_reference_tone(tmp_path):
    calls = []
    session = Session.new(
        play_tone=_fake_player(calls),
        ask_response=_fake_responder(["same"]),
        path=str(tmp_path / "session.json"),
        seed=0,
    )
    session.prime()
    assert calls == [(1000.0, 0.0)]
    assert session.prev_band_idx == session.model._ref_idx


def test_run_trial_advances_and_updates_model(tmp_path):
    calls = []
    session = Session.new(
        play_tone=_fake_player(calls),
        ask_response=_fake_responder(["louder", "same", "quieter"]),
        path=str(tmp_path / "session.json"),
        seed=0,
    )
    session.prime()
    for _ in range(10):
        session.run_trial()

    assert len(calls) == 11  # prime + 10 trials
    assert len(session.model.history) == 10


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "session.json")
    calls = []
    session = Session.new(
        play_tone=_fake_player(calls),
        ask_response=_fake_responder(["louder", "same", "quieter"]),
        path=path,
        seed=0,
    )
    session.prime()
    for _ in range(15):
        session.run_trial()

    offsets_before = session.offsets_by_freq()
    prev_idx_before = session.prev_band_idx
    trial_count_before = len(session.model.history)

    reloaded = Session.load(path, play_tone=_fake_player([]), ask_response=lambda: "same")

    assert reloaded.offsets_by_freq() == offsets_before
    assert reloaded.prev_band_idx == prev_idx_before
    assert len(reloaded.model.history) == trial_count_before

    # resuming should work seamlessly
    reloaded.run_trial()
    assert len(reloaded.model.history) == trial_count_before + 1


def test_inaudible_response_is_scored_as_quieter_and_recorded(tmp_path):
    calls = []
    session = Session.new(
        play_tone=_fake_player(calls),
        ask_response=lambda: "inaudible",
        path=str(tmp_path / "session.json"),
        seed=0,
    )
    session.prime()
    prev_idx_before = session.prev_band_idx
    prev_level_before = session.prev_level_db

    outcome = session.run_trial()

    assert outcome.response == "inaudible"
    assert outcome.scored_as == "quieter"
    assert session.inaudible_count == 1
    assert len(session.model.history) == 1  # it IS data now
    assert session.model.history[0].response == "quieter"

    # a tone the user couldn't hear can't anchor the next comparison, so
    # "last sound heard" does not advance to it
    assert session.prev_band_idx == prev_idx_before
    assert session.prev_level_db == prev_level_before
    assert session.needs_reference_replay is True


def test_reference_is_replayed_before_the_next_candidate_after_a_miss(tmp_path):
    calls = []
    responses = iter(["inaudible", "same"])
    session = Session.new(
        play_tone=_fake_player(calls),
        ask_response=lambda: next(responses),
        path=str(tmp_path / "session.json"),
        seed=0,
    )
    session.prime()  # calls[0]: reference tone
    prev_idx_before = session.prev_band_idx
    prev_level_before = session.prev_level_db

    session.run_trial()  # calls[1]: a candidate, missed
    calls.clear()

    session.run_trial()  # should replay the reference before its own candidate

    assert len(calls) == 2
    assert calls[0] == (session.bands[prev_idx_before].freq_hz, prev_level_before)
    assert session.needs_reference_replay is False


def test_repeat_response_replays_the_same_pair_without_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr("earcal.session.time.sleep", lambda s: None)
    calls = []
    responses = iter([REPEAT_RESPONSE, REPEAT_RESPONSE, "louder"])
    session = Session.new(
        play_tone=_fake_player(calls),
        ask_response=lambda: next(responses),
        path=str(tmp_path / "session.json"),
        seed=0,
    )
    session.prime()
    calls.clear()

    outcome = session.run_trial()

    assert outcome.response == "louder"
    assert outcome.scored_as == "louder"
    assert session.inaudible_count == 0
    assert len(session.model.history) == 1

    # calls: [candidate, replay-A, replay-candidate, replay-A, replay-candidate]
    assert len(calls) == 5
    candidate_tone = calls[0]
    assert calls[2] == candidate_tone
    assert calls[4] == candidate_tone


def test_inaudible_count_and_replay_flag_persist_across_save_load(tmp_path):
    path = str(tmp_path / "session.json")
    session = Session.new(
        play_tone=_fake_player([]),
        ask_response=lambda: "inaudible",
        path=path,
        seed=0,
    )
    session.prime()
    session.run_trial()
    session.run_trial()

    assert session.inaudible_count == 2
    reloaded = Session.load(path, play_tone=_fake_player([]), ask_response=lambda: "same")
    assert reloaded.inaudible_count == 2
    assert reloaded.needs_reference_replay is True


def test_tuned_offset_combines_measured_and_adjustment(tmp_path):
    session = Session.new(
        play_tone=_fake_player([]),
        ask_response=lambda: "same",
        path=str(tmp_path / "session.json"),
        seed=0,
    )
    idx = session.bands[3].index
    session.model.states[idx].offset_db = 4.0

    assert session.measured_offset_db(idx) == 4.0
    assert session.tuned_offset_db(idx) == 4.0  # no adjustment yet

    session.adjust_tune(idx, 1.0)
    session.adjust_tune(idx, 1.0)
    assert session.tuned_offset_db(idx) == 6.0
    assert session.measured_offset_db(idx) == 4.0  # measured curve itself untouched

    session.adjust_tune(idx, -0.5)
    assert session.tuned_offset_db(idx) == 5.5


def test_play_band_tuned_uses_base_plus_tuned_offset(tmp_path):
    calls = []
    session = Session.new(
        play_tone=_fake_player(calls),
        ask_response=lambda: "same",
        path=str(tmp_path / "session.json"),
        seed=0,
        base_level_db=-8.0,
    )
    idx = session.bands[5].index
    session.model.states[idx].offset_db = 3.0
    session.adjust_tune(idx, 2.0)

    session.play_band_tuned(idx)

    assert calls == [(session.bands[idx].freq_hz, -3.0)]  # -8 + 3 + 2


def test_tune_adjustments_persist_across_save_load(tmp_path):
    path = str(tmp_path / "session.json")
    session = Session.new(
        play_tone=_fake_player([]),
        ask_response=lambda: "same",
        path=path,
        seed=0,
    )
    idx = session.bands[2].index
    session.adjust_tune(idx, 3.5)
    session.save()

    reloaded = Session.load(path, play_tone=_fake_player([]), ask_response=lambda: "same")
    assert reloaded.tune_adjustments_db[idx] == 3.5
    assert reloaded.tuned_offset_db(idx) == 3.5
