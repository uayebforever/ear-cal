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


def test_inaudible_response_is_retried_with_replay_until_a_real_answer(tmp_path, monkeypatch):
    monkeypatch.setattr("earcal.session.time.sleep", lambda s: None)
    calls = []
    responses = iter(["inaudible", "inaudible", "same"])
    session = Session.new(
        play_tone=_fake_player(calls),
        ask_response=lambda: next(responses),
        path=str(tmp_path / "session.json"),
        seed=0,
    )
    session.prime()  # calls[0]: the reference tone
    prev_idx_before = session.prev_band_idx
    prev_level_before = session.prev_level_db
    calls.clear()

    outcome = session.run_trial()

    assert outcome.response == "same"
    assert outcome.misses == 2
    assert session.discarded_count == 2
    assert len(session.model.history) == 1  # only the final, real answer is recorded

    # calls: [candidate1, replay-A, candidate2, replay-A, candidate3]
    assert len(calls) == 5
    reference_tone = (
        session.bands[prev_idx_before].freq_hz,
        prev_level_before,
    )
    assert calls[1] == reference_tone
    assert calls[3] == reference_tone

    # "last sound heard" only advances to the tone that actually got a
    # real answer, not either of the misses.
    assert session.prev_band_idx == outcome.next_band.index
    assert session.prev_level_db == outcome.next_level_db


def test_repeat_response_replays_the_same_pair_without_discarding(tmp_path, monkeypatch):
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
    assert outcome.misses == 0
    assert session.discarded_count == 0
    assert len(session.model.history) == 1

    # calls: [candidate, replay-A, replay-candidate, replay-A, replay-candidate]
    assert len(calls) == 5
    candidate_tone = calls[0]
    assert calls[2] == candidate_tone
    assert calls[4] == candidate_tone


def test_discarded_count_persists_across_save_load(tmp_path, monkeypatch):
    monkeypatch.setattr("earcal.session.time.sleep", lambda s: None)
    path = str(tmp_path / "session.json")
    responses = itertools.cycle(["inaudible", "same"])
    session = Session.new(
        play_tone=_fake_player([]),
        ask_response=lambda: next(responses),
        path=path,
        seed=0,
    )
    session.prime()
    session.run_trial()
    session.run_trial()

    assert session.discarded_count == 2
    reloaded = Session.load(path, play_tone=_fake_player([]), ask_response=lambda: "same")
    assert reloaded.discarded_count == 2
