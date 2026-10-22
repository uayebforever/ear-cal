import itertools

from earcal.session import INAUDIBLE_RESPONSE, Session


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


def test_inaudible_response_is_discarded_not_recorded(tmp_path):
    calls = []
    session = Session.new(
        play_tone=_fake_player(calls),
        ask_response=_fake_responder(["same"]),
        path=str(tmp_path / "session.json"),
        seed=0,
    )
    session.prime()

    prev_idx_before = session.prev_band_idx
    prev_level_before = session.prev_level_db
    trial_num_before = session.model.trial_num

    session.ask_response = lambda: INAUDIBLE_RESPONSE
    outcome = session.run_trial()

    assert outcome.response == INAUDIBLE_RESPONSE
    assert session.discarded_count == 1
    assert len(session.model.history) == 0
    assert session.model.trial_num == trial_num_before
    # "last sound heard" is unchanged, since this one wasn't heard
    assert session.prev_band_idx == prev_idx_before
    assert session.prev_level_db == prev_level_before


def test_discarded_count_persists_across_save_load(tmp_path):
    path = str(tmp_path / "session.json")
    session = Session.new(
        play_tone=_fake_player([]),
        ask_response=lambda: INAUDIBLE_RESPONSE,
        path=path,
        seed=0,
    )
    session.prime()
    session.run_trial()
    session.run_trial()

    reloaded = Session.load(path, play_tone=_fake_player([]), ask_response=lambda: "same")
    assert reloaded.discarded_count == 2
