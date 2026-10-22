import itertools

from earcal.session import Session


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
