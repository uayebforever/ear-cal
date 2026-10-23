import csv
import itertools
import os

from earcal.report import write_report, write_tuned_plot
from earcal.session import Session


def test_write_report_produces_csv_and_plot(tmp_path):
    calls = []
    resp = itertools.cycle(["louder", "same", "quieter"])
    session = Session.new(
        play_tone=lambda f, l: calls.append((f, l)),
        ask_response=lambda: next(resp),
        path=str(tmp_path / "session.json"),
        seed=1,
    )
    session.prime()
    for _ in range(30):
        session.run_trial()

    out_dir = str(tmp_path / "out")
    write_report(session, out_dir)

    csv_path = os.path.join(out_dir, "earcal_curve.csv")
    plot_path = os.path.join(out_dir, "earcal_curve.png")
    assert os.path.exists(csv_path)
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(session.bands)
    freqs_in_csv = {float(r["freq_hz"]) for r in rows}
    assert freqs_in_csv == {b.freq_hz for b in session.bands}


def test_write_tuned_plot_produces_png_no_csv(tmp_path):
    session = Session.new(
        play_tone=lambda f, l: None,
        ask_response=lambda: "same",
        path=str(tmp_path / "session.json"),
        seed=2,
    )
    idx = session.bands[4].index
    session.adjust_tune(idx, 2.5)

    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "earcal_tuned_curve.png")
    write_tuned_plot(session, plot_path)

    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0
    assert os.listdir(out_dir) == ["earcal_tuned_curve.png"]
