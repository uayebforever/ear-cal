"""Turns a saved session into something useful: a CSV of the measured
per-band correction curve, and a plot of it against a generic reference
shape for context."""

from __future__ import annotations

import csv
import os

from earcal.bands import TYPICAL_SHAPE_HZ_DB, reference_band
from earcal.session import Session


def write_csv(session: Session, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["freq_hz", "offset_db", "step_db", "converged", "visits"])
        for band in session.bands:
            state = session.model.states[band.index]
            writer.writerow(
                [
                    band.freq_hz,
                    round(state.offset_db, 3),
                    round(state.step_db, 3),
                    state.converged(),
                    state.visits,
                ]
            )


def write_plot(session: Session, path: str) -> None:
    import matplotlib.pyplot as plt

    bands = session.bands
    freqs = [b.freq_hz for b in bands]
    offsets = [session.model.states[b.index].offset_db for b in bands]
    steps = [session.model.states[b.index].step_db for b in bands]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        freqs, offsets, yerr=steps, fmt="o-", capsize=3, label="your measured offset", color="C0"
    )

    typical_freqs = sorted(TYPICAL_SHAPE_HZ_DB)
    typical_vals = [TYPICAL_SHAPE_HZ_DB[f] for f in typical_freqs]
    ax.plot(typical_freqs, typical_vals, "--", color="gray", alpha=0.6, label="generic reference shape")

    ref_hz = reference_band(bands).freq_hz
    ax.axvline(ref_hz, color="gray", linewidth=0.5)
    ax.axhline(0.0, color="gray", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_xticks(freqs)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=60)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Correction needed relative to 1 kHz (dB)")
    ax.set_title(
        f"Personal loudness-correction curve "
        f"({session.model.trial_num} trials, "
        f"{session.convergence_fraction():.0%} converged)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(session: Session, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "earcal_curve.csv")
    plot_path = os.path.join(out_dir, "earcal_curve.png")

    write_csv(session, csv_path)
    write_plot(session, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")

    unconverged = [
        b.freq_hz
        for b in session.bands
        if not session.model.states[b.index].converged(tol=2.0)
    ]
    if unconverged:
        print(
            "Still noisy at: " + ", ".join(f"{f:g} Hz" for f in unconverged)
            + " - more trials at those bands would help."
        )
