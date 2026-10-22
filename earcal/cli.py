"""Blind terminal interface.

Deliberately text-only and minimal: the whole point is that the user
never sees which frequency or level is playing, only "what did you just
hear relative to the one before it?" Response keys are four adjacent
home-row-ish keys, pressed with no Enter needed: a = didn't hear,
s = quieter, d = same, f = louder.
"""

from __future__ import annotations

import argparse
import os
import sys

from earcal.audio_io import AudioUnavailableError, play_blocking, probe_output_device
from earcal.keypress import read_key
from earcal.session import INAUDIBLE_RESPONSE, Session
from earcal.synth import synth_tone

DEFAULT_SESSION_PATH = os.path.expanduser("~/.earcal/session.json")
DEFAULT_BASE_LEVEL_DB = -8.0
PRACTICE_TRIALS = 5
MAX_TRIALS_SAFETY_CAP = 4000

RESPONSE_KEYS = {
    "a": INAUDIBLE_RESPONSE,
    "s": "quieter",
    "d": "same",
    "f": "louder",
}
QUIT_KEYS = {"x"}
PROMPT = "[a] didn't hear   [s] quieter   [d] same   [f] louder   ([x] to stop): "


class QuitRequested(Exception):
    pass


def make_play_tone():
    def play_tone(freq_hz: float, level_db: float) -> None:
        wave = synth_tone(freq_hz, level_db)
        play_blocking(wave)

    return play_tone


def make_ask_response(prompt_prefix: str = "") -> callable:
    def ask() -> str:
        while True:
            print(f"{prompt_prefix}{PROMPT}", end="", flush=True)
            try:
                key = read_key()
            except KeyboardInterrupt:
                print()
                raise QuitRequested()
            print(key)
            if key in QUIT_KEYS:
                raise QuitRequested()
            if key in RESPONSE_KEYS:
                return RESPONSE_KEYS[key]
            print("  (didn't understand that - press a, s, d, or f)")

    return ask


def run_volume_calibration(play_tone, base_level_db: float) -> None:
    print()
    print("=== Step 1: set your listening volume ===")
    print("We'll play a steady reference tone. Using your SYSTEM/device volume")
    print("(not anything in this program), adjust it to a comfortable,")
    print("clearly-audible level - not loud. Once set, don't touch your")
    print("system volume again for the rest of this session; everything from")
    print("here on is calibrated relative to it.")
    print()
    while True:
        input("Press Enter to play the reference tone...")
        play_tone(1000.0, base_level_db)
        again = input("Replay [Enter] or continue [c]? ").strip().lower()
        if again == "c":
            break


def run_practice(play_tone, ask_response) -> None:
    print()
    print("=== Step 2: practice ===")
    print(f"{PRACTICE_TRIALS} practice comparisons with obvious differences, just to")
    print("get used to the task. These aren't recorded.")
    import numpy as np

    from earcal.bands import build_bands

    rng = np.random.default_rng()
    bands = build_bands()
    level = DEFAULT_BASE_LEVEL_DB
    for _ in range(PRACTICE_TRIALS):
        band = bands[rng.integers(0, len(bands))]
        level = DEFAULT_BASE_LEVEL_DB + rng.choice([-8.0, 8.0])
        play_tone(band.freq_hz, level)
        try:
            ask_response()
        except QuitRequested:
            raise
    print("Practice done. The real test now behaves exactly the same way,")
    print("except the differences will often be much smaller - that's expected.")


def run_main_loop(session: Session, ask_response) -> None:
    print()
    print("=== Step 3: the real test ===")
    print("Each new tone replaces the last one in your memory - compare it to")
    print("the one you just heard. Differences start out large and should get")
    print("harder to hear as the test learns your curve. Progress saves after")
    print("every trial, so it's safe to stop anytime with 'x' and resume later.")
    print("Didn't catch one, or got distracted? Press 'a' - it's thrown out,")
    print("not counted as a real answer, and doesn't change your results.")
    print()

    session.prime()
    trial = 0
    announced_converged = False
    while trial < MAX_TRIALS_SAFETY_CAP:
        conv = session.convergence_fraction()
        if conv >= 0.9 and not announced_converged:
            print(
                f"\n{conv:.0%} of bands have converged - you can stop anytime now "
                "with 'x', or keep going to refine further.\n"
            )
            announced_converged = True
        try:
            outcome = session.run_trial()
        except QuitRequested:
            break
        if outcome.response == INAUDIBLE_RESPONSE:
            print("  (discarded - not counted)")
            continue
        trial += 1
        print(f"  trial {session.model.trial_num} done ({conv:.0%} converged)")


def cmd_run(args: argparse.Namespace) -> None:
    device = probe_output_device()
    if device is None:
        print(
            "No audio output device found. This program needs real speakers or "
            "headphones to be useful - run it on your own machine, not a "
            "headless container.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Output device: {device}")

    os.makedirs(os.path.dirname(args.session) or ".", exist_ok=True)
    play_tone = make_play_tone()
    ask_response = make_ask_response()

    try:
        if os.path.exists(args.session) and not args.fresh:
            print(f"Resuming session from {args.session}")
            session = Session.load(args.session, play_tone, ask_response)
        else:
            print(f"Starting a new session at {args.session}")
            session = Session.new(
                play_tone, ask_response, path=args.session, base_level_db=DEFAULT_BASE_LEVEL_DB
            )
            run_volume_calibration(play_tone, DEFAULT_BASE_LEVEL_DB)
            run_practice(play_tone, ask_response)

        run_main_loop(session, ask_response)
    except QuitRequested:
        pass
    except AudioUnavailableError as exc:
        print(f"\nAudio playback failed: {exc}", file=sys.stderr)
        print("Any progress from completed trials was already saved.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSaved. {session.convergence_fraction():.0%} of bands converged, "
          f"{session.model.trial_num} trials so far "
          f"({session.discarded_count} discarded as not heard).")
    print(f"Run `earcal report --session {args.session}` for your curve.")


def cmd_report(args: argparse.Namespace) -> None:
    from earcal.report import write_report

    session = Session.load(args.session, play_tone=lambda *a: None, ask_response=lambda: "same")
    write_report(session, args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earcal")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the interactive calibration test")
    run_p.add_argument("--session", default=DEFAULT_SESSION_PATH)
    run_p.add_argument("--fresh", action="store_true", help="ignore any existing session file")
    run_p.set_defaults(func=cmd_run)

    report_p = sub.add_parser("report", help="write a plot + CSV from a saved session")
    report_p.add_argument("--session", default=DEFAULT_SESSION_PATH)
    report_p.add_argument("--out-dir", default=".")
    report_p.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
