# ear-cal

An adaptive test that measures how loud different frequencies sound *to
you*, and builds a personal loudness-correction curve from it.

## How it works

One tone plays at a time. You compare it only to the tone you just
heard - louder, the same, or quieter - never shown the frequency or
level. The test picks the next frequency and how different to make it
based on how confident it currently is about that part of your hearing:
big, obvious differences early on, shrinking automatically as your curve
comes into focus, down to differences you can no longer reliably tell
apart. That's convergence.

Frequencies are sampled at 1/3-octave resolution (the standard
acoustical spacing) from 100 Hz to 12.5 kHz - chosen because it's the
range typical speakers/headphones reproduce faithfully and ears judge
reliably; this isn't a clinical audiometer and it isn't calibrated to
absolute dB SPL. The tones themselves are a fundamental plus a few quiet
harmonics (not a pure sine, not a sampled piano) so every band has
*identical* timbre - a real piano sample's spectral shape changes a lot
with pitch, which would confound a loudness judgement with a timbre
judgement.

### The algorithm

Each band has an estimated dB offset relative to 1 kHz (the reference,
pinned at 0) and an uncertainty ("step") that starts large and shrinks,
reversal by reversal, like a classic adaptive staircase. Trial selection
favors whichever bands are least converged, with occasional forced
comparisons back to the 1 kHz reference to keep the whole curve
anchored and stop long chains of comparisons from drifting. Probe
direction and magnitude are randomized each trial - nothing about the
sequence is fixed, so there's no pattern to anticipate.

`earcal/simulate.py` runs this whole procedure against a simulated
listener with a known hearing curve and realistic response noise, so the
algorithm's convergence behavior can be (and was) validated without
spending any real listening time:

```
python3 -m earcal.simulate
```

The design/derivation of the update rule is in the docstrings of
`earcal/model.py`.

## Setup

```
pip install -e ".[dev]"
```

Needs real audio output - speakers or headphones, run on your own
machine. On Linux, `sounddevice` needs PortAudio available:

```
sudo apt install libportaudio2   # if `earcal run` can't find an output device
```

## Running it

```
earcal run
```

First run walks you through:
1. **Volume calibration** - play a reference tone, adjust your *system*
   volume (not anything in this program) to a comfortable level, then
   leave it alone for the rest of the session. Everything after this is
   relative to that level.
2. **Practice** - a handful of unscored trials with obvious differences,
   just to get used to responding.
3. **The real test** - `q` (quieter) / `s` (same) / `l` (louder), or `x`
   anytime to stop. Progress saves after every trial, so stopping and
   resuming later (`earcal run` again, same `--session` path) picks up
   exactly where you left off.

When you've got enough convergence (the test tells you when most bands
are there, but you can keep refining):

```
earcal report --session ~/.earcal/session.json --out-dir .
```

writes `earcal_curve.csv` (frequency, measured offset in dB, remaining
uncertainty, convergence flag) and `earcal_curve.png` (a plot of your
curve against a generic reference shape for context).

## Safety

Playback level is hard-clamped (`earcal/synth.py`, `MIN_LEVEL_DB` /
`MAX_LEVEL_DB`) and the whole test is meant to run at a comfortable,
moderate volume you set once at the start. If anything ever sounds too
loud, stop (`x`) and lower your system volume before resuming.

## Limitations

- Relative, not clinical: this measures *your* loudness differences
  across frequency on *your* playback gear, not calibrated dB SPL. It's
  not a substitute for an audiologist.
- The output curve is a starting point for a personal EQ correction, not
  a medical hearing-loss measurement.
