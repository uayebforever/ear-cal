"""Single-keypress reading, no Enter required.

Falls back to reading a line when stdin isn't an interactive terminal
(redirected input, most test environments), so callers stay testable.
"""

from __future__ import annotations

import sys


def read_key() -> str:
    """Block until a key is pressed and return it, lowercased."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip()[:1].lower()

    if sys.platform == "win32":
        return _read_key_windows()
    return _read_key_posix()


def _read_key_posix() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch.lower()


def _read_key_windows() -> str:
    import msvcrt

    return msvcrt.getwch().lower()
