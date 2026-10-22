import io

from earcal.keypress import read_key


class _FakeStdin(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_read_key_falls_back_to_line_when_not_a_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", _FakeStdin("D\n"))
    assert read_key() == "d"


def test_read_key_fallback_handles_empty_line(monkeypatch):
    monkeypatch.setattr("sys.stdin", _FakeStdin("\n"))
    assert read_key() == ""
