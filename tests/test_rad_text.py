"""Tests for legacy RAD text decoding."""

from opentop.rad._text import read_legacy_lines


def test_rad_text_falls_back_to_windows_1252(tmp_path):
    path = tmp_path / "legacy.awk"
    expected = "# Comment caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    path.write_bytes(f"{expected}\n".encode("cp1252"))

    assert read_legacy_lines(path) == [expected]
