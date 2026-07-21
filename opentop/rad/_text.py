"""Text decoding shared by legacy RAD readers."""

from __future__ import annotations

from pathlib import Path


def read_legacy_lines(path: Path) -> list[str]:
    """Decode RAD text as UTF-8 when possible, otherwise Windows-1252."""

    content = path.read_bytes()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp1252")
    return text.splitlines()
