"""Locate bundled GUI assets (the application icon)."""

from __future__ import annotations

from pathlib import Path

_ICON = Path(__file__).resolve().parent / "aerosoltools.ico"


def icon_path() -> str | None:
    """Absolute path to the app icon, or ``None`` if it is not bundled."""
    return str(_ICON) if _ICON.exists() else None
