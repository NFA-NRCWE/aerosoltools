"""Which occupational-exposure-limit list the GUI offers, and where newer ones live.

The package ships a parsed snapshot of the Danish limit-value order. A newer order
fetched from Retsinformation (the *Exposure limits* dialog) is saved as JSON in
the user's data folder and, being newer, becomes the list offered for new picks.
A pick already made in a project stores its own values and source (see
``view.exposure_limit_picker.pick_record``), so a later list never changes a
saved analysis.
"""

from __future__ import annotations

from pathlib import Path

from ...exposure_limits import ExposureLimitList, load_exposure_limits
from ..qt import QtCore

_active: ExposureLimitList | None = None


def user_dir() -> Path:
    """Folder holding exposure-limit lists fetched by the user."""
    base = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.GenericDataLocation
    )
    return Path(base or Path.home()) / "aerosoltools" / "exposure_limits"


def _newness(limits: ExposureLimitList) -> tuple:
    """Sort key: the most recently issued order last."""
    return (limits.source.date, limits.source.number)


def available_lists() -> list[ExposureLimitList]:
    """The bundled list plus every readable list in :func:`user_dir`, oldest first."""
    lists = [load_exposure_limits()]
    folder = user_dir()
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            try:
                lists.append(load_exposure_limits(path))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # an unreadable file must not break the picker
    return sorted(lists, key=_newness)


def active_list() -> ExposureLimitList:
    """The list offered for new picks: the most recently issued order available."""
    global _active
    if _active is None:
        _active = available_lists()[-1]
    return _active


def save_user_list(limits: ExposureLimitList) -> Path:
    """Save a fetched list to :func:`user_dir` and re-pick the active list.

    Returns:
        The file written.
    """
    folder = user_dir()
    folder.mkdir(parents=True, exist_ok=True)
    src = limits.source
    path = folder / f"dk_bek_{src.date[:4]}_{src.number}.json"
    limits.to_json(path)
    reload()
    return path


def reload() -> None:
    """Forget the cached active list (re-read on the next :func:`active_list`)."""
    global _active
    _active = None


def compare_lists(old: ExposureLimitList, new: ExposureLimitList) -> list[str]:
    """Readable differences between two lists: added, removed and changed limits.

    Returns:
        One line per difference (empty when the limits are identical).
    """

    def fmt(value) -> str:
        return "–" if value is None else f"{value:g}"

    before = {lim.name: lim for lim in old}
    after = {lim.name: lim for lim in new}
    lines = []
    for name, lim in after.items():
        if name not in before:
            lines.append(f"+ {name}: 8-h {fmt(lim.twa)} {lim.unit} (new)")
            continue
        was = before[name]
        changes = []
        if (was.twa, was.unit) != (lim.twa, lim.unit):
            changes.append(f"8-h {fmt(was.twa)} → {fmt(lim.twa)} {lim.unit}")
        if was.stel != lim.stel:
            changes.append(f"short-term {fmt(was.stel)} → {fmt(lim.stel)} {lim.unit}")
        if was.remarks != lim.remarks:
            changes.append(f"remarks '{was.remarks}' → '{lim.remarks}'")
        if changes:
            lines.append(f"~ {name}: " + "; ".join(changes))
    lines.extend(f"− {name} (removed)" for name in before if name not in after)
    return lines
