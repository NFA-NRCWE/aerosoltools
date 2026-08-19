"""Create a Desktop / Start-Menu shortcut for the aerosoltools GUI.

Run this once after installing the package::

    aerosoltools-gui-shortcut

It drops an "AerosolTools" shortcut on the Desktop and in the Start Menu, so the
GUI can be launched by double-click or via Windows search — no terminal or IDE
required. Re-running it simply refreshes the shortcuts.

This module is the ``aerosoltools-gui-shortcut`` console-script entry point, so
it deliberately lives at the ``gui`` package top level (next to the
``aerosoltools-gui`` entry point in ``__init__``): pip bakes the entry-point
module path into the generated launcher at install time, so moving this module
into a subpackage would break the launcher of every already-installed
environment until it is reinstalled.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .view.assets import icon_path

SHORTCUT_NAME = "AerosolTools.lnk"

# PowerShell that creates the shortcut(s) at the real (OneDrive-aware) Desktop
# and Start-Menu locations. All inputs are passed via environment variables to
# avoid quoting problems with spaces in paths.
_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$ws = New-Object -ComObject WScript.Shell
$targets = @([Environment]::GetFolderPath('Desktop'),
             [Environment]::GetFolderPath('Programs'))
foreach ($dir in $targets) {
    if (-not (Test-Path $dir)) { continue }
    $lnk = Join-Path $dir $env:AT_LNK_NAME
    $s = $ws.CreateShortcut($lnk)
    $s.TargetPath = $env:AT_TARGET
    $s.Arguments = $env:AT_ARGS
    if ($env:AT_ICON) { $s.IconLocation = $env:AT_ICON }
    $s.WorkingDirectory = $env:AT_WORKDIR
    $s.Description = 'AerosolTools - aerosol data viewer'
    $s.Save()
    Write-Output $lnk
}
"""


def _resolve_launcher() -> tuple[str, str]:
    """Return ``(target, arguments)`` for the shortcut.

    Prefer the ``aerosoltools-gui`` launcher that pip installs into the
    environment's ``Scripts`` folder (a windowless GUI launcher). Fall back to
    ``pythonw.exe -m aerosoltools.gui`` if that executable is not present.
    """
    scripts_exe = Path(sys.prefix) / "Scripts" / "aerosoltools-gui.exe"
    if scripts_exe.exists():
        return str(scripts_exe), ""

    # Fall back to pythonw (no console window) running the module.
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pythonw = Path(sys.prefix) / "pythonw.exe"
    launcher = str(pythonw) if pythonw.exists() else sys.executable
    return launcher, "-m aerosoltools.gui"


def create_shortcuts() -> list[str]:
    """Create the Desktop and Start-Menu shortcuts; return the paths created."""
    if os.name != "nt":
        raise RuntimeError("Shortcut creation is only supported on Windows.")

    target, args = _resolve_launcher()
    env = os.environ.copy()
    env["AT_LNK_NAME"] = SHORTCUT_NAME
    env["AT_TARGET"] = target
    env["AT_ARGS"] = args
    env["AT_ICON"] = icon_path() or ""
    env["AT_WORKDIR"] = str(Path(sys.prefix))

    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to create shortcut:\n" + (proc.stderr or proc.stdout)
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    """Console entry point: create the shortcuts and report where they landed."""
    try:
        created = create_shortcuts()
    except Exception as exc:
        print(f"Could not create the AerosolTools shortcut: {exc}")
        return 1

    if not created:
        print("No shortcut locations were available.")
        return 1

    print("Created AerosolTools shortcut(s):")
    for path in created:
        print(f"  {path}")
    print(
        "\nYou can now launch the app from the Desktop icon, or by pressing the "
        "Windows key and typing 'AerosolTools'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
