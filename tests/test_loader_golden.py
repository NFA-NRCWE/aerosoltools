"""Golden-output regression net for the instrument loaders.

The per-instrument loaders are being refactored onto a shared read pipeline
(see handoff Task 1). The existing loader tests only assert the result is
non-``None``; this module pins the *actual* loaded output — the numeric data,
columns, size axis and key metadata — so any silent numeric or metadata drift
during the refactor fails a test.

Each ``(loader, sample)`` pair is fingerprinted (see :func:`_fingerprint`) and
compared against the checked-in ``tests/data/loader_golden.json``. Values are
rounded before hashing so genuine changes are caught while last-ULP float noise
across numpy/pandas versions is tolerated.

To intentionally regenerate the golden file after a *reviewed* behaviour change::

    UPDATE_LOADER_GOLDEN=1 python -m pytest tests/test_loader_golden.py -q
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

import aerosoltools as at

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_GOLDEN = os.path.join(_DATA_DIR, "loader_golden.json")
_NAN = -987654.321  # sentinel so NaNs hash stably

# (loader function name, sample filename) — mirrors the smoke-test matrix.
LOADER_CASES = [
    ("load_aethalometer_file", "Sample_Aetholometer.csv"),
    ("load_aps_file", "Sample_APS_aero.txt"),
    ("load_aps_file", "Sample_APS_correlated.txt"),
    ("load_cpc_file", "Sample_CPC_Direct.txt"),
    ("load_discmini_file", "Sample_Discmini.txt"),
    ("load_elpi_file", "Sample_ELPI.txt"),
    ("load_elpi_file", "Sample_ELPI2.txt"),
    ("load_fmps_file", "Sample_FMPS.txt"),
    ("load_fmps_file", "Sample_FMPS2.txt"),
    ("load_fourtec_file", "Sample_Fourtec.xlsx"),
    ("load_grimm_file", "Sample_Grimm.txt"),
    ("load_ns_file", "Sample_NS.csv"),
    ("load_opcn3_file", "Sample_OPCN3.txt"),
    ("load_ops_file", "Sample_OPS.csv"),
    ("load_ops_file", "Sample_OPS2.txt"),
    ("load_partector_file", "Sample_Partector.txt"),
    ("load_ranger_file", "Sample_Ranger.csv"),
    ("load_smps_file", "Sample_SMPS.txt"),
]


def _round(arr: np.ndarray) -> np.ndarray:
    """Round to 6 decimals with NaNs filled, so hashing is version-stable."""
    return np.round(np.nan_to_num(np.asarray(arr, dtype=float), nan=_NAN), 6)


def _fingerprint(obj) -> dict:
    """Stable fingerprint of one loaded aerosol object."""
    df = obj.data
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    rounded = _round(numeric)
    idx = df.index
    fp: dict = {
        "type": type(obj).__name__,
        "shape": list(df.shape),
        "columns": [str(c) for c in df.columns],
        "dtype": str(getattr(obj, "dtype", None)),
        "unit": str(getattr(obj, "unit", None)),
        "serial": str(getattr(obj, "serial_number", None)),
        "index_start": str(idx[0]) if len(idx) else None,
        "index_end": str(idx[-1]) if len(idx) else None,
        "data_hash": hashlib.md5(np.ascontiguousarray(rounded)).hexdigest(),
        "data_sum": round(float(np.nansum(rounded)), 3),
    }
    for attr in ("bin_mids", "bin_edges", "density"):
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                if attr == "density":
                    fp[attr] = round(float(val), 6)
                else:
                    fp[attr] = [round(float(x), 4) for x in np.asarray(val)]
            except (TypeError, ValueError):
                pass
    return fp


def _fingerprint_case(func: str, fname: str):
    """Fingerprint a loader result (handles multi-object loaders like Ranger)."""
    obj = getattr(at, func)(os.path.join(_DATA_DIR, fname))
    if isinstance(obj, list):
        return [_fingerprint(o) for o in obj]
    return _fingerprint(obj)


def _build_all() -> dict:
    return {
        f"{func}:{fname}": _fingerprint_case(func, fname)
        for func, fname in LOADER_CASES
    }


if os.environ.get("UPDATE_LOADER_GOLDEN") == "1":

    def test_regenerate_golden():
        """Rewrite the golden file (opt-in via UPDATE_LOADER_GOLDEN=1)."""
        with open(_GOLDEN, "w", encoding="utf-8") as fh:
            json.dump(_build_all(), fh, indent=2, sort_keys=True)
        assert os.path.exists(_GOLDEN)

else:

    with open(_GOLDEN, encoding="utf-8") as _fh:
        _GOLDEN_DATA = json.load(_fh)

    @pytest.mark.parametrize("func, fname", LOADER_CASES)
    def test_loader_output_matches_golden(func, fname):
        """The loader's output still matches its checked-in fingerprint."""
        key = f"{func}:{fname}"
        assert key in _GOLDEN_DATA, f"No golden fingerprint for {key}; regenerate."
        assert _fingerprint_case(func, fname) == _GOLDEN_DATA[key]
