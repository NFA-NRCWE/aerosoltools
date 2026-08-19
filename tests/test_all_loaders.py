import os
import re

import pandas as pd
import pytest

from aerosoltools.loaders import (
    load_aethalometer_file,
    load_aps_file,
    load_cpc_file,
    load_discmini_file,
    load_discmini_raw_file,
    load_dusttrak_file,
    load_elpi_file,
    load_fmps_file,
    load_fourtec_file,
    load_grimm_file,
    load_ns_file,
    load_opcn3_file,
    load_ops_file,
    load_partector_file,
    load_ranger_file,
    load_simple_acsm_file,
    load_smps_file,
    load_tiger_file,
)
from aerosoltools.loaders.discmini import (
    _extract_serial_and_firmware,
    _normalize_serial,
)

#: Every loader paired with a sample file it can read. Shared by the smoke test
#: and the unit-spelling test so a new loader is covered by both at once.
LOADER_CASES = [
    (load_aethalometer_file, "Sample_Aethalometer.csv"),
    (load_aps_file, "Sample_APS_aero.txt"),
    (load_aps_file, "Sample_APS_correlated.txt"),
    (load_cpc_file, "Sample_CPC_Direct.txt"),
    (load_discmini_file, "Sample_Discmini.txt"),
    (load_elpi_file, "Sample_ELPI.txt"),
    (load_elpi_file, "Sample_ELPI2.txt"),
    (load_fmps_file, "Sample_FMPS.txt"),
    (load_fmps_file, "Sample_FMPS2.txt"),
    (load_fourtec_file, "Sample_Fourtec.xlsx"),
    (load_grimm_file, "Sample_Grimm.txt"),
    (load_ns_file, "Sample_NS.csv"),
    (load_opcn3_file, "Sample_OPCN3.txt"),
    (load_ops_file, "Sample_OPS.csv"),
    (load_ops_file, "Sample_OPS2.txt"),
    (load_partector_file, "Sample_Partector.txt"),
    (load_ranger_file, "Sample_Ranger.csv"),
    (load_smps_file, "Sample_SMPS.txt"),
    (load_simple_acsm_file, "Sample_ACSM.csv"),
    (load_tiger_file, "Sample_Tiger.csv"),
    (load_dusttrak_file, "Sample_DustTrak.csv"),
]


@pytest.mark.parametrize("loader_func, filename", LOADER_CASES)
def test_loader_smoke(loader_func, filename):
    test_file = os.path.join(os.path.dirname(__file__), "data", filename)
    assert os.path.exists(test_file), f"Missing test file: {filename}"

    data = loader_func(test_file)
    assert data is not None
    assert hasattr(data, "data"), f"{filename}: missing 'data'"
    assert hasattr(data, "metadata"), f"{filename}: missing 'metadata'"
    assert isinstance(data.data, pd.DataFrame), f"{filename}: data is not DataFrame"


def test_aps_correlated_and_aero_only():
    """APS: correlated -> Aerosol3d (both axes consistent); aero-only -> 2D."""
    from aerosoltools import Aerosol2D, Aerosol3d

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # Aerodynamic-only export loads as a plain 2D distribution.
    aero = load_aps_file(os.path.join(data_dir, "Sample_APS_aero.txt"))
    assert isinstance(aero, Aerosol2D)
    assert not isinstance(aero, Aerosol3d)
    assert len(aero.bin_mids) == 52
    assert aero.unit == "cm⁻³"

    # Correlated export carries both size axes plus their matrix.
    cor = load_aps_file(os.path.join(data_dir, "Sample_APS_correlated.txt"))
    assert isinstance(cor, Aerosol3d)
    assert cor.is_correlated
    assert len(cor.bin_mids) == 52
    assert cor.optical is not None and len(cor.optical.bin_mids) == 16
    # Aerodynamic total, optical total and matrix total agree per sample.
    a0 = float(cor.data["Total_conc"].iloc[0])
    o0 = float(cor.optical.data["Total_conc"].iloc[0])
    m0 = float(cor.correlation.iloc[0].sum())
    assert a0 == pytest.approx(o0, rel=1e-6)
    assert a0 == pytest.approx(m0, rel=1e-6)
    # Time cropping keeps both axes (and the matrix) in sync.
    cor.timecrop(cor.time[3], cor.time[10])
    assert len(cor.data) == len(cor.optical.data) == len(cor.correlation)


def test_correlation_cube():
    """Aerosol3d.correlation_cube reshapes the matrix and normalizes on request."""
    import numpy as np

    from aerosoltools import CorrelationCube, load_aps_file

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    cor = load_aps_file(os.path.join(data_dir, "Sample_APS_correlated.txt"))

    n_opt = len(cor.optical.bin_mids)
    n_aero = len(cor.bin_mids)
    cube = cor.correlation_cube()
    assert isinstance(cube, CorrelationCube)
    assert not cube.normalized
    assert cube.values.shape == (len(cor.correlation), n_opt, n_aero)
    assert cube.optical_edges.shape == (n_opt + 1,)
    assert cube.aero_edges.shape == (n_aero + 1,)
    # The physical total matches the raw matrix row-sums and the aero total.
    assert cube.total[0] == pytest.approx(float(cor.correlation.iloc[0].sum()))
    assert cube.total[0] == pytest.approx(float(cor.data["Total_conc"].iloc[0]))

    # Normalized cube divides each cell by both log bin widths; the total stays
    # the raw physical sum.
    norm = cor.correlation_cube(normalize=True)
    assert norm.normalized
    dlog_opt = np.diff(np.log10(cube.optical_edges))
    dlog_aero = np.diff(np.log10(cube.aero_edges))
    expected = cube.values / (dlog_opt[None, :, None] * dlog_aero[None, None, :])
    np.testing.assert_allclose(norm.values, expected, equal_nan=True)
    np.testing.assert_allclose(norm.total, cube.total)

    # An un-correlated Aerosol3d (no optical axis) has no correlation cube.
    from aerosoltools import Aerosol3d

    uncorrelated = Aerosol3d(cor.data.copy())
    assert not uncorrelated.is_correlated
    with pytest.raises(ValueError):
        uncorrelated.correlation_cube()


_RANGER_STATUS = "Status(f:fail a:aging w:warming-up z:zerocal c:spancal)"


def _write_ranger(path, blocks):
    """Write a synthetic Ranger export with the given (header, rows) blocks."""
    lines = ["Ranger Serial Number:,2605-1000088-A"]
    for header, rows in blocks:
        lines.append(header)
        lines.extend(rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def test_ranger_single_component_returns_single_object():
    """A Cl₂-only file loads as one Gas1D with ppm / Cl₂ metadata."""
    from aerosoltools import Gas1D

    data = load_ranger_file(_data_path("Sample_Ranger.csv"))
    assert isinstance(data, Gas1D)
    assert "Concentration" in data.data.columns
    assert data.metadata["unit"] == "ppm"
    assert data.metadata["dtype"] == "Cl₂"
    assert data.metadata["measurement"] == "Cl₂"
    # 'total_concentration' does not apply to a gas monitor any more.
    with pytest.raises(AttributeError):
        _ = data.total_concentration


def test_ranger_multi_header_splits_by_component(tmp_path):
    """Interleaved heads (PM/Cl₂/NO₂) split into one object per component."""
    from aerosoltools import Aerosol1D, Gas1D

    pm_header = (
        "UTC time,Local time,LocationID,PM1 (ug/m3),PM2.5 (ug/m3),"
        f"PMrsp (ug/m3),PM10 (ug/m3),TSP (ug/m3),{_RANGER_STATUS}"
    )
    cl2_header = f"UTC time,Local time,LocationID,CL2 (PPM),{_RANGER_STATUS}"
    no2_header = f"UTC time,Local time,LocationID,NO2 (PPM),{_RANGER_STATUS}"

    path = tmp_path / "Ranger_multi.csv"
    _write_ranger(
        path,
        [
            (
                pm_header,
                ["2026-05-19 08:42:00,2026-05-19 08:42:00,02,0.2,0.8,1.6,3.1,4.8,"],
            ),
            (
                cl2_header,
                [
                    "2026-06-01 10:53:00,2026-06-01 10:53:00,10,0.12,",
                    "2026-06-01 10:54:00,2026-06-01 10:54:00,10,0.11,",
                ],
            ),
            (no2_header, ["2026-06-02 09:00:00,2026-06-02 09:00:00,10,0.05,"]),
            # A second Cl₂ block must merge into the first Cl₂ object.
            (cl2_header, ["2026-06-03 08:00:00,2026-06-03 08:00:00,10,0.20,"]),
        ],
    )

    result = load_ranger_file(str(path))
    assert isinstance(result, list) and len(result) == 3

    by_measure = {o.metadata["measurement"]: o for o in result}
    assert set(by_measure) == {"PM", "Cl₂", "NO₂"}

    # PM head -> plain multi-channel Aerosol1D with one column per fraction.
    pm = by_measure["PM"]
    assert isinstance(pm, Aerosol1D) and not isinstance(pm, Gas1D)
    for col in ("PM1", "PM2.5", "PMrsp", "PM10", "TSP"):
        assert col in pm.data.columns
    assert pm.metadata["unit"]["PM10"] == "µg/m³"

    # Gas heads -> Gas1D; the two Cl₂ blocks are concatenated (2 + 1 rows).
    cl2 = by_measure["Cl₂"]
    assert isinstance(cl2, Gas1D)
    assert cl2.metadata["unit"] == "ppm"
    assert cl2.concentration.notna().sum() == 3

    assert by_measure["NO₂"].metadata["dtype"] == "NO₂"


def test_ranger_sniffer_identifies_non_chlorine_heads(tmp_path):
    """The sniffer recognizes a Ranger export even with no Cl₂ column."""
    from aerosoltools.gui.logic.loaders import identify_instrument, is_Ranger_file

    pm_header = f"UTC time,Local time,LocationID,PM10 (ug/m3),{_RANGER_STATUS}"
    path = tmp_path / "Ranger_pm.csv"
    _write_ranger(
        path, [(pm_header, ["2026-05-19 08:42:00,2026-05-19 08:42:00,02,3.1,"])]
    )

    assert is_Ranger_file(str(path)) is True
    assert identify_instrument(str(path)) == "Ranger"


def test_aethalometer_per_channel_units():
    """Aethalometer stores per-channel unit/dtype dicts (not a single scalar).

    The AerosolAlt convention is a name→value dict keyed by channel; the BC
    channels are ng/m³ mass concentrations and (when present) the Ångström
    exponent AAE is dimensionless. A scalar unit here would make
    ``summarize_activities`` iterate over the characters of the string.
    """
    import contextlib
    import io

    aeth = load_aethalometer_file(_data_path("Sample_Aethalometer.csv"))
    unit, dtype = aeth.metadata["unit"], aeth.metadata["dtype"]
    assert isinstance(unit, dict) and isinstance(dtype, dict)
    channels = [c for c in aeth.data.columns if c != "All data"]
    assert channels, "expected at least the spectral BC channels"
    for ch in channels:
        if "AAE" in ch:
            assert unit[ch] == "" and dtype[ch] == "AAE"
        else:
            assert unit[ch] == "ng/m³" and dtype[ch] == "dM"

    # With a proper unit dict the per-channel summary runs (it prints a table).
    with contextlib.redirect_stdout(io.StringIO()):
        aeth.summarize_activities()


def _scaled_copy(obj, factor):
    """A deep copy of ``obj`` with every numeric channel multiplied by ``factor``.

    Because the copy shares the original timestamps, ``match='exact'`` aligns
    every sample and the fitted gain recovers ``factor`` (through the origin).
    """
    import pandas as _pd

    ref = obj.copy_self()
    for c in ref._data.columns:
        if _pd.api.types.is_numeric_dtype(ref._data[c]):
            ref._data[c] = ref._data[c] * factor
    return ref


def test_calibrate_against_reference_total():
    """High-level total calibration recovers the gain and does not mutate target."""
    import aerosoltools as at

    target = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    reference = _scaled_copy(target, 2.0)
    original = target.total_concentration.copy()

    calibrated, model = at.calibrate_against_reference(target, reference)

    assert model.basis == "total"
    assert model.gains[0] == pytest.approx(2.0, rel=1e-3)
    assert model.r_squared[0] == pytest.approx(1.0, abs=1e-3)
    # inplace=False leaves the target untouched and returns a new object.
    assert calibrated is not target
    assert target.total_concentration.equals(original)
    # The calibrated total now matches the reference (2x the original).
    ref_total = reference.total_concentration.dropna()
    cal_total = calibrated.total_concentration.dropna()
    assert cal_total.reindex(ref_total.index).sub(ref_total).abs().max() < 1e-6
    # Metadata records the applied model.
    assert calibrated._meta["calibration"]["basis"] == "total"


def test_calibrate_inplace_mutates():
    """inplace=True corrects the target itself."""
    import aerosoltools as at

    target = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    reference = _scaled_copy(target, 3.0)
    out, model = at.calibrate_against_reference(target, reference, inplace=True)
    assert out is target
    assert model.gains[0] == pytest.approx(3.0, rel=1e-3)


def test_calibration_model_roundtrip():
    """CalibrationModel serializes and restores identically."""
    from aerosoltools.intercomparison import CalibrationModel, fit_calibration

    target = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    reference = _scaled_copy(target, 2.0)
    model = fit_calibration(target, reference, include_offset=True)
    restored = CalibrationModel.from_dict(model.to_dict())
    assert restored.to_dict() == model.to_dict()
    assert restored.gains == model.gains


def test_calibration_per_bin():
    """Per-bin calibration fits an independent gain per size bin."""
    from aerosoltools.intercomparison import calibrate_against_reference

    target = load_ops_file(_data_path("Sample_OPS.csv"))
    reference = _scaled_copy(target, 1.5)
    calibrated, model = calibrate_against_reference(target, reference, basis="per_bin")
    assert model.basis == "per_bin"
    assert len(model.gains) == len(target.size_data.columns)
    applied_gains = [g for g, ok in zip(model.gains, model.applied) if ok]
    assert applied_gains, "expected at least one bin to be calibrated"
    for g in applied_gains:
        assert g == pytest.approx(1.5, rel=1e-2)


def test_calibration_clips_negative():
    """clip_negative keeps calibrated concentrations >= 0 (total and per-bin)."""
    from aerosoltools.intercomparison import CalibrationModel

    cpc = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    total_model = CalibrationModel(
        basis="total",
        gains=[1.0],
        offsets=[-1e12],
        r_squared=[1.0],
        n_obs=[10],
        applied=[True],
        include_offset=True,
    )
    out = cpc.apply_calibration(total_model)
    assert (out.total_concentration.dropna() >= 0).all()

    ops = load_ops_file(_data_path("Sample_OPS.csv"))
    n = len(ops.size_data.columns)
    bin_model = CalibrationModel(
        basis="per_bin",
        gains=[1.0] * n,
        offsets=[-1e12] * n,
        r_squared=[1.0] * n,
        n_obs=[10] * n,
        applied=[True] * n,
        include_offset=True,
    )
    out2 = ops.apply_calibration(bin_model)
    assert (out2.size_data.to_numpy() >= 0).all()
    assert (out2.total_concentration.dropna() >= 0).all()


def test_calibration_rejects_incompatible():
    """Domain errors for unit / basis / bin-count mismatches."""
    from aerosoltools.intercomparison import CalibrationError, fit_calibration

    cpc = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    ops = load_ops_file(_data_path("Sample_OPS.csv"))

    # Per-bin needs both datasets size-resolved.
    with pytest.raises(CalibrationError):
        fit_calibration(cpc, _scaled_copy(cpc, 2.0), basis="per_bin")

    # Unknown basis.
    with pytest.raises(CalibrationError):
        fit_calibration(cpc, _scaled_copy(cpc, 2.0), basis="nonsense")

    # A CPC (number) and an OPS (size-resolved) do not share a unit contract for
    # a per-bin calibration — mismatched bin counts are rejected too.
    with pytest.raises(CalibrationError):
        fit_calibration(ops, cpc, basis="per_bin")


def test_psd_fit_and_evaluator_single_sourced():
    """fit_psd + the public lognormal_modes are the one model the GUI also uses."""
    import numpy as np

    import aerosoltools as at

    # Importing the GUI tabs package pulls in gui.qt, which selects the QtAgg
    # matplotlib backend -- so this assertion needs the optional gui extra, even
    # though everything it checks about the core is importable without it.
    pytest.importorskip("PyQt5")
    from aerosoltools.gui.tabs import _psdfit

    # The GUI overlay evaluator is literally the API function (one implementation).
    assert _psdfit.lognormal_modes is at.lognormal_modes

    ops = load_ops_file(_data_path("Sample_OPS.csv"))
    period = (ops.time[0], ops.time[-1])
    res = ops.fit_psd(period=period, mu=[500], sigma=[2.0], factor=[1000.0])

    # fit_psd returns a typed PSDFitResult that is still a plain 2-tuple, so the
    # historical `modes, err = fit_psd(...)` unpacking keeps working.
    assert isinstance(res, at.PSDFitResult) and isinstance(res, tuple)
    modes, err = res
    assert modes is res[0] is res.modes and err is res[1] is res.errors
    assert res.n_modes == len(modes["mu"])

    # result.evaluate reproduces the standalone lognormal_modes on the same modes.
    total, per_mode = res.evaluate(ops.bin_mids)
    total2, _ = at.lognormal_modes(ops.bin_mids, res.triples())
    assert np.allclose(total, total2)
    assert len(total) == len(ops.bin_mids)
    assert np.all(np.isfinite(total))
    assert len(per_mode) == len(modes["mu"])


def test_calibration_gui_and_script_agree():
    """The GUI applies the very same model the scripting API produces."""
    import aerosoltools as at
    from aerosoltools.intercomparison import fit_calibration

    target = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    reference = _scaled_copy(target, 2.0)

    model = fit_calibration(target, reference)
    via_method = target.apply_calibration(model)
    via_highlevel, _ = at.calibrate_against_reference(target, reference)

    a = via_method.total_concentration.dropna()
    b = via_highlevel.total_concentration.dropna()
    assert a.sub(b).abs().max() < 1e-9


def test_summary_cache_entry_typed_and_serializable():
    """The Summary-tab cache is a typed SummaryCacheEntry that round-trips JSON."""
    import json

    from aerosoltools.gui.state.projectio import (
        _restore_summary_state,
        _summary_state_payload,
    )
    from aerosoltools.gui.state.summary_cache import SummaryCacheEntry

    e = SummaryCacheEntry(
        signature={"k": 1},
        params={"p": "v"},
        columns=["A", "B"],
        records=[[1, 2], [3, 4]],
    )
    # to_dict/from_dict round-trip and table rebuild.
    assert SummaryCacheEntry.from_dict(e.to_dict()).to_dict() == e.to_dict()
    df = e.dataframe()
    assert list(df.columns) == ["A", "B"] and df.shape == (2, 2)
    # Staleness compares the stored signature to the current inputs.
    assert e.is_stale({"k": 2}) and not e.is_stale({"k": 1})

    # projectio serializes to JSON-safe primitives and restores typed entries.
    class _Proj:
        summary_state = {
            "active_kind": "Exposure summary",
            "cache": {"Exposure summary": e},
        }

    payload = _summary_state_payload(_Proj())
    json.dumps(payload)  # must not raise
    restored = _restore_summary_state(payload)
    got = restored["cache"]["Exposure summary"]
    assert isinstance(got, SummaryCacheEntry)
    assert got.dataframe().shape == (2, 2)


def test_psd_fit_spec_typed_and_serializable():
    """Stored PSD fits are typed PsdFitSpec objects that round-trip JSON."""
    import json

    from aerosoltools.gui.state.fit_specs import PsdFitSpec
    from aerosoltools.gui.state.projectio import _dump_psd_fits, _load_psd_fits

    spec = PsdFitSpec(
        modes=[{"mu": 80.0, "sigma": 1.6, "peak": 1.2e4, "bound": True}],
        optimized=True,
    )
    # to_dict/from_dict round-trip (modes coerced to plain floats/bools).
    payload = spec.to_dict()
    json.dumps(payload)  # must not raise
    back = PsdFitSpec.from_dict(payload)
    assert back.to_dict() == payload
    assert back.modes[0]["mu"] == 80.0 and back.optimized is True

    # projectio dumps the {activity: PsdFitSpec} map and reloads it typed,
    # dropping fits that have no modes.
    fits = {"Task 1": spec, "Empty": PsdFitSpec(modes=[], optimized=False)}
    dumped = _dump_psd_fits(fits)
    assert set(dumped) == {"Task 1"}
    reloaded = _load_psd_fits(dumped)
    assert isinstance(reloaded["Task 1"], PsdFitSpec)
    assert reloaded["Task 1"].modes[0]["sigma"] == 1.6


def test_decay_fit_spec_typed_and_serializable():
    """Stored decay fits round-trip live-dict -> JSON -> live-dict via DecayFitSpec."""
    import json

    import pandas as pd

    from aerosoltools.gui.state.fit_specs import DecayFitSpec
    from aerosoltools.gui.state.projectio import _dump_decay_fits, _load_decay_fits

    t0 = pd.Timestamp("2024-01-01 10:00:00")
    live = {
        "window": (pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 10:30")),
        "metric": "PNC",
        "model": "auto",
        "optimized": False,
        "per_bin": True,
        "overrides": {
            "t0": t0,
            "peak_time": None,
            "background": 100.0,
            "peakval": None,
            "rate": 0.5,
        },
        "_result": {"model": "single"},  # transient, must not be serialized
    }
    dumped = _dump_decay_fits([live])
    json.dumps(dumped)  # must not raise
    assert "_result" not in dumped[0]
    assert dumped[0]["overrides"]["t0"] == t0.isoformat()

    restored = _load_decay_fits(dumped)[0]
    # Timestamps parse back, floats/None preserved, transient key stays gone.
    assert restored["window"][0] == live["window"][0]
    assert restored["overrides"]["t0"] == t0
    assert restored["overrides"]["background"] == 100.0
    assert restored["overrides"]["peakval"] is None
    assert restored["per_bin"] is True and restored["optimized"] is False
    assert "_result" not in restored

    # A spec with no usable window is dropped on both directions.
    assert _dump_decay_fits([{"window": None}]) == []
    assert DecayFitSpec.from_dict({"window": ["x"]}) is None


def test_project_manifest_migration():
    """The manifest carries a schema version; load migrates old and rejects newer."""
    import pytest

    from aerosoltools.gui.state import projectio
    from aerosoltools.gui.state.projectio import VERSION, _migrate_manifest

    # A pre-versioning manifest (no 'version') is treated as v1 and upgraded.
    upgraded = _migrate_manifest({"format": "aerosoltools-project"})
    assert upgraded["version"] == VERSION

    # An explicit v1 manifest reaches the current version via the migration chain.
    assert _migrate_manifest({"version": 1})["version"] == VERSION

    # A manifest already at the current version is unchanged.
    assert _migrate_manifest({"version": VERSION})["version"] == VERSION

    # A manifest from a newer, forward-incompatible build is rejected clearly.
    with pytest.raises(ValueError, match="newer version"):
        _migrate_manifest({"version": VERSION + 1})

    # Every migration step advances the version by exactly one.
    for src, step in projectio._MIGRATIONS.items():
        assert step({"version": src}).get("version", src) in (src, src + 1)


def test_core_mixin_protocols_are_typing_only():
    """The _core host Protocols document the contract but never change runtime MRO."""
    from aerosoltools._core import _protocols
    from aerosoltools._core.corrections import CorrectionMixin
    from aerosoltools._core.statistics import SummaryMixin

    assert hasattr(_protocols, "AerosolData")
    assert hasattr(_protocols, "SizeResolvedData")
    # TYPE_CHECKING-guarded: at runtime the mixins still subclass only ``object``,
    # so composing them onto the facades is unaffected.
    assert SummaryMixin.__bases__ == (object,)
    assert CorrectionMixin.__bases__ == (object,)
    assert _protocols.AerosolData not in SummaryMixin.__mro__
    assert _protocols.SizeResolvedData not in CorrectionMixin.__mro__


def test_loader_exception_hierarchy():
    """Specific loader errors are LoaderError and stay ValueError-compatible."""
    from aerosoltools.loaders import (
        DelimiterDetectionError,
        EmptyFileError,
        FileFormatError,
        InstrumentDetectionError,
        LoaderError,
        TimestampError,
    )

    for exc in (
        FileFormatError,
        EmptyFileError,
        TimestampError,
        DelimiterDetectionError,
        InstrumentDetectionError,
    ):
        assert issubclass(exc, LoaderError)
        # Back-compat: existing `except ValueError` call sites still catch them.
        assert issubclass(exc, ValueError)


def test_loader_support_toolkit():
    """The shared read/construct helpers behave as the loaders expect."""
    import numpy as np
    import pandas as pd

    from aerosoltools import Aerosol2D
    from aerosoltools.loaders.support.construct import build_2d
    from aerosoltools.loaders.support.reading import read_table, sniff

    # sniff returns provided values verbatim, else detects.
    assert sniff("x", "utf-8", ",") == ("utf-8", ",")
    enc, delim = sniff(_data_path("Sample_OPS2.txt"))
    assert delim == ","

    # read_table is pd.read_csv with detection folded in.
    got = read_table(
        _data_path("Sample_OPS2.txt"), header=13, encoding=enc, delimiter=delim
    )
    exp = pd.read_csv(
        _data_path("Sample_OPS2.txt"), header=13, encoding=enc, delimiter=delim
    )
    assert got.equals(exp)

    # build_2d constructs, sets the canonical metadata, and (default) converts to
    # number + unnormalizes — same as the manual loader tail.
    mids = np.array([100.0, 200.0, 400.0])
    edges = np.array([70.0, 140.0, 280.0, 560.0])
    df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"]),
            "Total_conc": [3.0, 6.0],
            "100.0": [1.0, 2.0],
            "200.0": [1.0, 2.0],
            "400.0": [1.0, 2.0],
        }
    )
    obj = build_2d(
        df.copy(),
        bin_edges=edges,
        bin_mids=mids,
        instrument="TEST",
        serial_number="SN1",
        unit="cm⁻³",
        dtype="dN",
        density=1.2,
    )
    assert isinstance(obj, Aerosol2D)
    assert obj.metadata["instrument"] == "TEST"
    assert obj.metadata["serial_number"] == "SN1"
    assert float(obj.density) == 1.2
    assert list(obj.bin_mids) == [100.0, 200.0, 400.0]
    # A different concrete class can be requested.
    from aerosoltools import ELPI

    elpi = build_2d(
        df.copy(),
        bin_edges=edges,
        bin_mids=mids,
        instrument="ELPI",
        serial_number="SN2",
        unit="cm⁻³",
        dtype="dN",
        cls=ELPI,
    )
    assert isinstance(elpi, ELPI)


def test_public_load_file_autodetects():
    """at.load_file auto-detects the instrument and matches the explicit loader."""
    import aerosoltools as at
    from aerosoltools.loaders import InstrumentDetectionError

    auto = at.load_file(_data_path("Sample_OPS.csv"))
    explicit = load_ops_file(_data_path("Sample_OPS.csv"))
    assert type(auto) is type(explicit)
    assert auto.data.shape == explicit.data.shape

    # Explicit instrument name routes to that loader.
    assert type(at.load_file(_data_path("Sample_CPC_Direct.txt"), instrument="CPC"))

    # detect_instrument returns the canonical registry key (or None).
    assert at.detect_instrument(_data_path("Sample_OPS.csv")) == "OPS"
    assert at.detect_instrument(_data_path("Sample_Ranger.csv")) == "Ranger"

    # An unknown instrument name is a domain error, not a KeyError.
    with pytest.raises(InstrumentDetectionError):
        at.load_file(_data_path("Sample_OPS.csv"), instrument="NoSuchDevice")


def test_detect_delimiter_encoding_detection(tmp_path):
    """The sniffer decodes UTF-8/BOM files correctly, not as Latin-1 mojibake."""
    from aerosoltools.loaders.support.parsing import _detect_delimiter

    # A genuine UTF-8 file with multi-byte glyphs (µg/m³). Previously the sniffer
    # tried Latin-1 first (which decodes any bytes), so these were mis-decoded.
    utf8 = tmp_path / "utf8.csv"
    utf8.write_text("Conc (µg/m³),Temp\n1.0,20\n2.0,21\n3.0,22\n", encoding="utf-8")
    enc, delim = _detect_delimiter(str(utf8))
    assert delim == ","
    with open(str(utf8), encoding=enc) as fh:
        assert "µg/m³" in fh.read()  # round-trips, not "Âµg/mÂ³"

    # A UTF-8 BOM is honoured (utf-8-sig strips it).
    bom = tmp_path / "bom.csv"
    bom.write_text("A;B;C\n1;2;3\n4;5;6\n7;8;9\n", encoding="utf-8-sig")
    enc_bom, delim_bom = _detect_delimiter(str(bom))
    assert delim_bom == ";"
    with open(str(bom), encoding=enc_bom) as fh:
        assert fh.readline().startswith("A;B;C")  # no leading BOM char

    # A Latin-1 file with a high byte (µ = 0xB5) still decodes to the right glyph
    # via the cp1252/latin-1 fallback.
    latin = tmp_path / "latin.csv"
    latin.write_bytes("x,\xb5m\n1,2\n3,4\n5,6\n".encode("latin-1"))
    enc_l, _ = _detect_delimiter(str(latin))
    with open(str(latin), encoding=enc_l) as fh:
        assert "\xb5m" in fh.read()


def test_detect_delimiter_bounded_tail_read(tmp_path):
    """A large file is sniffed from a bounded slice, not read whole."""
    from aerosoltools.loaders.support import parsing
    from aerosoltools.loaders.support.parsing import _detect_delimiter

    # Build a file well over the 2x sniff-bytes threshold, tab-delimited data
    # rows with a differently-delimited header block up top.
    big = tmp_path / "big.tsv"
    rows = "\n".join(f"{i}\t{i * 2}\t{i * 3}\tµ" for i in range(200_000))
    big.write_text(
        "# header,with,commas\nName\tA\tB\tC\n" + rows + "\n", encoding="utf-8"
    )
    assert big.stat().st_size > 2 * parsing._SNIFF_BYTES

    enc, delim = _detect_delimiter(str(big))
    assert delim == "\t"  # sampled from the tab-delimited data tail
    assert enc.lower() in parsing._TAIL_SAFE

    # The bounded path must agree with a full read (force it via a huge budget).
    orig = parsing._SNIFF_BYTES
    try:
        parsing._SNIFF_BYTES = 10**9  # forces the whole-file branch
        enc_full, delim_full = _detect_delimiter(str(big))
    finally:
        parsing._SNIFF_BYTES = orig
    assert (enc_full, delim_full) == (enc, delim)


def test_discmini_serial_normalization():
    """The serial normalizer strips the inconsistent 'SN' prefix."""
    assert _normalize_serial("SN101923") == "101923"
    assert _normalize_serial("101670") == "101670"
    assert _normalize_serial("sn 109172") == "109172"
    assert _normalize_serial(None) is None


def test_discmini_serial_from_processed_and_raw_headers():
    """Serial + firmware parse from both processed and raw header variants."""
    processed = [
        "[testo DiSCmini java tool version 2,1 output file]",
        "[Data recorded with testo DiSCmini SN101923 running firmware 3,42]",
    ]
    serial, fw = _extract_serial_and_firmware(processed)
    assert serial == "101923" and fw == "3.42"

    # Raw header where the serial has no "SN" prefix must still match.
    raw = [
        "nw PERSONAL AEROSOL MONITOR Data written with SW-Ver 3.42",
        "CalData: 101670      6.02   25.98   -2.24    0.45    1.1530348.26    0.96",
    ]
    serial, fw = _extract_serial_and_firmware(raw)
    assert serial == "101670" and fw == "3.42"


def _write_synthetic_raw(path):
    """Write a minimal but realistic raw DiSCmini file for loader testing."""
    lines = [
        "nw PERSONAL AEROSOL MONITOR Data written with SW-Ver 3.42",
        "Filename: TEST.TXT",
        "Averaging Period: 1 sec",
        "Date and Time: 2024.01.01 12:00:00",
        "CalData: SN123456    0.28   30.73   -6.45    1.28    1.1319808.76    0.68",
        " NaCl test",
        "    0.28\t   30.73\t   -6.45\t    1.28\t    1.13\t19808.76\t    0.68\t",
        "Offsets:    -0.75\t   -0.69\t",
        "Sampled:   1000 pC\tC:     10\tW:      1",
        "Time\tDiffusion\tFilter\tTemp\tIdiff\tUcor\tFlow\tBatt\tStatus",
    ]
    # 20 valid measuring rows (Status 8B) with constant currents.
    for t in range(20):
        lines.append(f"{t}\t10.00\t35.00\t27.6\t9.82\t4.19\t1.00\t7.90\t8B")
    # 5 idle rows (Status 88, near-zero currents) that must be excluded.
    for t in range(20, 25):
        lines.append(f"{t}\t0.10\t0.10\t27.6\t0.00\t0.37\t0.37\t7.66\t88")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def test_discmini_raw_loader_pipeline(tmp_path):
    """Raw loader filters idle rows, averages, and reproduces LDSA exactly."""
    raw = tmp_path / "TEST.TXT"
    _write_synthetic_raw(str(raw))
    # Disable the zero-offset correction here so the exact cal-formula values
    # below are checked without the (separately tested) offset subtraction.
    dm = load_discmini_raw_file(
        str(raw), extra_data=True, period=10, zero_offset_correction=False
    )

    # 20 valid rows / period 10 = 2 output rows (idle rows excluded).
    assert dm.data.shape[0] == 2
    assert dm.serial_number == "123456"  # "SN" stripped
    assert dm._meta.get("firmware") == "3.42"
    assert dm._meta.get("size_inversion") == "cubic_ratio_polynomial"

    # LDSA = cal[6] * (Diff + Filter) = 0.68 * (10 + 35) = 30.6 (exact here).
    assert dm.data["LDSA"].iloc[0] == pytest.approx(30.6, abs=1e-6)
    # Size = cal0 + cal1*R + cal2*R^2 + cal3*R^3 with R = 35/10 = 3.5.
    assert dm.data["Size"].iloc[0] == pytest.approx(83.7, abs=0.1)
    # Idle rows would have dragged the currents toward zero if not excluded.
    assert dm.extra_data["Diffusion"].iloc[0] == pytest.approx(10.0, abs=1e-6)
    assert dm.total_concentration.iloc[0] > 0


def test_discmini_raw_window_gap_handling(tmp_path):
    """Windows with < 8 valid rows are dropped; real time is preserved across gaps."""
    import numpy as np

    header = [
        "nw PERSONAL AEROSOL MONITOR Data written with SW-Ver 3.42",
        "Filename: TEST.TXT",
        "Averaging Period: 1 sec",
        "Date and Time: 2024.01.01 00:00:00",
        "CalData: SN1    0.28   30.73   -6.45    1.28    1.1319808.76    0.68",
        " NaCl",
        "    0.28\t30.73\t-6.45\t1.28\t1.13\t19808.76\t0.68\t",
        "Offsets:  -0.75\t-0.69\t",
        "Sampled: 1 pC\tC: 1\tW: 1",
        "Time\tDiffusion\tFilter\tTemp\tIdiff\tUcor\tFlow\tBatt\tStatus",
    ]
    rows = []
    for t in range(40):
        # Window 2 (t=20..29): only t=20..24 valid (5 rows) -> below the >=8
        # threshold, so it must be dropped. All other windows are full.
        idle = 20 <= t <= 29 and t >= 25
        status = "88" if idle else "8B"
        rows.append(f"{t}\t10.00\t35.00\t27.6\t9.82\t4.19\t1.00\t7.90\t{status}")
    raw = tmp_path / "GAP.TXT"
    with open(raw, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header + rows) + "\n")

    dm = load_discmini_raw_file(str(raw), period=10)
    # Windows 0, 1, 3 kept (window 2 has only 5 valid rows -> dropped).
    assert dm.data.shape[0] == 3
    elapsed = (dm.time - dm.time[0]).total_seconds().to_numpy()
    # Centres at 4.5, 14.5, 34.5 s — the 24.5 window is skipped (70 s jump).
    assert np.allclose(elapsed, [0.0, 10.0, 30.0])


def _data_path(name):
    return os.path.join(os.path.dirname(__file__), "data", name)


def test_discmini_raw_loader_on_sample_file():
    """Load a real (truncated) raw DiSCmini file and sanity-check the output."""
    dm = load_discmini_raw_file(_data_path("Sample_Discmini_raw.txt"), extra_data=True)
    assert dm.serial_number == "101923"  # "SN101923" in the CalData line
    assert dm._meta.get("firmware") == "3.42"
    assert dm._meta.get("size_inversion") == "cubic_ratio_polynomial"
    # 300 valid rows / 10 = 30 averaged output rows.
    assert dm.data.shape[0] == 30
    for col in ("Total_conc", "Size", "LDSA"):
        assert col in dm.data.columns
        assert (dm.data[col] > 0).all()


def test_discmini_raw_matches_processed_size_and_number():
    """Raw-loader Size & Number reproduce the vendor output (< 3% median)."""
    import numpy as np

    dm = load_discmini_raw_file(
        _data_path("Sample_Discmini_raw.txt"), zero_offset_correction=False
    )
    proc = pd.read_csv(
        _data_path("Sample_Discmini_raw_output.txt"),
        sep="\t",
        skiprows=6,
        header=None,
        decimal=",",
        engine="python",
        names=["Time", "Number", "Size", "LDSA", "Filter", "Diff", "z"],
    )
    n = min(len(dm.data), len(proc))
    for mine_col, ven_col in [("Size", "Size"), ("Total_conc", "Number")]:
        mine = dm.data[mine_col].to_numpy()[:n]
        ven = pd.to_numeric(proc[ven_col], errors="coerce").to_numpy()[:n]
        rel = np.abs(mine - ven) / ven
        assert np.median(rel) < 0.03, f"{mine_col} median rel err too high"


def test_discmini_zero_offset_correction_recovers_and_applies():
    """Zero-offset correction reads the idle-period zeros and shifts the currents."""
    from aerosoltools.loaders.discmini import _parse_raw_header, _zero_offset_series

    raw = _data_path("Sample_Discmini_raw.txt")
    on = load_discmini_raw_file(raw, zero_offset_correction=True)
    off = load_discmini_raw_file(raw, zero_offset_correction=False)
    # The flag is recorded and the correction changes the computed currents.
    assert on._meta.get("zero_offset_correction") is True
    assert off._meta.get("zero_offset_correction") is False
    assert (on.data["LDSA"].to_numpy() != off.data["LDSA"].to_numpy()).any()

    # The offset series is anchored on the header Offsets even without an idle
    # period, and grows a new anchor per hourly idle run when present.
    with open(raw, encoding="latin-1") as fh:
        header = [next(fh) for _ in range(15)]
    meta = _parse_raw_header(header)
    times, off_d, off_f = _zero_offset_series(
        pd.DataFrame(
            {
                "Time": [0],
                "Diffusion": [0.0],
                "Filter": [0.0],
                "Idiff": [9.8],
                "Status": ["8B"],
            }
        ),
        meta["offsets"],
    )
    assert times[0] == 0.0
    assert off_d[0] == pytest.approx(meta["offsets"][0])


def test_nonparticle_classes_exported():
    """The four non-particle classes are part of the public top-level API."""
    import aerosoltools as at

    for name in ("Gas1D", "Aethalometer", "Environmental1D", "Partector"):
        assert hasattr(at, name), f"{name} not exported"
        assert name in at.__all__


def test_aethalometer_class_and_channels():
    """Aethalometer loads as its own class with per-wavelength BCc accessors."""
    from aerosoltools import Aethalometer

    aeth = load_aethalometer_file(_data_path("Sample_Aethalometer.csv"))
    assert isinstance(aeth, Aethalometer)
    # Per-wavelength direct accessors resolve to the matching BCc column.
    assert aeth.ir_bcc.name == "IR BCc"
    assert aeth.uv_bcc.name == "UV BCc"
    # The internal primary hook is the IR-equivalent channel.
    assert aeth._primary.name == "IR BCc"
    # 'total_concentration' does not apply to black carbon.
    with pytest.raises(AttributeError):
        _ = aeth.total_concentration


def test_partector_class_and_ldsa():
    """Partector loads as its own class exposing LDSA and no total concentration."""
    from aerosoltools import Partector

    par = load_partector_file(_data_path("Sample_Partector.txt"))
    assert isinstance(par, Partector)
    assert par.ldsa.name == "LDSA"
    assert par._primary.name == "LDSA"
    assert par.tem_samples is not None
    with pytest.raises(AttributeError):
        _ = par.total_concentration


def test_environmental_class_and_channels():
    """Fourtec loads as Environmental1D with temperature/RH accessors."""
    from aerosoltools import Environmental1D

    ft = load_fourtec_file(_data_path("Sample_Fourtec.xlsx"))
    assert isinstance(ft, Environmental1D)
    assert ft.temperature.name == "Temperature"
    assert ft.rh.name == "RH"
    # Temperature is the default primary channel.
    assert ft._primary.name == "Temperature"
    with pytest.raises(AttributeError):
        _ = ft.total_concentration


def test_nonparticle_summary_uses_selected_channel():
    """AltMixin summaries work on a non-particle channel without total_concentration."""
    import contextlib
    import io

    par = load_partector_file(_data_path("Sample_Partector.txt"))
    with contextlib.redirect_stdout(io.StringIO()):
        summary = par.summarize(parameter="LDSA")
    assert "Mean" in summary.columns
    assert (summary["N datapoints"] >= 0).all()


def test_discmini_ldsa_correction_scales_with_size():
    """The optional LDSA correction raises LDSA and grows with particle size."""
    import numpy as np

    from aerosoltools.loaders.discmini import _ldsa_size_factor

    # Factor is ~1 for small particles and increases with size (bounded).
    assert _ldsa_size_factor(np.array([20.0]))[0] == pytest.approx(1.0, abs=0.01)
    assert (
        _ldsa_size_factor(np.array([130.0]))[0] > _ldsa_size_factor(np.array([50.0]))[0]
    )
    # Bounded against extrapolation well beyond the fitted range.
    assert _ldsa_size_factor(np.array([1000.0]))[0] <= 1.06

    raw = _data_path("Sample_Discmini_raw.txt")
    on = load_discmini_raw_file(raw, ldsa_correction=True)
    off = load_discmini_raw_file(raw, ldsa_correction=False)
    assert on._meta.get("ldsa_correction") is True
    assert off._meta.get("ldsa_correction") is False
    # The correction is a >= 1 multiplier, so corrected LDSA is never smaller.
    assert (on.data["LDSA"].to_numpy() >= off.data["LDSA"].to_numpy() - 1e-9).all()
    assert (on.data["LDSA"].to_numpy() > off.data["LDSA"].to_numpy()).any()


def test_discmini_processed_has_no_phantom_column_and_only_numeric_series():
    """The trailing-tab 'Unnamed' column is dropped and never offered as a series."""
    from aerosoltools.gui.logic.helpers import plottable_columns

    dm = load_discmini_file(_data_path("Sample_Discmini.txt"), extra_data=True)
    # No phantom/empty column survives the loader.
    assert not any(str(c).startswith("Unnamed") for c in dm.extra_data.columns)
    # Extra currents are coerced to numbers, and the series picker offers them.
    labels = [label for label, _, _ in plottable_columns(dm)]
    assert any("Filter" in x for x in labels)

    # An all-NA column is never offered (would otherwise crash on float(NA)).
    dm.extra_data["Blank"] = pd.array([pd.NA] * len(dm.time), dtype="string")
    labels = [label for label, _, _ in plottable_columns(dm)]
    assert not any("Blank" in x for x in labels)


def test_discmini_raw_matches_processed_ldsa():
    """Raw-loader LDSA reproduces the vendor-processed LDSA (charger physics)."""
    import numpy as np

    # Compare the pure inversion (no zero-offset correction) against the vendor;
    # the offset correction is exercised separately.
    dm = load_discmini_raw_file(
        _data_path("Sample_Discmini_raw.txt"), zero_offset_correction=False
    )
    proc = pd.read_csv(
        _data_path("Sample_Discmini_raw_output.txt"),
        sep="\t",
        skiprows=6,
        header=None,
        decimal=",",
        engine="python",
        names=["Time", "Number", "Size", "LDSA", "Filter", "Diff", "z"],
    )
    proc_ldsa = pd.to_numeric(proc["LDSA"], errors="coerce").to_numpy()
    n = min(len(dm.data), len(proc_ldsa))
    mine = dm.data["LDSA"].to_numpy()[:n]
    ven = proc_ldsa[:n]
    rel = np.abs(mine - ven) / ven
    # LDSA = cal6 * total current reproduces the vendor value to < 1.5%.
    assert np.median(rel) < 0.015


def test_discmini_sniffer_distinguishes_raw_and_processed():
    """The content sniffer routes raw vs processed DiSCmini to the right loader."""
    from aerosoltools.gui.logic.loaders import (
        identify_instrument,
        is_DiSCmini_file,
        is_DiSCmini_raw_file,
    )

    raw = _data_path("Sample_Discmini_raw.txt")
    processed = _data_path("Sample_Discmini.txt")

    assert is_DiSCmini_raw_file(raw) is True
    assert is_DiSCmini_raw_file(processed) is False
    assert is_DiSCmini_file(processed) is True
    assert identify_instrument(raw) == "DiSCmini (raw)"
    assert identify_instrument(processed) == "DiSCmini"


def test_aerosolalt_removed_from_public_api():
    """The legacy AerosolAlt catch-all is gone from the package namespace."""
    import aerosoltools as at

    assert not hasattr(at, "AerosolAlt")
    assert "AerosolAlt" not in at.__all__


def test_discmini_is_own_class_with_accessors():
    """DiSCmini loads as its own particle class with .size/.ldsa accessors."""
    from aerosoltools import Aerosol1D, DiSCmini

    dm = load_discmini_file(_data_path("Sample_Discmini.txt"))
    assert isinstance(dm, DiSCmini) and isinstance(dm, Aerosol1D)
    # Still a particle instrument: total_concentration works.
    assert float(dm.total_concentration.dropna().iloc[0]) > 0
    assert dm.size.name == "Size"
    assert dm.ldsa.name == "LDSA"


def test_dusttrak_class_accessors_and_primary():
    """DustTrak exposes PM accessors and its primary is the total PM mass."""
    from aerosoltools import DustTrak

    t = pd.date_range("2024-01-01", periods=4, freq="1min")
    df = pd.DataFrame(
        {
            "Datetime": t,
            "PM1": [1.0] * 4,
            "PM2.5": [2.0] * 4,
            "PM4": [3.0] * 4,
            "PM10": [4.0] * 4,
            "Total": [5.0] * 4,
        }
    )
    d = DustTrak(df)
    # Primary / total_concentration is the "Total" channel, not the first column.
    assert float(d.total_concentration.iloc[0]) == 5.0
    assert float(d._primary.iloc[0]) == 5.0
    assert float(d.pm2_5.iloc[0]) == 2.0
    assert float(d.total.iloc[0]) == 5.0


def test_elpi_is_own_class_and_density_hook_recomputes_diameters():
    """ELPI set_density recomputes BOTH the size axis and the concentrations.

    set_density is the single public API hook: the ELPI subclass overrides
    _recompute_diameters_for_density so a density change moves the (density-
    dependent) diameters and re-attributes the per-bin number via the charger
    efficiency. The API and the GUI both go through this one method.
    """
    import numpy as np

    from aerosoltools import ELPI, Aerosol2D

    elpi = load_elpi_file(_data_path("Sample_ELPI.txt"))
    assert isinstance(elpi, ELPI) and isinstance(elpi, Aerosol2D)

    mids_before = [float(x) for x in elpi.bin_mids]
    bins_before = elpi.size_data.to_numpy(dtype=float)
    total_before = float(np.nansum(bins_before))

    elpi.set_density(2.0)

    mids_after = [float(x) for x in elpi.bin_mids]
    bins_after = elpi.size_data.to_numpy(dtype=float)

    # (1) The density-dependent diameters must move (the ELPI hook fired).
    assert mids_before != mids_after
    assert float(elpi.density) == 2.0
    # (2) The per-bin number must be re-attributed, not just relabelled.
    assert not np.allclose(bins_before, bins_after, equal_nan=True)
    # (3) Total_conc stays consistent with the (recomputed) size bins.
    assert float(np.nansum(bins_after)) != total_before
    np.testing.assert_allclose(
        elpi.total_concentration.to_numpy(dtype=float),
        np.nansum(bins_after, axis=1),
        equal_nan=True,
    )

    # The hook is a no-op for a non-ELPI 2D object: only the stored density
    # changes, the size axis does not.
    ops = load_ops_file(_data_path("Sample_OPS2.txt"))
    ops_mids = [float(x) for x in ops.bin_mids]
    ops.set_density(2.0)
    assert [float(x) for x in ops.bin_mids] == ops_mids
    assert float(ops.density) == 2.0


def test_metric_unit_registry_converts_within_dimension():
    """The unit registry classifies dimensions and converts within a dimension."""
    from aerosoltools._core.metrics import (
        canonical_unit,
        classify_unit,
        convert_value,
        unit_key,
    )

    assert classify_unit("ng/m³")[0] == "mass"
    assert classify_unit("cm⁻³")[0] == "number"
    # LaTeX and plain LDSA spellings collapse to one key.
    assert unit_key("nm$^{2}$/cm$^{3}$") == unit_key("nm²/cm³")
    # Scale conversions within a dimension.
    assert convert_value(1000.0, "ng/m³", "µg/m³") == pytest.approx(1.0)
    assert convert_value(2.0, "ppm", "ppb") == pytest.approx(2000.0)
    assert canonical_unit("ng/m³") == "µg/m³"
    # Cross-dimension conversion is a no-op (never fabricates numbers).
    assert convert_value(5.0, "cm⁻³", "µg/m³") == 5.0


def test_available_metrics_are_instrument_aware():
    """Each class advertises only the metrics it actually measures."""
    cpc = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    assert [m.key for m in cpc.available_metrics()] == ["PNC"]

    ops = load_ops_file(_data_path("Sample_OPS.csv"))
    ops_keys = [m.key for m in ops.available_metrics()]
    assert "PNC" in ops_keys and "MASS" in ops_keys and "PM2.5" in ops_keys

    dm = load_discmini_file(_data_path("Sample_Discmini.txt"))
    assert {m.key for m in dm.available_metrics()} == {"PNC", "Size", "LDSA"}
    assert set(dm._default_metrics()) == {"PNC", "LDSA"}

    par = load_partector_file(_data_path("Sample_Partector.txt"))
    par_keys = {m.key for m in par.available_metrics()}
    assert "LDSA" in par_keys and "TEM" not in par_keys  # bool flag excluded
    assert par._default_metrics() == ["LDSA"]

    aeth = load_aethalometer_file(_data_path("Sample_Aethalometer.csv"))
    assert "IR BCc" in [m.key for m in aeth.available_metrics()]
    assert aeth._default_metrics() == ["IR BCc"]


def test_pnc_metric_is_rejected_for_non_number_instruments():
    """'PNC' works on number instruments and raises on gas/BC/LDSA (the leak fix)."""
    cpc = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    ops = load_ops_file(_data_path("Sample_OPS.csv"))
    # No exception, and a number unit.
    assert cpc._get_metric_series("PNC")[1] == "cm⁻³"
    assert ops._get_metric_series("PNC")[1] == "cm⁻³"

    aeth = load_aethalometer_file(_data_path("Sample_Aethalometer.csv"))
    par = load_partector_file(_data_path("Sample_Partector.txt"))
    for obj in (aeth, par):
        with pytest.raises(ValueError):
            obj._get_metric_series("PNC")


def test_gas_metric_keyed_by_species():
    """A gas dataset advertises (and summarises) its species, not a generic key."""
    heads = load_ranger_file(_data_path("Ranger_2605-1000088-A_20260612-0603.csv"))
    gas = next(o for o in heads if type(o).__name__ == "Gas1D")
    keys = [m.key for m in gas.available_metrics()]
    assert keys == [str(gas.dtype)]  # e.g. "Cl₂"
    series, unit = gas._get_metric_series(str(gas.dtype))
    assert unit == gas.unit and len(series) == len(gas.time)


# ---------------------------------------------------------------------------
# Tiger (handheld VOC monitor) and simple ACSM
# ---------------------------------------------------------------------------


def test_tiger_loads_as_gas1d_following_the_gas1d_contract():
    """Tiger returns a Gas1D whose reading is reachable via ``.concentration``.

    Gas1D's contract is a single column named "Concentration" with the measured
    species carried in ``dtype`` -- the same shape the Ranger loader produces.
    A loader that names the column after the species instead (here "TVOC")
    still constructs, but every generic accessor built on that contract breaks,
    so assert the contract rather than just that the file parses.
    """
    from aerosoltools import Gas1D

    tiger = load_tiger_file(_data_path("Sample_Tiger.csv"))

    assert isinstance(tiger, Gas1D)
    assert "Concentration" in tiger.data.columns
    assert tiger.metadata["instrument"] == "Tiger"
    assert tiger.dtype == "TVOC"
    assert tiger.unit == "ppb"

    # The accessor and anything built on it must work.
    assert not tiger.concentration.empty
    assert pd.api.types.is_numeric_dtype(tiger.concentration)
    assert len(tiger.summarize_activities()) >= 1


def test_tiger_corrects_the_instrument_year():
    """The Tiger firmware writes a nonsense year; the loader overrides it."""
    tiger = load_tiger_file(_data_path("Sample_Tiger.csv"), year=2026)

    assert tiger.time[0].year == 2026
    assert tiger.time.is_monotonic_increasing


def test_acsm_loads_with_all_species_accessors():
    """ACSM returns an ACSM_simple exposing each chemical species."""
    from aerosoltools import ACSM_simple

    acsm = load_simple_acsm_file(_data_path("Sample_ACSM.csv"))

    assert isinstance(acsm, ACSM_simple)
    assert acsm.metadata["instrument"] == "ACSM"

    for species in ("org", "sulfate", "nitrate", "ammonia", "chlorine"):
        series = getattr(acsm, species)
        assert not series.empty, f"{species} is empty"
        assert pd.api.types.is_numeric_dtype(series), f"{species} is not numeric"


def test_acsm_is_non_particle():
    """ACSM measures mass by species, not a particle number concentration."""
    acsm = load_simple_acsm_file(_data_path("Sample_ACSM.csv"))

    with pytest.raises(AttributeError):
        _ = acsm.total_concentration


# ---------------------------------------------------------------------------
# Automatic instrument detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("Sample_ACSM.csv", "ACSM"),
        ("Sample_Tiger.csv", "Tiger"),
        ("Sample_DustTrak.csv", "DustTrak"),
        ("Sample_Aethalometer.csv", "Aethalometer"),
        ("Sample_APS_correlated.txt", "APS"),
        ("Sample_ELPI.txt", "ELPI"),
        ("Sample_NS.csv", "NanoScan (NS)"),
        ("Sample_OPCN3.txt", "OPC-N3"),
        ("Sample_OPS.csv", "OPS"),
        ("Sample_Partector.txt", "Partector"),
        ("Sample_Ranger.csv", "Ranger"),
        ("Sample_SMPS.txt", "SMPS"),
    ],
)
def test_detect_instrument_identifies_sample_files(filename, expected):
    """Content sniffing identifies each sample file."""
    from aerosoltools import detect_instrument

    assert detect_instrument(_data_path(filename)) == expected


def test_detection_of_new_instruments_does_not_depend_on_filename():
    """ACSM and Tiger are recognized from content, not from their names.

    The sample files are named by convention, which the filename-hint fallback
    would match on its own. Copy them to a neutral name so only the content
    sniffer can succeed.
    """
    import shutil
    import tempfile

    from aerosoltools import detect_instrument

    for filename, expected in [
        ("Sample_ACSM.csv", "ACSM"),
        ("Sample_Tiger.csv", "Tiger"),
    ]:
        with tempfile.TemporaryDirectory() as tmp:
            neutral = os.path.join(tmp, "unnamed_export.csv")
            shutil.copyfile(_data_path(filename), neutral)
            assert detect_instrument(neutral) == expected


def test_load_file_dispatches_new_instruments():
    """``load_file`` routes an unidentified path to the right loader."""
    from aerosoltools import ACSM_simple, Gas1D, load_file

    assert isinstance(load_file(_data_path("Sample_Tiger.csv")), Gas1D)
    assert isinstance(load_file(_data_path("Sample_ACSM.csv")), ACSM_simple)


def test_new_instruments_are_registered():
    """Both new loaders are reachable through the public registry."""
    from aerosoltools import INSTRUMENT_LOADERS

    assert INSTRUMENT_LOADERS["ACSM"] is load_simple_acsm_file
    assert INSTRUMENT_LOADERS["Tiger"] is load_tiger_file


# ---------------------------------------------------------------------------
# Unit spelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loader_func, filename", LOADER_CASES)
def test_loaders_emit_canonically_spelled_units(loader_func, filename):
    """Every emitted unit uses the µ/superscript spelling, not an ASCII variant.

    The GUI merges comparable metrics across instruments by unit string, and
    ``_UNIT_TABLE`` only recognises the spellings registered in it. Emitting
    ``"ug/m³"`` from one loader and ``"µg/m³"`` from another also shows two
    different axis labels for the same quantity, so the spelling is pinned here
    rather than left to whoever writes the next loader.
    """
    from aerosoltools._core.metrics import classify_unit

    obj = loader_func(_data_path(filename))
    if isinstance(obj, list):  # multi-component files (e.g. Ranger)
        objs = obj
    else:
        objs = [obj]

    for one in objs:
        units = []
        raw = one._meta.get("unit")
        units += list(raw.values()) if isinstance(raw, dict) else [raw]
        units += list((one._meta.get("column_units") or {}).values())

        for unit in units:
            if not isinstance(unit, str) or not unit:
                continue
            assert "ug/" not in unit, f"{filename}: ASCII micro in {unit!r}"
            assert not re.search(
                r"(?<![\d.])m3\b|cm3\b|nm2\b", unit
            ), f"{filename}: ASCII exponent in {unit!r}"
            # And it must still be a unit the registry can place, so the GUI can
            # convert it when merging.
            assert classify_unit(unit) is not None


def test_loader_sources_use_the_canonical_micro_spelling():
    """No loader *source* writes an ASCII-micro unit literal.

    The runtime test above can only check what the bundled samples exercise, and
    several loaders pick their unit from a table keyed by the file's declared
    weighting -- so a wrong ``dM``/``Ma`` entry stays invisible until someone
    loads a file exported in mass units, of which there is no sample. This
    checks the literals directly instead.
    """
    import pathlib

    loaders = pathlib.Path(__file__).resolve().parents[1] / "src/aerosoltools/loaders"
    offenders = []
    for path in sorted(loaders.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            # "ug/m3" (ASCII 3) and "µg/m3" appear as *keys* in the loaders'
            # input-normalisation tables, mapping a spelling found in a file to
            # the canonical one; those are correct, so only flag non-key uses.
            for bad in ('"ug/m³"', '"µg/m3"'):
                if bad in line and f"{bad}:" not in line:
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "non-canonical unit literals:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# OPC-N3
# ---------------------------------------------------------------------------


def test_opcn3_pm_focus_returns_pm_timeseries():
    """PM_focus=True yields a 1-D PM series instead of the size distribution."""
    from aerosoltools import Aerosol1D, Aerosol2D

    sized = load_opcn3_file(_data_path("Sample_OPCN3.txt"))
    pm = load_opcn3_file(_data_path("Sample_OPCN3.txt"), PM_focus=True)

    # The default keeps the size bins; PM_focus drops them for the PM channels.
    assert isinstance(sized, Aerosol2D)
    assert isinstance(pm, Aerosol1D) and not isinstance(pm, Aerosol2D)
    assert [c for c in pm.data.columns if c != "All data"] == [
        "Total_conc",
        "PM1",
        "PM2.5",
        "PM10",
    ]

    # Both views cover the same record and agree on the total concentration.
    assert len(pm.data) == len(sized.data)
    assert pm.data["Total_conc"].equals(sized.data["Total_conc"])

    assert pm.metadata["instrument"] == "OPCN"
    assert pm.metadata["dtype"]["PM1"] == "dM"
    assert pm.metadata["dtype"]["Total_conc"] == "dN"


def test_opcn3_pm_focus_extra_data_toggle():
    """Non-core channels follow the extra_data flag on the PM path too."""
    with_extra = load_opcn3_file(_data_path("Sample_OPCN3.txt"), PM_focus=True)
    without = load_opcn3_file(
        _data_path("Sample_OPCN3.txt"), PM_focus=True, extra_data=False
    )

    assert not with_extra.extra_data.empty
    assert without.extra_data.empty


# ---------------------------------------------------------------------------
# DustTrak DRX
# ---------------------------------------------------------------------------


def test_dusttrak_loads_all_pm_channels():
    """DustTrak DRX exposes the five PM mass fractions as accessors."""
    from aerosoltools import DustTrak

    dust = load_dusttrak_file(_data_path("Sample_DustTrak.csv"))

    assert isinstance(dust, DustTrak)
    assert dust.metadata["instrument"] == "DustTrak DRX"
    assert dust.metadata["serial_number"] == "8533151303"

    for channel in ("pm1", "pm2_5", "pm4", "pm10", "total"):
        series = getattr(dust, channel)
        assert not series.empty, f"{channel} is empty"
        assert pd.api.types.is_numeric_dtype(series), f"{channel} is not numeric"

    # Fractions are cumulative, so each cut must be >= the one below it.
    assert (dust.pm1 <= dust.pm2_5).all()
    assert (dust.pm2_5 <= dust.pm4).all()
    assert (dust.pm4 <= dust.pm10).all()


def test_dusttrak_converts_mg_to_ug():
    """The export is in mg/m3; the loader normalises to ug/m3."""
    dust = load_dusttrak_file(_data_path("Sample_DustTrak.csv"))

    # First data row of the sample reads PM1 = 0.123 mg/m3.
    assert dust.pm1.iloc[0] == pytest.approx(123.0)
    assert dust.metadata["unit"]["PM1"] == "µg/m³"


def test_dusttrak_keeps_extra_columns_by_default():
    """Non-core DRX channels are kept, like every other loader's extra_data."""
    dust = load_dusttrak_file(_data_path("Sample_DustTrak.csv"))

    # Alarms/Errors are not PM channels, so they belong in extra_data -- and
    # they must not leak into the main frame.
    assert list(dust.extra_data.columns) == ["Alarms", "Errors"]
    assert len(dust.extra_data) == len(dust.data)
    assert not {"Alarms", "Errors"} & set(dust.data.columns)

    # Opting out still works and leaves extra_data empty.
    bare = load_dusttrak_file(_data_path("Sample_DustTrak.csv"), extra_data=False)
    assert bare.extra_data.empty


def test_dusttrak_reconstructs_timestamps_from_header():
    """Rows carry elapsed seconds; absolute time comes from the header."""
    dust = load_dusttrak_file(_data_path("Sample_DustTrak.csv"))

    # Header: Test Start Date 24/06/2024, Test Start Time 09:22:57,
    # Test Interval 0:10, Number of Samples 234.
    assert len(dust.data) == 234
    assert dust.time[0] == pd.Timestamp("2024-06-24 09:23:07")
    assert dust.time.is_monotonic_increasing
    assert (dust.time[1] - dust.time[0]).total_seconds() == 10


def test_dusttrak_is_not_mistaken_for_ops():
    """A DustTrak DRX header looks like an OPS header at first glance.

    Both open with "Instrument Name" followed by a model line, and OPS is
    sniffed first, so a DustTrak file was previously routed to the OPS loader.
    The instrument name itself is the discriminator.
    """
    from aerosoltools import detect_instrument
    from aerosoltools.loaders.registry import is_DustTrak_file, is_OPS_file

    dusttrak = _data_path("Sample_DustTrak.csv")

    assert is_DustTrak_file(dusttrak)
    assert not is_OPS_file(dusttrak)
    assert detect_instrument(dusttrak) == "DustTrak"

    # ...and the OPS sniffer still accepts both real OPS header variants.
    assert is_OPS_file(_data_path("Sample_OPS.csv"))
    assert is_OPS_file(_data_path("Sample_OPS2.txt"))
    assert not is_DustTrak_file(_data_path("Sample_OPS.csv"))


def test_load_file_dispatches_dusttrak():
    """Auto-detection routes a DustTrak export to the DustTrak loader."""
    from aerosoltools import DustTrak, load_file

    assert isinstance(load_file(_data_path("Sample_DustTrak.csv")), DustTrak)


# ---------------------------------------------------------------------------
# Plotting and decay helpers used by the example notebooks
# ---------------------------------------------------------------------------


def test_plot_psd_color_and_label_distinguish_two_datasets():
    """Two datasets on one axes need a colour each to be told apart.

    plot_psd assigns colours per *activity*, so overlaying two datasets that
    both have only "All data" drew them in the same colour -- they read as one
    line doubling back. The color/label arguments override that.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aerosoltools import load_aps_file

    aps = load_aps_file(_data_path("Sample_APS_correlated.txt"))

    fig, ax = plt.subplots()
    aps.aerodynamic.plot_psd(
        ax=ax, activities=["All data"], color="tab:blue", label="aerodynamic"
    )
    aps.optical.plot_psd(
        ax=ax, activities=["All data"], color="tab:orange", label="optical"
    )

    colors = [line.get_color() for line in ax.get_lines()]
    labels = [line.get_label() for line in ax.get_lines()]

    assert colors == ["tab:blue", "tab:orange"]
    assert labels == ["aerodynamic", "optical"]
    plt.close(fig)

    # Without them, both curves still fall back to the per-activity colour.
    fig, ax = plt.subplots()
    aps.aerodynamic.plot_psd(ax=ax, activities=["All data"])
    aps.optical.plot_psd(ax=ax, activities=["All data"])
    assert len({line.get_color() for line in ax.get_lines()}) == 1
    plt.close(fig)


def test_decay_curve_reconstructs_the_fitted_model():
    """decay_curve is exported and rebuilds the curve fit_decay reported."""
    import numpy as np

    from aerosoltools import decay_curve, load_ns_file

    ns = load_ns_file(_data_path("Sample_NS.csv"))
    fit = ns.fit_decay(
        period=("2023-09-11 14:45:00", "2023-09-11 15:35:00"), metric="PNC"
    )

    curve = decay_curve(fit.model, np.array([0.0, fit.peak_time_s]), fit.model_popt)

    # At t=0 the model sits at the background; at the peak time it reaches the
    # reported peak concentration.
    assert curve[0] == pytest.approx(fit.background, rel=1e-6)
    assert curve[1] == pytest.approx(fit.peak_concentration, rel=1e-3)


def test_decay_fit_on_the_documented_sample_is_sound():
    """The NS sample is the example used in the docs; guard its fit quality."""
    from aerosoltools import load_ns_file

    ns = load_ns_file(_data_path("Sample_NS.csv"))
    fit = ns.fit_decay(
        period=("2023-09-11 14:45:00", "2023-09-11 15:35:00"),
        metric="PNC",
        volume=30.0,
    )

    assert fit.decay_r_squared > 0.99
    assert fit.peak_concentration > 3 * fit.background
    assert fit.decay_rate_per_hour > 0
    assert fit.source_strength > 0
