"""ICRP 66 regional lung-deposition model (GitHub #26).

The parameterisation is checked against the published behaviour of the ICRP 66
Human Respiratory Tract Model rather than against itself: the characteristic
~0.3 µm minimum in total deposition, the alveolar peak in the ultrafine range,
head-airway dominance in the coarse range, and the inhalability asymptote.
"""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import aerosoltools as at  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
UM = 1000.0  # nm per µm


@pytest.fixture(scope="module")
def ns():
    return at.load_file(os.path.join(DATA, "Sample_NS.csv"))


# -- the published curve shape -----------------------------------------------


@pytest.mark.parametrize("region", ["head", "tb", "alveolar", "total"])
def test_fractions_stay_in_range(region):
    d = np.geomspace(1.0, 200 * UM, 500)
    frac = at.icrp_deposition_fraction(d, region)
    assert np.all(np.isfinite(frac))
    assert frac.min() >= 0.0 and frac.max() <= 1.0


def test_total_deposition_has_its_minimum_near_300_nm():
    """The signature of the ICRP curve: too heavy to diffuse, too light to impact."""
    d = np.geomspace(5.0, 50 * UM, 2000)
    total = at.icrp_deposition_fraction(d, "total")
    d_min = d[int(np.argmin(total))]
    assert 200.0 < d_min < 600.0
    assert total.min() < 0.2


def test_deposition_is_high_for_both_ultrafine_and_coarse():
    assert at.icrp_deposition_fraction(10.0, "total") > 0.8
    assert at.icrp_deposition_fraction(5 * UM, "total") > 0.8
    assert at.icrp_deposition_fraction(300.0, "total") < 0.2


def test_alveolar_peaks_in_the_ultrafine_range():
    """Deep-lung deposition peaks around 20 nm, well above its 300 nm value."""
    d = np.geomspace(5.0, 10 * UM, 2000)
    alveolar = at.icrp_deposition_fraction(d, "alveolar")
    d_peak = d[int(np.argmax(alveolar))]
    assert 10.0 < d_peak < 60.0
    assert at.icrp_deposition_fraction(
        20.0, "alveolar"
    ) > 4 * at.icrp_deposition_fraction(300.0, "alveolar")


def test_head_airways_dominate_the_coarse_range():
    for d in (5 * UM, 10 * UM, 20 * UM):
        head = at.icrp_deposition_fraction(d, "head")
        assert head > at.icrp_deposition_fraction(d, "tb")
        assert head > at.icrp_deposition_fraction(d, "alveolar")


def test_regions_sum_to_the_total():
    """Two independent fits in the standard; they agree to a few percent."""
    d = np.geomspace(5.0, 50 * UM, 300)
    parts = sum(at.icrp_deposition_fraction(d, r) for r in ("head", "tb", "alveolar"))
    total = at.icrp_deposition_fraction(d, "total")
    assert np.max(np.abs(parts - total)) < 0.07


# -- inhalability ------------------------------------------------------------


def test_inhalable_fraction_matches_the_published_values():
    assert at.inhalable_fraction(1 * UM) == pytest.approx(1.0, abs=1e-3)
    assert at.inhalable_fraction(10 * UM) == pytest.approx(0.838, abs=0.01)
    assert at.inhalable_fraction(100 * UM) == pytest.approx(0.502, abs=0.01)


def test_inhalable_flag_only_matters_for_coarse_particles():
    fine = 100.0
    coarse = 50 * UM
    assert at.icrp_deposition_fraction(fine, "total", inhalable=True) == pytest.approx(
        at.icrp_deposition_fraction(fine, "total", inhalable=False), rel=1e-6
    )
    assert at.icrp_deposition_fraction(
        coarse, "total", inhalable=True
    ) < at.icrp_deposition_fraction(coarse, "total", inhalable=False)


# -- API surface -------------------------------------------------------------


def test_region_aliases_are_accepted():
    d = 100.0
    assert at.icrp_deposition_fraction(d, "et") == at.icrp_deposition_fraction(
        d, "head"
    )
    assert at.icrp_deposition_fraction(d, "pulmonary") == at.icrp_deposition_fraction(
        d, "alveolar"
    )
    assert at.icrp_deposition_fraction(d, "TB") == at.icrp_deposition_fraction(d, "tb")


def test_unknown_region_raises():
    with pytest.raises(ValueError, match="Unknown region"):
        at.icrp_deposition_fraction(100.0, "spleen")


def test_deposition_fraction_is_indexed_by_bin_midpoint(ns):
    series = ns.deposition_fraction("alveolar")
    assert len(series) == len(ns.bin_mids)
    assert np.allclose(series.index.to_numpy(), np.asarray(ns.bin_mids, dtype=float))


def test_deposited_distribution_never_exceeds_the_source(ns):
    deposited = ns.deposited_size_distribution("total")
    assert deposited.shape == ns.size_data.shape
    assert np.all(
        np.nan_to_num(deposited.to_numpy())
        <= np.nan_to_num(ns.size_data.to_numpy()) + 1e-9
    )


def test_deposition_calc_adds_a_metric_column(ns):
    work = ns.copy_self()
    work.deposition_calc("alveolar", dtype="dN")
    work.deposition_calc("total", dtype="dN")
    assert "DEPN_alveolar" in work.extra_data.columns
    assert "DEPN_total" in work.extra_data.columns
    # Deposited <= airborne, and the alveolar part <= the total.
    alveolar = work.extra_data["DEPN_alveolar"]
    total = work.extra_data["DEPN_total"]
    assert (alveolar.dropna() <= total.dropna() + 1e-9).all()
    assert (total.dropna() <= work.total_concentration.dropna() + 1e-9).all()


def test_deposited_dose_scales_with_breathing_rate(ns):
    work = ns.copy_self()
    base = work.deposited_dose("alveolar", breathing_rate=1.2)
    double = work.deposited_dose("alveolar", breathing_rate=2.4)
    assert base["dose"] > 0
    assert double["dose"] == pytest.approx(2 * base["dose"], rel=1e-9)
    assert base["hours"] > 0
    assert base["region"] == "alveolar"


def test_deposited_dose_accepts_an_activity(ns):
    work = ns.copy_self()
    work.mark_activities({"Window": [(work.time[0], work.time[len(work.time) // 2])]})
    part = work.deposited_dose("total", period="Window")
    whole = work.deposited_dose("total")
    assert 0 < part["hours"] < whole["hours"]
    assert part["n_samples"] < whole["n_samples"]


def test_plot_deposition_draws_every_region(ns):
    fig, ax = ns.plot_deposition()
    assert len(ax.get_legend().get_texts()) == 4
    assert ax.get_xscale() == "log"
    fig, ax = ns.plot_deposition(regions="alveolar", show_bins=False)
    assert len(ax.get_legend().get_texts()) == 1


def test_deposited_dose_uses_the_right_volume_basis(ns):
    """cm-3 needs the 1e6 conversion for a m3/h breathing rate; ug/m3 does not."""
    work = ns.copy_self()
    mass = work.deposited_dose("alveolar", dtype="dM", breathing_rate=1.2)
    assert mass["concentration_unit"] == "µg/m³"
    assert mass["dose_unit"] == "µg"
    assert mass["dose"] == pytest.approx(
        mass["deposited_concentration"] * 1.2 * mass["hours"], rel=1e-9
    )

    number = work.deposited_dose("alveolar", dtype="dN", breathing_rate=1.2)
    assert number["dose"] == pytest.approx(
        number["deposited_concentration"] * 1e6 * 1.2 * number["hours"], rel=1e-9
    )
