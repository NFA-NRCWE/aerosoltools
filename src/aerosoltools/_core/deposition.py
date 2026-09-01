"""ICRP 66 respiratory-tract deposition (regional lung deposition fractions).

Answers "of the particles in this air, what fraction actually deposits, and
where?" — as a function of particle size, split into the three regions of the
ICRP Publication 66 Human Respiratory Tract Model:

* **Head airways** (``"head"``, extrathoracic: nose, mouth, pharynx, larynx).
  Catches the large particles by impaction and the smallest by diffusion.
* **Tracheobronchial** (``"tb"``: trachea down to the terminal bronchioles).
* **Alveolar** (``"alveolar"``, the gas-exchange region) — the deposition that
  matters most for systemic uptake and for many health endpoints.
* **Total** (``"total"``) — everything that deposits anywhere in the tract.

Implemented as the standard analytical parameterisation of ICRP 66 (ICRP
Publication 66, 1994, Annex F; reproduced as eq. 11.13–11.16 in Hinds, *Aerosol
Technology*, 2nd ed., §11.4). It describes a **reference adult, nose breathing,
at light exercise** — the reference-worker scenario the ICRP model is normally
applied with in occupational hygiene, and the one the curves in the standard are
drawn for.

Two things worth knowing before using the numbers:

* **Inhalability.** The fractions are of the particles *in the ambient air*, not
  of the particles that made it past the nose: the inhalable fraction (IF) is
  folded in, so coarse particles that are never drawn in are correctly excluded.
  Pass ``inhalable=False`` to get deposition per *inhaled* particle instead.
* **Which diameter.** ICRP 66 uses aerodynamic diameter where impaction and
  sedimentation dominate (above ~0.5 µm) and thermodynamic (diffusion-equivalent)
  diameter below. The parameterisation is a single fit across both, so it is
  applied here to whatever diameter the dataset carries. That is exact for an
  APS/ELPI (aerodynamic) in the coarse range and for an SMPS/NanoScan (mobility
  ≈ thermodynamic) in the fine range — the two regimes each instrument is used
  in anyway — but an optical diameter (OPS, Grimm) is neither, so treat those
  results as indicative.

The model is a function of size only, so it lives on the size-resolved classes
(:class:`~aerosoltools.aerosol2d.Aerosol2D` and its subclasses).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .decay import _numerator_and_cm3_factor

#: Canonical region names, and the aliases accepted for each.
_REGION_ALIASES = {
    "head": "head",
    "ha": "head",
    "et": "head",
    "extrathoracic": "head",
    "head_airways": "head",
    "tb": "tb",
    "trachea": "tb",
    "bronchial": "tb",
    "tracheobronchial": "tb",
    "al": "alveolar",
    "alveolar": "alveolar",
    "pulmonary": "alveolar",
    "a": "alveolar",
    "total": "total",
    "all": "total",
    "tot": "total",
    "respiratory": "total",
}

#: Human-readable labels, used in plots, column names and summaries.
REGION_LABELS = {
    "head": "Head airways",
    "tb": "Tracheobronchial",
    "alveolar": "Alveolar",
    "total": "Total",
}

#: Reference-worker breathing rate (m³/h) — ICRP 66 light exercise, the
#: scenario the deposition parameterisation describes. Used only to turn a
#: deposited concentration into an inhaled dose; it never changes a fraction.
REFERENCE_BREATHING_RATE_M3_H = 1.2


def _canonical_region(region: str) -> str:
    """Map a region name/alias onto its canonical key."""
    key = str(region).strip().lower().replace(" ", "_")
    try:
        return _REGION_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown region {region!r}. Use one of "
            f"{', '.join(sorted(set(_REGION_ALIASES.values())))}."
        ) from None


def inhalable_fraction(diameter_nm) -> NDArray[np.float64]:
    """Fraction of ambient particles that are drawn into the airways.

    ICRP 66's inhalability for a reference worker::

        IF = 1 - 0.5 * (1 - 1 / (1 + 0.00076 * d**2.8))

    with ``d`` the diameter in µm. It is ~1 for fine particles and falls away
    through the coarse range — about 0.84 at 10 µm, tending to 0.5 for the very
    coarse (0.50 at 100 µm).

    Args:
        diameter_nm: Particle diameter(s) in **nanometres**. Scalar or array.

    Returns:
        numpy.ndarray: Inhalable fraction, same shape as the input.
    """
    d = np.asarray(diameter_nm, dtype=float) / 1000.0  # nm -> µm
    with np.errstate(over="ignore", invalid="ignore"):
        inhalable = 1.0 - 0.5 * (1.0 - 1.0 / (1.0 + 0.00076 * d**2.8))
    return np.clip(np.nan_to_num(inhalable, nan=0.0), 0.0, 1.0)


def icrp_deposition_fraction(
    diameter_nm, region: str = "total", inhalable: bool = True
) -> NDArray[np.float64]:
    """Deposition fraction in one region of the respiratory tract, by size.

    The ICRP 66 analytical parameterisation for a reference adult, nose
    breathing, at light exercise. See the module docstring for its scope and
    for which particle diameter it expects.

    Args:
        diameter_nm: Particle diameter(s) in **nanometres**. Scalar or array.
        region: ``"head"`` (extrathoracic), ``"tb"`` (tracheobronchial),
            ``"alveolar"``, or ``"total"``. Common aliases are accepted
            (``"et"``, ``"bronchial"``, ``"pulmonary"``, …).
        inhalable: When ``True`` (default) the result is the fraction of the
            particles **in the ambient air** that deposit in the region, i.e.
            the inhalable fraction is folded in. When ``False`` it is the
            fraction of the **inhaled** particles that deposit there.

    Returns:
        numpy.ndarray: Deposition fraction in ``[0, 1]``, same shape as the
        input.

    Raises:
        ValueError: If ``region`` is not recognised.

    Notes:
        Theory:
            Three mechanisms shape the curves and give total deposition its
            characteristic minimum near 0.3 µm: **diffusion**, which dominates
            below ~0.1 µm and drives the ultrafine particles onto the airway
            walls (deep-lung deposition of a 20 nm particle is far higher than
            of a 300 nm one); **impaction**, which removes coarse particles in
            the head airways; and **sedimentation** in the slow-moving distal
            airways. Particles around 0.3 µm are too heavy to diffuse and too
            light to impact, so most of them are simply breathed out again.

    Examples:
        The classic deposition curve, and the ultrafine/coarse contrast::

            import numpy as np
            from aerosoltools import icrp_deposition_fraction

            d = np.array([10, 300, 1000, 10000])          # nm
            icrp_deposition_fraction(d, "alveolar")
            icrp_deposition_fraction(d, "total")
    """
    key = _canonical_region(region)
    d = np.asarray(diameter_nm, dtype=float) / 1000.0  # nm -> µm

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ln_d = np.log(np.where(d > 0, d, np.nan))

        if key == "head":
            frac = 1.0 / (1.0 + np.exp(6.84 + 1.183 * ln_d)) + 1.0 / (
                1.0 + np.exp(0.924 - 1.885 * ln_d)
            )
        elif key == "tb":
            frac = (0.00352 / d) * (
                np.exp(-0.234 * (ln_d + 3.40) ** 2)
                + 63.9 * np.exp(-0.819 * (ln_d - 1.61) ** 2)
            )
        elif key == "alveolar":
            frac = (0.0155 / d) * (
                np.exp(-0.416 * (ln_d + 2.84) ** 2)
                + 19.11 * np.exp(-0.482 * (ln_d - 1.362) ** 2)
            )
        else:  # total
            frac = (
                0.0587
                + 0.911 / (1.0 + np.exp(4.77 + 1.485 * ln_d))
                + 0.943 / (1.0 + np.exp(0.508 - 2.58 * ln_d))
            )

        # As published, the head-airways and total expressions are written with
        # the inhalable fraction factored out in front, while the TB and
        # alveolar ones are per inhaled particle. The expressions above are all
        # the per-inhaled-particle form, so applying IF here uniformly both
        # reproduces the published head/total curves and keeps one basis across
        # the four -- the three regions then sum to the total (to ~3 %, the
        # accuracy of the fits themselves).
        if inhalable:
            frac = frac * inhalable_fraction(diameter_nm)

    frac = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(frac, 0.0, 1.0)


class LungDepositionMixin:
    """Regional lung deposition (ICRP 66) for a size-resolved dataset."""

    def deposition_fraction(
        self, region: str = "total", inhalable: bool = True
    ) -> pd.Series:
        """Per-size-bin deposition fraction for this dataset's size axis.

        Args:
            region: ``"head"``, ``"tb"``, ``"alveolar"`` or ``"total"``.
            inhalable: Fold in the inhalable fraction (see
                :func:`icrp_deposition_fraction`).

        Returns:
            pandas.Series: Deposition fraction per bin, indexed by bin
            midpoint in nm and named after the region.

        Examples:
            .. code-block:: python

                ns = at.load_file("Sample_NS.csv")
                ns.deposition_fraction("alveolar").head()
        """
        key = _canonical_region(region)
        mids = np.asarray(self.bin_mids, dtype=float)
        values = icrp_deposition_fraction(mids, key, inhalable=inhalable)
        return pd.Series(
            values,
            index=pd.Index(mids, name="Dp, nm"),
            name=f"{REGION_LABELS[key]} deposition fraction",
        )

    def deposited_size_distribution(
        self,
        region: str = "total",
        dtype: Optional[str] = None,
        inhalable: bool = True,
    ) -> pd.DataFrame:
        """The size distribution of the particles that actually deposit.

        Each size bin of the current distribution is scaled by that bin's
        deposition fraction, so the result is a *deposited* PSD with the same
        bins, unit and time index as the source.

        Args:
            region: ``"head"``, ``"tb"``, ``"alveolar"`` or ``"total"``.
            dtype: Distribution basis to deposit (``"dN"``, ``"dS"``, ``"dV"``,
                ``"dM"``). Defaults to the dataset's current dtype.
            inhalable: Fold in the inhalable fraction.

        Returns:
            pandas.DataFrame: Deposited concentration per bin, indexed by time
            with the bin headers as columns.
        """
        work = self if dtype is None else self.dtype_converter(dtype, False)
        frac = icrp_deposition_fraction(
            np.asarray(work.bin_mids, dtype=float),
            _canonical_region(region),
            inhalable=inhalable,
        )
        data = work.size_data
        return data.multiply(frac, axis="columns")

    def deposition_calc(
        self,
        region: str = "total",
        dtype: Optional[str] = None,
        inhalable: bool = True,
    ):
        """Add a deposited-concentration time series to :attr:`extra_data`.

        The size-integrated amount that deposits in ``region``, per volume of
        **ambient air**, at every time step — the deposition analogue of
        :meth:`~aerosoltools._core.fractions.FractionMixin.PM_calc`. Multiply by
        a breathing rate and an exposure time to get a dose, or use
        :meth:`deposited_dose`, which does exactly that.

        Args:
            region: ``"head"``, ``"tb"``, ``"alveolar"`` or ``"total"``.
            dtype: Distribution basis to integrate (``"dN"``, ``"dS"``,
                ``"dV"``, ``"dM"``). Defaults to the dataset's current dtype.
            inhalable: Fold in the inhalable fraction.

        Returns:
            Aerosol2D: ``self``, with a new column in :attr:`extra_data` named
            ``"DEP<X>_<region>"`` — for example ``"DEPN_alveolar"`` for a
            deposited number concentration, ``"DEPM_total"`` for mass.

        Examples:
            Alveolar-deposited number concentration alongside the usual metrics:

            .. code-block:: python

                ns.deposition_calc("alveolar", dtype="dN")
                ns.extra_data["DEPN_alveolar"].describe()
        """
        key = _canonical_region(region)
        work = self if dtype is None else self.dtype_converter(dtype, False)
        base = str(work.dtype).replace("/dlogDp", "")
        if "/dlogDp" in str(work.dtype):
            work = work.unnormalize_logdp(inplace=False)

        deposited = work.deposited_size_distribution(
            region=key, dtype=None, inhalable=inhalable
        )
        values = np.nansum(deposited.to_numpy(dtype=float), axis=1)
        values = np.where(values == 0.0, np.nan, values)
        series = self._ensure_data_robustness(values)

        label = f"DEP{base[-1].upper()}_{key}"
        if self._extra_data.empty:
            self._extra_data = pd.DataFrame(index=self.time)
        elif not self._extra_data.index.equals(self.time):
            self._extra_data = self._extra_data.reindex(self.time)
        self._extra_data[label] = series
        self._register_derived(label)
        return self

    def deposited_dose(
        self,
        region: str = "total",
        period: Union[str, tuple, None] = None,
        dtype: Optional[str] = None,
        breathing_rate: float = REFERENCE_BREATHING_RATE_M3_H,
        inhalable: bool = True,
    ) -> dict:
        """Total amount deposited in a region over a period.

        Integrates the deposited concentration over time and multiplies by the
        breathing rate::

            dose = mean(deposited concentration) x breathing rate x duration

        Args:
            region: ``"head"``, ``"tb"``, ``"alveolar"`` or ``"total"``.
            period: Activity name, ``(start, end)`` pair, or ``None`` (default)
                for the whole record.
            dtype: Distribution basis (``"dN"``, ``"dS"``, ``"dV"``, ``"dM"``).
                Defaults to the dataset's current dtype.
            breathing_rate: Ventilation in m³/h. Defaults to
                :data:`REFERENCE_BREATHING_RATE_M3_H` (1.2 m³/h), the ICRP 66
                light-exercise reference worker the deposition curves describe.
                Note that this scales the *dose* only — the deposition
                fractions themselves are fixed to that scenario.
            inhalable: Fold in the inhalable fraction.

        Returns:
            dict: ``region``, ``dose``, ``dose_unit``, ``deposited_concentration``
            (the period mean), ``concentration_unit``, ``hours``,
            ``breathing_rate_m3_h`` and ``n_samples``.

        Raises:
            ValueError: If ``period`` is neither a known activity nor a valid
                ``(start, end)`` pair.

        Examples:
            .. code-block:: python

                ns.deposited_dose("alveolar", period="Emission")
                # {'region': 'alveolar', 'dose': 2.7e+10, 'dose_unit': 'count', ...}

                # On a mass basis the dose comes out in µg:
                ns.deposited_dose("alveolar", dtype="dM")["dose_unit"]  # 'µg'
        """
        key = _canonical_region(region)
        work = self if dtype is None else self.dtype_converter(dtype, False)
        if "/dlogDp" in str(work.dtype):
            work = work.unnormalize_logdp(inplace=False)

        deposited = work.deposited_size_distribution(
            region=key, dtype=None, inhalable=inhalable
        )
        per_step = pd.Series(
            np.nansum(deposited.to_numpy(dtype=float), axis=1), index=self.time
        )

        if period is None:
            mask = pd.Series(True, index=self.time)
        elif isinstance(period, str):
            if period not in self.activities:
                raise ValueError(f"Activity '{period}' not found.")
            mask = self.data[period].astype(bool)
        elif isinstance(period, tuple) and len(period) == 2:
            start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
            mask = (self.time >= start) & (self.time <= end)
        else:
            raise ValueError("period must be an activity name or a (start, end) tuple.")

        selected = per_step[np.asarray(mask)]
        selected = selected[np.isfinite(selected)]
        if selected.empty:
            hours, mean_conc = 0.0, float("nan")
        else:
            span = selected.index[-1] - selected.index[0]
            step = pd.Timedelta(seconds=0)
            if len(selected) > 1:
                step = pd.Timedelta(np.median(np.diff(selected.index.to_numpy())))
            hours = (span + step).total_seconds() / 3600.0
            mean_conc = float(selected.mean())

        # The breathing rate is in m3/h, so the concentration has to be on a
        # per-m3 basis. Reuse the decay module's unit splitter rather than
        # re-deriving it -- it already knows which units are per cm3.
        unit = str(work.unit)
        numerator, to_m3 = _numerator_and_cm3_factor(unit)
        dose = mean_conc * to_m3 * breathing_rate * hours

        return {
            "region": key,
            "dose": dose,
            "dose_unit": numerator,
            "deposited_concentration": mean_conc,
            "concentration_unit": unit,
            "hours": hours,
            "breathing_rate_m3_h": float(breathing_rate),
            "n_samples": int(selected.size),
        }

    def plot_deposition(
        self,
        regions: Union[str, list, tuple] = ("head", "tb", "alveolar", "total"),
        ax=None,
        inhalable: bool = True,
        show_bins: bool = True,
    ):
        """Plot the ICRP 66 deposition curves over this dataset's size range.

        Args:
            regions: Region name or sequence of them. Defaults to all four.
            ax: Existing Matplotlib axes; a new figure is made when ``None``.
            inhalable: Fold in the inhalable fraction.
            show_bins: Mark the dataset's own bin midpoints on each curve, so
                it is visible which part of the curve the instrument resolves.

        Returns:
            tuple: ``(figure, axes)``.

        Examples:
            .. code-block:: python

                fig, ax = ns.plot_deposition()
                fig.savefig("deposition.png", dpi=150)
        """
        import matplotlib.pyplot as plt

        if isinstance(regions, str):
            regions = [regions]
        keys = [_canonical_region(r) for r in regions]

        mids = np.asarray(self.bin_mids, dtype=float)
        lo, hi = float(np.min(mids)), float(np.max(mids))
        grid = np.geomspace(min(lo, 10.0), max(hi, 10000.0), 400)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        else:
            fig = ax.get_figure()

        for key in keys:
            curve = icrp_deposition_fraction(grid, key, inhalable=inhalable)
            (line,) = ax.plot(grid, curve, lw=2.0, label=REGION_LABELS[key])
            if show_bins:
                ax.plot(
                    mids,
                    icrp_deposition_fraction(mids, key, inhalable=inhalable),
                    ls="None",
                    marker="o",
                    ms=3.5,
                    color=line.get_color(),
                    alpha=0.7,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Dp, nm")
        basis = "ambient" if inhalable else "inhaled"
        ax.set_ylabel("Deposition fraction\n(of {} particles)".format(basis))
        ax.set_ylim(0, 1)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, title="ICRP 66 region", title_fontsize=8)
        ax.set_title(
            "Lung deposition — ICRP 66 reference worker, nose breathing",
            fontsize=10,
        )
        return fig, ax
