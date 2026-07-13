"""AerosolAlt instrument overrides: calibration and summary statistics.

Behaviour specific to alternative-metric instruments (e.g. black-carbon
mass, optical depth or custom channels), composed into
:class:`~aerosoltools.AerosolAlt`.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import pandas as pd
from tabulate import tabulate

try:
    from typing import override  # Python 3.12+
except ImportError:  # pragma: no cover - typing_extensions fallback
    from typing_extensions import override  # noqa: F401


class AltMixin:
    """Calibration and summaries for alternative-metric instruments."""

    @override
    def calibrate(
        self, parameter: Union[int, str] = 0, m: float = 1, b: float = 0, inplace=True
    ):
        """
        Apply a correction to a chosen parameter and mark the data as calibrated
        by a linear function constructed as: x_cor = m * x + b

        Args:
            parameter (int | str, optional):
                Index or column name of the signal to plot. If ``int``, it is
                interpreted as a positional index into :attr:`data.columns`. If
                ``str``, it is treated as a column label. Defaults to ``0``.
            m : float
                The calibration value to be multiplied to the data for correction.
            b : float
                A constant offset to be removed. By default is zero and should be
                used cautionsly.

        Returns:

            AerosolAlt: A calibrated dataset. Either as a copy or as the original.
                A new dictionary has been added to the meta named 'calibrated',
                to indicate that the chosen parameter has been caibrated.

        """

        out = self if inplace else self.copy_self()

        # Resolve which column to use based on the requested parameter.
        if isinstance(parameter, int):
            if parameter >= len(self._raw_data.columns):
                raise LookupError("Chosen parameter is invalid")
            parameter = self.data.columns[parameter]
        elif isinstance(parameter, str):
            pass
        else:
            raise LookupError("Chosen parameter is invalid")

        # Apply the correction to the chosen parameter
        if isinstance(m, (float, int)):
            out._data[parameter] = self._ensure_data_robustness(
                out.data[parameter] * m + b
            )
        else:
            raise ValueError("Mismatch between m and expected type")

        if "calibrated" not in out._meta:
            out._meta["calibrated"] = {}

        if b == 0:
            out._meta["calibrated"][parameter] = m
        else:
            out._meta["calibrated"][parameter] = {"m": m, "b": b}

        return out

    def summarize(
        self,
        filename: Optional[str] = None,
        *,
        parameter: Union[int, str] = 0,
    ) -> pd.DataFrame:
        """Description:
            Summarise basic statistics of a selected scalar channel by activity.

        Args:
            filename (str | None, optional):
                Path to an Excel file to write the summary table to. If
                ``None`` (default), the summary is not saved to disk and is
                only printed and returned.
            parameter (int | str, optional):
                Index or column name of the signal to summarise. If an
                ``int`` is given, it is interpreted as a positional index into
                :attr:`data.columns`. If a ``str`` is given, it is interpreted
                as a column label. Defaults to ``0``.

        Returns:
            pandas.DataFrame:
                A summary table with one row per activity (including
                ``"All data"``), containing at least:

                ``["Segment", "Min", "Max", "Mean", "Std", "N datapoints"]``.

        Raises:
            LookupError:
                If ``parameter`` does not correspond to a valid column index or
                column label.

        Notes:
            Detailed description:
                This method extends :meth:`Aerosol1D.summarize` by allowing the
                user to choose which column in :attr:`data` is summarised. It
                uses activity flags previously defined on the object to compute
                statistics for each segment.

                Internally, the method:

                - Resolves ``parameter``:

                  - if integer, used as a positional index into
                    :attr:`data.columns`,
                  - if string, used as a column label,

                  raising :class:`LookupError` if the resolution fails or if
                  the index is out of range.

                - Iterates over all activities in :attr:`activities` (including
                  the default ``"All data"`` segment), and for each:

                  - selects rows belonging to that activity, using either:

                    - an activity mask in :attr:`data[activity]`, and
                    - the selected parameter column,

                  - computes:

                    - minimum,
                    - maximum,
                    - mean,
                    - standard deviation,
                    - number of data points.

                - Collects these into a :class:`pandas.DataFrame` and rounds
                  the numeric values to three decimal places for readability.
                - Prints a nicely formatted version of the table to the
                  console using :func:`tabulate`, labelling columns as
                  ``"Segment"``, ``"Min"``, ``"Max"``, ``"Mean"``, ``"Std"``,
                  and ``"N datapoints"``.
                - If ``filename`` is provided, writes the rounded summary to an
                  Excel file (without index) and prints the output path.

                The method then returns the rounded summary DataFrame so it can
                be used programmatically (e.g. in reports or further analysis).

        Examples:
            A typical use is to summarise different scalar channels for
            predefined activities (e.g. measurement phases, locations,
            scenarios):

            .. code-block:: python

                import aerosoltools as at

                # Load an LDSA-like time series
                alt = at.Load_Partector_file("data/Partector_log.txt")

                # Suppose activities have been marked already on 'alt'
                # Summarise LDSA by activity
                summary_ldsa = alt.summarize(parameter="LDSA")

                # Summarise Flow channel and save to Excel
                summary_flow = alt.summarize(
                    filename="partector_flow_summary.xlsx",
                    parameter="Flow",
                )
        """
        # Resolve which column to summarise based on the requested parameter.
        if isinstance(parameter, int):
            if parameter >= len(self._raw_data.columns):
                raise LookupError("Chosen parameter is invalid")
            parameter = self.data.columns[parameter]
        elif isinstance(parameter, str):
            pass
        else:
            raise LookupError("Chosen parameter is invalid")

        rows: list[list[object]] = []

        # Loop through all activities (including "All data") and collect stats.
        # ``parameter`` is already resolved to a column label above, so select
        # that column directly (no reliance on ``total_concentration``, which is
        # meaningless for non-particle Alt-based classes such as Aethalometer).
        for activity in self.activities:
            subset = self.data[self.data[activity]][parameter]

            if not subset.empty:
                rows.append(
                    [
                        activity,
                        subset.min(),
                        subset.max(),
                        subset.mean(),
                        subset.std(),
                        len(subset),
                    ]
                )

        # Build the summary table and round for readability.
        summary = pd.DataFrame(
            rows, columns=["Segment", "Min", "Max", "Mean", "Std", "N datapoints"]
        )
        summary_rounded = summary.round(3)

        # Print a nicely formatted version to the console.
        print(f"\nSummary of {parameter}:\n")
        print(
            tabulate(
                summary_rounded,  # type: ignore
                headers="keys",
                tablefmt="pretty",
                floatfmt=".3f",
            )  # type: ignore[arg-type]
        )

        # Optionally save to Excel.
        if filename:
            summary_rounded.to_excel(filename, index=False)
            print(f"\nSummary saved to: {filename}")

        return summary_rounded

    @override
    def summarize_activities(
        self,
        metrics: Optional[list[str]] = None,
        filename: Optional[str] = None,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Description:
            Summarize size-resolved aerosol metrics per activity.

        Args:
            metrics (list[str] | None): List of metric names to compute.
                If None, a default set is used reporting every value in the
                data sheet.
           filename (str | None): Optional Excel file path. If provided,
               the summary table is written to this file (one sheet,
               activities as rows). If None, no file is written.

        Returns:
            pandas.DataFrame: Summary table with:
                * "Segment"
                * "Duration (HH:MM)"
                * For each metric M: "M mean [unit]" and "M std [unit]".

        Raises:
            ValueError: If a metric name cannot be interpreted (for
                example malformed PMx string) or is unsupported.
            ValueError: If internal preparation for a Pₓ metric fails
                (for example missing PSD columns or inconsistent bin
                metadata).

        Notes:
            Detailed description:
                For each activity, the method computes the total duration
                and the requested set of metrics. Each metric is reported
                as mean and standard deviation over the activity.
                A transposed version of the table is printed to the terminal
                for quick inspection.

            Theory:
                The method combines time-weighted activity durations with
                standard PSD-derived metrics: bulk concentrations, Pₓ, and
                central size metrics (mode, median, geometric mean).
                Pₓ metrics are determined using penetration curves given in
                EN 481 / ISO 7708, mimicing cyclone cut-offs rather than
                sharp size cut-offs for PMₓ calculations.

        Examples:
            Generate a task-level summary of exposure metrics:

            .. code-block:: python

                elpi.summarize_activities(
                    metrics=["Total_conc", "PM2.5", "PM10"],
                    filename="activity_summary_OPCN3.xlsx",

                )
        """

        # --- defaults --------------------------------------------------------
        if metrics is None:
            metrics = [i for i in self._meta["unit"]]

        # --- helper: duration in minutes per time step (shared helper) -------
        dt_mins = self._dt_minutes()

        # --- compute per-activity --------------------------------------------
        rows: list[list[float | str]] = []
        # bin_mids = np.asarray(self.bin_mids, dtype=float)

        for activity in self.activities:
            mask = self.data[activity]
            if mask.sum() == 0:
                continue

            # Duration of this activity (min and HH:MM)
            duration_minutes = float(dt_mins.loc[mask].sum())
            duration_hhmm = self._format_hhmm(duration_minutes)

            # Assemble row
            row: list[float | str] = [activity, duration_hhmm]
            for name in metrics:
                value_df = self.data[name].loc[mask]
                row += [
                    round(float(value_df.mean()), 2),
                    round(float(value_df.std()), 2),
                ]

            rows.append(row)

        # --- column headers with explicit units ---------------------------------
        columns: list[str] = ["Segment", "Duration (HH:MM)"]

        for name in metrics:

            unit = self._meta["unit"][name]
            label = f"{name} [{unit}]"
            columns += [label, f"{label} std"]

        summary = pd.DataFrame(rows, columns=columns)
        # --- append to file if requested ---------------------------------------
        if filename and not summary.empty:
            fname = str(filename)
            lower = fname.lower()
            if sheet_name:
                shname = str(sheet_name)
            else:
                shname = f"{metrics} summary"

            if lower.endswith(".csv"):
                if os.path.exists(fname):
                    summary.to_csv(fname, mode="a", header=False, index=False)
                else:
                    summary.to_csv(fname, mode="w", header=True, index=False)
            elif lower.endswith((".xlsx", ".xls")):
                if os.path.exists(fname):
                    with pd.ExcelWriter(
                        filename,
                        engine="openpyxl",
                        mode="a",
                        if_sheet_exists="replace",  # requires pandas ≥ 1.4
                    ) as writer:
                        summary.to_excel(writer, sheet_name=shname, index=False)
                else:
                    summary.to_excel(fname, sheet_name=shname, index=False)
            else:
                raise ValueError(
                    f"Unsupported file extension for '{filename}'. Use .csv or .xlsx."
                )
            print(f"\nActivity summary saved to: {filename}")

        summary_t = summary.set_index("Segment").T
        print("\nSummary of aerosol properties (transposed):\n")
        print(tabulate(summary_t, headers="keys", tablefmt="pretty", floatfmt=".3f"))  # type: ignore

        return summary
