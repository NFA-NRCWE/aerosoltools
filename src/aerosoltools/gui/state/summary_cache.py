"""Typed cache for the Summary tab's computed tables (GUI + project persistence).

The Summary tab computes a combined exposure/activity table from the scientific
``data.summarize_*`` methods, then caches it on the project so reopening a project
shows the values (and restores the STEL/OEL/window inputs) without recomputing —
and flags them *stale* when the datasets, tasks, or settings change.

That cache used to be an undocumented ``{"signature", "params", "columns",
"records"}`` dict. :class:`SummaryCacheEntry` gives it an explicit type and owns
the two operations the tab needs (rebuild the table, test staleness), so the
scientific result (a ``DataFrame`` from the data objects) stays separate from
this GUI-owned display/persistence cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SummaryCacheEntry:
    """One computed Summary-tab table plus the inputs that produced it.

    Attributes:
        signature: Fingerprint of everything that affects the result (selected
            datasets' state, activities, scopes, and the kind's input params).
            Compared against the current inputs to detect staleness.
        params: The kind-specific input fields (exposure limits/windows, or the
            activity stats/metric selection), stored so they restore on reload.
        columns: Column names of the cached table.
        records: JSON-safe table rows (one list per row, aligned to ``columns``).
    """

    signature: dict
    params: dict
    columns: list
    records: list

    def dataframe(self) -> pd.DataFrame:
        """Rebuild the cached table as a DataFrame."""
        return pd.DataFrame.from_records(self.records, columns=self.columns)

    def is_stale(self, current_signature: dict) -> bool:
        """True when the current inputs no longer match the cached result."""
        return self.signature != current_signature

    def to_dict(self) -> dict:
        """JSON-safe dict for project persistence."""
        return {
            "signature": self.signature,
            "params": self.params,
            "columns": list(self.columns),
            "records": self.records,
        }

    @classmethod
    def from_dict(cls, data) -> "SummaryCacheEntry":
        """Rebuild an entry from :meth:`to_dict` output (tolerant of missing keys)."""
        data = data or {}
        return cls(
            signature=dict(data.get("signature") or {}),
            params=dict(data.get("params") or {}),
            columns=list(data.get("columns") or []),
            records=list(data.get("records") or []),
        )
