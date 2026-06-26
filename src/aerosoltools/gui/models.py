"""Qt table model for displaying pandas DataFrames in the GUI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .qt import QtCore


class PandasTableModel(QtCore.QAbstractTableModel):
    """A read-only :class:`QAbstractTableModel` backed by a pandas DataFrame.

    Cells are rendered lazily, so even large frames remain responsive because
    only the visible viewport is formatted at any time.
    """

    def __init__(self, df: pd.DataFrame | None = None, float_format: str = "{:.4g}"):
        """Wrap ``df`` (or an empty frame) and remember the float display format."""
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()
        self._float_format = float_format
        # Optional per-column header tooltips (column name -> hover text), used
        # to explain otherwise-cryptic headers such as size-bin midpoints (nm).
        self._header_tooltips: dict = {}

    def set_header_tooltips(self, tooltips: dict | None) -> None:
        """Set hover text for column headers (column name -> tooltip string)."""
        self._header_tooltips = dict(tooltips or {})

    # -- data management ---------------------------------------------------
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Replace the underlying DataFrame and refresh the view."""
        self.beginResetModel()
        self._df = df if df is not None else pd.DataFrame()
        self.endResetModel()

    @property
    def dataframe(self) -> pd.DataFrame:
        """The DataFrame currently backing the model."""
        return self._df

    # -- Qt model interface ------------------------------------------------
    def rowCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        """Return the row count (0 for any non-root parent)."""
        return 0 if parent.isValid() else len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        """Return the column count (0 for any non-root parent)."""
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):  # noqa: N802
        """Return the formatted display text / alignment for a cell."""
        if not index.isValid():
            return None
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            value = self._df.iat[index.row(), index.column()]
            return self._format(value)
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return None

    def headerData(
        self, section, orientation, role=QtCore.Qt.DisplayRole
    ):  # noqa: N802
        """Return the column name / row label (or a header tooltip) for a header."""
        if orientation == QtCore.Qt.Horizontal:
            name = str(self._df.columns[section])
            if role == QtCore.Qt.ToolTipRole:
                return self._header_tooltips.get(name)
            if role == QtCore.Qt.DisplayRole:
                return name
            return None
        if role != QtCore.Qt.DisplayRole:
            return None
        # Show timestamps (or whatever the index is) on the row header.
        label = self._df.index[section]
        if isinstance(label, pd.Timestamp):
            return label.strftime("%Y-%m-%d %H:%M:%S")
        return str(label)

    # -- helpers -----------------------------------------------------------
    def _format(self, value) -> str:
        """Format one cell value (blank for NaN/None, ISO for timestamps)."""
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return self._float_format.format(value)
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return str(value)
