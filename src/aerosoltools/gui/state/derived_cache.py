"""Per-dataset cache of basis-converted copies (Qt-free).

The display tabs used to deep-copy + ``dtype_converter`` the active object on
*every* redraw. This memoises the converted view per ``(dataset, generation,
basis)`` so an unchanged redraw — or a pane switch back to a basis already
computed — is free. Correctness rests on the dataset's generation counter: a
mutation bumps ``Dataset.generation`` (via ``Dataset.touch``), which changes the
key, so a stale converted view can never be handed out. A missed bump would only
cost a recompute, never show wrong data.

The cache uses **only core object methods** (``copy_self`` / ``dtype_converter``
/ ``unit_of``), so ``state`` stays below ``logic`` in the layering.

**The returned object is cache-owned and must be treated read-only.** Callers
that mutate a converted copy (e.g. adding Pₓ columns, normalising for display)
must ``copy_self()`` it first — otherwise they corrupt the shared entry.
"""

from __future__ import annotations


class DerivedCache:
    """Memoises basis-converted copies of datasets' objects."""

    def __init__(self):
        # dataset id -> (generation, {basis: (converted_view, unit)})
        self._store: dict[int, tuple[int, dict[str, tuple]]] = {}

    def converted(self, dataset, basis: str) -> tuple:
        """Return ``(view, unit)`` for ``dataset.obj`` on ``basis``.

        ``basis == "dN"`` returns the loaded object itself (already dN — no
        copy). Any other basis returns a **cache-owned, read-only** converted
        copy; mutate a ``copy_self()`` of it, never the returned object.

        Args:
            dataset: The project ``Dataset`` (provides ``obj`` and ``generation``).
            basis: Target distribution basis (``"dN"``/``"dS"``/``"dV"``/``"dM"``).

        Returns:
            ``(view, unit)``; ``(None, "")`` when the dataset has no object.
        """
        obj = getattr(dataset, "obj", None)
        if obj is None:
            return None, ""
        if basis == "dN":
            # The loaded object is already dN; hand it back without a copy.
            return obj, obj.unit_of()

        gen = dataset.generation
        entry = self._store.get(id(dataset))
        if entry is None or entry[0] != gen:
            # New dataset or its data changed: start a fresh generation bucket.
            entry = (gen, {})
            self._store[id(dataset)] = entry
        bases = entry[1]
        if basis not in bases:
            view = obj.copy_self()
            view.dtype_converter(basis)
            bases[basis] = (view, view.unit_of())
        return bases[basis]

    def invalidate(self, dataset) -> None:
        """Drop any cached conversions for ``dataset`` (e.g. on removal)."""
        self._store.pop(id(dataset), None)

    def clear(self) -> None:
        """Drop the whole cache (e.g. on project load/close)."""
        self._store.clear()
