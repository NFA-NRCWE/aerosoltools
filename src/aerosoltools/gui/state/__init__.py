"""Project state, persistence and caches (Qt-free).

The data model behind the GUI: the :class:`~aerosoltools.gui.state.project.
Project` / ``Dataset`` container, its save/load to a self-contained folder
(:mod:`~aerosoltools.gui.state.projectio`), and the typed persisted result
caches (:mod:`~aerosoltools.gui.state.summary_cache`,
:mod:`~aerosoltools.gui.state.fit_specs`). Kept free of widget code so the state
can be reasoned about and tested independently of Qt.
"""
