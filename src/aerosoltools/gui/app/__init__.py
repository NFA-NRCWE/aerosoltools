"""Application shell: the main window and the dataset sidebar.

These modules own the top-level GUI lifecycle — the :class:`~aerosoltools.gui.
app.main_window.MainWindow` that wires everything together and the sidebar that
lists the loaded datasets. Everything else (tabs, state, view widgets, domain
logic) is orchestrated from here.
"""
