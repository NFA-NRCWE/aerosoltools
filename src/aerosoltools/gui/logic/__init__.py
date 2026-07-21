"""Domain logic and workflows behind the GUI (Qt-light).

The non-widget behaviour the tabs and window drive: file loading / instrument
detection (:mod:`~aerosoltools.gui.logic.loaders`), the calibration workflow and
dialog glue (:mod:`~aerosoltools.gui.logic.calibration`), the crop/rebin/smooth
adjustments box (:mod:`~aerosoltools.gui.logic.adjustments`) and shared data
helpers (:mod:`~aerosoltools.gui.logic.helpers`). These sit between the pure
:mod:`~aerosoltools.gui.state` model and the :mod:`~aerosoltools.gui.view`
widgets.
"""
