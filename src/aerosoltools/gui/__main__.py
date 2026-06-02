"""Enable ``python -m aerosoltools.gui [path] [-i INSTRUMENT]``."""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
