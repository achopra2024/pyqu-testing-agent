"""Entry point for ``python -m pyqu``."""

import sys

from pyqu.cli import main

sys.exit(0 if main() else 1)
