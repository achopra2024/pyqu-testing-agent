"""Entry point: ``python __main__.py``."""

import sys

from cli import main

sys.exit(0 if main() else 1)
