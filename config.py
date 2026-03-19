"""Global constants and default settings for the pipeline."""

VENV_DIR = ".pynguin_venv"
PYNGUIN_PACKAGE = "pynguin==0.38.0"
HYPOTHESIS_PACKAGE = "hypothesis==6.119.4"
DEFAULT_ALGORITHM = "DYNAMOSA"
DEFAULT_SEARCH_TIME = 120
DEFAULT_OUTPUT_DIR = "pynguin_tests"
DEFAULT_REPORT_DIR = "pynguin-report"
DEFAULT_EXPORT_STRATEGY = "PY_TEST"

DEFAULT_EXCLUDE_DIRS = {
    "__pycache__", ".git", ".svn", ".hg",
    "node_modules", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".eggs", "*.egg-info",
    "venv", ".venv", "env", ".env",
    ".pynguin_venv", ".proj",
    "pynguin_tests", "pynguin-report",
}
