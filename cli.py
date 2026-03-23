"""Command-line interface: argument parsing and main entry point."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import List, Optional

from config import (
    DEFAULT_ALGORITHM,
    DEFAULT_EXPORT_STRATEGY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_SEARCH_TIME,
    VENV_DIR,
)
from runner import PynguinRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a venv, install dependencies, and run Pynguin.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              # Auto-detect deps and test a specific module
              python run_pynguin.py --project-path ./myproj --module-name pkg.module

              # Generate tests for EVERY module in the project
              python run_pynguin.py --project-path /path/to/project --all-modules

              # Provide an explicit requirements file for the target project
              python run_pynguin.py --project-path ./myproj \\
                  --requirements ./myproj/requirements.txt --module-name core

              # Pass extra Pynguin flags after --
              python run_pynguin.py --project-path . --module-name utils \\
                  -- --maximum-test-executions 500

            Dependency Detection (when --requirements is not provided)
            ----------------------------------------------------------
              1. Looks for requirements*.txt, setup.py, pyproject.toml
                 inside --project-path
              2. Falls back to pipreqs: scans actual imports and resolves
                 them to correct PyPI package names automatically
        """),
    )

    parser.add_argument(
        "--project-path",
        default=".",
        help="Root path of the project under test (default: current directory)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--module-name",
        default="services",
        help="Dotted module name to generate tests for (default: services)",
    )
    group.add_argument(
        "--all-modules",
        action="store_true",
        help="Discover and test ALL Python modules in the project",
    )
    parser.add_argument(
        "--venv-dir",
        default=VENV_DIR,
        help=f"Directory for the virtual environment (default: {VENV_DIR})",
    )
    parser.add_argument(
        "--requirements",
        default=None,
        help="Path to a pip requirements file (default: auto-detect from project)",
    )
    parser.add_argument(
        "--algorithm",
        default=DEFAULT_ALGORITHM,
        choices=["DYNAMOSA", "MIO", "MOSA", "RANDOM", "WHOLE_SUITE"],
        help=f"Pynguin search algorithm (default: {DEFAULT_ALGORITHM})",
    )
    parser.add_argument(
        "--max-search-time",
        type=int,
        default=DEFAULT_SEARCH_TIME,
        help=f"Maximum search time in seconds (default: {DEFAULT_SEARCH_TIME})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated test files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--report-dir",
        default=DEFAULT_REPORT_DIR,
        help=f"Directory for Pynguin statistics/reports (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--export-strategy",
        default=DEFAULT_EXPORT_STRATEGY,
        choices=["PY_TEST", "UNITTEST"],
        help=f"Test export format (default: {DEFAULT_EXPORT_STRATEGY})",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a pynguin TOML config file (overrides most CLI flags)",
    )
    parser.add_argument(
        "--force-venv",
        action="store_true",
        help="Delete and recreate the virtual environment from scratch",
    )
    parser.add_argument(
        "--force-deps",
        action="store_true",
        help="Force reinstall all dependencies even if already installed",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> bool:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    runner = PynguinRunner(
        project_path=args.project_path,
        module_name=args.module_name,
        venv_dir=args.venv_dir,
        requirements_file=args.requirements,
        algorithm=args.algorithm,
        max_search_time=args.max_search_time,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        export_strategy=args.export_strategy,
        config_file=args.config,
        extra_pynguin_args=extra,
    )

    all_modules = getattr(args, "all_modules", False)

    print("=" * 60)
    print("  Pynguin Automated Runner")
    print("=" * 60)
    print(f"  Project path   : {Path(runner.project_path).resolve()}")
    if all_modules:
        print(f"  Target         : ALL modules (auto-discovery)")
    else:
        print(f"  Module         : {runner.module_name}")
    print(f"  Venv           : {runner._venv_path}")
    if runner.requirements_file:
        print(f"  Requirements   : {Path(runner.requirements_file).resolve()}")
    else:
        print(f"  Dependencies   : auto-detect (project manifests / pipreqs)")
    print(f"  Algorithm      : {runner.algorithm}")
    print(f"  Search time    : {runner.max_search_time}s per module")
    print(f"  Output dir     : {Path(runner.output_dir).resolve()}")
    if runner.config_file:
        print(f"  Config file    : {Path(runner.config_file).resolve()}")
    print("=" * 60)

    passed = runner.run_all(
        force_venv=args.force_venv,
        force_deps=args.force_deps,
        all_modules=all_modules,
    )

    print("\n" + "=" * 60)
    if passed:
        print("  TESTING PHASE: PASSED (True)")
    else:
        print("  TESTING PHASE: FAILED (False)")
    print("=" * 60)

    return passed
