"""Module discovery: walk a project tree and return dotted module names."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

from config import DEFAULT_EXCLUDE_DIRS


def discover_modules(
    project_path: Path,
    exclude_dirs: Optional[Set[str]] = None,
    exclude_files: Optional[Set[str]] = None,
) -> List[str]:
    """Walk *project_path* and return dotted module names for every .py file.

    Skips directories in *exclude_dirs* and files in *exclude_files*.
    Also skips files that are not valid Python (syntax errors).
    """
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    if exclude_files is None:
        exclude_files = {"setup.py", "conftest.py", "manage.py"}

    project_path = project_path.resolve()
    modules: List[str] = []

    for py_file in sorted(project_path.rglob("*.py")):
        rel = py_file.relative_to(project_path)

        if any(part in exclude_dirs for part in rel.parts[:-1]):
            continue

        if rel.name in exclude_files:
            continue

        if rel.name.startswith("test_") or rel.name.endswith("_test.py"):
            continue

        if rel.name == "__init__.py":
            continue

        parts = list(rel.with_suffix("").parts)
        module_name = ".".join(parts)
        modules.append(module_name)

    return modules


def find_before_after_pairs(
    modules: List[str],
    project_path: Path,
) -> List[Dict[str, str]]:
    """Find matching before/after module pairs from discovered module names.

    Case 1: project contains ``before/`` and ``after/`` subdirs so modules
            are named ``before.pkg.mod`` / ``after.pkg.mod``.
    Case 2: project IS the ``before/`` dir — look for a sibling ``after/``.
    """
    # ── Case 1: modules carry the before/after prefix ──
    before_mods: Dict[str, str] = {}
    after_mods: Dict[str, str] = {}
    for mod in modules:
        parts = mod.split(".")
        if parts[0] == "before":
            before_mods[".".join(parts[1:])] = mod
        elif parts[0] == "after":
            after_mods[".".join(parts[1:])] = mod

    pairs: List[Dict[str, str]] = []
    for common, bmod in before_mods.items():
        if common in after_mods:
            pairs.append({
                "before_mod": bmod,
                "after_mod": after_mods[common],
                "common": common,
                "import_mode": "direct",
            })
    if pairs:
        return pairs

    # ── Case 2: project_path is the before/ dir itself ──
    if project_path.name == "before":
        after_dir = project_path.parent / "after"
        if after_dir.exists():
            for mod in modules:
                after_file = after_dir / Path(*mod.split(".")).with_suffix(".py")
                if after_file.exists():
                    pairs.append({
                        "before_mod": mod,
                        "after_mod": mod,
                        "common": mod,
                        "import_mode": "importlib",
                        "before_path": str(project_path),
                        "after_path": str(after_dir),
                    })

    return pairs
