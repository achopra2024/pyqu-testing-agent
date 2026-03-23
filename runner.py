"""PynguinRunner: orchestrates venv, deps, Pynguin, post-processing, and pytest."""

from __future__ import annotations

import ast
import glob
import os
import re
import shutil
import tempfile
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    DEFAULT_ALGORITHM,
    DEFAULT_EXPORT_STRATEGY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_SEARCH_TIME,
    HYPOTHESIS_PACKAGE,
    PYNGUIN_PACKAGE,
    VENV_DIR,
)
from discovery import discover_modules, find_before_after_pairs
from helpers import (
    find_dependency_sources,
    is_windows,
    run,
    venv_executable,
    venv_pip,
    venv_python,
)
from comparator import run_all_comparisons
from hypothesis_gen import analyze_params_for_hypothesis, generate_hypothesis_file
from preprocessor import (
    STUB_PREFIX,
    clear_stub_defaults,
    postprocess_test_file,
    preprocess_file,
)


@dataclass
class PynguinRunner:
    """Orchestrates venv creation, dependency installation, and Pynguin execution."""

    project_path: str
    module_name: str
    venv_dir: str = VENV_DIR
    requirements_file: Optional[str] = None
    algorithm: str = DEFAULT_ALGORITHM
    max_search_time: int = DEFAULT_SEARCH_TIME
    output_dir: str = DEFAULT_OUTPUT_DIR
    report_dir: str = DEFAULT_REPORT_DIR
    export_strategy: str = DEFAULT_EXPORT_STRATEGY
    config_file: Optional[str] = None
    extra_pynguin_args: List[str] = field(default_factory=list)

    _venv_path: Path = field(init=False, repr=False)
    _python: Path = field(init=False, repr=False)
    _pip: List[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._venv_path = Path(self.venv_dir).resolve()
        self._python = venv_python(self._venv_path)
        self._pip = venv_pip(self._venv_path)

    # ── Stage 1: Virtual environment ─────────────────────────────────────

    def create_venv(self, *, force: bool = False) -> None:
        if self._venv_path.exists():
            if force:
                print(f"[venv] Removing existing venv at {self._venv_path} ...")
                shutil.rmtree(self._venv_path)
            else:
                print(f"[venv] Reusing existing venv at {self._venv_path}")
                return

        print(f"[venv] Creating virtual environment at {self._venv_path} ...")
        builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=not is_windows())
        builder.create(str(self._venv_path))
        print("[venv] Virtual environment created successfully.")

        run(self._pip + ["install", "--upgrade", "pip", "setuptools", "wheel"])

    # ── Stage 2: Install dependencies ────────────────────────────────────

    def _deps_fingerprint(self) -> str:
        """Build a fingerprint string from the project path and requirements source.

        If the fingerprint matches what was stored during the last install,
        we can safely skip re-installing.
        """
        import hashlib

        project = Path(self.project_path).resolve()
        parts = [str(project), PYNGUIN_PACKAGE, HYPOTHESIS_PACKAGE]

        if self.requirements_file:
            req_path = Path(self.requirements_file).resolve()
            if req_path.is_file():
                parts.append(req_path.read_text(encoding="utf-8", errors="ignore"))
        else:
            sources = find_dependency_sources(project)
            for rf in sources["req_files"]:
                if rf.is_file():
                    parts.append(rf.read_text(encoding="utf-8", errors="ignore"))
            for manifest in (sources["setup_py"], sources["pyproject"], sources["setup_cfg"]):
                if manifest and manifest.is_file():
                    parts.append(manifest.read_text(encoding="utf-8", errors="ignore"))

        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]

    def _deps_already_installed(self) -> bool:
        """Return True if deps were already installed for the same project/requirements."""
        marker = self._venv_path / ".pyqu_deps_installed"
        if not marker.is_file():
            return False
        stored = marker.read_text(encoding="utf-8").strip()
        return stored == self._deps_fingerprint()

    def _mark_deps_installed(self) -> None:
        """Write a marker so the next run can skip dependency installation."""
        marker = self._venv_path / ".pyqu_deps_installed"
        marker.write_text(self._deps_fingerprint(), encoding="utf-8")

    def install_dependencies(self) -> None:
        if self._deps_already_installed():
            print("[deps] Dependencies already installed (unchanged project/requirements). Skipping.")
            return

        project = Path(self.project_path).resolve()
        installed = False

        if self.requirements_file:
            req_path = Path(self.requirements_file).resolve()
            if req_path.is_file():
                print(f"[deps] Installing from user-specified requirements: {req_path}")
                run(self._pip + ["install", "-r", str(req_path)])
                installed = True
            else:
                print(f"[deps] WARNING: --requirements file not found: {req_path}")

        if not installed:
            sources = find_dependency_sources(project)

            for rf in sources["req_files"]:
                print(f"[deps] Found {rf.name} in target project -- installing ...")
                run(self._pip + ["install", "-r", str(rf)])
                installed = True

            if not installed and (sources["setup_py"] or sources["pyproject"] or sources["setup_cfg"]):
                manifest = sources["setup_py"] or sources["pyproject"] or sources["setup_cfg"]
                print(f"[deps] Found {manifest.name} in target project -- editable install ...")
                result = run(
                    self._pip + ["install", "-e", str(project)],
                    check=False,
                )
                if result.returncode == 0:
                    installed = True
                else:
                    print("[deps] Editable install failed; falling back to import scanning.")

        if not installed:
            self._install_via_pipreqs(project)

        print(f"[deps] Installing {PYNGUIN_PACKAGE} ...")
        run(self._pip + ["install", PYNGUIN_PACKAGE])
        print(f"[deps] {PYNGUIN_PACKAGE} installed.")

        print("[deps] Installing hypothesis (compatible version) ...")
        run(
            self._pip + ["install", "--force-reinstall", HYPOTHESIS_PACKAGE],
            check=False,
        )
        print("[deps] hypothesis installed.")

        self._verify_imports(project)
        self._mark_deps_installed()

    # ── Stage 2b: pipreqs fallback ───────────────────────────────────────

    def _install_via_pipreqs(self, project: Path) -> None:
        """Use *pipreqs* to scan project imports and install detected deps."""
        print("[deps] No dependency manifest found — using pipreqs to scan imports ...")

        result = run(self._pip + ["install", "pipreqs"], check=False)
        if result.returncode != 0:
            print("[deps] WARNING: could not install pipreqs; skipping import scan.")
            return

        tmp_req = project / ".pipreqs_requirements.txt"
        try:
            pipreqs_cmd = [
                str(self._python), "-m", "pipreqs.pipreqs",
                "--savepath", str(tmp_req),
                "--force",
                "--mode", "no-pin",
                "--encoding", "utf-8",
                str(project),
            ]
            result = run(pipreqs_cmd, check=False)

            if result.returncode != 0 or not tmp_req.is_file():
                print("[deps] pipreqs failed to generate requirements.")
                return

            content = tmp_req.read_text(encoding="utf-8").strip()
            if not content:
                print("[deps] pipreqs detected no third-party dependencies.")
                return

            print("[deps] pipreqs detected dependencies:")
            for line in content.splitlines():
                if line.strip():
                    print(f"       {line.strip()}")

            print("[deps] Installing pipreqs-detected dependencies ...")
            run(self._pip + ["install", "-r", str(tmp_req)], check=False)
        finally:
            if tmp_req.is_file():
                tmp_req.unlink()

    # ── Stage 2c: verify imports and install missing deps ──────────────

    # Common import-name -> PyPI-package mappings that pip/pipreqs miss
    _IMPORT_TO_PACKAGE = {
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "skimage": "scikit-image",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "attr": "attrs",
        "bs4": "beautifulsoup4",
        "dateutil": "python-dateutil",
        "gi": "PyGObject",
        "wx": "wxPython",
        "Crypto": "pycryptodome",
        "serial": "pyserial",
        "usb": "pyusb",
        "dotenv": "python-dotenv",
    }

    def _verify_imports(self, project: Path) -> None:
        """Try importing every project dependency; chase and install missing modules.

        Instead of just checking whether the import name is installed, this
        actually runs ``import X`` in the venv Python and parses stderr for
        ``ModuleNotFoundError`` — catching *transitive* missing deps like
        cv2 (needed by tensorlayer but not by the project directly).
        """
        imports: set = set()
        for py_file in project.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])

        local_packages = {p.stem for p in project.iterdir() if p.is_dir() or p.suffix == ".py"}
        skip = local_packages | {"before", "after"}

        third_party = sorted(imports - skip)
        if not third_party:
            return

        print(f"[deps] Verifying {len(third_party)} import(s) can load in the venv ...")
        max_rounds = 5
        for round_num in range(1, max_rounds + 1):
            newly_installed: List[str] = []

            for imp in list(third_party):
                result = run(
                    [str(self._python), "-c", f"import {imp}"],
                    check=False,
                    capture=True,
                )
                if result.returncode == 0:
                    continue

                stderr = result.stderr or ""
                missing_mod = self._extract_missing_module(stderr)

                if missing_mod:
                    pkg = self._IMPORT_TO_PACKAGE.get(missing_mod, missing_mod)
                    print(f"[deps] Round {round_num}: import '{imp}' failed — "
                          f"missing '{missing_mod}', installing {pkg} ...")
                    run(self._pip + ["install", pkg], check=False)
                    newly_installed.append(pkg)
                else:
                    pkg = self._IMPORT_TO_PACKAGE.get(imp, imp)
                    print(f"[deps] Round {round_num}: import '{imp}' failed, "
                          f"installing {pkg} ...")
                    run(self._pip + ["install", pkg], check=False)
                    newly_installed.append(pkg)

            if not newly_installed:
                print(f"[deps] All verifiable imports OK.")
                break

            print(f"[deps] Installed {len(newly_installed)} package(s) in round {round_num}, "
                  f"re-checking ...")
        else:
            print(f"[deps] WARNING: some imports may still fail after {max_rounds} rounds.")

    @staticmethod
    def _extract_missing_module(output: str) -> Optional[str]:
        """Parse a ``ModuleNotFoundError`` from any output and return the module name."""
        for line in reversed(output.strip().splitlines()):
            m = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", line)
            if m:
                return m.group(1).split(".")[0]
            m = re.search(r"No module named ['\"]([^'\"]+)['\"]", line)
            if m:
                return m.group(1).split(".")[0]
        return None

    # ── Stage 2d: ensure __init__.py files exist ─────────────────────────

    def _ensure_init_files(self, project: Optional[Path] = None) -> None:
        """Create missing ``__init__.py`` files so Pynguin can resolve imports."""
        if project is None:
            project = Path(self.project_path).resolve()
        created: List[str] = []

        for py_file in project.rglob("*.py"):
            rel = py_file.relative_to(project)
            if rel.name == "__init__.py":
                continue
            for parent in rel.parents:
                if parent == Path("."):
                    continue
                init = project / parent / "__init__.py"
                if not init.exists():
                    init.touch()
                    created.append(str(parent))

        if created:
            unique = sorted(set(created))
            print(f"[init] Created missing __init__.py in {len(unique)} package(s):")
            for p in unique:
                print(f"       {p}/")

    # ── Stage 3a: Pre-process project (AST stub injection) ───────────────

    def _preprocess_project(self) -> Optional[Path]:
        """Create a temporary copy of the project with injected type stubs."""
        clear_stub_defaults()
        project = Path(self.project_path).resolve()
        temp_dir = Path(tempfile.mkdtemp(prefix="pynguin_pp_"))
        temp_project = temp_dir / "project"
        shutil.copytree(project, temp_project)

        modified = False
        for py_file in temp_project.rglob("*.py"):
            if py_file.name.startswith("test_") or py_file.name == "__init__.py":
                continue
            if preprocess_file(py_file):
                modified = True

        if not modified:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("[preprocess] No untyped attribute-access patterns found; "
                  "running Pynguin on the original project.")
            return None

        self._ensure_init_files(temp_project)
        print(f"[preprocess] Temporary project with stubs: {temp_project}")
        return temp_project

    # ── Stage 3b: Post-process generated tests ───────────────────────────

    def _postprocess_tests(self) -> None:
        """Replace stub-class references in generated tests with SimpleNamespace."""
        output = Path(self.output_dir).resolve()
        if not output.exists():
            return
        for test_file in sorted(output.glob("test_*.py")):
            postprocess_test_file(test_file)

    # ── Stage 3c: Run Pynguin ────────────────────────────────────────────

    def _build_pynguin_env(self, *extra_paths: str) -> dict:
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(self._venv_path)
        env["PYNGUIN_DANGER_AWARE"] = "1"
        if is_windows():
            env["PATH"] = str(self._venv_path / "Scripts") + os.pathsep + env.get("PATH", "")
        else:
            env["PATH"] = str(self._venv_path / "bin") + os.pathsep + env.get("PATH", "")

        project = Path(self.project_path).resolve()
        paths: List[str] = [str(project), str(project.parent)]
        for p in extra_paths:
            resolved = str(Path(p).resolve())
            if resolved not in paths:
                paths.append(resolved)
        existing = env.get("PYTHONPATH", "")
        if existing:
            paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(paths)

        return env

    def run_pynguin(
        self,
        module_name: Optional[str] = None,
        project_path_override: Optional[str] = None,
    ) -> bool:
        """Run Pynguin for a single module. Returns True on success."""
        target = module_name or self.module_name
        project_dir = project_path_override or self.project_path
        pynguin_exe = venv_executable(self._venv_path, "pynguin")

        if not pynguin_exe.exists():
            pynguin_cmd = [str(self._python), "-m", "pynguin"]
        else:
            pynguin_cmd = [str(pynguin_exe)]

        project = Path(project_dir).resolve()
        if not project.exists():
            raise FileNotFoundError(f"Project path does not exist: {project}")

        output = Path(self.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)

        report = Path(self.report_dir).resolve()
        report.mkdir(parents=True, exist_ok=True)

        if self.config_file:
            cfg = Path(self.config_file).resolve()
            if not cfg.is_file():
                raise FileNotFoundError(f"Config file not found: {cfg}")
            cmd = pynguin_cmd + ["--config", str(cfg)]
        else:
            cmd = pynguin_cmd + [
                "--project-path", str(project),
                "--module-name", target,
                "--output-path", str(output),
                "--report-dir", str(report),
                "--algorithm", self.algorithm,
                "--maximum-search-time", str(self.max_search_time),
                "--export-strategy", self.export_strategy,
            ]

        cmd.extend(self.extra_pynguin_args)
        extra = [str(project)] if project_path_override else []
        env = self._build_pynguin_env(*extra)

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            print(f"\n[pynguin] Running Pynguin on module '{target}' "
                  f"(attempt {attempt}/{max_retries}) ...")
            result = run(cmd, env=env, check=False, capture=True)

            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            print(combined.rstrip())

            if result.returncode == 0:
                print(f"[pynguin] Done. Generated tests are in: {output}")
                return True

            missing = self._extract_missing_module(combined)
            if not missing:
                break

            pkg = self._IMPORT_TO_PACKAGE.get(missing, missing)
            print(f"\n[pynguin] Module '{target}' failed due to missing '{missing}'. "
                  f"Auto-installing {pkg} and retrying ...")
            run(self._pip + ["install", pkg], check=False)

        print(f"[pynguin] WARNING: Pynguin exited with code {result.returncode} for '{target}'")
        return False

    def run_all_modules(self, project_path_override: Optional[str] = None) -> None:
        """Discover every module in the project and run Pynguin on each.

        When before/after pairs are detected, Pynguin only runs on ``before``
        modules.  The generated tests are then mirrored for ``after`` modules
        so both versions are tested with *identical* test cases.
        """
        project_dir = project_path_override or self.project_path
        project = Path(project_dir).resolve()
        all_modules = discover_modules(project)

        if not all_modules:
            print("[pynguin] No Python modules found in the project.")
            return

        original_project = Path(self.project_path).resolve()
        pairs = find_before_after_pairs(all_modules, original_project)
        after_mods = {p["after_mod"] for p in pairs}

        if after_mods:
            modules = [m for m in all_modules if m not in after_mods]
            print(f"\n[pynguin] Detected {len(pairs)} before/after pair(s).")
            print(f"          Skipping 'after' modules (will reuse 'before' tests).")
        else:
            modules = all_modules

        total = len(modules)
        print(f"\n[pynguin] Will generate tests for {total} module(s):")
        for i, mod in enumerate(modules, 1):
            print(f"          {i:>3}. {mod}")

        succeeded: List[str] = []
        failed: List[str] = []

        for i, mod in enumerate(modules, 1):
            print(f"\n{'=' * 60}")
            print(f"  [{i}/{total}]  Module: {mod}")
            print(f"{'=' * 60}")
            ok = self.run_pynguin(
                module_name=mod,
                project_path_override=project_path_override,
            )
            (succeeded if ok else failed).append(mod)

        print(f"\n{'=' * 60}")
        print(f"  Test Generation Summary")
        print(f"{'=' * 60}")
        print(f"  Total modules  : {total}")
        print(f"  Succeeded      : {len(succeeded)}")
        print(f"  Failed/skipped : {len(failed)}")
        if failed:
            print(f"\n  Failed modules:")
            for mod in failed:
                print(f"    - {mod}")
        print(f"\n  Generated tests in: {Path(self.output_dir).resolve()}")
        print(f"{'=' * 60}")

    # ── Stage 3d: Mirror before tests for after modules ──────────────────

    def _mirror_before_tests_for_after(self) -> None:
        """Create after test files by copying before tests with swapped imports."""
        project = Path(self.project_path).resolve()
        modules = discover_modules(project)
        pairs = find_before_after_pairs(modules, project)

        if not pairs:
            return

        output = Path(self.output_dir).resolve()
        if not output.exists():
            return

        mirrored = 0
        for pair in pairs:
            if pair["import_mode"] != "direct":
                continue

            before_mod = pair["before_mod"]
            after_mod = pair["after_mod"]

            safe_before = before_mod.replace(".", "_")
            before_test = output / f"test_{safe_before}.py"

            if not before_test.is_file():
                continue

            source = before_test.read_text(encoding="utf-8")
            modified = source.replace(
                f"import {before_mod} ",
                f"import {after_mod} ",
            )

            safe_after = after_mod.replace(".", "_")
            after_test = output / f"test_{safe_after}.py"
            after_test.write_text(modified, encoding="utf-8")
            mirrored += 1
            print(
                f"[mirror] {before_test.name} \u2192 {after_test.name} "
                f"(same tests, different module)"
            )

        if mirrored:
            print(f"[mirror] Mirrored {mirrored} test file(s) for after modules.")

    # ── Stage 3e: Generate Hypothesis property tests ─────────────────────

    def generate_hypothesis_tests(self) -> None:
        """Auto-generate Hypothesis before/after comparison tests."""
        project = Path(self.project_path).resolve()
        modules = discover_modules(project)
        pairs = find_before_after_pairs(modules, project)

        if not pairs:
            if project.name == "before":
                print(
                    "[hypothesis] Tip: use --project-path pointing to the "
                    "parent folder that contains both before/ and after/ "
                    "to enable before-vs-after comparison tests."
                )
            else:
                print("[hypothesis] No before/after module pairs found; skipping.")
            return

        output = Path(self.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)

        generated = 0
        for pair in pairs:
            before_mod = pair["before_mod"]
            mod_parts = before_mod.split(".")
            before_file = project / Path(*mod_parts).with_suffix(".py")

            if not before_file.is_file():
                continue

            try:
                source = before_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            analysis = analyze_params_for_hypothesis(source)
            if not analysis:
                continue

            content = generate_hypothesis_file(pair, analysis)
            safe_name = pair["common"].replace(".", "_")
            test_file = output / f"test_hypothesis_{safe_name}.py"
            test_file.write_text(content, encoding="utf-8")
            print(f"[hypothesis] Generated: {test_file.name}")
            generated += 1

        if generated:
            print(f"[hypothesis] Created {generated} Hypothesis test file(s).")

    # ── Stage 4: Regression comparison (before vs after) ───────────────

    def run_comparison(self) -> bool:
        """Run the before/after regression comparison as a separate phase.

        Compares each ``test_before_*.py`` against its ``test_after_*.py``
        test-by-test.  The phase PASSES only when no regressions are found
        (i.e. no test that passed in 'before' fails in 'after').
        """
        output = Path(self.output_dir).resolve()
        env = self._build_pynguin_env()
        return run_all_comparisons(output, self._python, env)

    # ── Stage 5: Run pytest on test files ────────────────────────────────

    @staticmethod
    def _load_existing_test_map(output: Path) -> Dict[str, dict]:
        """Read ``.existing_test_map`` and return per-file metadata.

        Returns ``{filename: {"module": str, "relevant": [str], "skipped": [str]}}``.
        """
        map_file = output / ".existing_test_map"
        if not map_file.is_file():
            return {}

        mapping: Dict[str, dict] = {}
        for line in map_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" not in line:
                continue

            fname, rest = line.split("=", 1)
            parts = rest.split("|")
            module = parts[0]
            relevant: List[str] = []
            skipped: List[str] = []
            for part in parts[1:]:
                if part.startswith("relevant=") and part[9:]:
                    relevant = part[9:].split(",")
                elif part.startswith("skipped=") and part[8:]:
                    skipped = part[8:].split(",")

            mapping[fname] = {
                "module": module,
                "relevant": relevant,
                "skipped": skipped,
            }

        return mapping

    def run_pytest(self) -> bool:
        """Run pytest in two rounds: pre-existing tests, then generated tests.

        For pre-existing tests linked to a source module, only the test
        functions that actually call module symbols are executed (via
        ``pytest -k``).  Irrelevant functions are reported and skipped.

        Returns True only if both rounds pass (or a round has no files).
        """
        output = Path(self.output_dir).resolve()
        if not output.exists():
            print("[pytest] No test files found -- skipping pytest.")
            return False

        existing_files = sorted(output.glob("existing_test_*.py"))
        generated_files = sorted(
            f for f in output.glob("test_*.py")
            if not f.name.startswith("existing_")
        )

        if not existing_files and not generated_files:
            print("[pytest] No test files found -- skipping pytest.")
            return False

        env = self._build_pynguin_env()

        test_map = self._load_existing_test_map(output)
        all_passed = True

        # ── Round 1: pre-existing project tests ──
        if existing_files:
            print(f"\n{'=' * 60}")
            print(f"  ROUND 1: PRE-EXISTING PROJECT TESTS "
                  f"({len(existing_files)} file(s))")
            print(f"{'=' * 60}")

            for tf in existing_files:
                info = test_map.get(tf.name, {})
                module = info.get("module", "UNLINKED")
                relevant = info.get("relevant", [])
                skipped = info.get("skipped", [])

                print(f"\n  {'-' * 56}")
                if module == "UNLINKED":
                    print(f"  {tf.name}")
                    print(f"    Source module : (unlinked)")
                    print(f"    Running       : ALL test functions")
                    cmd = [str(self._python), "-m", "pytest", str(tf),
                           "-v", "--tb=long", "-rA"]
                else:
                    print(f"  {tf.name}  ->  {module}")
                    total = len(relevant) + len(skipped)
                    print(f"    Functions     : {total} total, "
                          f"{len(relevant)} relevant, {len(skipped)} skipped")

                    if relevant:
                        print(f"    RUN:")
                        for fn in relevant:
                            print(f"      + {fn}")
                    if skipped:
                        print(f"    SKIP (not related to {module}):")
                        for fn in skipped:
                            print(f"      - {fn}")

                    if relevant:
                        k_expr = " or ".join(relevant)
                        cmd = [str(self._python), "-m", "pytest", str(tf),
                               "-k", k_expr, "-v", "--tb=long", "-rA"]
                    elif not relevant and not skipped:
                        cmd = [str(self._python), "-m", "pytest", str(tf),
                               "-v", "--tb=long", "-rA"]
                    else:
                        print(f"    -> No relevant functions found; skipping file.")
                        continue

                result = run(cmd, env=env, check=False)
                if result.returncode == 0:
                    print(f"  [PASSED] {tf.name}")
                else:
                    print(f"  [FAILED] {tf.name} (exit code {result.returncode})")
                    all_passed = False
        else:
            print("\n[pytest] No pre-existing tests found in the project.")

        # ── Round 2: Pynguin-generated tests ──
        if generated_files:
            print(f"\n{'=' * 60}")
            print(f"  ROUND 2: PYNGUIN-GENERATED TESTS "
                  f"({len(generated_files)} file(s))")
            print(f"{'=' * 60}")
            for tf in generated_files:
                print(f"  {tf.name}")

            file_args = [str(f) for f in generated_files]
            cmd = [str(self._python), "-m", "pytest"] + file_args + [
                "-v", "--tb=long", "-rA",
            ]
            result = run(cmd, env=env, check=False)
            if result.returncode == 0:
                print("[pytest] All generated tests PASSED.")
            else:
                print(f"[pytest] Some generated tests FAILED "
                      f"(exit code {result.returncode}).")
                all_passed = False
        else:
            print("\n[pytest] No Pynguin-generated tests produced.")

        return all_passed

    # ── Discover & match existing project tests ─────────────────────────

    @staticmethod
    def _extract_imports(test_file: Path) -> List[str]:
        """Return all imported module names from a Python test file."""
        try:
            source = test_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            return []

        modules: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.append(node.module)
        return modules

    @staticmethod
    def _get_module_symbols(module_path: Path) -> set:
        """Return the set of top-level function and class names in a source module."""
        try:
            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            return set()

        symbols: set = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
        return symbols

    @staticmethod
    def _filter_relevant_test_functions(
        test_file: Path,
        target_symbols: set,
    ) -> Tuple[List[str], List[str]]:
        """Split test functions into relevant vs irrelevant for a source module.

        A test function is *relevant* if its body contains a call to any
        name in *target_symbols* (directly or via attribute access like
        ``module.func()``).

        Returns ``(relevant_names, skipped_names)``.
        """
        try:
            source = test_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            return [], []

        relevant: List[str] = []
        skipped: List[str] = []

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue

            found = False
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name and name in target_symbols:
                    found = True
                    break

            if found:
                relevant.append(node.name)
            else:
                skipped.append(node.name)

        return relevant, skipped

    def _match_test_to_module(
        self,
        test_file: Path,
        source_modules: List[str],
    ) -> Optional[str]:
        """Try to link a test file to a source module.

        Strategy (in priority order):
        1. **Import scan** — parse the test file's imports and see if any
           match a known source module (or a parent of one).
        2. **Name convention** — strip ``test_`` / ``_test`` from the
           filename and check if any source module's leaf name matches.
        """
        # ── Strategy 1: import scanning ──
        imported = self._extract_imports(test_file)
        for imp in imported:
            for mod in source_modules:
                if mod == imp or mod.startswith(imp + ".") or imp.startswith(mod + "."):
                    return mod

        # ── Strategy 2: name convention ──
        stem = test_file.stem
        name = re.sub(r"^test_", "", stem)
        name = re.sub(r"_test$", "", name)
        name_parts = name.lower().split("_")

        for mod in source_modules:
            mod_leaf = mod.rsplit(".", 1)[-1].lower()
            if mod_leaf in name_parts:
                return mod
            mod_flat = mod.lower().replace(".", "_")
            if mod_flat == name.lower():
                return mod

        return None

    def _collect_existing_tests(
        self,
    ) -> Tuple[Dict[str, List[Path]], List[Path]]:
        """Find test files that ship with the project and link them to modules.

        Returns ``(matched, unmatched)`` where:
        - *matched* maps ``module_name -> [test_file, ...]``
        - *unmatched* is a list of test files that couldn't be linked
        """
        project = Path(self.project_path).resolve()
        output = Path(self.output_dir).resolve()

        candidates: List[Path] = []
        search_roots = [project]
        for name in ("tests", "test"):
            d = project / name
            if d.is_dir():
                search_roots.append(d)

        for root in search_roots:
            for pat in ("test_*.py", "*_test.py"):
                for f in root.rglob(pat):
                    try:
                        if f.resolve().is_relative_to(output):
                            continue
                    except (ValueError, TypeError):
                        pass
                    candidates.append(f)

        candidates = sorted(set(candidates))

        source_modules = discover_modules(project)

        matched: Dict[str, List[Path]] = {}
        unmatched: List[Path] = []

        for tf in candidates:
            mod = self._match_test_to_module(tf, source_modules)
            if mod:
                matched.setdefault(mod, []).append(tf)
            else:
                unmatched.append(tf)

        return matched, unmatched

    def _stash_existing_tests(
        self,
    ) -> Optional[Tuple[Path, Dict[str, List[str]], List[str]]]:
        """Copy pre-existing project tests to a temp dir so they survive cleanup.

        Returns ``(stash_dir, matched_map, unmatched_names)`` or None.
        The *matched_map* maps ``module_name -> [stashed_filename, ...]``.
        """
        matched, unmatched = self._collect_existing_tests()
        all_files = [f for files in matched.values() for f in files] + unmatched
        if not all_files:
            return None

        stash = Path(tempfile.mkdtemp(prefix="pyqu_existing_tests_"))
        used_names: set = set()

        def _safe_copy(src: Path) -> str:
            dest_name = src.name
            if dest_name in used_names:
                dest_name = f"{src.stem}_dup{src.suffix}"
            used_names.add(dest_name)
            shutil.copy2(src, stash / dest_name)
            return dest_name

        stashed_matched: Dict[str, List[str]] = {}
        for mod, files in matched.items():
            stashed_matched[mod] = [_safe_copy(f) for f in files]

        stashed_unmatched: List[str] = [_safe_copy(f) for f in unmatched]

        # ── Print discovery report ──
        print(f"\n[existing-tests] Found {len(all_files)} pre-existing test file(s):")
        if stashed_matched:
            for mod, names in sorted(stashed_matched.items()):
                for n in names:
                    print(f"  LINKED   {n}  ->  {mod}")
        if stashed_unmatched:
            for n in stashed_unmatched:
                print(f"  UNLINKED {n}  (no matching source module found)")

        return stash, stashed_matched, stashed_unmatched

    def _resolve_module_path(self, module_name: str) -> Optional[Path]:
        """Turn a dotted module name into a file path relative to the project."""
        project = Path(self.project_path).resolve()
        rel = Path(*module_name.split(".")).with_suffix(".py")
        full = project / rel
        return full if full.is_file() else None

    def _restore_existing_tests(
        self,
        stash_info: Optional[Tuple[Path, Dict[str, List[str]], List[str]]],
    ) -> None:
        """Copy stashed tests into the output dir with an ``existing_`` prefix.

        For linked files, performs function-level filtering: only test
        functions that call symbols from the matched source module are
        marked as relevant.  Writes a mapping file that ``run_pytest``
        uses to run only the relevant functions.
        """
        if stash_info is None:
            return

        stash, matched_map, unmatched_names = stash_info
        if not stash.exists():
            return

        output = Path(self.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)

        restored = 0
        mapping_lines: List[str] = []

        for mod, names in sorted(matched_map.items()):
            mod_path = self._resolve_module_path(mod)
            target_symbols = self._get_module_symbols(mod_path) if mod_path else set()

            for name in names:
                src = stash / name
                dest = output / f"existing_{name}"
                if not src.is_file() or dest.exists():
                    continue
                shutil.copy2(src, dest)
                restored += 1

                if target_symbols:
                    relevant, skipped = self._filter_relevant_test_functions(
                        dest, target_symbols,
                    )
                    funcs_str = ",".join(relevant) if relevant else ""
                    skipped_str = ",".join(skipped) if skipped else ""
                    mapping_lines.append(
                        f"existing_{name}={mod}|relevant={funcs_str}|skipped={skipped_str}"
                    )

                    if relevant or skipped:
                        print(f"  [filter] {name} -> {mod}")
                        print(f"           {len(relevant)} relevant, "
                              f"{len(skipped)} skipped")
                        if skipped:
                            for s in skipped:
                                print(f"           SKIP  {s}")
                else:
                    mapping_lines.append(f"existing_{name}={mod}|relevant=|skipped=")

        for name in unmatched_names:
            src = stash / name
            dest = output / f"existing_{name}"
            if src.is_file() and not dest.exists():
                shutil.copy2(src, dest)
                mapping_lines.append(f"existing_{name}=UNLINKED|relevant=|skipped=")
                restored += 1

        if mapping_lines:
            (output / ".existing_test_map").write_text(
                "\n".join(mapping_lines) + "\n", encoding="utf-8",
            )

        shutil.rmtree(stash, ignore_errors=True)

        if restored:
            print(f"[existing-tests] Restored {restored} pre-existing test(s) "
                  f"into {output} (prefixed with 'existing_')")

    # ── Cleanup ──────────────────────────────────────────────────────────

    def _cleanup_temp_files(self) -> None:
        """Remove stale temp dirs and previous Pynguin outputs before a run."""
        tmp_root = tempfile.gettempdir()
        removed = 0
        for entry in glob.glob(os.path.join(tmp_root, "tmp*")):
            pydoc_out = os.path.join(entry, "pydoc.out")
            if os.path.isdir(entry) and os.path.exists(pydoc_out):
                try:
                    shutil.rmtree(entry)
                    removed += 1
                except OSError:
                    pass
        if removed:
            print(f"[cleanup] Removed {removed} stale Pynguin temp dir(s).")

        for d in (self.output_dir, self.report_dir):
            p = Path(d).resolve()
            if p.exists():
                shutil.rmtree(p)
                print(f"[cleanup] Removed previous output: {p}")

    # ── Run all stages ───────────────────────────────────────────────────

    def run_all(self, *, force_venv: bool = False, force_deps: bool = False, all_modules: bool = False) -> bool:
        """Run every stage and return True if the testing phase passed."""
        existing_stash = self._stash_existing_tests()
        self._cleanup_temp_files()
        self.create_venv(force=force_venv)
        if force_deps:
            marker = self._venv_path / ".pyqu_deps_installed"
            if marker.is_file():
                marker.unlink()
        self.install_dependencies()
        self._ensure_init_files()

        temp_project = self._preprocess_project()
        project_override = str(temp_project) if temp_project else None

        try:
            if all_modules:
                self.run_all_modules(project_path_override=project_override)
            else:
                self.run_pynguin(project_path_override=project_override)
        finally:
            if temp_project:
                shutil.rmtree(temp_project.parent, ignore_errors=True)

        self._postprocess_tests()
        self._mirror_before_tests_for_after()
        self.generate_hypothesis_tests()
        self._restore_existing_tests(existing_stash)
        self.run_pytest()
        passed = self.run_comparison()
        return passed
