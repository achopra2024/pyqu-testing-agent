"""PynguinRunner: orchestrates venv, deps, Pynguin, post-processing, and pytest."""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import (
    DEFAULT_ALGORITHM,
    DEFAULT_EXPORT_STRATEGY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_SEARCH_TIME,
    HYPOTHESIS_PACKAGE,
    PYNGUIN_PACKAGE,
    VENV_DIR,
)
from .discovery import discover_modules, find_before_after_pairs
from .helpers import (
    find_dependency_sources,
    is_windows,
    run,
    venv_executable,
    venv_pip,
    venv_python,
)
from .comparator import run_all_comparisons
from .hypothesis_gen import analyze_params_for_hypothesis, generate_hypothesis_file
from .preprocessor import (
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

    def install_dependencies(self) -> None:
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

    # ── Stage 2c: ensure __init__.py files exist ─────────────────────────

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

    def _build_pynguin_env(self) -> dict:
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(self._venv_path)
        env["PYNGUIN_DANGER_AWARE"] = "1"
        if is_windows():
            env["PATH"] = str(self._venv_path / "Scripts") + os.pathsep + env.get("PATH", "")
        else:
            env["PATH"] = str(self._venv_path / "bin") + os.pathsep + env.get("PATH", "")
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
        env = self._build_pynguin_env()

        print(f"\n[pynguin] Running Pynguin on module '{target}' ...")
        result = run(cmd, env=env, check=False)
        if result.returncode == 0:
            print(f"[pynguin] Done. Generated tests are in: {output}")
            return True
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
        project = Path(self.project_path).resolve()
        env = self._build_pynguin_env()
        env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")

        return run_all_comparisons(output, self._python, env)

    # ── Stage 5: Run pytest on generated tests ───────────────────────────

    def run_pytest(self) -> bool:
        """Run pytest against the generated test files."""
        output = Path(self.output_dir).resolve()
        if not output.exists() or not any(output.glob("test_*.py")):
            print("[pytest] No generated test files found -- skipping pytest.")
            return False

        test_files = sorted(output.glob("test_*.py"))
        print(f"\n[pytest] Running pytest on {len(test_files)} generated test file(s):")
        for tf in test_files:
            print(f"         {tf.name}")

        project = Path(self.project_path).resolve()
        env = self._build_pynguin_env()
        env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [str(self._python), "-m", "pytest", str(output), "-v", "--tb=long", "-rA"]
        result = run(cmd, env=env, check=False)

        if result.returncode == 0:
            print("[pytest] All generated tests PASSED.")
            return True

        print(f"[pytest] Some tests failed (exit code {result.returncode}).")
        return False

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

    def run_all(self, *, force_venv: bool = False, all_modules: bool = False) -> bool:
        """Run every stage and return True if the testing phase passed."""
        self._cleanup_temp_files()
        self.create_venv(force=force_venv)
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
        self.run_pytest()
        passed = self.run_comparison()
        return passed
