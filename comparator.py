"""Regression detector: compare before/after test results test-by-test.

Rules
-----
  - Both pass   -> OK
  - Both fail   -> OK  (same behavior preserved)
  - Before pass, after fail -> REGRESSION  (new failure introduced)
  - Before fail, after pass -> IMPROVED    (bug fixed — that's fine)

The testing phase FAILS if there is at least one REGRESSION.
"""

from __future__ import annotations

import ast
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .helpers import run


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Outcome of a single test from JUnit XML."""
    status: str           # passed, failed, error, xfailed, skipped
    message: str = ""     # error/failure message (empty when passing)


@dataclass
class TestInputInfo:
    """Readable description of what a test case does."""
    func_called: str = ""
    args_readable: List[str] = field(default_factory=list)
    expects_error: str = ""        # e.g. "ValueError"
    expected_to_fail: bool = False


@dataclass
class TestVerdict:
    """Full comparison result for a single test case."""
    name: str
    before_result: TestResult
    after_result: TestResult
    input_info: TestInputInfo
    verdict: str          # "OK", "REGRESSION", "IMPROVED"


# ── JUnit XML parsing ───────────────────────────────────────────────────────

def _parse_junit_xml(xml_path: Path) -> Dict[str, TestResult]:
    """Parse a JUnit XML file and return ``{test_name: TestResult}``."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results: Dict[str, TestResult] = {}
    for testcase in root.iter("testcase"):
        name = testcase.get("name", "unknown")

        failure_el = testcase.find("failure")
        error_el = testcase.find("error")
        skipped_el = testcase.find("skipped")

        if failure_el is not None:
            msg = _clean_message(failure_el.get("message", ""), failure_el.text)
            results[name] = TestResult("failed", msg)
        elif error_el is not None:
            msg = _clean_message(error_el.get("message", ""), error_el.text)
            results[name] = TestResult("error", msg)
        elif skipped_el is not None:
            skip_type = skipped_el.get("type", "")
            msg = skipped_el.get("message", "")
            if "xfail" in skip_type:
                results[name] = TestResult("xfailed", msg)
            else:
                results[name] = TestResult("skipped", msg)
        else:
            results[name] = TestResult("passed")

    return results


def _clean_message(message: str, traceback_text: str | None) -> str:
    """Extract a short, readable error message from pytest output.

    Tries to pull the last ``ErrorType: description`` line from the
    traceback, falling back to the ``message`` attribute.
    """
    if traceback_text:
        for line in reversed(traceback_text.strip().splitlines()):
            line = line.strip()
            if re.match(r"^[A-Z]\w*(Error|Exception|Warning):", line):
                return line
            if re.match(r"^E\s+[A-Z]\w*(Error|Exception|Warning):", line):
                return line.lstrip("E ").strip()

    if message:
        first_line = message.strip().splitlines()[0].strip()
        if len(first_line) > 120:
            return first_line[:117] + "..."
        return first_line

    return ""


def _is_ok(status: str) -> bool:
    """Whether a test status counts as 'passing' for comparison."""
    return status in ("passed", "xfailed", "skipped")


# ── Test source parsing (extract readable inputs) ────────────────────────────

def _ast_value_to_str(node: ast.AST, var_table: Dict[str, str]) -> str:
    """Convert an AST value node into a human-readable string."""
    if isinstance(node, ast.Constant):
        return repr(node.value)

    if isinstance(node, ast.Name):
        return var_table.get(node.id, node.id)

    if isinstance(node, ast.List):
        elts = [_ast_value_to_str(e, var_table) for e in node.elts]
        return f"[{', '.join(elts)}]"

    if isinstance(node, ast.Tuple):
        elts = [_ast_value_to_str(e, var_table) for e in node.elts]
        return f"({', '.join(elts)})"

    if isinstance(node, ast.Dict):
        pairs = []
        for k, v in zip(node.keys, node.values):
            ks = _ast_value_to_str(k, var_table) if k else "..."
            vs = _ast_value_to_str(v, var_table)
            pairs.append(f"{ks}: {vs}")
        return "{" + ", ".join(pairs) + "}"

    if isinstance(node, ast.Call):
        # types.SimpleNamespace(Y=[0.0, 1.0])
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "types"
                and node.func.attr == "SimpleNamespace"):
            parts = []
            for kw in node.keywords:
                val = _ast_value_to_str(kw.value, var_table)
                parts.append(f"{kw.arg}={val}")
            return "object(" + ", ".join(parts) + ")"

        # module_X.SomeClass(...) or module_X.some_func()
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            mod = node.func.value.id
            fn = node.func.attr
            args = [_ast_value_to_str(a, var_table) for a in node.args]
            kws = [f"{kw.arg}={_ast_value_to_str(kw.value, var_table)}" for kw in node.keywords]
            all_args = ", ".join(args + kws)
            return f"{mod}.{fn}({all_args})"

        # plain function call
        if isinstance(node.func, ast.Name):
            args = [_ast_value_to_str(a, var_table) for a in node.args]
            return f"{node.func.id}({', '.join(args)})"

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"-{_ast_value_to_str(node.operand, var_table)}"

    return "<complex value>"


def _extract_test_inputs(test_file: Path) -> Dict[str, TestInputInfo]:
    """Parse a test source file and extract readable input info per test."""
    try:
        source = test_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}

    # Figure out what module_0 maps to (e.g. "before.Orange.evaluation.scoring")
    module_map: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    module_map[alias.asname] = alias.name

    results: Dict[str, TestInputInfo] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue

        info = TestInputInfo()

        # Check for @pytest.mark.xfail
        for dec in node.decorator_list:
            if "xfail" in ast.dump(dec):
                info.expected_to_fail = True
                break

        # Build a variable lookup table by walking assignments in order
        var_table: Dict[str, str] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                if isinstance(stmt.targets[0], ast.Name):
                    var_name = stmt.targets[0].id
                    var_table[var_name] = _ast_value_to_str(stmt.value, var_table)

        # Find the call to module_0.XXX(...) — the function under test
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if not isinstance(child.func, ast.Attribute):
                continue
            if not isinstance(child.func.value, ast.Name):
                continue
            if not child.func.value.id.startswith("module_0"):
                continue

            info.func_called = child.func.attr
            for arg in child.args:
                info.args_readable.append(_ast_value_to_str(arg, var_table))
            for kw in child.keywords:
                info.args_readable.append(
                    f"{kw.arg}={_ast_value_to_str(kw.value, var_table)}"
                )
            break  # take the first module_0 call

        # Check for pytest.raises(SomeError)
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if (isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "pytest"
                        and child.func.attr == "raises"):
                    if child.args and isinstance(child.args[0], ast.Name):
                        info.expects_error = child.args[0].id

        results[node.name] = info

    return results


# ── Core comparison ──────────────────────────────────────────────────────────

def _compare(
    before_results: Dict[str, TestResult],
    after_results: Dict[str, TestResult],
    input_info: Dict[str, TestInputInfo],
) -> List[TestVerdict]:
    """Compare two result dicts and produce per-test verdicts."""
    all_names = sorted(set(before_results) | set(after_results))
    verdicts: List[TestVerdict] = []

    empty_result = TestResult("missing")
    empty_info = TestInputInfo()

    for name in all_names:
        b = before_results.get(name, empty_result)
        a = after_results.get(name, empty_result)
        inp = input_info.get(name, empty_info)

        b_ok = _is_ok(b.status)
        a_ok = _is_ok(a.status)

        if b_ok and a_ok:
            verdict = "OK"
        elif not b_ok and not a_ok:
            verdict = "OK"
        elif b_ok and not a_ok:
            verdict = "REGRESSION"
        else:
            verdict = "IMPROVED"

        verdicts.append(TestVerdict(
            name=name,
            before_result=b,
            after_result=a,
            input_info=inp,
            verdict=verdict,
        ))

    return verdicts


# ── Human-readable status descriptions ───────────────────────────────────────

_STATUS_EXPLAIN = {
    "passed":  "Passed (ran successfully)",
    "failed":  "FAILED",
    "error":   "ERROR (crashed)",
    "xfailed": "Expected failure occurred (OK)",
    "skipped": "Skipped",
    "missing": "Not found",
}


def _status_line(result: TestResult) -> str:
    """Build a one-line explanation of a test result."""
    base = _STATUS_EXPLAIN.get(result.status, result.status)
    if result.message and result.status in ("failed", "error"):
        return f"{base} — {result.message}"
    return base


# ── Report formatting ────────────────────────────────────────────────────────

_VERDICT_TAG = {
    "OK":         "OK",
    "REGRESSION": "REGRESSION",
    "IMPROVED":   "IMPROVED",
}


def _format_report(
    before_file: str,
    after_file: str,
    verdicts: List[TestVerdict],
) -> str:
    """Build a detailed, non-coder friendly comparison report."""
    sep = "=" * 70
    thin = "-" * 70

    regressions = [v for v in verdicts if v.verdict == "REGRESSION"]
    improvements = [v for v in verdicts if v.verdict == "IMPROVED"]
    ok_count = sum(1 for v in verdicts if v.verdict == "OK")
    total = len(verdicts)

    overall = "PASSED" if not regressions else "FAILED"

    lines: List[str] = [
        "",
        sep,
        "  BEFORE vs AFTER  —  REGRESSION COMPARISON",
        sep,
        "",
        f"  Before tests : {before_file}",
        f"  After tests  : {after_file}",
        "",
    ]

    # ── Per-test detailed cards ──
    for v in verdicts:
        tag = _VERDICT_TAG[v.verdict]
        inp = v.input_info

        lines.append(thin)

        marker = ""
        if inp.expected_to_fail:
            marker = "  [marked: expected to fail]"
        elif inp.expects_error:
            marker = f"  [expects {inp.expects_error}]"

        lines.append(f"  {v.name}{marker}")
        lines.append("")

        # What was tested
        if inp.func_called:
            args_str = ", ".join(inp.args_readable) if inp.args_readable else "..."
            lines.append(f"    What was tested : {inp.func_called}({args_str})")
        else:
            lines.append(f"    What was tested : (could not detect)")

        # Inputs
        if inp.args_readable:
            lines.append(f"    Inputs:")
            for i, arg in enumerate(inp.args_readable, 1):
                lines.append(f"      arg {i} = {arg}")

        lines.append("")
        lines.append(f"    Before : {_status_line(v.before_result)}")
        lines.append(f"    After  : {_status_line(v.after_result)}")
        lines.append(f"    Result : [{tag}]")

    # ── Summary ──
    lines.append(thin)
    lines.append("")
    lines.append(f"  SUMMARY")
    lines.append(f"  {'-' * 40}")
    lines.append(f"  Total tests compared : {total}")
    lines.append(f"  Same behavior (OK)   : {ok_count}")
    lines.append(f"  Regressions          : {len(regressions)}")
    lines.append(f"  Improvements         : {len(improvements)}")
    lines.append("")

    if regressions:
        lines.append("  REGRESSIONS DETECTED:")
        for v in regressions:
            lines.append(f"    X  {v.name}")
            lines.append(f"       Before: {_status_line(v.before_result)}")
            lines.append(f"       After : {_status_line(v.after_result)}")
        lines.append("")

    if improvements:
        lines.append("  IMPROVEMENTS (tests fixed by 'after' version):")
        for v in improvements:
            lines.append(f"    +  {v.name}")
            lines.append(f"       Before: {_status_line(v.before_result)}")
            lines.append(f"       After : {_status_line(v.after_result)}")
        lines.append("")

    lines.append(f"  VERDICT: {overall}")
    if regressions:
        lines.append(
            f"           {len(regressions)} test(s) passed in BEFORE "
            f"but FAIL in AFTER — the 'after' version introduced a bug."
        )
    elif overall == "PASSED":
        lines.append(
            "           The 'after' version behaves the same (or better) "
            "than the 'before' version."
        )
    lines.append(sep)

    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────

def run_comparison(
    before_test_file: Path,
    after_test_file: Path,
    python_exe: Path,
    env: dict,
) -> Tuple[bool, str]:
    """Run pytest on both files separately, compare results, return report.

    Returns ``(passed, report_text)``.
    ``passed`` is True when there are zero regressions.
    """
    tmp = Path(tempfile.gettempdir())
    before_xml = tmp / "pytest_before.xml"
    after_xml = tmp / "pytest_after.xml"

    for xml in (before_xml, after_xml):
        if xml.exists():
            xml.unlink()

    run(
        [str(python_exe), "-m", "pytest", str(before_test_file),
         "--junit-xml", str(before_xml), "-v", "--tb=short", "-q"],
        env=env, check=False,
    )

    run(
        [str(python_exe), "-m", "pytest", str(after_test_file),
         "--junit-xml", str(after_xml), "-v", "--tb=short", "-q"],
        env=env, check=False,
    )

    if not before_xml.exists() or not after_xml.exists():
        msg = "[comparison] Could not collect results (JUnit XML missing)."
        return False, msg

    before_results = _parse_junit_xml(before_xml)
    after_results = _parse_junit_xml(after_xml)

    input_info = _extract_test_inputs(before_test_file)

    verdicts = _compare(before_results, after_results, input_info)
    report = _format_report(
        before_test_file.name, after_test_file.name, verdicts,
    )

    has_regressions = any(v.verdict == "REGRESSION" for v in verdicts)
    return not has_regressions, report


def run_all_comparisons(
    output_dir: Path,
    python_exe: Path,
    env: dict,
) -> bool:
    """Find all before/after test file pairs and compare each one.

    Returns True if every pair passed (zero regressions across all pairs).
    """
    if not output_dir.exists():
        return True

    before_tests = sorted(output_dir.glob("test_before_*.py"))
    if not before_tests:
        print("[comparison] No before/after test pairs found; skipping comparison.")
        return True

    all_passed = True

    for before_file in before_tests:
        suffix = before_file.name.replace("test_before_", "")
        after_file = output_dir / f"test_after_{suffix}"

        if not after_file.is_file():
            print(f"[comparison] No matching after file for {before_file.name}; skipping.")
            continue

        passed, report = run_comparison(
            before_file, after_file, python_exe, env,
        )
        print(report)

        if not passed:
            all_passed = False

    return all_passed
