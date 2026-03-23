"""Hypothesis property-test generation for before/after comparison."""

from __future__ import annotations

import ast
from typing import Dict, List, Set

from preprocessor import _detect_module_imports, _infer_attr_default

# ── Strategy mappings ────────────────────────────────────────────────────────

_DEFAULT_TO_STRATEGY: Dict[str, str] = {
    "[0.0, 1.0]": (
        "st.lists(st.floats(min_value=-1e3, max_value=1e3,"
        " allow_nan=False, allow_infinity=False), min_size=1, max_size=20)"
    ),
    '"default"': "st.text(min_size=1, max_size=50)",
    "0.0": (
        "st.floats(min_value=-1e3, max_value=1e3,"
        " allow_nan=False, allow_infinity=False)"
    ),
    "True": "st.booleans()",
    '{"k": 0}': (
        "st.dictionaries(st.text(min_size=1, max_size=10),"
        " st.integers(-100, 100), min_size=1, max_size=5)"
    ),
    "[0, 1]": "st.lists(st.integers(-1000, 1000), min_size=1, max_size=20)",
}

_FALLBACK_STRATEGY = "st.lists(st.integers(-1000, 1000), min_size=1, max_size=20)"


# ── Strategy inference ───────────────────────────────────────────────────────


def _infer_simple_param_strategy(
    func_node: ast.AST,
    param_name: str,
    import_aliases: Dict[str, str],
) -> str:
    """Infer a Hypothesis strategy for a parameter without attribute access."""
    array_libs = {"numpy", "np", "scipy", "torch", "tf", "pandas", "pd"}
    resolved = {
        alias for alias, pkg in import_aliases.items()
        if pkg.split(".")[0] in array_libs or alias in array_libs
    }

    has_len = False
    is_iterated = False
    in_arithmetic = False
    passed_to_array_lib = False

    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len" and any(
                isinstance(a, ast.Name) and a.id == param_name for a in node.args
            ):
                has_len = True

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in resolved:
                for a in node.args:
                    if isinstance(a, ast.Name) and a.id == param_name:
                        passed_to_array_lib = True

        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name):
            if node.iter.id == param_name:
                is_iterated = True

        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == param_name:
                has_len = True

        if isinstance(node, ast.BinOp):
            for side in (node.left, node.right):
                if isinstance(side, ast.Name) and side.id == param_name:
                    in_arithmetic = True

    if passed_to_array_lib:
        return _DEFAULT_TO_STRATEGY["[0.0, 1.0]"]
    if has_len or is_iterated:
        return _DEFAULT_TO_STRATEGY["[0.0, 1.0]"]
    if in_arithmetic:
        return _DEFAULT_TO_STRATEGY["0.0"]
    return _FALLBACK_STRATEGY


def analyze_params_for_hypothesis(source: str) -> Dict[str, Dict[str, dict]]:
    """Analyse every function in *source* and return strategy info for all params.

    Returns::

        {
            "func_name": {
                "param_name": {
                    "type":     "namespace" | "simple",
                    "attrs":    {attr: strategy_str, ...},  # only for namespace
                    "strategy": "st.xxx(...)",              # only for simple
                },
                ...
            },
            ...
        }
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    import_aliases = _detect_module_imports(tree)
    result: Dict[str, Dict[str, dict]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        params = [
            arg.arg for arg in node.args.args if arg.arg not in ("self", "cls")
        ]
        if not params:
            continue

        func_info: Dict[str, dict] = {}
        for param in params:
            attrs: Set[str] = set()
            methods: Set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if (isinstance(child.func.value, ast.Name)
                            and child.func.value.id == param):
                        methods.add(child.func.attr)
                elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                    if child.value.id == param:
                        attrs.add(child.attr)
            attrs -= methods

            if attrs:
                attr_strats: Dict[str, str] = {}
                for attr in sorted(attrs):
                    default = _infer_attr_default(node, param, attr, import_aliases)
                    attr_strats[attr] = _DEFAULT_TO_STRATEGY.get(
                        default, _FALLBACK_STRATEGY,
                    )
                func_info[param] = {"type": "namespace", "attrs": attr_strats}
            else:
                strat = _infer_simple_param_strategy(node, param, import_aliases)
                func_info[param] = {"type": "simple", "strategy": strat}

        if func_info:
            result[node.name] = func_info

    return result


# ── Test-file generation ─────────────────────────────────────────────────────


def generate_hypothesis_file(
    pair: Dict[str, str],
    func_analysis: Dict[str, Dict[str, dict]],
) -> str:
    """Generate the full content of a Hypothesis property-test file."""
    before_mod_name = pair["before_mod"]
    after_mod_name = pair["after_mod"]
    sep = "=" * 62

    lines: List[str] = [
        '"""Auto-generated Hypothesis property tests (before vs after)."""',
        "import types",
        "import pytest",
        "from hypothesis import given, strategies as st, settings",
        "",
    ]

    if pair["import_mode"] == "direct":
        lines.append(f"import {before_mod_name} as before_mod")
        lines.append(f"import {after_mod_name} as after_mod")
    else:
        lines.extend([
            "import importlib.util",
            "from pathlib import Path",
            "",
            "",
            "def _load(name, filepath):",
            "    spec = importlib.util.spec_from_file_location(name, str(filepath))",
            "    mod = importlib.util.module_from_spec(spec)",
            "    spec.loader.exec_module(mod)",
            "    return mod",
            "",
            "",
            f"_before_root = Path(r\"{pair['before_path']}\")",
            f"_after_root  = Path(r\"{pair['after_path']}\")",
            f"_rel = Path(*\"{pair['common']}\".split(\".\")).with_suffix(\".py\")",
            "",
            'before_mod = _load("before_mod", _before_root / _rel)',
            'after_mod  = _load("after_mod",  _after_root  / _rel)',
        ])

    # ── helper to build a readable failure report ──
    lines.extend(["", ""])
    lines.append("def _failure_report(")
    lines.append("    func_name, before_mod_name, after_mod_name,")
    lines.append("    scenario, inputs_dict,")
    lines.append("    before_outcome, after_outcome,")
    lines.append("):")
    lines.append('    """Build a clear, human-readable failure report."""')
    lines.append(f'    sep = "{sep}"')
    lines.append('    parts = [')
    lines.append('        f"\\n{{sep}}",')
    lines.append('        f"  BEHAVIORAL DIFFERENCE DETECTED",')
    lines.append('        f"{{sep}}",')
    lines.append('        f"",')
    lines.append('        f"  Function : {{func_name}}()",')
    lines.append('        f"  Before   : {{before_mod_name}}",')
    lines.append('        f"  After    : {{after_mod_name}}",')
    lines.append('        f"",')
    lines.append('        f"  WHAT HAPPENED:",')
    lines.append('        f"    {{scenario}}",')
    lines.append('        f"",')
    lines.append('        f"  INPUTS:",')
    lines.append('    ]')
    lines.append('    for name, val in inputs_dict.items():')
    lines.append('        val_str = repr(val)')
    lines.append('        if hasattr(val, "__dict__"):')
    lines.append('            attrs = ", ".join(f"{k}={v!r}" for k, v in vars(val).items())')
    lines.append('            val_str = f"object({attrs})"')
    lines.append('        parts.append(f"    {name:>20s} = {val_str}")')
    lines.append('    parts.extend([')
    lines.append('        f"",')
    lines.append('        f"  RESULTS:",')
    lines.append('        f"    {\'BEFORE\':>20s} \u2192 {before_outcome}",')
    lines.append('        f"    {\'AFTER\':>20s} \u2192 {after_outcome}",')
    lines.append('        f"",')
    lines.append(f'        f"{{sep}}",')
    lines.append('    ])')
    lines.append('    return "\\n".join(parts)')
    lines.extend(["", ""])

    for func_name, params in func_analysis.items():
        # ── composite strategies for namespace params ──
        for pname, info in params.items():
            if info["type"] != "namespace":
                continue
            lines.append("@st.composite")
            lines.append(f"def _{func_name}_{pname}(draw):")
            for attr, strat in info["attrs"].items():
                lines.append(f"    {attr} = draw({strat})")
            attrs_kw = ", ".join(f"{a}={a}" for a in info["attrs"])
            lines.append(f"    return types.SimpleNamespace({attrs_kw})")
            lines.extend(["", ""])

        # ── build @given arguments ──
        given_parts: List[str] = []
        for pname, info in params.items():
            if info["type"] == "namespace":
                given_parts.append(f"{pname}=_{func_name}_{pname}()")
            else:
                given_parts.append(f"{pname}={info['strategy']}")

        param_names = ", ".join(params.keys())
        given_str = ", ".join(given_parts)

        # ── test: before and after produce identical results ──
        lines.append("@settings(max_examples=200, deadline=None)")
        lines.append(f"@given({given_str})")
        lines.append(f"def test_{func_name}_before_equals_after({param_names}):")
        lines.append(f'    """before and after versions of {func_name} must behave identically."""')
        lines.append(f"    before_exc = None")
        lines.append(f"    after_exc = None")
        lines.append(f"    before_result = None")
        lines.append(f"    after_result = None")
        lines.append(f"")
        lines.append(f"    try:")
        lines.append(f"        before_result = before_mod.{func_name}({param_names})")
        lines.append(f"    except Exception as e:")
        lines.append(f'        before_exc = f"{{type(e).__name__}}: {{e}}"')
        lines.append(f"")
        lines.append(f"    try:")
        lines.append(f"        after_result = after_mod.{func_name}({param_names})")
        lines.append(f"    except Exception as e:")
        lines.append(f'        after_exc = f"{{type(e).__name__}}: {{e}}"')
        lines.append(f"")

        inputs_dict = "{" + ", ".join(
            f'"{p}": {p}' for p in params.keys()
        ) + "}"

        lines.append(f"    _inputs = {inputs_dict}")
        lines.append(f"")

        # Scenario 1: before errors, after doesn't
        lines.append(f"    if before_exc and not after_exc:")
        lines.append(f"        pytest.fail(_failure_report(")
        lines.append(f'            "{func_name}", "{before_mod_name}", "{after_mod_name}",')
        lines.append(f'            "BEFORE raised an error but AFTER returned normally.",')
        lines.append(f"            _inputs,")
        lines.append(f'            f"ERROR: {{before_exc}}",')
        lines.append(f'            f"returned {{after_result!r}}",')
        lines.append(f"        ))")
        lines.append(f"")

        # Scenario 2: after errors, before doesn't
        lines.append(f"    if after_exc and not before_exc:")
        lines.append(f"        pytest.fail(_failure_report(")
        lines.append(f'            "{func_name}", "{before_mod_name}", "{after_mod_name}",')
        lines.append(f'            "AFTER raised an error but BEFORE returned normally.",')
        lines.append(f"            _inputs,")
        lines.append(f'            f"returned {{before_result!r}}",')
        lines.append(f'            f"ERROR: {{after_exc}}",')
        lines.append(f"        ))")
        lines.append(f"")

        # Scenario 3: both error but different types
        lines.append(f"    if before_exc and after_exc:")
        lines.append(f"        b_type = before_exc.split(':')[0]")
        lines.append(f"        a_type = after_exc.split(':')[0]")
        lines.append(f"        if b_type != a_type:")
        lines.append(f"            pytest.fail(_failure_report(")
        lines.append(f'                "{func_name}", "{before_mod_name}", "{after_mod_name}",')
        lines.append(f'                "Both raised errors but of DIFFERENT types.",')
        lines.append(f"                _inputs,")
        lines.append(f'                f"ERROR: {{before_exc}}",')
        lines.append(f'                f"ERROR: {{after_exc}}",')
        lines.append(f"            ))")
        lines.append(f"        return")
        lines.append(f"")

        # Scenario 4: results differ
        lines.append(f"    if isinstance(before_result, float):")
        lines.append(f"        if before_result != pytest.approx(after_result):")
        lines.append(f"            pytest.fail(_failure_report(")
        lines.append(f'                "{func_name}", "{before_mod_name}", "{after_mod_name}",')
        lines.append(f'                "Both returned a value but the values are DIFFERENT.",')
        lines.append(f"                _inputs,")
        lines.append(f'                f"returned {{before_result!r}}",')
        lines.append(f'                f"returned {{after_result!r}}",')
        lines.append(f"            ))")
        lines.append(f"    else:")
        lines.append(f"        if before_result != after_result:")
        lines.append(f"            pytest.fail(_failure_report(")
        lines.append(f'                "{func_name}", "{before_mod_name}", "{after_mod_name}",')
        lines.append(f'                "Both returned a value but the values are DIFFERENT.",')
        lines.append(f"                _inputs,")
        lines.append(f'                f"returned {{before_result!r}}",')
        lines.append(f'                f"returned {{after_result!r}}",')
        lines.append(f"            ))")
        lines.extend(["", ""])

    return "\n".join(lines)
