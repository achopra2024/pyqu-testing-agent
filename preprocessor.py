"""AST-based pre-processor: analyse modules, inject type stubs, post-process tests."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Set

# ── Constants ────────────────────────────────────────────────────────────────

STUB_PREFIX = "_PynguinStub_"

_STRING_METHODS = frozenset({
    "upper", "lower", "strip", "lstrip", "rstrip", "split", "rsplit",
    "join", "replace", "find", "rfind", "index", "rindex", "startswith",
    "endswith", "encode", "decode", "format", "count", "capitalize",
    "title", "swapcase", "isdigit", "isalpha", "isalnum", "zfill",
})

# Registry populated during pre-processing so post-processing can inject
# the correct default attribute values into SimpleNamespace() calls.
#   key  : stub class name  (e.g. "_PynguinStub_CA_data")
#   value: list of (attr_name, default_repr) tuples
_STUB_DEFAULTS: Dict[str, List[tuple]] = {}


# ── Internal helpers ─────────────────────────────────────────────────────────


def _detect_module_imports(tree: ast.AST) -> Dict[str, str]:
    """Return a mapping of local alias -> top-level package for every import."""
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = node.module
    return aliases


def _is_param_attr(node: ast.AST, param_name: str, attr_name: str) -> bool:
    """Check whether *node* is ``param_name.attr_name``."""
    return (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == param_name
            and node.attr == attr_name)


def _infer_attr_default(
    func_node: ast.AST,
    param_name: str,
    attr_name: str,
    import_aliases: Dict[str, str],
) -> str:
    """Infer a reasonable default value for ``param.attr`` from usage context.

    Walks *func_node* looking for how ``param.attr`` is consumed:
      - passed to numpy/scipy/pandas/torch functions -> ``[0.0, 1.0]``
      - has string method calls (``.upper()``, etc.)  -> ``"default"``
      - used in arithmetic with ints/floats            -> ``0.0``
      - used as a boolean / compared to True/False     -> ``True``
      - used as a dict (``.keys()``, ``.items()``)     -> ``{"k": 0}``
      - fallback                                       -> ``[0, 1]``
    """
    array_libs = {"numpy", "np", "scipy", "torch", "tf", "pandas", "pd"}
    resolved_array_aliases = set()
    for alias, pkg in import_aliases.items():
        base = pkg.split(".")[0]
        if base in array_libs or alias in array_libs:
            resolved_array_aliases.add(alias)

    for node in ast.walk(func_node):
        # param.attr passed to a numpy/torch/etc. function
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in resolved_array_aliases:
                    for arg in node.args:
                        if _is_param_attr(arg, param_name, attr_name):
                            return "[0.0, 1.0]"

        # param.attr.<string_method>()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            inner = node.func.value
            if (isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == param_name
                    and inner.attr == attr_name
                    and node.func.attr in _STRING_METHODS):
                return '"default"'

        # param.attr used as dict (.keys/.values/.items/[key])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            inner = node.func.value
            if (isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == param_name
                    and inner.attr == attr_name
                    and node.func.attr in ("keys", "values", "items", "get", "pop")):
                return '{"k": 0}'

        # param.attr in BinOp with a number -> numeric
        if isinstance(node, ast.BinOp):
            for side in (node.left, node.right):
                if _is_param_attr(side, param_name, attr_name):
                    other = node.right if side is node.left else node.left
                    if isinstance(other, ast.Constant) and isinstance(other.value, (int, float)):
                        return "0.0"

        # param.attr compared to True/False/bool -> boolean
        if isinstance(node, ast.Compare):
            if _is_param_attr(node.left, param_name, attr_name):
                for comp in node.comparators:
                    if isinstance(comp, ast.Constant) and isinstance(comp.value, bool):
                        return "True"

    return "[0, 1]"


# ── Public API ───────────────────────────────────────────────────────────────


def analyze_module(source: str) -> Dict[str, Dict[str, Dict[str, set]]]:
    """Parse *source* and detect attribute-access patterns on function params.

    Returns::

        {
            "func_name": {
                "param_name": {
                    "attrs":   {"Y", "shape", ...},
                    "methods": {"transform", ...},
                },
                ...
            },
            ...
        }

    Only parameters that have attribute accesses are included; ordinary
    scalars/lists that Pynguin can already handle are skipped.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    import_aliases = _detect_module_imports(tree)
    result: Dict[str, Dict[str, Dict[str, set]]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        param_names: Set[str] = set()
        for arg in node.args.args:
            if arg.arg not in ("self", "cls"):
                param_names.add(arg.arg)
        for arg in node.args.kwonlyargs:
            param_names.add(arg.arg)

        if not param_names:
            continue

        annotated = {
            arg.arg for arg in node.args.args if arg.annotation is not None
        }
        if param_names <= annotated:
            continue

        info: Dict[str, Dict[str, set]] = {
            p: {"attrs": set(), "methods": set()}
            for p in param_names
            if p not in annotated
        }
        if not info:
            continue

        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if isinstance(child.func.value, ast.Name) and child.func.value.id in info:
                    info[child.func.value.id]["methods"].add(child.func.attr)
            elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                if child.value.id in info:
                    info[child.value.id]["attrs"].add(child.attr)

        for p in info:
            info[p]["attrs"] -= info[p]["methods"]

        for p in info:
            defaults: Dict[str, str] = {}
            for attr in info[p]["attrs"]:
                defaults[attr] = _infer_attr_default(
                    node, p, attr, import_aliases,
                )
            info[p]["attr_defaults"] = defaults

        info = {p: v for p, v in info.items() if v["attrs"] or v["methods"]}
        if info:
            result[node.name] = info

    return result


def generate_stubs(
    analysis: Dict[str, Dict[str, Dict[str, set]]],
    source: str,
) -> str:
    """Generate stub class definitions and ``__annotations__`` assignments.

    The stubs are appended to the end of the module source so that Pynguin
    can discover and construct them.  A mapping of stub-class-name -> default
    attributes is saved in ``_STUB_DEFAULTS`` for post-processing.
    """
    lines: List[str] = [
        "",
        "",
        "# --- Auto-generated Pynguin type stubs (do not edit) ---",
    ]

    for func_name, params in analysis.items():
        for param_name, info in params.items():
            class_name = f"{STUB_PREFIX}{func_name}_{param_name}"
            attrs = sorted(info["attrs"])
            methods = sorted(info["methods"])
            attr_defaults: Dict[str, str] = info.get("attr_defaults", {})

            _STUB_DEFAULTS[class_name] = [
                (a, attr_defaults.get(a, "[0, 1]")) for a in attrs
            ]

            lines.append(f"")
            lines.append(f"class {class_name}:")

            if attrs:
                init_params = ", ".join(f"{a}=None" for a in attrs)
                lines.append(f"    def __init__(self, {init_params}):")
                for a in attrs:
                    default = attr_defaults.get(a, "[0, 1]")
                    lines.append(
                        f"        self.{a} = {a} if {a} is not None else {default}"
                    )
            else:
                lines.append("    def __init__(self):")
                lines.append("        pass")

            for m in methods:
                lines.append(f"    def {m}(self, *args, **kwargs):")
                lines.append(f"        return None")

            lines.append(f"")
            lines.append(
                f"{func_name}.__annotations__['{param_name}'] = {class_name}"
            )

    lines.append("")
    return "\n".join(lines)


def preprocess_file(filepath: Path) -> bool:
    """Analyse a single Python file and inject stubs.  Returns True if modified."""
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    analysis = analyze_module(source)
    if not analysis:
        return False

    stubs = generate_stubs(analysis, source)
    filepath.write_text(source + stubs, encoding="utf-8")

    stub_count = sum(len(p) for p in analysis.values())
    print(f"[preprocess] Injected {stub_count} stub class(es) into {filepath.name}")
    return True


# ── Post-processing (replace stubs with SimpleNamespace) ─────────────────────


def _build_default_kwargs(class_name: str) -> str:
    """Return the keyword arguments string to reproduce a stub's defaults."""
    defaults = _STUB_DEFAULTS.get(class_name, [])
    if not defaults:
        return ""
    return ", ".join(f"{name}={val}" for name, val in defaults)


def _positional_to_keyword(class_name: str, args_text: str) -> str:
    """Convert positional arguments to keyword arguments using the stub schema.

    ``SimpleNamespace`` does not accept positional args, so if Pynguin
    generated ``_Stub(val)`` we rewrite it as ``SimpleNamespace(attr=val)``.
    """
    defaults = _STUB_DEFAULTS.get(class_name, [])
    attr_names = [name for name, _ in defaults]

    parts: List[str] = []
    for raw_arg in args_text.split(","):
        arg = raw_arg.strip()
        if not arg:
            continue
        if "=" in arg:
            parts.append(arg)
        elif attr_names:
            parts.append(f"{attr_names[0]}={arg}")
        else:
            continue

    return ", ".join(parts)


def postprocess_test_file(filepath: Path) -> None:
    """Replace stub-class references in a generated test with SimpleNamespace."""
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    if STUB_PREFIX not in source:
        return

    def _replace_stub_call(match: re.Match) -> str:
        full = match.group(0)
        name_match = re.search(re.escape(STUB_PREFIX) + r"\w+", full)
        if not name_match:
            return "types.SimpleNamespace()"
        class_name = name_match.group(0)
        args_text = match.group("args").strip()

        if not args_text:
            default_kwargs = _build_default_kwargs(class_name)
            return f"types.SimpleNamespace({default_kwargs})"

        converted = _positional_to_keyword(class_name, args_text)
        return f"types.SimpleNamespace({converted})"

    pattern = (
        r"\b\w+\."
        + re.escape(STUB_PREFIX)
        + r"\w+\((?P<args>[^)]*)\)"
    )
    modified = re.sub(pattern, _replace_stub_call, source)

    if "import types" not in modified:
        modified = "import types\n" + modified

    filepath.write_text(modified, encoding="utf-8")
    print(f"[postprocess] Replaced stub references in {filepath.name}")


def clear_stub_defaults() -> None:
    """Clear the global stub-defaults registry between runs."""
    _STUB_DEFAULTS.clear()
