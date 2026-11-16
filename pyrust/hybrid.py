"""Hybrid tier: native functions inside a CPython program.

When a file cannot be transpiled as a whole (it uses a library or a language
feature PyRust has no Rust equivalent for), it still runs on CPython, but its
self-contained numeric functions are compiled to a native library and swapped
in transparently.

Correctness contract: a native function is only ever a *fast path*.

* Only pure functions are compiled (no I/O, no globals, no side effects), so
  re-running one is always safe.
* The wrapper checks the runtime argument types on every call. Anything that
  does not match the compiled signature goes to the original Python function.
* If the native code hits anything Python would treat differently (64-bit
  overflow, division by zero, a domain error, an index error, ...) it reports
  failure and the original Python function runs instead, producing the exact
  Python result or exception.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pyrust.errors import PyRustError
from pyrust.types import BOOL, FLOAT, INT

# Builtins with no side effects whose native implementation matches CPython.
PURE_BUILTINS = {
    "len", "range", "abs", "min", "max", "int", "float", "bool", "str",
    "sorted", "list", "set", "dict", "enumerate", "zip", "round", "divmod",
    "all", "any", "reversed", "ord", "chr",
}
if sys.version_info < (3, 12):
    # 3.12+ uses compensated summation for floats; a naive native sum would
    # differ in the last bits.
    PURE_BUILTINS.add("sum")

_SCALAR_NAMES = {INT: "int", FLOAT: "float", BOOL: "bool"}

_BANNED_NODES: tuple[type, ...] = (
    ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await, ast.Try,
    ast.With, ast.AsyncWith, ast.AsyncFor, ast.Lambda, ast.FunctionDef,
    ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Starred,
    ast.LShift, ast.RShift, ast.MatMult, ast.JoinedStr, ast.GeneratorExp,
)
for _name in ("TryStar", "Match"):
    if hasattr(ast, _name):
        _BANNED_NODES += (getattr(ast, _name),)


@dataclass
class AccelFunction:
    name: str
    params: list[str]
    ret: str


@dataclass
class HybridPlan:
    functions: list[AccelFunction] = field(default_factory=list)
    rust_source: str = ""
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.functions]


# --------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------


def _module_bindings(tree: ast.Module) -> dict[str, int]:
    """How many times each name is bound at module level (or via `global`)."""
    counts: dict[str, int] = {}

    def bump(name: str, n: int = 1) -> None:
        counts[name] = counts.get(name, 0) + n

    def visit(node: ast.AST, top: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if top:
                bump(node.name)
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Global, ast.Nonlocal)):
                    for n in sub.names:
                        bump(n, 2)  # rebindable from elsewhere: never "bound once"
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bump(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bump((alias.asname or alias.name).split(".")[0])
        for child in ast.iter_child_nodes(node):
            visit(child, top)

    for stmt in tree.body:
        visit(stmt, True)
    return counts


def _math_alias_ok(tree: ast.Module, bindings: dict[str, int]) -> bool:
    if bindings.get("math", 0) != 1:
        return False
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "math" and alias.asname is None:
                    return True
    return False


def _module_consts(tree: ast.Module, bindings: dict[str, int]) -> dict[str, ast.Assign]:
    consts: dict[str, ast.Assign] = {}
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and type(stmt.value.value) in (int, float, bool)
            and bindings.get(stmt.targets[0].id) == 1
        ):
            if type(stmt.value.value) is int and not (-(2**63) <= stmt.value.value < 2**63):
                continue
            consts[stmt.targets[0].id] = stmt
    return consts


def _static_reason(fn: ast.FunctionDef, bindings: dict[str, int]) -> str | None:
    if fn.decorator_list:
        return "is decorated"
    a = fn.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        return "uses *args/**kwargs/keyword-only/positional-only parameters"
    if any(not isinstance(d, ast.Constant) for d in a.defaults):
        return "has non-literal default arguments"
    if fn.name == "main":
        return "is the program entry point"
    if bindings.get(fn.name, 0) != 1:
        return "is rebound elsewhere in the module"
    return None


def _body_reason(
    fn: ast.FunctionDef,
    candidates: set[str],
    bindings: dict[str, int],
    consts: dict[str, ast.Assign],
    math_ok: bool,
) -> str | None:
    """Why `fn` cannot be compiled as a pure native function (None = it can)."""
    local_names = {x.arg for x in fn.args.posonlyargs + fn.args.args}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            local_names.add(node.id)

    call_funcs: set[int] = set()
    attr_bases: set[int] = set()
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(node, _BANNED_NODES):
                return f"uses {type(node).__name__} (line {getattr(node, 'lineno', fn.lineno)})"
            if isinstance(node, ast.Call):
                if node.keywords:
                    return f"passes keyword arguments (line {node.lineno})"
                f = node.func
                if isinstance(f, ast.Name):
                    call_funcs.add(id(f))
                    if f.id in local_names:
                        return f"calls a local value '{f.id}'"
                    if f.id in candidates:
                        continue
                    if f.id in bindings:
                        return f"calls '{f.id}', which stays on CPython"
                    if f.id not in PURE_BUILTINS:
                        return f"calls '{f.id}', which is not a pure builtin"
                elif isinstance(f, ast.Attribute):
                    base = f.value
                    if isinstance(base, ast.Name) and base.id not in local_names:
                        if not (base.id == "math" and math_ok):
                            return f"calls '{base.id}.{f.attr}', which stays on CPython"
                else:
                    return f"calls a computed function (line {node.lineno})"
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                attr_bases.add(id(node.value))

    for stmt in fn.body:
        for node in ast.walk(stmt):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            if node.id in local_names or id(node) in call_funcs:
                continue
            if node.id == "math" and math_ok and id(node) in attr_bases:
                continue
            if node.id in consts:
                continue
            return f"reads global '{node.id}'"
    return None


_LOOPS = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp)


def _worth_exporting(candidates: dict[str, ast.FunctionDef]) -> set[str]:
    """Functions with enough work inside to repay the Python<->native bridge.

    Crossing the bridge costs more than a straight-line function saves, so only
    functions that loop, recurse, or call one that does are swapped in. The
    others are still compiled and used by native callers.
    """
    calls: dict[str, set[str]] = {}
    for name, fn in candidates.items():
        calls[name] = {
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in candidates
        }

    def reaches(start: str) -> set[str]:
        seen: set[str] = set()
        stack = list(calls[start])
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(calls[cur])
        return seen

    reach = {name: reaches(name) for name in candidates}
    heavy = {
        name
        for name, fn in candidates.items()
        if name in reach[name] or any(isinstance(n, _LOOPS) for n in ast.walk(fn))
    }
    return {name for name in candidates if name in heavy or reach[name] & heavy}


# --------------------------------------------------------------------------
# Call-site type hints (best effort: the runtime guard makes guesses safe)
# --------------------------------------------------------------------------


class _Hints:
    def __init__(self, tree: ast.Module, candidates: set[str]) -> None:
        self.candidates = candidates
        self.hints: dict[str, list[tuple[str, ...]]] = {}
        self._scan_scope(tree.body, skip_candidates=True)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in candidates and node in tree.body:
                    continue
                self._scan_scope(node.body, skip_candidates=False)

    def _scan_scope(self, body: list[ast.stmt], skip_candidates: bool) -> None:
        nodes: list[ast.AST] = []
        stack: list[ast.AST] = list(body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            nodes.append(node)
            stack.extend(ast.iter_child_nodes(node))

        stores: dict[str, list[ast.AST | str]] = {}
        for node in nodes:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                stores.setdefault(node.targets[0].id, []).append(node.value)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                stores.setdefault(node.target.id, []).append(
                    ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()), op=node.op, right=node.value)
                )
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                it = node.iter
                is_range = isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
                stores.setdefault(node.target.id, []).append("int" if is_range else "?")
        handled = {id(n.targets[0]) for n in nodes if isinstance(n, ast.Assign) and len(n.targets) == 1}
        handled |= {id(n.target) for n in nodes if isinstance(n, (ast.AugAssign, ast.For))}
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in handled:
                stores.setdefault(node.id, []).append("?")

        env: dict[str, str] = {}
        for _ in range(3):
            for name, values in stores.items():
                kinds = {v if isinstance(v, str) else (self._simple(v, env, name) or "?") for v in values}
                kinds.discard("self")
                if len(kinds) == 1 and "?" not in kinds:
                    env[name] = next(iter(kinds))
                else:
                    env.pop(name, None)

        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.candidates:
                if node.keywords:
                    continue
                kinds = [self._simple(a, env, None) for a in node.args]
                if all(k in ("int", "float", "bool") for k in kinds):
                    sig = tuple(k for k in kinds if k)
                    self.hints.setdefault(node.func.id, [])
                    if sig not in self.hints[node.func.id]:
                        self.hints[node.func.id].append(sig)

    def _simple(self, node: ast.AST, env: dict[str, str], self_name: str | None) -> str | None:
        if isinstance(node, ast.Constant):
            return {bool: "bool", int: "int", float: "float"}.get(type(node.value))
        if isinstance(node, ast.Name):
            if node.id == self_name:
                return env.get(node.id, "self")
            return env.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            k = self._simple(node.operand, env, self_name)
            return "int" if k == "bool" else k
        if isinstance(node, ast.BinOp):
            lk = self._simple(node.left, env, self_name)
            rk = self._simple(node.right, env, self_name)
            if lk == "self":
                lk = rk
            if rk == "self":
                rk = lk
            if lk is None or rk is None or lk == "self":
                return None
            if isinstance(node.op, ast.Div):
                return "float"
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
                return "float" if "float" in (lk, rk) else "int"
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id in ("len", "int", "ord") or (node.func.id == "round" and len(node.args) == 1):
                return "int"
            if node.func.id == "float":
                return "float"
        return None


_HINT_LITERAL = {"int": "0", "float": "0.0", "bool": "True"}


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def _unparse_fn(fn: ast.FunctionDef) -> str:
    return ast.unparse(fn)


def plan_hybrid(source: str, filename: str = "<string>") -> HybridPlan:
    """Work out which top-level functions can run natively, and generate their Rust."""
    from pyrust.codegen import generate_rust_module
    from pyrust.inference import _TypeInferer, _positional_params
    from pyrust.ownership import analyze_ownership
    from pyrust.parser import parse_source
    from pyrust.types import ANY, UNKNOWN

    plan = HybridPlan()
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return plan

    bindings = _module_bindings(tree)
    consts = _module_consts(tree, bindings)
    math_ok = _math_alias_ok(tree, bindings)
    all_fns = {s.name: s for s in tree.body if isinstance(s, ast.FunctionDef)}

    candidates: dict[str, ast.FunctionDef] = {}
    for name, fn in all_fns.items():
        reason = _static_reason(fn, bindings)
        if reason:
            plan.skipped[name] = reason
        else:
            candidates[name] = fn

    def prune() -> None:
        changed = True
        while changed:
            changed = False
            for name in list(candidates):
                reason = _body_reason(candidates[name], set(candidates), bindings, consts, math_ok)
                if reason:
                    plan.skipped[name] = reason
                    del candidates[name]
                    changed = True

    prune()
    int_guess: dict[str, tuple[str, ...]] = {}

    for _attempt in range(len(all_fns) * 2 + 4):
        if not candidates:
            return plan
        hints = _Hints(tree, set(candidates)).hints
        lines: list[str] = []
        spans: list[tuple[int, int, str]] = []
        if math_ok:
            lines.append("import math")
        used_consts = [n for n in consts if any(
            isinstance(x, ast.Name) and x.id == n for fn in candidates.values() for x in ast.walk(fn)
        )]
        for n in used_consts:
            lines.append(ast.unparse(consts[n]))
        for name, fn in candidates.items():
            text = _unparse_fn(fn).splitlines()
            spans.append((len(lines) + 1, len(lines) + len(text), name))
            lines.extend(text)
        for name in candidates:
            sigs = list(hints.get(name, []))
            if name in int_guess:
                sigs.append(int_guess[name])
            for sig in sigs:
                lines.append(f"{name}({', '.join(_HINT_LITERAL[k] for k in sig)})")
        synth = "\n".join(lines) + "\n"

        try:
            parsed = parse_source(synth, filename=filename, module_name="accel")
            inferer = _TypeInferer(parsed)
            env = inferer.run()

            # Parameters nothing pinned down: guess int (the runtime guard
            # sends every other type to the original function).
            guessed = False
            for name, fn in candidates.items():
                if name in int_guess:
                    continue
                ft = env.func_types[name]
                if any(p in (UNKNOWN, ANY) for p in ft.params):
                    int_guess[name] = tuple(
                        _SCALAR_NAMES.get(p, "int") for p in ft.params[: len(_positional_params(fn))]
                    )
                    guessed = True
            if guessed:
                continue

            inferer._finalize_unknown_params()
            ownership = analyze_ownership(parsed, env)
            rust = generate_rust_module(parsed, env, ownership, is_entry=False, all_modules=["accel"])
        except PyRustError as exc:
            culprit = next((n for lo, hi, n in spans if exc.lineno and lo <= exc.lineno <= hi), None)
            if culprit is None:
                for name in list(candidates):
                    plan.skipped[name] = f"native compile failed: {exc.message}"
                candidates.clear()
                return plan
            plan.skipped[culprit] = f"not supported natively yet: {exc.message}"
            del candidates[culprit]
            prune()
            continue
        except Exception as exc:  # inference/codegen bug: never take the program down
            for name in list(candidates):
                plan.skipped[name] = f"native compile failed: {type(exc).__name__}: {exc}"
            candidates.clear()
            return plan

        worth = _worth_exporting(candidates)
        for name, fn in candidates.items():
            ft = env.func_types[name]
            n_pos = len(_positional_params(fn))
            params = [_SCALAR_NAMES.get(p) for p in ft.params[:n_pos]]
            ret = _SCALAR_NAMES.get(ft.ret)
            if name not in worth:
                plan.skipped[name] = (
                    "too small to benefit (no loop or recursion); calling into native code would cost more "
                    "than it saves"
                )
                continue
            if ret is None or any(p is None for p in params) or len(ft.params) != n_pos:
                plan.skipped[name] = (
                    "compiled natively as a helper only (arguments/result are not plain int/float/bool)"
                )
                continue
            plan.functions.append(AccelFunction(name, [p for p in params if p], ret))
        plan.rust_source = rust if plan.functions else ""
        return plan
    return plan


# --------------------------------------------------------------------------
# Crate generation and build
# --------------------------------------------------------------------------

_RUST_ARG = {"int": "i64", "float": "f64", "bool": "u8"}
_RUST_OUT = {"int": "i64", "float": "f64", "bool": "u8"}


_RUST_FAIL = {"int": "i64::MIN", "float": "f64::NAN", "bool": "2"}


def _lib_rs(plan: HybridPlan) -> str:
    out = [
        "#![allow(dead_code, unused_imports, unused_variables, unused_mut, non_snake_case)]",
        "mod pyrust_rt;",
        "mod accel;",
        "",
        "thread_local! {",
        "    static PYR_FAILED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };",
        "}",
        "",
        "#[no_mangle]",
        'pub extern "C" fn pyr_init() {',
        "    // Failures are reported to the Python wrapper, which re-runs the",
        "    // original function; keep stderr clean.",
        "    std::panic::set_hook(Box::new(|_| {}));",
        "}",
        "",
        "/// 1 if the last export on this thread bailed out (and clears the flag).",
        "#[no_mangle]",
        'pub extern "C" fn pyr_take_error() -> i32 {',
        "    PYR_FAILED.with(|f| f.replace(false)) as i32",
        "}",
        "",
    ]
    for fn in plan.functions:
        params = [f"a{i}: {_RUST_ARG[t]}" for i, t in enumerate(fn.params)]
        args = [f"a{i} != 0" if t == "bool" else f"a{i}" for i, t in enumerate(fn.params)]
        conv = "v as u8" if fn.ret == "bool" else "v"
        out += [
            "#[no_mangle]",
            f'pub extern "C" fn pyr_export_{fn.name}({", ".join(params)}) -> {_RUST_OUT[fn.ret]} {{',
            f"    match std::panic::catch_unwind(|| accel::{fn.name}({', '.join(args)})) {{",
            f"        Ok(v) => {conv},",
            "        Err(_) => {",
            "            PYR_FAILED.with(|f| f.set(true));",
            f"            {_RUST_FAIL[fn.ret]}",
            "        }",
            "    }",
            "}",
            "",
        ]
    return "\n".join(out)


_CARGO_TOML = """[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[profile.release]
opt-level = 3
lto = "thin"
overflow-checks = true
panic = "unwind"
"""


def _dylib_name(crate: str) -> str:
    if os.name == "nt":
        return f"{crate}.dll"
    if sys.platform == "darwin":
        return f"lib{crate}.dylib"
    return f"lib{crate}.so"


from pyrust.plancache import toolchain_salt  # noqa: E402  (re-exported)


def prepare_hybrid(script: Path, cargo_bin: str | None) -> tuple[dict | None, HybridPlan | None]:
    """Plan + build (cached) the native library for `script`.

    Returns `(manifest, plan)`; manifest is None when nothing runs natively.
    `plan` is None when the result came from the cache.
    """
    from pyrust.runtime_template import RUNTIME_RS

    script = Path(script).resolve()
    source = script.read_text(encoding="utf-8")
    digest = hashlib.sha256((toolchain_salt() + "\0" + source).encode("utf-8")).hexdigest()[:16]
    stem = "".join(c if c.isalnum() else "_" for c in script.stem.lower()) or "app"
    crate = f"pyr_{stem}_accel"
    crate_dir = script.parent / ".pyrust_cache" / f"{stem}_hybrid_{digest}"
    manifest_path = crate_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest["functions"] or Path(manifest["dll"]).exists():
                return (manifest if manifest["functions"] else None), None
        except (OSError, ValueError, KeyError):
            pass

    plan = plan_hybrid(source, str(script))
    manifest: dict = {"dll": "", "functions": [], "skipped": plan.skipped}
    if plan.functions and cargo_bin:
        src = crate_dir / "src"
        src.mkdir(parents=True, exist_ok=True)
        (crate_dir / "Cargo.toml").write_text(_CARGO_TOML.format(name=crate), encoding="utf-8")
        (src / "pyrust_rt.rs").write_text(RUNTIME_RS, encoding="utf-8")
        (src / "accel.rs").write_text(plan.rust_source, encoding="utf-8")
        (src / "lib.rs").write_text(_lib_rs(plan), encoding="utf-8")
        proc = subprocess.run(
            [cargo_bin, "build", "--quiet", "--release"],
            cwd=str(crate_dir), capture_output=True, text=True, check=False,
        )
        dll = crate_dir / "target" / "release" / _dylib_name(crate)
        if proc.returncode == 0 and dll.exists():
            manifest["dll"] = str(dll)
            manifest["functions"] = [
                {"name": f.name, "params": f.params, "ret": f.ret} for f in plan.functions
            ]
        else:
            manifest["error"] = proc.stderr.strip()[-4000:]
            for f in plan.functions:
                plan.skipped[f.name] = "generated Rust failed to compile (PyRust bug); running on CPython"
            plan.functions = []
    elif plan.functions:
        return None, plan  # no cargo: do not cache, it may be installed later
    try:
        crate_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError:
        pass
    return (manifest if manifest["functions"] else None), plan
