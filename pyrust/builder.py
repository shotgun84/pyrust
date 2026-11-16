from __future__ import annotations

import ast
import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pyrust.codegen import generate_rust_module
from pyrust.errors import PyRustError
from pyrust.inference import _TypeInferer, _positional_params, infer_module
from pyrust.ownership import analyze_ownership
from pyrust.parser import IGNORED_IMPORTS, PyRustSyntaxError, find_entry, load_project, project_files
from pyrust.runtime_template import RUNTIME_RS
from pyrust.types import FuncType, unify_types

# Cargo refuses these as binary target names (they collide with folders in
# target/), and Windows cannot create files with the device names.
_RESERVED_BIN_NAMES = {"build", "deps", "examples", "incremental", "con", "prn", "aux", "nul"} | {
    f"{dev}{i}" for dev in ("com", "lpt") for i in range(1, 10)
}


def _sanitize_crate_name(name: str) -> str:
    """A Cargo package/binary name for a script or folder name (Unicode letters are kept)."""
    cleaned = "".join(
        c if c == "_" or (c.isalnum() and f"a{c}".isidentifier()) else "_" for c in name.lower()
    ).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"pyrust_{cleaned or 'app'}"
    if cleaned in _RESERVED_BIN_NAMES:
        cleaned = f"{cleaned}_app"
    return cleaned


def _cargo_toml_template(crate_name: str) -> str:
    return f"""[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"

[profile.release]
opt-level = 3
lto = "thin"
# Python ints never wrap: abort on i64 overflow instead of printing wrong numbers.
overflow-checks = true
"""


from pyrust.plancache import (  # noqa: E402
    cache_dir,
    find_tool,
    load_plan,
    save_plan,
    toolchain_salt as _toolchain_salt,
)

# Marks a folder as a crate PyRust generated, so `-o` may overwrite it.
CRATE_MARKER = ".pyrust-crate"


def _crate_name_for(target_path: Path) -> str:
    target_path = Path(target_path)
    return _sanitize_crate_name(target_path.stem if target_path.is_file() else target_path.name)


def _project_hash(modules: list) -> str:
    digest = hashlib.sha256()
    # Generated code depends on the transpiler itself: never reuse a binary
    # built by a different PyRust (or Rust toolchain).
    digest.update(_toolchain_salt().encode("utf-8"))
    for mod in modules:
        digest.update(mod.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(mod.source.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


# Standard-library modules with a native implementation. Any other import that
# is not one of the project's own modules keeps the program on CPython.
SUPPORTED_STDLIB_MODULES = {"math", "random", "json", "sys", "os", "time", "platform", "statistics", "gc"}


def compute_project_hash(target_path: Path) -> str:
    return _project_hash(load_project(target_path, SUPPORTED_STDLIB_MODULES))


def default_crate_dir(target_path: Path) -> Path:
    """The cache folder holding the crate for `target_path` as it is now."""
    target_path = Path(target_path).resolve()
    base_dir = target_path.parent if target_path.is_file() else target_path
    return Path(cache_dir(str(base_dir))) / f"{_crate_name_for(target_path)}_{compute_project_hash(target_path)}"


def _check_imports_resolvable(modules: list) -> None:
    known = {mod.name for mod in modules} | SUPPORTED_STDLIB_MODULES | IGNORED_IMPORTS
    for mod in modules:
        for imp in mod.imports:
            top = imp.split(".")[0]
            if top not in known and top not in {"main"}:
                raise PyRustError(
                    f"import '{imp}' has no native Rust equivalent, so it runs on CPython",
                    filename=str(mod.path) if mod.path else mod.name,
                )
        for _local, src_mod in getattr(mod, "from_imports", {}).items():
            top = src_mod.split(".")[0]
            if top not in known and top not in {"main"}:
                raise PyRustError(
                    f"import '{src_mod}' has no native Rust equivalent, so it runs on CPython",
                    filename=str(mod.path) if mod.path else mod.name,
                )


def _is_simple_constant(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, bool, str))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
    return False


def _check_import_time_code(modules: list) -> None:
    """Imported modules only contribute definitions to the native crate.

    Code a non-entry module runs at import time (a print, a loop, a computed
    global) would silently vanish, so such programs stay on CPython.
    """
    from pyrust.parser import _is_docstring_stmt, _is_main_guard

    for mod in modules[1:]:
        for idx, stmt in enumerate(mod.tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Pass)):
                continue
            if idx == 0 and _is_docstring_stmt(stmt):
                continue
            if isinstance(stmt, ast.If) and _is_main_guard(stmt):
                continue  # does not run on import
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and _is_simple_constant(stmt.value)
            ) or (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.value is not None
                and _is_simple_constant(stmt.value)
            ):
                continue
            raise PyRustError(
                f"module '{mod.name}' runs code when imported (line {stmt.lineno}), "
                "which only the CPython tier reproduces",
                filename=str(mod.path) if mod.path else mod.name,
            )


def _prepare_output_dir(output_dir: Path) -> Path:
    """Validate a user-chosen `-o` folder: never clobber a crate PyRust did not write."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise PyRustError(f"Output path {output_dir} exists and is not a directory")
    if output_dir.is_dir() and not (output_dir / CRATE_MARKER).exists():
        existing = [n for n in ("Cargo.toml", "src/main.rs") if (output_dir / n).exists()]
        if existing:
            raise PyRustError(
                f"{output_dir} already contains {' and '.join(existing)} that PyRust did not generate; "
                "refusing to overwrite it. Choose a new or empty directory."
            )
    return output_dir


def _write_crate(output_dir: Path, crate_name: str, files: dict[str, str]) -> None:
    try:
        (output_dir / "src").mkdir(parents=True, exist_ok=True)
        (output_dir / CRATE_MARKER).write_text("Generated by PyRust; safe to overwrite.\n", encoding="utf-8")
        (output_dir / "Cargo.toml").write_text(_cargo_toml_template(crate_name), encoding="utf-8")
        (output_dir / "src" / "pyrust_rt.rs").write_text(RUNTIME_RS, encoding="utf-8")
        for name, text in files.items():
            (output_dir / "src" / name).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise PyRustError(f"Cannot write the Rust project to {output_dir}: {exc.strerror or exc}") from exc


def emit_cargo_project(target_path: Path, output_dir: Path | None = None) -> Path:
    """Transpile a single `.pyr`/`.py` script or directory of modules into a standalone Cargo project."""
    target_path = Path(target_path).resolve()
    modules = load_project(target_path, SUPPORTED_STDLIB_MODULES)
    _check_imports_resolvable(modules)
    _check_import_time_code(modules)
    crate_name = _crate_name_for(target_path)

    if output_dir is None:
        base_dir = target_path.parent if target_path.is_file() else target_path
        output_dir = Path(cache_dir(str(base_dir))) / f"{crate_name}_{_project_hash(modules)}"
    else:
        output_dir = _prepare_output_dir(output_dir)

    # Infer all modules and propagate cross-module call argument types
    inferers = {mod.name: _TypeInferer(mod) for mod in modules}
    envs = {name: inf.run() for name, inf in inferers.items()}

    import ast

    def _refresh_all() -> None:
        """Re-run every module's inference with the latest cross-module signatures."""
        for mod in modules:
            fresh_inf = _TypeInferer(mod)
            for imp_name in mod.imports:
                if imp_name in envs:
                    for fn_name, fn_type in envs[imp_name].func_types.items():
                        fresh_inf.env.func_types[f"{imp_name}.{fn_name}"] = fn_type
                    for cls_name, cls_type in envs[imp_name].class_types.items():
                        fresh_inf.env.class_types[f"{imp_name}.{cls_name}"] = cls_type
                    for key, names in envs[imp_name].func_param_names.items():
                        fresh_inf.env.func_param_names[f"{imp_name}.{key}"] = list(names)
                    for key, dflts in envs[imp_name].func_defaults.items():
                        fresh_inf.env.func_defaults[f"{imp_name}.{key}"] = list(dflts)
                    for cname, ctype in envs[imp_name].const_types.items():
                        fresh_inf.env.imported_consts[f"{imp_name}.{cname}"] = ctype
            if mod.name in envs:
                fresh_inf.env.func_types.update(envs[mod.name].func_types)
            envs[mod.name] = fresh_inf.run()
            inferers[mod.name] = fresh_inf

    # Propagate call-site types across modules to a fixed point so that
    # dependency chains (main -> a -> b) fully resolve no matter the order.
    for _round in range(max(2, len(modules))):
        for mod in modules:
            caller_env = envs[mod.name]
            all_expr_types = dict(caller_env.module_scope.expr_types)
            for fscope in caller_env.func_scopes.values():
                all_expr_types.update(fscope.expr_types)
            for node in ast.walk(mod.tree):
                if not isinstance(node, ast.Call):
                    continue
                target_mod: str | None = None
                fn_name = ""
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    target_mod = node.func.value.id
                    fn_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    alias = mod.from_imports.get(node.func.id)
                    if alias is not None:
                        target_mod = alias
                        fn_name = node.func.id
                if target_mod is not None:
                    if target_mod in inferers and fn_name in envs[target_mod].func_types:
                        target_inf = inferers[target_mod]
                        target_env = envs[target_mod]
                        ft = target_env.func_types[fn_name]
                        fn_scope = target_env.func_scopes[fn_name]
                        fn_node = target_inf.func_nodes[fn_name]
                        arg_types = [all_expr_types.get(id(a)) for a in node.args]
                        pos_defs = _positional_params(fn_node)
                        new_params = target_inf._propagate_into_callable(
                            pos_defs,
                            fn_node.args.vararg.arg if fn_node.args.vararg else None,
                            fn_scope,
                            list(ft.params),
                            arg_types,
                        )
                        target_env.func_types[fn_name] = FuncType(
                            params=new_params, ret=ft.ret, has_vararg=ft.has_vararg
                        )
                        target_inf._infer_function(fn_node)
        _refresh_all()

    # Finalize: any parameter never constrained by calls, annotations,
    # defaults, or body usage becomes dynamic (compilable dead code).
    for mod in modules:
        inferers[mod.name]._finalize_unknown_params()

    all_names = [mod.name for mod in modules]
    files: dict[str, str] = {}
    for idx, mod in enumerate(modules):
        is_entry = idx == 0
        env = envs[mod.name]
        ownership = analyze_ownership(mod, env)
        rust_code = generate_rust_module(mod, env, ownership, is_entry=is_entry, all_modules=all_names)
        if is_entry:
            # Generated code is not the user's to lint: keep cargo's output to real errors.
            rust_code = "#![allow(warnings)]\n" + rust_code
        files["main.rs" if is_entry else f"{mod.name}.rs"] = rust_code

    _write_crate(output_dir, crate_name, files)
    return output_dir


def find_cargo_executable() -> str | None:
    return find_tool("cargo")


def find_rustc_executable() -> str | None:
    return find_tool("rustc")


def doctor() -> dict[str, bool | str]:
    cargo_bin = find_cargo_executable()
    rustc_bin = find_rustc_executable()
    rustc_version = "not found"
    cargo_version = "not found"

    if rustc_bin:
        try:
            proc = subprocess.run([rustc_bin, "--version"], capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                rustc_version = proc.stdout.strip()
        except OSError:
            pass

    if cargo_bin:
        try:
            proc = subprocess.run([cargo_bin, "--version"], capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                cargo_version = proc.stdout.strip()
        except OSError:
            pass

    return {
        "python_version": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 11),
        "cargo_available": cargo_bin is not None,
        "rustc_available": rustc_bin is not None,
        "cargo_version": cargo_version,
        "rustc_version": rustc_version,
    }


def rust_install_hint() -> str:
    if os.name == "nt":
        return "    winget install Rustlang.Rustup\n  or visit https://rustup.rs"
    return "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n  or visit https://rustup.rs"


def _environment_error(message: str) -> PyRustError:
    """A failure caused by the machine (no cargo, killed build, linker missing...), not by the program.

    Such failures are never remembered: the next run tries again.
    """
    exc = PyRustError(message)
    exc.transient = True  # type: ignore[attr-defined]
    return exc


# A rustc diagnostic located in the generated sources: the generated code itself
# is wrong, so building it again with the same toolchain cannot succeed.
_GENUINE_ERROR = re.compile(r"^error(\[E\d+\])?: .*\n\s*--> src[/\\]", re.M)


def _build_error_summary(stderr: str) -> str:
    lines = [ln.strip() for ln in stderr.splitlines()]
    for pattern in (r"^error\[E\d+\]: ", r"^error: (?!could not compile|aborting due to)"):
        for line in lines:
            if re.match(pattern, line):
                return line
    return next((ln for ln in lines if ln), "cargo exited with an error")


def _failure_marker(crate_dir: Path, release: bool) -> Path:
    # The crate folder name already includes the PyRust + toolchain salt, so a
    # marker never outlives a Rust upgrade.
    return crate_dir / f"build_failed_{'release' if release else 'debug'}.txt"


def _bin_path(crate_dir: Path, crate_name: str, release: bool) -> Path:
    exe_name = f"{crate_name}.exe" if os.name == "nt" else crate_name
    return crate_dir / "target" / ("release" if release else "debug") / exe_name


def native_build_status(target_path: Path, release: bool = True) -> tuple[str, str | None]:
    """('built', binary) | ('failed', summary) | ('unbuilt', None) for the cached native build."""
    crate_dir = default_crate_dir(target_path)
    binary = _bin_path(crate_dir, _crate_name_for(Path(target_path).resolve()), release)
    if binary.exists():
        return "built", str(binary)
    marker = _failure_marker(crate_dir, release)
    if marker.exists():
        text = marker.read_text(encoding="utf-8", errors="replace")
        return "failed", _build_error_summary(text)
    return "unbuilt", None


def build_project(
    target_path: Path,
    output_dir: Path | None = None,
    release: bool = True,
) -> Path:
    target_path = Path(target_path).resolve()
    crate_dir = emit_cargo_project(target_path, output_dir=output_dir)
    cargo_bin = find_cargo_executable()
    if not cargo_bin:
        exc = PyRustError(
            "Rust toolchain ('cargo') not found on PATH.\n"
            f"  Your Rust project was successfully generated at: {crate_dir}\n"
            "  To compile and run native binaries, install Rust with:\n" + rust_install_hint()
        )
        exc.no_cargo = True  # type: ignore[attr-defined]
        raise exc

    bin_path = _bin_path(crate_dir, _crate_name_for(target_path), release)
    if bin_path.exists() and output_dir is None:
        return bin_path

    # A failed build of this exact source (and toolchain) is remembered so that
    # falling back to CPython does not pay for a doomed `cargo build` on every run.
    failed_marker = _failure_marker(crate_dir, release)
    if failed_marker.exists() and output_dir is None:
        text = failed_marker.read_text(encoding="utf-8", errors="replace")
        raise PyRustError(f"Cargo build failed: {_build_error_summary(text)}\n{text}")

    cmd = [cargo_bin, "build", "--quiet"]
    if release:
        cmd.append("--release")

    try:
        proc = subprocess.run(
            cmd, cwd=str(crate_dir), capture_output=True, text=True, errors="replace", check=False
        )
    except OSError as exc:
        raise _environment_error(f"Cargo could not be started: {exc}") from exc
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        summary = _build_error_summary(stderr)
        if proc.returncode < 0:
            raise _environment_error(f"Cargo build was killed (signal {-proc.returncode})")
        if not _GENUINE_ERROR.search(stderr):
            raise _environment_error(f"Cargo build failed: {summary}\n{stderr}")
        try:
            failed_marker.write_text(stderr, encoding="utf-8")
        except OSError:
            pass
        raise PyRustError(f"Cargo build failed: {summary}\n{stderr}")
    if not bin_path.exists():
        raise _environment_error(f"Cargo reported success but {bin_path} is missing")
    return bin_path


def entry_script(target_path: Path) -> Path:
    """The file CPython should execute for a script or project directory."""
    target_path = Path(target_path).resolve()
    if target_path.is_file():
        return target_path
    return find_entry(target_path)


_entry_script = entry_script  # backwards-compatible name


_RUNNER_BOOT = (
    "import sys, importlib.util as u;"
    "s = u.spec_from_file_location('_pyrust_hybrid_runner', sys.argv[1]);"
    "m = u.module_from_spec(s); s.loader.exec_module(m);"
    "sys.exit(m.main(sys.argv[2:]))"
)


def run_on_cpython(script: Path, args: list[str] | None = None, manifest: Path | None = None) -> int:
    """Run `script` on CPython in a fresh interpreter (with native functions if `manifest`)."""
    runner = Path(__file__).resolve().parent / "hybrid_runner.py"
    cmd = [
        sys.executable,
        "-c",
        _RUNNER_BOOT,
        str(runner),
        str(script),
        str(manifest) if manifest else "-",
        str(Path(script).parent),
        *(args or []),
    ]
    return run_child(cmd)


def run_child(cmd: list[str]) -> int:
    """Run a child process to completion; Ctrl-C reaches it directly, so just keep waiting."""
    proc = subprocess.Popen(cmd)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            continue


def _first_line(exc: Exception) -> str:
    text = exc.message if isinstance(exc, PyRustError) else str(exc)
    line = text.strip().splitlines()[0] if text.strip() else type(exc).__name__
    if isinstance(exc, PyRustError) and exc.lineno is not None:
        line = f"line {exc.lineno}: {line}"
    return line if len(line) <= 140 else line[:137] + "..."


@dataclasses.dataclass
class ExecutionPlan:
    """How a target will actually run. `tier` is 'native', 'hybrid', or 'python'."""

    tier: str
    target: Path
    binary: Path | None = None
    script: Path | None = None
    manifest: dict | None = None
    manifest_path: Path | None = None
    native_blocker: str | None = None
    skipped: dict[str, str] = dataclasses.field(default_factory=dict)
    # 'built' | 'unbuilt' for a native plan made without building (pyrust check).
    native_status: str | None = None
    # False when the blocker is a passing environment problem: do not cache this plan.
    cacheable: bool = True
    # The entry module is not valid Python.
    syntax_error: bool = False
    # Cargo is not installed (a native-capable program would run on CPython).
    no_cargo: bool = False
    hybrid_error: str | None = None
    crate_dir: Path | None = None

    @property
    def native_functions(self) -> list[str]:
        return [f["name"] for f in (self.manifest or {}).get("functions", [])]


def plan_execution(target_path: Path, release: bool = True, build: bool = True) -> ExecutionPlan:
    """Pick the fastest tier that runs `target_path` with exact Python behaviour.

    1. native  - the whole program transpiles to Rust.
    2. hybrid  - CPython runs the program; its pure numeric functions are native.
    3. python  - plain CPython.
    """
    from pyrust.hybrid import prepare_hybrid

    target_path = Path(target_path).resolve()
    script = entry_script(target_path) if target_path.exists() else None
    blocker: str
    cacheable = True
    syntax_error = no_cargo = False
    crate_dir: Path | None = None
    try:
        if build:
            binary = build_project(target_path, release=release)
            return ExecutionPlan("native", target_path, binary=binary, script=script)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            emit_cargo_project(target_path, output_dir=Path(tmp))
        status, detail = native_build_status(target_path, release)
        if status == "failed":
            raise PyRustError(f"Cargo build failed: {detail}")
        plan = ExecutionPlan("native", target_path, script=script, native_status=status)
        plan.no_cargo = status == "unbuilt" and find_cargo_executable() is None
        if status == "built":
            plan.binary = Path(detail)
        return plan
    except PyRustError as exc:
        if not target_path.exists():
            raise
        blocker = _first_line(exc)
        cacheable = not getattr(exc, "transient", False)
        no_cargo = bool(getattr(exc, "no_cargo", False))
        syntax_error = isinstance(exc, PyRustSyntaxError) and exc.filename == str(script)
        if blocker.startswith("Cargo build failed"):
            try:
                crate_dir = default_crate_dir(target_path)
            except Exception:  # noqa: BLE001
                crate_dir = None
    except RecursionError:
        blocker = "program is too deeply nested for the transpiler"
    except Exception as exc:  # transpiler bug: the program must still run
        blocker = f"internal transpiler error ({type(exc).__name__}: {exc})"

    if no_cargo:
        blocker = "Rust toolchain ('cargo') not found, so the native build is skipped"
    plan = ExecutionPlan(
        "python",
        target_path,
        script=script,
        native_blocker=blocker,
        cacheable=cacheable,
        syntax_error=syntax_error,
        no_cargo=no_cargo,
        crate_dir=crate_dir,
    )
    if syntax_error:
        return plan
    try:
        manifest, hplan = prepare_hybrid(script, find_cargo_executable() if build else None)
    except Exception as exc:  # never let the accelerator break a runnable program
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        plan.hybrid_error = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
        plan.skipped["*"] = f"hybrid planning failed ({plan.hybrid_error})"
        return plan
    if hplan is not None:
        plan.skipped = dict(hplan.skipped)
        if not build and hplan.functions:
            manifest = {"functions": [{"name": f.name, "params": f.params, "ret": f.ret} for f in hplan.functions]}
    elif manifest is not None:
        plan.skipped = dict(manifest.get("skipped", {}))
    if manifest is not None:
        plan.tier = "hybrid"
        plan.manifest = manifest
        if manifest.get("dll"):
            plan.manifest_path = Path(manifest["dll"]).parents[2] / "manifest.json"
    return plan


def _recipe_from_plan(plan: ExecutionPlan) -> dict:
    dll = (plan.manifest or {}).get("dll") or None
    return {
        "tier": plan.tier,
        "binary": str(plan.binary) if plan.binary else None,
        "script": str(plan.script) if plan.script else None,
        "manifest_path": str(plan.manifest_path) if plan.manifest_path else None,
        # Checked by load_plan: a deleted native library must trigger a rebuild.
        "dll": str(dll) if dll and plan.manifest_path else None,
        "blocker": plan.native_blocker,
        "names": plan.native_functions,
        "crate_dir": str(plan.crate_dir) if plan.crate_dir else None,
        "hybrid_error": plan.hybrid_error,
    }


def resolve_run_plan(target_path: Path, release: bool = True) -> dict:
    """The (cached) launch recipe for `target_path` as a plain dict.

    Keys: tier, binary, script, manifest_path, dll, blocker, names, ...
    """
    cached = load_plan(str(target_path), release)
    if cached is not None:
        return cached
    plan = plan_execution(target_path, release=release)
    recipe = _recipe_from_plan(plan)
    if not plan.cacheable:
        return recipe
    deps, absent = project_files(Path(target_path))
    from pyrust.plancache import file_digest

    recipe["deps"] = [[str(p), file_digest(str(p))] for p in deps]
    recipe["absent"] = [str(p) for p in absent]
    plan_file = save_plan(str(target_path), recipe, release)
    if plan_file is not None:
        prune_cache(Path(target_path).resolve(), Path(plan_file), recipe)
    return recipe


_HEX16 = "[0-9a-f]{16}"


def prune_cache(target_path: Path, plan_file: Path, recipe: dict) -> None:
    """Delete this program's superseded builds and plans (each edit would otherwise leave ~1 MB behind)."""
    folder = plan_file.parent
    keep: set[Path] = {plan_file}
    for key, depth in (("binary", 3), ("manifest_path", 1), ("crate_dir", 0)):
        value = recipe.get(key)
        if value:
            keep.add(Path(value).parents[depth - 1] if depth else Path(value))
    crate = re.escape(_crate_name_for(target_path))
    script = Path(recipe.get("script") or target_path)
    hstem = re.escape("".join(c if c.isalnum() else "_" for c in script.stem.lower()) or "app")
    stem = re.escape(plan_file.name.rsplit("_", 1)[0])  # plan_<name>
    patterns = [
        re.compile(rf"{crate}_{_HEX16}"),
        re.compile(rf"{hstem}_hybrid_{_HEX16}"),
        re.compile(rf"{stem}_{_HEX16}\.bin"),
    ]
    try:
        entries = list(folder.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry in keep or not any(p.fullmatch(entry.name) for p in patterns):
            continue
        if entry.is_dir():
            owner = entry / ".pyrust-target"
            # Only folders this exact program created (two scripts may share a stem).
            if owner.exists() and owner.read_text(encoding="utf-8", errors="replace") == str(target_path):
                shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except OSError:
                pass
    for kept in keep:
        if kept.is_dir() and kept.parent == folder:
            try:
                (kept / ".pyrust-target").write_text(str(target_path), encoding="utf-8")
            except OSError:
                pass


def run_project(
    target_path: Path,
    args: list[str] | None = None,
    release: bool = True,
    native_only: bool = False,
    quiet: bool = False,
) -> int:
    if native_only:
        bin_path = build_project(target_path, release=release)
        return run_child([str(bin_path), *(args or [])])

    from pyrust.cli import describe_fallback

    recipe = resolve_run_plan(target_path, release=release)
    if recipe["tier"] == "native":
        return run_child([recipe["binary"], *(args or [])])
    if not quiet:
        describe_fallback(recipe)
    manifest = Path(recipe["manifest_path"]) if recipe["manifest_path"] else None
    return run_on_cpython(Path(recipe["script"]), args, manifest)


# Folders that hold environments, dependencies, VCS data or caches, never the user's own sources.
_MIGRATE_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", ".tox", ".nox", "__pycache__", ".pyrust_cache",
    "site-packages", "dist-packages", "node_modules", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".eggs", "pyrust_vendor",
}


def _is_environment_dir(path: Path) -> bool:
    return (
        path.name in _MIGRATE_SKIP_DIRS
        or path.name.endswith(".egg-info")
        or (path / "pyvenv.cfg").exists()  # any virtual environment, whatever its name
    )


@dataclasses.dataclass
class MigrationResult:
    converted: list[Path]
    skipped: list[tuple[Path, str]]  # (file, reason): left untouched


def migrate_directory_to_pyr(directory: Path, copy_only: bool = True) -> list[Path]:
    """Convert `.py` files in a directory to `.pyr` files for drag-and-drop migration."""
    return migrate_directory(directory, copy_only=copy_only).converted


def migrate_directory(directory: Path, copy_only: bool = True) -> MigrationResult:
    """Copy (or rename) each `.py` under `directory` to `.pyr`.

    Existing `.pyr` files are never overwritten, environment/VCS/cache folders are
    skipped, and `--rename` leaves packages (folders with `__init__.py`) alone,
    since `.pyr` modules inside a package cannot be imported by plain Python.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        what = "is not a directory" if directory.exists() else "does not exist"
        raise PyRustError(f"Cannot migrate {directory}: it {what}")
    result = MigrationResult([], [])
    for root, dirs, files in os.walk(directory):
        root_path = Path(root)
        dirs[:] = sorted(d for d in dirs if not _is_environment_dir(root_path / d))
        in_package = (root_path / "__init__.py").exists()
        for name in sorted(files):
            if not name.endswith(".py") or name == "__init__.py":
                continue
            py_file = root_path / name
            pyr_file = py_file.with_suffix(".pyr")
            if pyr_file.exists():
                result.skipped.append((py_file, f"{pyr_file.name} already exists"))
                continue
            if not copy_only and in_package:
                result.skipped.append((py_file, "inside a package; renaming would break `import`"))
                continue
            if copy_only:
                shutil.copy2(py_file, pyr_file)
            else:
                py_file.rename(pyr_file)
            result.converted.append(pyr_file)
    return result
