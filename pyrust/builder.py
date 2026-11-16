from __future__ import annotations

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
from pyrust.parser import IGNORED_IMPORTS, load_project
from pyrust.runtime_template import RUNTIME_RS
from pyrust.types import FuncType, unify_types


def _sanitize_crate_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"pyrust_{cleaned or 'app'}"
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


from pyrust.plancache import load_plan, save_plan, toolchain_salt as _toolchain_salt  # noqa: E402


def compute_project_hash(target_path: Path) -> str:
    modules = load_project(target_path)
    digest = hashlib.sha256()
    # Generated code depends on the transpiler itself: never reuse a binary
    # built by a different PyRust.
    digest.update(_toolchain_salt().encode("utf-8"))
    for mod in modules:
        digest.update(mod.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(mod.source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


# Third-party/C modules PyRust cannot transpile: importing any of these is a
# hard error at transpile time (honest instead of broken Rust).
SUPPORTED_STDLIB_MODULES = {"math", "random", "json", "sys", "os", "time", "platform", "statistics", "gc"}


def _check_imports_resolvable(modules: list) -> None:
    known = {mod.name for mod in modules} | SUPPORTED_STDLIB_MODULES | IGNORED_IMPORTS
    for mod in modules:
        for imp in mod.imports:
            top = imp.split(".")[0]
            if top not in known and top not in {"main"}:
                from_imports = getattr(mod, "from_imports", {})
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


def emit_cargo_project(target_path: Path, output_dir: Path | None = None) -> Path:
    """Transpile a single `.pyr`/`.py` script or directory of modules into a standalone Cargo project."""
    target_path = Path(target_path).resolve()
    modules = load_project(target_path)
    _check_imports_resolvable(modules)
    crate_name = _sanitize_crate_name(target_path.stem if target_path.is_file() else target_path.name)

    if output_dir is None:
        base_dir = target_path.parent if target_path.is_file() else target_path
        project_hash = compute_project_hash(target_path)
        output_dir = base_dir / ".pyrust_cache" / f"{crate_name}_{project_hash}"
    else:
        output_dir = Path(output_dir).resolve()

    src_dir = output_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "Cargo.toml").write_text(_cargo_toml_template(crate_name), encoding="utf-8")
    (src_dir / "pyrust_rt.rs").write_text(RUNTIME_RS, encoding="utf-8")

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
    for idx, mod in enumerate(modules):
        is_entry = idx == 0
        env = envs[mod.name]
        ownership = analyze_ownership(mod, env)
        rust_code = generate_rust_module(mod, env, ownership, is_entry=is_entry, all_modules=all_names)
        out_file = src_dir / ("main.rs" if is_entry else f"{mod.name}.rs")
        out_file.write_text(rust_code, encoding="utf-8")

    return output_dir


def find_cargo_executable() -> str | None:
    found = shutil.which("cargo")
    if found:
        return found
    cargo_home_bin = Path.home() / ".cargo" / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
    if cargo_home_bin.exists():
        return str(cargo_home_bin)
    return None


def find_rustc_executable() -> str | None:
    found = shutil.which("rustc")
    if found:
        return found
    cargo_home_bin = Path.home() / ".cargo" / "bin" / ("rustc.exe" if os.name == "nt" else "rustc")
    if cargo_home_bin.exists():
        return str(cargo_home_bin)
    return None


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


def build_project(
    target_path: Path,
    output_dir: Path | None = None,
    release: bool = True,
) -> Path:
    crate_dir = emit_cargo_project(target_path, output_dir=output_dir)
    cargo_bin = find_cargo_executable()
    if not cargo_bin:
        raise PyRustError(
            "Rust toolchain ('cargo') not found on PATH.\n"
            f"  Your Rust project was successfully generated at: {crate_dir}\n"
            "  To compile and run native binaries, install Rust with:\n"
            "    winget install Rustlang.Rustup\n"
            "  or visit https://rustup.rs"
        )

    crate_name = _sanitize_crate_name(
        Path(target_path).stem if Path(target_path).is_file() else Path(target_path).name
    )
    profile_dir = "release" if release else "debug"
    exe_name = f"{crate_name}.exe" if os.name == "nt" else crate_name
    bin_path = crate_dir / "target" / profile_dir / exe_name

    if bin_path.exists() and output_dir is None:
        return bin_path

    # A failed build of this exact source is remembered so that falling back
    # to CPython does not pay for a doomed `cargo build` on every run.
    failed_marker = crate_dir / f"build_failed_{profile_dir}.txt"
    if failed_marker.exists() and output_dir is None:
        raise PyRustError(
            "Cargo build failed:\n" + failed_marker.read_text(encoding="utf-8", errors="replace")
        )

    cmd = [cargo_bin, "build", "--quiet"]
    if release:
        cmd.append("--release")

    proc = subprocess.run(cmd, cwd=str(crate_dir), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        try:
            failed_marker.write_text(proc.stderr.strip(), encoding="utf-8")
        except OSError:
            pass
        raise PyRustError(f"Cargo build failed:\n{proc.stderr.strip()}")

    return bin_path


def _entry_script(target_path: Path) -> Path:
    """The file CPython should execute for a script or project directory."""
    target_path = Path(target_path).resolve()
    if target_path.is_file():
        return target_path
    modules = load_project(target_path)
    if modules[0].path is None:
        raise PyRustError(f"No entry module found in {target_path}")
    return modules[0].path


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
    return subprocess.run(cmd, check=False).returncode


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
    blocker: str
    try:
        if build:
            return ExecutionPlan("native", target_path, binary=build_project(target_path, release=release))
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            emit_cargo_project(target_path, output_dir=Path(tmp))
        return ExecutionPlan("native", target_path)
    except PyRustError as exc:
        if not target_path.exists():
            raise
        blocker = _first_line(exc)
    except RecursionError:
        blocker = "program is too deeply nested for the transpiler"
    except Exception as exc:  # transpiler bug: the program must still run
        blocker = f"internal transpiler error ({type(exc).__name__}: {exc})"

    script = _entry_script(target_path)
    plan = ExecutionPlan("python", target_path, script=script, native_blocker=blocker)
    try:
        manifest, hplan = prepare_hybrid(script, find_cargo_executable() if build else None)
    except Exception as exc:  # never let the accelerator break a runnable program
        plan.skipped["*"] = f"hybrid planning failed ({type(exc).__name__}: {exc})"
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


def resolve_run_plan(target_path: Path, release: bool = True) -> dict:
    """The (cached) launch recipe for `target_path` as a plain dict.

    Keys: tier, binary, script, manifest_path, blocker, names.
    """
    cached = load_plan(str(target_path), release)
    if cached is not None:
        return cached
    plan = plan_execution(target_path, release=release)
    recipe = {
        "tier": plan.tier,
        "binary": str(plan.binary) if plan.binary else None,
        "script": str(plan.script) if plan.script else None,
        "manifest_path": str(plan.manifest_path) if plan.manifest_path else None,
        "blocker": plan.native_blocker,
        "names": plan.native_functions,
    }
    # Without cargo the answer may change once Rust is installed: do not pin it.
    if plan.tier == "native" or find_cargo_executable() is not None:
        save_plan(str(target_path), recipe, release)
    return recipe


def run_project(
    target_path: Path,
    args: list[str] | None = None,
    release: bool = True,
    native_only: bool = False,
    quiet: bool = False,
) -> int:
    if native_only:
        bin_path = build_project(target_path, release=release)
        return subprocess.run([str(bin_path), *(args or [])], check=False).returncode

    from pyrust.cli import describe_fallback

    recipe = resolve_run_plan(target_path, release=release)
    if recipe["tier"] == "native":
        return subprocess.run([recipe["binary"], *(args or [])], check=False).returncode
    if not quiet:
        describe_fallback(recipe)
    manifest = Path(recipe["manifest_path"]) if recipe["manifest_path"] else None
    return run_on_cpython(Path(recipe["script"]), args, manifest)


def migrate_directory_to_pyr(directory: Path, copy_only: bool = True) -> list[Path]:
    """Convert `.py` files in a directory to `.pyr` files for drag-and-drop migration."""
    directory = Path(directory).resolve()
    converted: list[Path] = []
    for py_file in sorted(directory.rglob("*.py")):
        if py_file.name == "__init__.py" or ".pyrust_cache" in py_file.parts:
            continue
        pyr_file = py_file.with_suffix(".pyr")
        if copy_only:
            shutil.copy2(py_file, pyr_file)
        else:
            py_file.rename(pyr_file)
        converted.append(pyr_file)
    return converted
