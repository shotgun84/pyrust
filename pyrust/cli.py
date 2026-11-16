from __future__ import annotations

import os
import sys

# NOTE: the transpiler (pyrust.builder and friends) is imported lazily. A repeat
# `pyrust run` of an unchanged program only needs the plan cache, so it starts
# in roughly the time Python itself takes to start.


def _build_parser():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="pyrust",
        description="PyRust: run Python at native speed where possible, as plain Python everywhere else.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a .pyr/.py file or project on the fastest tier available")
    # A plain string: the program's sys.argv[0] is the target exactly as typed.
    run_p.add_argument("target", help="Path to .pyr/.py file or project directory")
    run_p.add_argument("--debug", action="store_true", help="Compile in debug mode instead of --release")
    run_p.add_argument(
        "--native-only",
        action="store_true",
        help="Fail instead of falling back to CPython when the program cannot be fully transpiled",
    )
    run_p.add_argument("-q", "--quiet", action="store_true", help="Do not report which tier is used")
    run_p.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the program. One leading `--` is removed "
        "(`pyrust run app.py -- -v` passes `-v`; use `-- --` to pass a literal `--`)",
    )

    check_p = sub.add_parser("check", help="Report how a program will run (native / hybrid / CPython) and why")
    check_p.add_argument("target", type=Path, help="Path to .pyr/.py file or project directory")

    trans_p = sub.add_parser("transpile", help="Transpile a .pyr/.py file or directory into a Cargo project")
    trans_p.add_argument("target", type=Path, help="Path to .pyr/.py file or project directory")
    trans_p.add_argument("-o", "--output", type=Path, default=None, help="Output directory for Cargo crate")
    trans_p.add_argument("--stdout", action="store_true", help="Print generated main.rs to stdout")

    build_p = sub.add_parser("build", help="Transpile and compile a standalone native binary via Cargo")
    build_p.add_argument("target", type=Path, help="Path to .pyr/.py file or project directory")
    build_p.add_argument("-o", "--output", type=Path, default=None, help="Output directory for Cargo crate")
    build_p.add_argument("--debug", action="store_true", help="Compile in debug mode instead of --release")

    mig_p = sub.add_parser("migrate", help="Create or rename .py files in a directory to .pyr")
    mig_p.add_argument("directory", type=Path, help="Directory containing .py files")
    mig_p.add_argument("--rename", action="store_true", help="Rename .py to .pyr in-place instead of copying")

    sub.add_parser("doctor", help="Check Python and Rust toolchain availability")
    return parser


_MAX_NAMES = 6


def describe_fallback(recipe: dict) -> None:
    """One stderr note explaining why a program is not fully native (PYRUST_QUIET silences it)."""
    if os.environ.get("PYRUST_QUIET"):
        return
    if recipe["tier"] == "hybrid":
        names = recipe.get("names") or []
        shown = ", ".join(names[:_MAX_NAMES])
        if len(names) > _MAX_NAMES:
            shown += f", ... ({len(names) - _MAX_NAMES} more)"
        how = f"CPython + {len(names)} native function(s): {shown}"
    elif recipe.get("hybrid_error"):
        how = f"CPython (native functions unavailable: {recipe['hybrid_error']})"
    else:
        how = "CPython (no function qualified for native compilation)"
    print(f"[pyrust] not fully native: {recipe.get('blocker')}", file=sys.stderr)
    print(f"[pyrust] running on {how}", file=sys.stderr)


def _fast_run(argv: list[str], isolate: bool) -> int | None:
    """`pyrust run [-q] target [args...]` straight from the plan cache.

    Returns None when anything is unusual, handing over to the full CLI.
    """
    if len(argv) < 2 or argv[0] != "run":
        return None
    rest = argv[1:]
    quiet = False
    while rest and rest[0] in ("-q", "--quiet"):
        quiet = True
        rest = rest[1:]
    if not rest or rest[0].startswith("-"):
        return None
    target, args = rest[0], rest[1:]
    if args and args[0] == "--":
        args = args[1:]

    from pyrust.plancache import load_plan

    recipe = load_plan(target)
    if recipe is None:
        return None
    return _launch(recipe, target, args, quiet, isolate)


def _program_identity(target: str, script: str | None) -> tuple[str, str, bool]:
    """(sys.argv[0], __file__, is-directory-entry) as `python <target>` would set them."""
    if script and os.path.isdir(target):
        name = os.path.basename(script)
        argv0 = target if os.path.splitext(name)[0] == "__main__" else os.path.join(target, name)
        return argv0, os.path.join(os.path.abspath(target), name), True
    main_file = target if os.path.isabs(target) else os.path.join(os.getcwd(), target)
    return target, main_file, False


def _exec_native(binary: str, argv0: str, args: list[str], isolate: bool) -> int:
    """Run the native program. On POSIX the launcher *becomes* the program (exec), so
    signals, Ctrl-C, exit status and sys.argv[0] are exactly the program's own."""
    if isolate and os.name != "nt":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        try:
            os.execv(binary, [argv0, *args])
        except OSError:
            pass  # e.g. noexec mount: fall through to a child process
    import subprocess

    proc = subprocess.Popen([argv0, *args], executable=binary)
    while True:
        try:
            rc = proc.wait()
            break
        except KeyboardInterrupt:
            continue  # the child got the same Ctrl-C; its exit status decides
    if rc < 0 and isolate and os.name != "nt":
        _die_by_signal(-rc)
    return rc


def _die_by_signal(signum: int) -> None:
    import signal

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    try:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except (OSError, ValueError):
        pass


def _launch(recipe: dict, target: str, args: list[str], quiet: bool, isolate: bool = False) -> int:
    script = recipe.get("script")
    argv0, main_file, directory_entry = _program_identity(str(target), script)
    if recipe["tier"] == "native":
        return _exec_native(recipe["binary"], argv0, args, isolate)
    if not quiet:
        describe_fallback(recipe)
    from pyrust import hybrid_runner

    return hybrid_runner.main(
        [script, recipe.get("manifest_path") or "-", os.path.dirname(script), *args],
        argv0=argv0,
        main_file=main_file,
        directory_entry=directory_entry,
        quiet=quiet,
        isolate=isolate,
    )


def format_error(exc: Exception, command: str | None = None) -> str:
    from pyrust.errors import PyRustError

    if isinstance(exc, PyRustError):
        if exc.filename in (None, "<string>"):
            loc = f"line {exc.lineno}" if exc.lineno is not None else ""
        else:
            loc = exc.filename
            if exc.lineno is not None:
                loc += f":{exc.lineno}" + (f":{exc.col}" if exc.col is not None else "")
        text = f"{loc}: {exc.message}" if loc else exc.message
        if exc.source_line:
            text += f"\n  {exc.source_line.rstrip()}"
            if exc.col is not None and exc.col >= 0:
                text += "\n  " + " " * exc.col + "^"
    else:
        text = str(exc)
    if command in ("build", "transpile"):
        text = text.replace(", so it runs on CPython", "")
        text = text.replace("(Python imports the local file), so it runs on CPython", "(Python imports the local file)")
    return text


def _check(target) -> int:
    from pyrust import builder

    plan = builder.plan_execution(target, build=False)
    if plan.syntax_error:
        print(f"Error: {plan.native_blocker}", file=sys.stderr)
        print("The program is not valid Python, so it cannot run on any tier.", file=sys.stderr)
        return 1
    if plan.tier == "native":
        if plan.native_status == "unbuilt":
            print("Tier: NATIVE (not yet compiled) - the whole program transpiles to Rust.")
            if plan.no_cargo:
                print("  Rust ('cargo') was not found: until it is installed, the program runs on CPython.")
            else:
                print("  The first `pyrust run` compiles it.")
        else:
            print("Tier: NATIVE - the whole program compiles to Rust.")
        return 0
    print(f"Whole-program native compile is blocked: {plan.native_blocker}")
    if plan.tier == "hybrid":
        print(f"Tier: HYBRID - runs on CPython with {len(plan.native_functions)} native function(s):")
        for spec in plan.manifest["functions"]:
            sig = ", ".join(spec["params"])
            print(f"  native  {spec['name']}({sig}) -> {spec['ret']}")
    else:
        print("Tier: PYTHON - runs on CPython at normal speed.")
    for name, reason in sorted(plan.skipped.items()):
        print(f"  python  {name}: {reason}")
    return 0


def _run_command(ns, isolate: bool) -> int:
    from pathlib import Path

    from pyrust import builder

    if ns.command == "doctor":
        info = builder.doctor()
        print("PyRust Toolchain Doctor")
        print("-----------------------")
        print(f"Python version : {info['python_version']} ({'OK' if info['python_ok'] else 'Requires >=3.11'})")
        print(f"Rust compiler  : {info['rustc_version']}")
        print(f"Cargo builder  : {info['cargo_version']}")
        if not info["cargo_available"]:
            print("\nRust is optional: without it every program still runs, on plain CPython.")
            print("Install it to unlock native speed:")
            print(builder.rust_install_hint())
        return 0

    if ns.command == "transpile":
        if ns.stdout and ns.output is None:
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                crate_dir = builder.emit_cargo_project(ns.target, output_dir=Path(tmp) / "crate")
                print((crate_dir / "src" / "main.rs").read_text(encoding="utf-8"))
            return 0
        crate_dir = builder.emit_cargo_project(ns.target, output_dir=ns.output)
        if ns.stdout:
            print((crate_dir / "src" / "main.rs").read_text(encoding="utf-8"))
        else:
            print(f"Transpiled to Cargo project: {crate_dir}")
        return 0

    if ns.command == "build":
        bin_path = builder.build_project(ns.target, output_dir=ns.output, release=not ns.debug)
        print(f"Built native binary: {bin_path}")
        return 0

    if ns.command == "run":
        args = ns.args[1:] if ns.args[:1] == ["--"] else ns.args
        target = ns.target
        if ns.native_only:
            binary = builder.build_project(Path(target), release=not ns.debug)
            script = str(builder.entry_script(Path(target)))
            return _launch({"tier": "native", "binary": str(binary), "script": script}, target, args, True, isolate)
        recipe = builder.resolve_run_plan(Path(target), release=not ns.debug)
        return _launch(recipe, target, args, ns.quiet, isolate)

    if ns.command == "check":
        return _check(ns.target)

    if ns.command == "migrate":
        result = builder.migrate_directory(ns.directory, copy_only=not ns.rename)
        print(f"Migrated {len(result.converted)} file(s) to .pyr")
        if result.skipped:
            print(f"Skipped {len(result.skipped)} file(s):")
            for path, reason in result.skipped:
                print(f"  {path}: {reason}")
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    # Called with no arguments from the `pyrust` command: this process exists only
    # to run the program, so it may exec the native binary and reset sys.modules.
    isolate = argv is None
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        fast = _fast_run(argv, isolate)
        if fast is not None:
            return fast

        ns = _build_parser().parse_args(argv)

        from pyrust.errors import PyRustError

        try:
            return _run_command(ns, isolate)
        except PyRustError as exc:
            print(format_error(exc, ns.command), file=sys.stderr)
            return 1
        except PermissionError as exc:
            print(f"Permission denied: {exc.filename or exc}", file=sys.stderr)
            return 1
    except BrokenPipeError:
        # Output piped into e.g. `head`, which stopped reading: exit quietly.
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 1
    except KeyboardInterrupt:
        # Interrupted while PyRust itself was working (planning, compiling): no traceback.
        if not isolate:
            raise
        if os.name == "nt":
            return 0xC000013A - (1 << 32)  # STATUS_CONTROL_C_EXIT, like python.exe
        import signal

        _die_by_signal(signal.SIGINT)
        return 130


if __name__ == "__main__":
    # `python -m pyrust.cli` puts the current folder first on sys.path; a user's
    # inspect.py or json.py there must not replace the modules PyRust itself
    # imports. (PyRust's own submodules load through the package path.)
    if sys.path and not sys.flags.safe_path and os.path.abspath(sys.path[0] or ".") == os.getcwd():
        sys.path[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__no_modules_here__")
    raise SystemExit(main())
