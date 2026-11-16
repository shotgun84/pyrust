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
    run_p.add_argument("target", type=Path, help="Path to .pyr/.py file or project directory")
    run_p.add_argument("--debug", action="store_true", help="Compile in debug mode instead of --release")
    run_p.add_argument(
        "--native-only",
        action="store_true",
        help="Fail instead of falling back to CPython when the program cannot be fully transpiled",
    )
    run_p.add_argument("-q", "--quiet", action="store_true", help="Do not report which tier is used")
    run_p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the program")

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


def describe_fallback(recipe: dict) -> None:
    """One stderr note explaining why a program is not fully native (PYRUST_QUIET silences it)."""
    if os.environ.get("PYRUST_QUIET"):
        return
    if recipe["tier"] == "hybrid":
        names = recipe.get("names") or []
        how = f"CPython + {len(names)} native function(s): {', '.join(names)}"
    else:
        how = "CPython (no function qualified for native compilation)"
    print(f"[pyrust] not fully native: {recipe.get('blocker')}", file=sys.stderr)
    print(f"[pyrust] running on {how}", file=sys.stderr)


def _fast_run(argv: list[str]) -> int | None:
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
    return _launch(recipe, args, quiet)


def _launch(recipe: dict, args: list[str], quiet: bool) -> int:
    if recipe["tier"] == "native":
        import subprocess

        return subprocess.run([recipe["binary"], *args], check=False).returncode
    if not quiet:
        describe_fallback(recipe)
    from pyrust import hybrid_runner

    script = recipe["script"]
    return hybrid_runner.main([script, recipe.get("manifest_path") or "-", os.path.dirname(script), *args])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fast = _fast_run(argv)
    if fast is not None:
        return fast

    ns = _build_parser().parse_args(argv)

    from pyrust.errors import PyRustError

    try:
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
                print("Install it to unlock native speed:  https://rustup.rs")
            return 0

        if ns.command == "transpile":
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
            if ns.native_only:
                return builder.run_project(ns.target, args=args, release=not ns.debug, native_only=True)
            recipe = builder.resolve_run_plan(ns.target, release=not ns.debug)
            return _launch(recipe, args, ns.quiet)

        if ns.command == "check":
            plan = builder.plan_execution(ns.target, build=False)
            if plan.tier == "native":
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

        if ns.command == "migrate":
            converted = builder.migrate_directory_to_pyr(ns.directory, copy_only=not ns.rename)
            print(f"Migrated {len(converted)} file(s) to .pyr")
            return 0

    except PyRustError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
