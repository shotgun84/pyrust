from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from pyrust.errors import PyRustError

SUPPORTED_EXTENSIONS = (".pyr", ".py")

# Imports that need no Rust module: pure-typing/decorator helpers whose
# semantics PyRust implements natively (or ignores safely).
IGNORED_IMPORTS = {"typing", "__future__", "dataclasses", "abc"}


class PyRustSyntaxError(PyRustError):
    """The source is not valid Python (so CPython would reject it too)."""


@dataclass
class ParsedModule:
    name: str
    path: Path | None
    source: str
    tree: ast.Module
    functions: list[ast.FunctionDef] = field(default_factory=list)
    classes: list[ast.ClassDef] = field(default_factory=list)
    main_body: list[ast.stmt] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    from_imports: dict[str, str] = field(default_factory=dict)
    filename: str = "<string>"


def _is_main_guard(node: ast.If) -> bool:
    """Detect `if __name__ == "__main__":` or `if "__main__" == __name__:`."""
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    left, right = test.left, test.comparators[0]
    if (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ):
        return True
    if (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    ):
        return True
    return False


def _is_docstring_stmt(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def parse_source(
    source: str,
    filename: str = "<string>",
    module_name: str = "main",
    path: Path | None = None,
) -> ParsedModule:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        lines = source.splitlines()
        line_text = lines[exc.lineno - 1] if exc.lineno and 1 <= exc.lineno <= len(lines) else None
        raise PyRustSyntaxError(
            f"Syntax error: {exc.msg}",
            lineno=exc.lineno,
            col=(exc.offset - 1) if exc.offset else None,
            filename=filename,
            source_line=line_text,
        ) from exc

    functions: list[ast.FunctionDef] = []
    classes: list[ast.ClassDef] = []
    main_body: list[ast.stmt] = []
    imports: list[str] = []
    from_imports: dict[str, str] = {}

    for idx, stmt in enumerate(tree.body):
        if idx == 0 and _is_docstring_stmt(stmt):
            continue
        if isinstance(stmt, ast.FunctionDef):
            functions.append(stmt)
        elif isinstance(stmt, ast.AsyncFunctionDef):
            raise PyRustError(
                "Async functions are not supported yet; use standard def functions.",
                lineno=stmt.lineno,
                col=stmt.col_offset,
                filename=filename,
            )
        elif isinstance(stmt, ast.ClassDef):
            classes.append(stmt)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                top_name = alias.name.split(".")[0]
                if top_name not in imports:
                    imports.append(top_name)
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module:
                top_name = stmt.module.split(".")[0]
                if top_name not in imports:
                    imports.append(top_name)
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    from_imports[local_name] = stmt.module
        elif isinstance(stmt, ast.If) and _is_main_guard(stmt):
            for inner in stmt.body:
                if isinstance(inner, ast.FunctionDef):
                    functions.append(inner)
                elif isinstance(inner, ast.ClassDef):
                    classes.append(inner)
                else:
                    main_body.append(inner)
        else:
            main_body.append(stmt)

    return ParsedModule(
        name=module_name,
        path=path,
        source=source,
        tree=tree,
        functions=functions,
        classes=classes,
        main_body=main_body,
        imports=imports,
        from_imports=from_imports,
        filename=filename,
    )


def read_source(path: Path) -> str:
    """Decode a source file the way Python does: UTF-8 BOM and PEP 263 coding cookies."""
    import importlib.util

    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise PyRustError(f"Cannot read {path}: {exc.strerror or exc}", filename=str(path)) from exc
    try:
        return importlib.util.decode_source(data)
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        detail = exc.msg if isinstance(exc, SyntaxError) else str(exc)
        raise PyRustSyntaxError(f"Cannot decode source: {detail}", filename=str(path)) from exc


def _parse_file(path: Path, module_name: str) -> ParsedModule:
    return parse_source(read_source(path), filename=str(path), module_name=module_name, path=path)


# Entry module of a project directory, in priority order. `__main__` first:
# that is what `python <dir>` runs.
ENTRY_NAMES = ("__main__", "main", "app", "run")


def _module_files(directory: Path) -> dict[str, Path]:
    """stem -> file for the importable `.pyr`/`.py` files in `directory` (`.pyr` wins, like the import hook)."""
    found: dict[str, Path] = {}
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return found
    for file_path in entries:
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS or not file_path.is_file():
            continue
        stem = file_path.stem
        if stem == "__init__":
            continue
        if stem not in found or suffix == ".pyr":
            found[stem] = file_path
    return found


def find_entry(directory: Path) -> Path:
    """The file `pyrust run <directory>` executes (without parsing anything)."""
    directory = Path(directory).resolve()
    candidates = _module_files(directory)
    if not candidates:
        raise PyRustError(f"No .pyr or .py files found in directory: {directory}", filename=str(directory))
    for name in ENTRY_NAMES:
        if name in candidates:
            return candidates[name]
    return candidates[sorted(candidates)[0]]


def _imported_names(tree: ast.Module) -> list[str]:
    """Top-level module names imported anywhere in `tree` (absolute imports only)."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module]
        else:
            continue
        for mod in mods:
            top = mod.split(".")[0]
            if top not in names:
                names.append(top)
    return names


# Names Python resolves from the program's own folder before the standard
# library: everything except modules compiled into the interpreter.
def _shadowable(name: str) -> bool:
    import sys

    return name not in sys.builtin_module_names


def _local_package(base_dir: Path, name: str) -> Path | None:
    """`<base>/<name>/__init__.*` or a compiled extension module: something PyRust cannot transpile."""
    import importlib.machinery

    pkg = base_dir / name
    if pkg.is_dir():
        for suffix in (".pyr", *importlib.machinery.all_suffixes()):
            if (pkg / f"__init__{suffix}").is_file():
                return pkg
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        if (base_dir / f"{name}{suffix}").is_file():
            return base_dir / f"{name}{suffix}"
    return None


class _Resolver:
    """Finds the project's own modules reachable from the entry module, as Python's import would."""

    def __init__(self, base_dir: Path, entry: Path, stdlib_shims: frozenset[str] = frozenset()):
        self.base_dir = base_dir
        self.entry = entry
        self.local = _module_files(base_dir)
        self.vendor = _module_files(base_dir / "pyrust_vendor")
        self.vendor.pop(entry.stem, None)
        self.stdlib_shims = stdlib_shims
        self.absent: list[Path] = []

    def resolve(self, name: str) -> tuple[str, Path | None]:
        """('module', path) | ('package', path) | ('external', None)."""
        if name == self.entry.stem or not _shadowable(name):
            return "external", None
        if name == "main":
            # The entry module is "main" in the generated crate; another main.py cannot coexist.
            return ("conflict", self.local["main"]) if "main" in self.local else ("external", None)
        pkg = _local_package(self.base_dir, name)
        if pkg is not None:
            return "package", pkg
        if name in self.local:
            return "module", self.local[name]
        # Python would pick up a file created here later: remember to re-plan then.
        for suffix in (".py", ".pyr"):
            self.absent.append(self.base_dir / f"{name}{suffix}")
        self.absent.append(self.base_dir / name / "__init__.py")
        if name in self.vendor:
            return "module", self.vendor[name]
        return "external", None


def project_files(target_path: Path) -> tuple[list[Path], list[Path]]:
    """(local module files the program imports, paths whose appearance would change its imports).

    Never raises: used to key the run-plan cache even for programs that do not transpile.
    """
    target_path = Path(target_path).resolve()
    try:
        entry = find_entry(target_path) if target_path.is_dir() else target_path
    except PyRustError:
        return [], []
    resolver = _Resolver(entry.parent, entry)
    seen: dict[Path, None] = {}
    queue = [entry]
    while queue:
        path = queue.pop()
        try:
            tree = ast.parse(read_source(path), filename=str(path))
        except (PyRustError, SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        for name in _imported_names(tree):
            kind, found = resolver.resolve(name)
            if kind == "module" and found not in seen and found != entry:
                seen[found] = None
                queue.append(found)
    return list(seen), resolver.absent


def load_project(target_path: Path, stdlib_shims: frozenset[str] | set[str] = frozenset()) -> list[ParsedModule]:
    """Load a `.pyr`/`.py` script or project directory: the entry module first, then every
    project module reachable from it through imports (unrelated files are never read).

    `stdlib_shims`: standard-library modules with a native implementation. A local
    file with one of these names is what Python would import instead, so it is refused.
    """
    target_path = Path(target_path).resolve()
    if not target_path.exists():
        raise PyRustError(f"Path does not exist: {target_path}", filename=str(target_path))

    if target_path.is_file():
        if target_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise PyRustError(
                f"Unsupported file extension '{target_path.suffix}'. Expected .pyr or .py",
                filename=str(target_path),
            )
        entry = target_path
    else:
        entry = find_entry(target_path)

    resolver = _Resolver(entry.parent, entry, frozenset(stdlib_shims))
    modules = [_parse_file(entry, "main")]
    loaded: dict[str, Path] = {}
    queue = [modules[0]]
    while queue:
        mod = queue.pop(0)
        for name in _imported_names(mod.tree):
            if name in loaded:
                continue
            kind, found = resolver.resolve(name)
            if kind == "package":
                raise PyRustError(
                    f"import '{name}' is a local package or compiled module, which only runs on CPython",
                    filename=str(found),
                )
            if kind == "conflict":
                raise PyRustError(
                    f"importing the local module '{found.name}' from entry '{entry.name}' only runs on CPython",
                    filename=str(found),
                )
            if kind != "module":
                continue
            if name in resolver.stdlib_shims:
                raise PyRustError(
                    f"local module '{found.name}' shadows the standard library module '{name}' "
                    "(Python imports the local file), so it runs on CPython",
                    filename=str(found),
                )
            loaded[name] = found
            sub = _parse_file(found, name)
            modules.append(sub)
            queue.append(sub)
    return modules
