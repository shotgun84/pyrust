from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from pyrust.errors import PyRustError

SUPPORTED_EXTENSIONS = (".pyr", ".py")

# Imports that need no Rust module: pure-typing/decorator helpers whose
# semantics PyRust implements natively (or ignores safely).
IGNORED_IMPORTS = {"typing", "__future__", "dataclasses", "abc"}


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
        raise PyRustError(
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


def load_project(target_path: Path) -> list[ParsedModule]:
    """Load either a single `.pyr`/`.py` script or a directory of `.pyr`/`.py` modules."""
    target_path = Path(target_path).resolve()
    if not target_path.exists():
        raise PyRustError(f"Path does not exist: {target_path}", filename=str(target_path))

    base_dir = target_path if target_path.is_dir() else target_path.parent
    # Vendored pure-Python dependencies live in pyrust_vendor/*.py(r).
    vendor_files: dict[str, Path] = {}
    vendor_dir = base_dir / "pyrust_vendor"
    if vendor_dir.is_dir():
        for ext in (".py", ".pyr"):
            for file_path in sorted(vendor_dir.glob(f"*{ext}")):
                if file_path.stem == "__init__":
                    continue
                vendor_files.setdefault(file_path.stem, file_path)

    if target_path.is_file():
        if target_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise PyRustError(
                f"Unsupported file extension '{target_path.suffix}'. Expected .pyr or .py",
                filename=str(target_path),
            )
        source = target_path.read_text(encoding="utf-8")
        main_mod = parse_source(
            source,
            filename=str(target_path),
            module_name="main",
            path=target_path,
        )
        vendor_mods = [
            parse_source(fp.read_text(encoding="utf-8"), filename=str(fp), module_name=stem, path=fp)
            for stem, fp in sorted(vendor_files.items())
            if stem != target_path.stem
        ]
        return [main_mod] + vendor_mods

    # Directory mode: discover all .pyr and .py files (preferring .pyr over .py when stem collides)
    candidates: dict[str, Path] = {}
    for ext in (".py", ".pyr"):
        for file_path in sorted(target_path.glob(f"*{ext}")):
            if file_path.stem == "__init__":
                continue
            candidates[file_path.stem] = file_path
    # Vendored modules fill in names the project itself does not define.
    for stem, fp in vendor_files.items():
        candidates.setdefault(stem, fp)

    if not candidates:
        raise PyRustError(
            f"No .pyr or .py files found in directory: {target_path}",
            filename=str(target_path),
        )

    # Determine entry module: main -> app -> run -> first file alphabetically
    entry_stem = next(
        (name for name in ("main", "app", "run", "__main__") if name in candidates),
        next(iter(candidates.keys())),
    )

    modules: list[ParsedModule] = []
    entry_file = candidates.pop(entry_stem)
    modules.append(
        parse_source(
            entry_file.read_text(encoding="utf-8"),
            filename=str(entry_file),
            module_name="main",
            path=entry_file,
        )
    )
    for stem, file_path in sorted(candidates.items()):
        modules.append(
            parse_source(
                file_path.read_text(encoding="utf-8"),
                filename=str(file_path),
                module_name=stem,
                path=file_path,
            )
        )
    return modules
