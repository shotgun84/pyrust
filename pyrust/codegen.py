from __future__ import annotations

import ast
import math as _pymath
import re

from pyrust.errors import PyRustError
from pyrust.inference import (
    MATH_CONST_RUST,
    MATH_CONSTS,
    MATH_FLOAT_FUNCS,
    ModuleTypeEnv,
    ScopeInfo,
    _positional_params,
    infer_module,
)
from pyrust.aliasing import AliasInfo, analyze_aliasing, is_mutable
from pyrust.ownership import MUTATING_METHODS, OwnershipPlan, analyze_ownership
from pyrust.parser import IGNORED_IMPORTS, ParsedModule, parse_source
from pyrust.types import (
    ANY,
    BOOL,
    FLOAT,
    INT,
    NONE,
    STR,
    UNKNOWN,
    ClassType,
    DictType,
    FuncType,
    ListType,
    PyType,
    SetType,
    TupleType,
)


def _is_negative_expr(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value < 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return True
    return False


class _WalrusExtractor(ast.NodeTransformer):
    """Replace `x := value` with `x`, recording each walrus in order."""

    def __init__(self) -> None:
        self.replacements: list[tuple[ast.NamedExpr, ast.Name]] = []

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.expr:
        self.generic_visit(node)
        replacement = ast.copy_location(ast.Name(id=node.target.id, ctx=ast.Load()), node)
        self.replacements.append((node, replacement))
        return replacement


class _RustGenerator:
    def __init__(
        self,
        parsed: ParsedModule,
        env: ModuleTypeEnv,
        ownership: dict[str, OwnershipPlan],
        is_entry: bool = True,
        all_modules: list[str] | None = None,
    ) -> None:
        self.parsed = parsed
        self.env = env
        self.ownership = ownership
        self.is_entry = is_entry
        self.all_modules: list[str] = list(all_modules) if all_modules else []
        self.lines: list[str] = []
        self.indent = 0
        self.self_var_name = "self"
        self.module_consts: set[str] = set()
        self._else_stack: list[str | None] = []
        self._loop_counter = 0
        self.in_function = False
        self._fn_locals: set[str] = set()
        self.alias = AliasInfo()
        self._ref_names: set[str] = set()  # names bound to `&mut T` (ref params, by-ref loop vars)
        self._fn_key: str | None = None
        self._place_override: str | None = None
        self._mut_methods_cache: dict[str, set[str]] | None = None

    def emit(self, line: str = "") -> None:
        if line:
            self.lines.append(("    " * self.indent) + line)
        else:
            self.lines.append("")

    def _discover_module_consts(self) -> list[tuple[str, PyType, str]]:
        """Identify unmutated top-level scalar constants referenced inside functions."""
        referenced_in_funcs: set[str] = set()
        for fn in self.parsed.functions:
            local_names = {a.arg for a in fn.args.args}
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    local_names.add(node.id)
            for node in ast.walk(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in local_names:
                    referenced_in_funcs.add(node.id)

        consts: list[tuple[str, PyType, str]] = []
        scope = self.env.module_scope
        plan = self.ownership.get("__main__", OwnershipPlan())
        assigned: dict[str, int] = {}
        for node in ast.walk(ast.Module(body=self.parsed.main_body, type_ignores=[])):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned[node.id] = assigned.get(node.id, 0) + 1
        for stmt in self.parsed.main_body:
            target = value = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                target, value = stmt.target, stmt.value
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, (int, float, bool))
            ):
                name = target.id
                vtype = scope.expr_types.get(id(value), scope.var_types.get(name, UNKNOWN))
                # Imported modules export every constant (`helper.CONST`).
                wanted = name in referenced_in_funcs or not self.is_entry
                if (
                    wanted
                    and assigned.get(name, 0) == 1
                    and name not in plan.mutable_vars
                    and vtype in (INT, FLOAT, BOOL)
                    and scope.var_types.get(name, vtype) == vtype
                ):
                    stmt = ast.Assign(targets=[target], value=value)
                    code = self._expr(stmt.value, scope, plan, expected_type=vtype)
                    consts.append((name, vtype, code))
                    self.module_consts.add(name)
        return consts

    def generate(self) -> str:
        from pyrust.definite import check_function, check_module

        for fn in self.parsed.functions:
            check_function(fn, self.parsed.filename)
        for cls in self.parsed.classes:
            for item in cls.body:
                if isinstance(item, ast.FunctionDef):
                    check_function(item, self.parsed.filename)
        if self.is_entry:
            check_module(self.parsed.tree, self.parsed.filename)
        self.alias = analyze_aliasing(self.parsed, self.env, include_main=self.is_entry)
        self._augment_mutability()
        self.emit("#![allow(dead_code, unused_imports, unused_variables, unused_mut, non_snake_case, unused_parens, arithmetic_overflow, unconditional_panic)]")
        if self.is_entry:
            self.emit("mod pyrust_rt;")
            declared: set[str] = set()
            for name in self.all_modules:
                if name == self.parsed.name or name in declared:
                    continue
                declared.add(name)
                if name.split(".")[0] in IGNORED_IMPORTS:
                    continue
                if name not in ("math", "sys", "time", "random", "os", "typing", "json", "platform", "statistics", "gc"):
                    self.emit(f"mod {name};")
            for imp in self.parsed.imports:
                if imp in declared:
                    continue
                declared.add(imp)
                if imp.split(".")[0] in IGNORED_IMPORTS:
                    continue
                if imp not in ("math", "sys", "time", "random", "os", "typing", "json", "platform", "statistics", "gc"):
                    self.emit(f"mod {imp};")
        self.emit("use crate::pyrust_rt::*;")
        if not self.is_entry:
            # #[macro_export] macros live at the crate root; submodules import them.
            self.emit("use crate::{py_print, py_println};")
        self.emit("use std::collections::HashSet;")
        self.emit()

        for cname, ctype, cval in self._discover_module_consts():
            self.emit(f"pub const {cname}: {ctype.to_rust_type()} = {cval};")
        if self.module_consts:
            self.emit()

        for cls_node in self.parsed.classes:
            self._emit_class(cls_node)
            self.emit()

        for fn_node in self.parsed.functions:
            self._emit_function(fn_node)
            self.emit()

        if self.is_entry:
            self.emit("fn main() {")
            self.emit(f"    py_run_main(py_module_main, {self._replay_info()});")
            self.emit("}")
            self.emit()
            self.emit("fn py_module_main() {")
            self.indent += 1
            scope = self.env.module_scope
            plan = self.ownership.get("__main__", OwnershipPlan())
            declared = set(scope.params) | set(self.module_consts)
            self._emit_hoisted_declarations(scope, plan, declared)
            self._emit_block(self.parsed.main_body, scope, plan, declared, current_class=None)
            self.indent -= 1
            self.emit("}")

        return "\n".join(self.lines) + "\n"

    def _replay_info(self) -> str:
        """(python, script, pyrust package parent) for replaying on CPython."""
        import os
        import sys as _sys

        import pyrust

        path = getattr(self.parsed, "path", None)
        if not path or not _sys.executable:
            return "None"
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(pyrust.__file__)))

        def lit(x: str) -> str:
            return self._rust_str_lit(str(x))

        return f"Some(({lit(_sys.executable)}, {lit(os.path.abspath(str(path)))}, {lit(pkg_parent)}))"

    def _emit_hoisted_declarations(
        self,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        declared: set[str],
    ) -> None:
        for var_name in sorted(scope.hoisted_vars):
            if var_name not in declared:
                vtype = scope.var_types.get(var_name, ANY)
                rust_t = vtype.to_rust_type()
                self.emit(f"let mut {var_name}: {rust_t} = Default::default();")
                declared.add(var_name)

    def _emit_class(self, cls_node: ast.ClassDef) -> None:
        cls_type = self.env.class_types[cls_node.name]
        self._check_class_decorators(cls_node)
        self.emit("#[derive(Debug, Clone, Default, PartialEq)]")
        self.emit(f"pub struct {cls_node.name} {{")
        self.indent += 1
        for fname, ftype in cls_type.fields.items():
            self.emit(f"pub {fname}: {ftype.to_rust_type()},")
        self.indent -= 1
        self.emit("}")
        self.emit()

        self.emit(f"impl {cls_node.name} {{")
        self.indent += 1
        emitted: set[str] = set()
        init_node = next(
            (m for m in cls_node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"),
            None,
        )
        if init_node is not None:
            self._emit_class_init(cls_node.name, init_node, cls_type)
            emitted.add("__init__")
        else:
            found_init = self._find_method_def(cls_node, "__init__")
            if found_init is not None:
                _, base_init = found_init
                self._emit_class_init(cls_node.name, base_init, cls_type)
                emitted.add("__init__")
            elif not cls_type.fields:
                # Method-only class: default-constructible.
                self.emit("pub fn new() -> Self {")
                self.indent += 1
                self.emit(f"{cls_node.name}::default()")
                self.indent -= 1
                self.emit("}")
                emitted.add("__init__")
            else:
                self._emit_synthetic_init(cls_node.name, cls_type)
                emitted.add("__init__")
        for item in cls_node.body:
            if not isinstance(item, ast.FunctionDef) or item.name == "__init__":
                continue
            self._emit_class_method(cls_node.name, item, cls_type)
            emitted.add(item.name)
        for mname in cls_type.methods:
            if mname in emitted:
                continue
            found = self._find_method_def(cls_node, mname)
            if found is not None:
                _, base_item = found
                self._emit_class_method(cls_node.name, base_item, cls_type)
                emitted.add(mname)
        self.indent -= 1
        self.emit("}")
        if cls_node.name in self._dataclass_names():
            fields = self._dataclass_field_names(cls_node)
            renames = getattr(self.parsed, "rust_renames", {}) or {}
            parts = ", ".join(f"{renames.get(f, f)}={{}}" for f in fields)
            args = "".join(f", self.{f}.py_repr()" for f in fields)
            self.emit()
            shown = renames.get(cls_node.name, cls_node.name)
            self.emit(f"impl PyRepr for {cls_node.name} {{")
            self.emit(f'    fn py_repr(&self) -> String {{ format!("{shown}({parts})"{args}) }}')
            self.emit("}")

    def _dataclass_field_names(self, cls_node: ast.ClassDef) -> list[str]:
        names: list[str] = []
        if cls_node.bases and isinstance(cls_node.bases[0], ast.Name):
            base = next((c for c in self.parsed.classes if c.name == cls_node.bases[0].id), None)
            if base is not None and base.name in self._dataclass_names():
                names = self._dataclass_field_names(base)
        for item in cls_node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                ann = ast.unparse(item.annotation)
                if "ClassVar" in ann:
                    continue
                if isinstance(item.value, ast.Call) and any(kw.arg == "repr" for kw in item.value.keywords):
                    raise PyRustError(
                        "dataclass field(repr=...) is not supported natively",
                        lineno=item.lineno,
                        col=item.col_offset,
                        filename=self.parsed.filename,
                    )
                if item.target.id not in names:
                    names.append(item.target.id)
        return names

    def _find_method_def(
        self, cls_node: ast.ClassDef, method_name: str, _seen: set[str] | None = None
    ) -> tuple[str, ast.FunctionDef] | None:
        """Find a method definition up the single-inheritance chain (same file)."""
        for item in cls_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                return (cls_node.name, item)
        seen = _seen or set()
        if cls_node.name in seen or not cls_node.bases or len(cls_node.bases) != 1:
            return None
        seen.add(cls_node.name)
        base = cls_node.bases[0]
        base_cls = None
        if isinstance(base, ast.Name):
            for c in self.parsed.classes:
                if c.name == base.id:
                    base_cls = c
                    break
        if base_cls is None:
            return None
        return self._find_method_def(base_cls, method_name, seen)

    def _emit_synthetic_init(self, cls_name: str, cls_type: ClassType) -> None:
        """Emit `new()` from dataclass-style fields (init = all fields)."""
        init_ft = cls_type.methods.get("__init__")
        names = self.env.func_param_names.get(f"{cls_name}.__init__", list(cls_type.fields.keys()))
        if init_ft is None:
            raise PyRustError(
                f"class '{cls_name}' needs an __init__ method or @dataclass fields "
                "with defaults so PyRust can construct it",
                filename=self.parsed.filename,
            )
        param_strs = [
            f"{a}: {pt.to_rust_type()}" for a, pt in zip(names, init_ft.params)
        ]
        self.emit(f"pub fn new({', '.join(param_strs)}) -> Self {{")
        self.indent += 1
        self.emit("let _py_frame = PyFrame::enter();")
        self.emit(f"{cls_name} {{")
        self.indent += 1
        for fname in cls_type.fields:
            if fname in names:
                self.emit(f"{fname}: {fname},")
            else:
                self.emit(f"{fname}: Default::default(),")
        self.indent -= 1
        self.emit("}")
        self.indent -= 1
        self.emit("}")

    @staticmethod
    def _all_param_names(fn_node: ast.FunctionDef, skip_self: bool = False) -> list[str]:
        names = [a.arg for a in _positional_params(fn_node)]
        if skip_self and "self" in names:
            names.remove("self")
        if fn_node.args.vararg is not None:
            names.append(fn_node.args.vararg.arg)
        return names

    def _def_param_strs(
        self,
        fn_node: ast.FunctionDef,
        ft: FuncType,
        plan: OwnershipPlan,
        skip_self: bool = False,
    ) -> list[str]:
        pos = _positional_params(fn_node)
        if skip_self:
            pos = [a for a in pos if a.arg != "self"]
        refs = self.alias.ref_params.get(self._fn_key or "", set())
        strs = [
            f"{a.arg}: &mut {pt.to_rust_type()}" if a.arg in refs else
            f"{('mut ' if a.arg in plan.mutable_vars else '')}{a.arg}: {pt.to_rust_type()}"
            for a, pt in zip(pos, ft.params)
        ]
        if fn_node.args.vararg is not None:
            vname = fn_node.args.vararg.arg
            vtype = ft.params[len(pos)] if len(ft.params) > len(pos) else ListType(UNKNOWN)
            vt = vtype.to_rust_type() if isinstance(vtype, ListType) else "Vec<PyValue>"
            strs.append(f"{('mut ' if vname in plan.mutable_vars else '')}{vname}: {vt}")
        return strs

    @staticmethod
    def _decorator_names(decorator_list: list[ast.expr]) -> list[str]:
        names: list[str] = []
        for d in decorator_list:
            if isinstance(d, ast.Name):
                names.append(d.id)
            elif isinstance(d, ast.Attribute):
                names.append(d.attr)
        return names

    def _check_class_decorators(self, cls_node: ast.ClassDef) -> None:
        for name in self._decorator_names(cls_node.decorator_list):
            if name != "dataclass":
                raise PyRustError(
                    f"@{name} is not supported by PyRust yet (only @dataclass is)",
                    lineno=cls_node.lineno,
                    col=cls_node.col_offset,
                    filename=self.parsed.filename,
                )

    def _check_fn_decorators(self, fn_node: ast.FunctionDef, what: str) -> None:
        for name in self._decorator_names(fn_node.decorator_list):
            if name not in ("abstractmethod",):
                raise PyRustError(
                    f"@{name} is not supported by PyRust yet on {what}",
                    lineno=fn_node.lineno,
                    col=fn_node.col_offset,
                    filename=self.parsed.filename,
                )

    def _emit_class_init(self, cls_name: str, fn_node: ast.FunctionDef, cls_type: ClassType) -> None:
        prev_in_function, self.in_function = self.in_function, True
        prev_locals, self._fn_locals = self._fn_locals, self._locals_of(fn_node)
        key = f"{self._defining_class(cls_name, fn_node)}.{fn_node.name}"
        prev_key, self._fn_key = self._fn_key, key
        prev_refs, self._ref_names = self._ref_names, set(self.alias.ref_params.get(key, set()))
        try:
            self._emit_class_init_body(cls_name, fn_node, cls_type)
        finally:
            self.in_function = prev_in_function
            self._fn_locals = prev_locals
            self._fn_key = prev_key
            self._ref_names = prev_refs

    def _emit_class_init_body(self, cls_name: str, fn_node: ast.FunctionDef, cls_type: ClassType) -> None:
        scope = self.env.class_method_scopes[(cls_name, "__init__")]
        plan = self.ownership.get(f"{cls_name}.__init__", OwnershipPlan())
        init_ft = cls_type.methods["__init__"]
        non_self = [a for a in _positional_params(fn_node) if a.arg != "self"]
        param_strs = self._def_param_strs(fn_node, init_ft, plan, skip_self=True)
        self.emit(f"pub fn new({', '.join(param_strs)}) -> Self {{")
        self.indent += 1
        self.emit("let _py_frame = PyFrame::enter();")

        simple_only = all(
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Attribute)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == "self"
            for stmt in fn_node.body
        )

        if simple_only and not scope.hoisted_vars:
            field_exprs: dict[str, str] = {}
            for stmt in fn_node.body:
                assert isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Attribute)
                attr = stmt.targets[0].attr
                field_exprs[attr] = self._expr(stmt.value, scope, plan, cls_type.fields.get(attr))
            self.emit(f"{cls_name} {{")
            self.indent += 1
            for fname in cls_type.fields:
                val_code = field_exprs.get(fname, "Default::default()")
                self.emit(f"{fname}: {val_code},")
            self.indent -= 1
            self.emit("}")
        else:
            self.emit(f"let mut self_ = {cls_name}::default();")
            declared = set(self._all_param_names(fn_node, skip_self=True))
            self._emit_hoisted_declarations(scope, plan, declared)
            prev_self = self.self_var_name
            self.self_var_name = "self_"
            self._emit_block(fn_node.body, scope, plan, declared, current_class=cls_name)
            self.self_var_name = prev_self
            self.emit("self_")

        self.indent -= 1
        self.emit("}")

    def _called_names(self) -> set[str]:
        if not hasattr(self, "_called_cache"):
            names: set[str] = set()
            base_refs = {id(b) for c in ast.walk(self.parsed.tree) if isinstance(c, ast.ClassDef) for b in c.bases}
            for node in ast.walk(self.parsed.tree):
                if id(node) in base_refs:
                    continue  # `class B(A)` does not instantiate A
                if isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Name):
                        names.add(f.id)
                    elif isinstance(f, ast.Attribute):
                        names.add(f.attr)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    names.add(node.id)  # passed around as a value: assume reachable
            self._called_cache = names
        return self._called_cache

    def _emit_or_stub(self, emit, signature: str, reachable: bool) -> None:
        """Emit a function; if it cannot be compiled but is provably never
        called, emit a stub that fails loudly instead of blocking the program."""
        mark, indent = len(self.lines), self.indent
        try:
            emit()
        except PyRustError:
            if reachable:
                raise
            del self.lines[mark:]
            self.indent = indent
            self.emit(signature + " {")
            self.emit('    py_unsupported("calling a function PyRust could not compile")')
            self.emit("}")

    def _emit_class_method(self, cls_name: str, fn_node: ast.FunctionDef, cls_type: ClassType) -> None:
        reachable = cls_name in self._called_names() and fn_node.name in self._called_names()
        sig = self._stub_signature(cls_name, fn_node, cls_type)
        self._emit_or_stub(lambda: self._emit_class_method_checked(cls_name, fn_node, cls_type), sig, reachable)

    def _stub_signature(self, cls_name: str, fn_node: ast.FunctionDef, cls_type: ClassType) -> str:
        m_ft = cls_type.methods.get(fn_node.name)
        if m_ft is None:
            return f"pub fn {fn_node.name}(&self)"
        params = [a for a in _positional_params(fn_node) if a.arg != "self"]
        ps = ", ".join(f"_{a.arg}: {t.to_rust_type()}" for a, t in zip(params, m_ft.params))
        ret = f" -> {m_ft.ret.to_rust_type()}" if m_ft.ret != NONE else ""
        return f"pub fn {fn_node.name}(&self{', ' if ps else ''}{ps}){ret}"

    def _emit_class_method_checked(self, cls_name: str, fn_node: ast.FunctionDef, cls_type: ClassType) -> None:
        self._check_fn_decorators(fn_node, f"method {cls_name}.{fn_node.name}")
        if fn_node.name.startswith("__") and fn_node.name.endswith("__"):
            # Python calls these implicitly (print, ==, len, ...). Emitting them
            # as plain methods would compile but silently ignore them.
            raise PyRustError(
                f"special method {cls_name}.{fn_node.name} is not supported by PyRust yet "
                "(only __init__ is): its implicit calls would be silently skipped",
                lineno=fn_node.lineno,
                col=fn_node.col_offset,
                filename=self.parsed.filename,
            )
        prev_in_function, self.in_function = self.in_function, True
        prev_locals, self._fn_locals = self._fn_locals, self._locals_of(fn_node)
        key = f"{self._defining_class(cls_name, fn_node)}.{fn_node.name}"
        prev_key, self._fn_key = self._fn_key, key
        prev_refs, self._ref_names = self._ref_names, set(self.alias.ref_params.get(key, set()))
        try:
            self._emit_class_method_body(cls_name, fn_node, cls_type)
        finally:
            self.in_function = prev_in_function
            self._fn_locals = prev_locals
            self._fn_key = prev_key
            self._ref_names = prev_refs

    def _emit_class_method_body(self, cls_name: str, fn_node: ast.FunctionDef, cls_type: ClassType) -> None:
        scope = self.env.class_method_scopes[(cls_name, fn_node.name)]
        plan = self.ownership.get(f"{cls_name}.{fn_node.name}", OwnershipPlan())
        m_ft = cls_type.methods[fn_node.name]
        non_self = [a for a in _positional_params(fn_node) if a.arg != "self"]
        # Read-only methods take &self so `an.big` + `an.big_share()` can
        # coexist in one format! (else E0502 mut+immut borrow conflict).
        self_recv = "&mut self" if "self" in plan.mutable_vars else "&self"
        param_strs = [self_recv] + self._def_param_strs(fn_node, m_ft, plan, skip_self=True)
        ret_str = f" -> {m_ft.ret.to_rust_type()}" if m_ft.ret != NONE else ""
        self.emit(f"pub fn {fn_node.name}({', '.join(param_strs)}){ret_str} {{")
        self.indent += 1
        self.emit("let _py_frame = PyFrame::enter();")
        declared = set(self._all_param_names(fn_node))
        self._emit_hoisted_declarations(scope, plan, declared)
        self._emit_block(fn_node.body, scope, plan, declared, current_class=cls_name, expected_ret=m_ft.ret)
        self.indent -= 1
        self.emit("}")

    def alias_mut_methods(self) -> dict[str, set[str]]:
        if self._mut_methods_cache is None:
            from pyrust.ownership import mutating_methods

            self._mut_methods_cache = mutating_methods(self.parsed)
        return self._mut_methods_cache

    def _defining_class(self, cls_name: str, fn_node: ast.FunctionDef) -> str:
        for c in self.parsed.classes:
            if any(item is fn_node for item in c.body):
                return c.name
        return cls_name

    def _alias_key(self, call_key: str) -> str:
        """Map a call key (`f`, `Cls.m`, `Cls.__init__`) to the key of the
        function that defines it (methods may be inherited)."""
        if "." not in call_key:
            return call_key
        cls, m = call_key.split(".", 1)
        from pyrust.inference import _find_method_node_in

        found = _find_method_node_in({c.name: c for c in self.parsed.classes}, cls, m)
        return f"{found[0]}.{m}" if found else call_key

    def _augment_mutability(self) -> None:
        """Names that are iterated by reference or passed as `&mut` must be
        declared `mut` in their scope."""
        def scan(stmts: list[ast.stmt], plan: OwnershipPlan, scope: ScopeInfo) -> None:
            for node in ast.walk(ast.Module(body=stmts, type_ignores=[])):
                if isinstance(node, ast.For) and id(node) in self.alias.byref_loops:
                    it = node.iter
                    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name):
                        it = it.args[0]
                    elif isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute):
                        it = it.func.value
                    r = self._root_of(it)
                    if r:
                        plan.mutable_vars.add(r)
                if isinstance(node, ast.Call):
                    key = self._call_alias_key(node, scope)
                    refs = self.alias.ref_params.get(key, set()) if key else set()
                    if not refs:
                        continue
                    from pyrust.inference import _positional_params as pp

                    fn = self._fn_node_for(key)
                    names = [a.arg for a in pp(fn)] if fn else []
                    if names and names[0] == "self":
                        names = names[1:]
                    for pname, a in zip(names, node.args):
                        if pname in refs and self._root_of(a):
                            plan.mutable_vars.add(self._root_of(a))
                    for kw in node.keywords:
                        if kw.arg in refs and self._root_of(kw.value):
                            plan.mutable_vars.add(self._root_of(kw.value))

        scan(self.parsed.main_body, self.ownership.setdefault("__main__", OwnershipPlan()), self.env.module_scope)
        for fn in self.parsed.functions:
            scan(fn.body, self.ownership.setdefault(fn.name, OwnershipPlan()), self.env.func_scopes[fn.name])
        for (cls_name, mname), scope in self.env.class_method_scopes.items():
            from pyrust.inference import _find_method_node_in

            found = _find_method_node_in({c.name: c for c in self.parsed.classes}, cls_name, mname)
            if found:
                scan(found[1].body, self.ownership.setdefault(f"{cls_name}.{mname}", OwnershipPlan()), scope)

    @staticmethod
    def _root_of(node: ast.expr) -> str | None:
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def _fn_node_for(self, key: str) -> ast.FunctionDef | None:
        if "." in key:
            cls, m = key.split(".", 1)
            for c in self.parsed.classes:
                if c.name == cls:
                    for item in c.body:
                        if isinstance(item, ast.FunctionDef) and item.name == m:
                            return item
            return None
        return next((f for f in self.parsed.functions if f.name == key), None)

    def _call_alias_key(self, node: ast.Call, scope: ScopeInfo) -> str | None:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.env.func_types:
                return node.func.id
            if node.func.id in self.env.class_types:
                return self._alias_key(f"{node.func.id}.__init__")
        if isinstance(node.func, ast.Attribute):
            rt = scope.expr_types.get(id(node.func.value))
            if isinstance(rt, ClassType):
                return self._alias_key(f"{rt.name}.{node.func.attr}")
        return None

    def _place(self, node: ast.expr, scope: ScopeInfo, plan: OwnershipPlan, pre: list[str]) -> str:
        """A Rust place expression (assignable / &mut-borrowable) for a
        Python target chain such as `a[i].f[j]`. Index expressions are
        evaluated into temporaries (in Python's left-to-right order) and
        appended to `pre`."""
        if isinstance(node, ast.Name):
            if node.id == "self":
                return self.self_var_name
            if node.id in self._ref_names:
                return f"(*{node.id})"
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._place(node.value, scope, plan, pre)}.{node.attr}"
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            bt = scope.expr_types.get(id(node.value), UNKNOWN)
            base = self._place(node.value, scope, plan, pre)
            self._loop_counter += 1
            tmp = f"__pk{self._loop_counter}"
            if isinstance(bt, ListType):
                pre.append(f"let {tmp} = {self._expr(node.slice, scope, plan, expected_type=INT)};")
                return f"(*py_index_mut(&mut {base}, {tmp}))"
            if isinstance(bt, DictType):
                kt = scope.expr_types.get(id(node.slice), UNKNOWN)
                if kt != bt.key:
                    raise PyRustError(
                        "dict subscript with a key of a different/dynamic type is not supported natively",
                        lineno=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                        filename=self.parsed.filename,
                    )
                pre.append(f"let {tmp} = {self._expr(node.slice, scope, plan, expected_type=bt.key)};")
                return f"(*{base}.index_mut(&{tmp}))"
        raise PyRustError(
            "this assignment target is not supported natively",
            lineno=getattr(node, "lineno", None),
            col=getattr(node, "col_offset", None),
            filename=self.parsed.filename,
        )

    @staticmethod
    def _has_subscript(node: ast.expr) -> bool:
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            if isinstance(node, ast.Subscript):
                return True
            node = node.value
        return False

    def _locals_of(self, fn_node: ast.FunctionDef) -> set[str]:
        from pyrust.definite import scope_locals

        params = {a.arg for a in fn_node.args.posonlyargs + fn_node.args.args + fn_node.args.kwonlyargs}
        if fn_node.args.vararg:
            params.add(fn_node.args.vararg.arg)
        return scope_locals(fn_node.body) | params

    def _emit_function(self, fn_node: ast.FunctionDef) -> None:
        reachable = fn_node.name in self._called_names() or not self.is_entry
        ft = self.env.func_types[fn_node.name]
        params = ", ".join(f"_{a.arg}: {t.to_rust_type()}" for a, t in zip(_positional_params(fn_node), ft.params))
        ret = f" -> {ft.ret.to_rust_type()}" if ft.ret != NONE else ""
        name = "py_main" if fn_node.name == "main" else fn_node.name
        sig = f"pub fn {name}({params}){ret}"
        if fn_node.args.vararg is not None:
            reachable = True
        self._emit_or_stub(lambda: self._emit_function_checked(fn_node), sig, reachable)

    def _emit_function_checked(self, fn_node: ast.FunctionDef) -> None:
        self._check_fn_decorators(fn_node, f"function {fn_node.name}")
        prev_in_function, self.in_function = self.in_function, True
        prev_locals, self._fn_locals = self._fn_locals, self._locals_of(fn_node)
        prev_key, self._fn_key = self._fn_key, fn_node.name
        prev_refs, self._ref_names = self._ref_names, set(self.alias.ref_params.get(fn_node.name, set()))
        try:
            self._emit_function_body(fn_node)
        finally:
            self.in_function = prev_in_function
            self._fn_locals = prev_locals
            self._fn_key = prev_key
            self._ref_names = prev_refs

    def _emit_function_body(self, fn_node: ast.FunctionDef) -> None:
        scope = self.env.func_scopes[fn_node.name]
        plan = self.ownership.get(fn_node.name, OwnershipPlan())
        ft = self.env.func_types[fn_node.name]
        param_strs = self._def_param_strs(fn_node, ft, plan)
        ret_str = f" -> {ft.ret.to_rust_type()}" if ft.ret != NONE else ""
        # A top-level `def main` would collide with the entry point:
        # emit it as `py_main` and rewrite bare calls to match.
        emit_name = "py_main" if fn_node.name == "main" else fn_node.name
        self.emit(f"pub fn {emit_name}({', '.join(param_strs)}){ret_str} {{")
        self.indent += 1
        self.emit("let _py_frame = PyFrame::enter();")
        declared = set(self._all_param_names(fn_node)) | set(self.module_consts)
        self._emit_hoisted_declarations(scope, plan, declared)
        self._emit_block(fn_node.body, scope, plan, declared, current_class=None, expected_ret=ft.ret)
        self.indent -= 1
        self.emit("}")

    def _emit_block(
        self,
        stmts: list[ast.stmt],
        scope: ScopeInfo,
        plan: OwnershipPlan,
        declared: set[str],
        current_class: str | None,
        expected_ret: PyType | None = None,
    ) -> None:
        if not stmts:
            self.emit("// pass")
            return
        for s in stmts:
            self._emit_stmt(s, scope, plan, declared, current_class, expected_ret)

    def _emit_stmt(
        self,
        stmt: ast.stmt,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        declared: set[str],
        current_class: str | None,
        expected_ret: PyType | None = None,
    ) -> None:
        if isinstance(stmt, ast.Pass):
            self.emit("// pass")
            return

        if isinstance(stmt, ast.Break):
            if self._else_stack and self._else_stack[-1] is not None:
                self.emit(f"{self._else_stack[-1]} = false;")
            self.emit("break;")
            return

        if isinstance(stmt, ast.Continue):
            self.emit("continue;")
            return

        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                self.emit("return;")
            else:
                val_str = self._expr(stmt.value, scope, plan, expected_ret)
                self.emit(f"return {val_str};")
            return

        if isinstance(stmt, ast.Assign):
            if (
                len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in self.module_consts
            ):
                return
            if len(stmt.targets) == 1:
                self._emit_assignment(stmt.targets[0], stmt.value, scope, plan, declared)
                return
            # a = b = value: evaluate once; a mutable value would be shared.
            vt = scope.expr_types.get(id(stmt.value), UNKNOWN)
            if not self._immutable(vt):
                raise PyRustError(
                    "chained assignment of a mutable value (a = b = []) makes both names share one object, "
                    "which native code does not model yet",
                    lineno=stmt.lineno,
                    col=stmt.col_offset,
                    filename=self.parsed.filename,
                )
            self._loop_counter += 1
            tmp = f"__chain{self._loop_counter}"
            self.emit(f"let {tmp} = {self._expr(stmt.value, scope, plan)};")
            for target in stmt.targets:
                ref = ast.Name(id=tmp if vt.is_copy else f"{tmp}.clone()", ctx=ast.Load())
                scope.expr_types[id(ref)] = vt
                self._emit_assignment(target, ref, scope, plan, declared)
            return

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id in self.module_consts and not self.in_function:
                return
            if stmt.value is not None:
                self._emit_assignment(stmt.target, stmt.value, scope, plan, declared)
            return

        if isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Subscript):
                target_t = scope.expr_types.get(id(stmt.target), UNKNOWN)
                synthetic = ast.BinOp(left=stmt.target, op=stmt.op, right=stmt.value)
                scope.expr_types[id(synthetic)] = target_t
                self._emit_assignment(stmt.target, synthetic, scope, plan, declared)
                return

            if isinstance(stmt.target, ast.Attribute) and (
                self._has_subscript(stmt.target.value) or self._root_of(stmt.target) in self._ref_names
            ):
                target_t = scope.expr_types.get(id(stmt.target), UNKNOWN)
                synthetic = ast.BinOp(left=stmt.target, op=stmt.op, right=stmt.value)
                scope.expr_types[id(synthetic)] = target_t
                self._emit_assignment(stmt.target, synthetic, scope, plan, declared)
                return
            lhs = self._recv_no_clone(stmt.target, scope, plan)
            target_t = scope.expr_types.get(id(stmt.target), UNKNOWN)
            value_t = scope.expr_types.get(id(stmt.value), UNKNOWN)
            if isinstance(stmt.target, ast.Name):
                target_t = scope.var_types.get(stmt.target.id, target_t)
            if isinstance(stmt.op, ast.Add) and target_t == STR and value_t == STR:
                self.emit(f"{lhs}.push_str(&{self._expr(stmt.value, scope, plan)});")
                return
            if isinstance(stmt.op, ast.Add) and isinstance(target_t, ListType):
                # list += iterable extends in place (unlike list = list + other).
                if not (isinstance(value_t, ListType) and value_t.elem == target_t.elem):
                    raise PyRustError(
                        "list += with a non-list or differently-typed value is not supported by PyRust yet",
                        lineno=stmt.lineno,
                        col=stmt.col_offset,
                        filename=self.parsed.filename,
                    )
                self.emit(f"{{ let __ext = {self._expr(stmt.value, scope, plan)}.clone(); {lhs}.extend(__ext); }}")
                return
            simple = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*="}
            sym = simple.get(type(stmt.op))
            if sym and target_t in (INT, FLOAT) and value_t in (INT, FLOAT, BOOL) and not (
                target_t == INT and value_t == FLOAT
            ):
                self.emit(f"{lhs} {sym} {self._expr(stmt.value, scope, plan, target_t)};")
                return
            synthetic = ast.BinOp(left=stmt.target, op=stmt.op, right=stmt.value)
            scope.expr_types[id(synthetic)] = target_t
            self.emit(f"{lhs} = {self._expr(synthetic, scope, plan, target_t)};")
            return

        if isinstance(stmt, ast.Raise):
            self._emit_raise(stmt, scope, plan)
            return

        if isinstance(stmt, ast.Delete):
            self._emit_delete(stmt, scope, plan)
            return

        if isinstance(stmt, ast.If):
            walrus_lines, test = self._extract_walrus(stmt.test, scope, plan, declared)
            for line in walrus_lines:
                self.emit(line)
            cond_str = self._cond_expr(test, scope, plan)
            self.emit(f"if {cond_str} {{")
            self.indent += 1
            self._emit_block(stmt.body, scope, plan, set(declared), current_class, expected_ret)
            self.indent -= 1
            if stmt.orelse:
                self.emit("} else {")
                self.indent += 1
                self._emit_block(stmt.orelse, scope, plan, set(declared), current_class, expected_ret)
                self.indent -= 1
            self.emit("}")
            return

        if isinstance(stmt, ast.While):
            walrus_lines, test = self._extract_walrus(stmt.test, scope, plan, declared)
            else_var = self._begin_loop_else(stmt.orelse)
            if else_var is not None:
                self.emit("{")
                self.indent += 1
                self.emit(f"let mut {else_var} = true;")
            if walrus_lines:
                self.emit("while {")
                self.indent += 1
                for line in walrus_lines:
                    self.emit(line)
                cond_str = self._cond_expr(test, scope, plan)
                # NOTE: no trailing semicolon — the block must evaluate to bool.
                self.emit(f"{cond_str}")
                self.indent -= 1
                self.emit("} {")
            else:
                cond_str = self._cond_expr(test, scope, plan)
                self.emit(f"while {cond_str} {{")
            self.indent += 1
            self._emit_block(stmt.body, scope, plan, set(declared), current_class, expected_ret)
            self.indent -= 1
            self.emit("}")
            self._end_loop_else(stmt.orelse, else_var, scope, plan, declared, current_class, expected_ret)
            return

        if isinstance(stmt, ast.For) and id(stmt) in self.alias.byref_loops:
            self._emit_byref_for(stmt, scope, plan, declared, current_class, expected_ret)
            return

        if isinstance(stmt, ast.For):
            target_str = self._pattern_target(stmt.target, plan)
            iter_str = self._for_iter_expr(stmt.iter, scope, plan)
            # If loop target is dynamic PyValue but iterator yields concrete
            # primitives, map into PyValue so `probe = v` typechecks.
            if isinstance(stmt.target, ast.Name):
                tgt_t = scope.var_types.get(stmt.target.id, UNKNOWN)
                it_t = scope.expr_types.get(id(stmt.iter), UNKNOWN)
                it_elem = it_t.elem if isinstance(it_t, (ListType, SetType)) else None
                if tgt_t == ANY and it_elem in (INT, FLOAT, BOOL, STR):
                    # range emits `(a..b)` / `(a..b).step_by()` (already an
                    # iterator); everything else is `.iter().cloned()`.
                    iter_str = f"{iter_str}.map(PyValue::from)"
            else_var = self._begin_loop_else(stmt.orelse)
            tgt_names = [n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name)]
            rebind = None
            if tgt_names and all(n in declared for n in tgt_names) and \
                    all(isinstance(e, ast.Name) for e in ([stmt.target] if isinstance(stmt.target, ast.Name) else getattr(stmt.target, "elts", [None]))):
                # Python's loop target is the enclosing variable: it keeps the
                # last value after the loop.
                self._loop_counter += 1
                rebind = f"__it{self._loop_counter}"
                pattern = stmt.target.id if isinstance(stmt.target, ast.Name) else "(" + ", ".join(tgt_names) + ")"
                target_str = rebind
            if else_var is None:
                self.emit(f"for {target_str} in {iter_str} {{")
            else:
                self.emit("{")
                self.indent += 1
                self.emit(f"let mut {else_var} = true;")
                self.emit(f"for {target_str} in {iter_str} {{")
            self.indent += 1
            if rebind is not None:
                self.emit(f"{pattern} = {rebind};")
            inner_declared = set(declared)
            if isinstance(stmt.target, ast.Name):
                inner_declared.add(stmt.target.id)
            self._emit_block(stmt.body, scope, plan, inner_declared, current_class, expected_ret)
            self.indent -= 1
            self.emit("}")
            self._end_loop_else(stmt.orelse, else_var, scope, plan, declared, current_class, expected_ret)
            return

        if isinstance(stmt, ast.Assert):
            cond_str = self._cond_expr(stmt.test, scope, plan)
            if stmt.msg:
                msg_str = self._print_arg(stmt.msg, scope, plan)
                self.emit(f'if !({cond_str}) {{ py_raise(format!("AssertionError: {{}}", {msg_str})); }}')
            else:
                self.emit(f'if !({cond_str}) {{ py_raise("AssertionError".to_string()); }}')
            return

        if isinstance(stmt, ast.Try):
            raise PyRustError(
                "try/except is not supported by PyRust yet: Rust has no exceptions, "
                "so Python-style error handling cannot be transpiled",
                lineno=stmt.lineno,
                col=stmt.col_offset,
                filename=self.parsed.filename,
            )

        if isinstance(stmt, ast.With):
            if (
                len(stmt.items) == 1
                and isinstance(stmt.items[0].context_expr, ast.Call)
                and isinstance(stmt.items[0].context_expr.func, ast.Name)
                and stmt.items[0].context_expr.func.id == "open"
                and isinstance(stmt.items[0].optional_vars, ast.Name)
            ):
                open_call = stmt.items[0].context_expr
                var_name = stmt.items[0].optional_vars.id
                file_code = self._expr(open_call, scope, plan)
                if var_name not in declared:
                    self.emit(f"let mut {var_name}: PyFile = {file_code};")
                    declared.add(var_name)
                else:
                    self.emit(f"{var_name} = {file_code};")
                self._emit_block(stmt.body, scope, plan, set(declared), current_class, expected_ret)
                self.emit(f"{var_name}.close();")
                return
            raise PyRustError(
                "only `with open(path) as f:` is supported by PyRust yet",
                lineno=stmt.lineno,
                col=stmt.col_offset,
                filename=self.parsed.filename,
            )

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raise PyRustError(
                "nested def/class is not supported by PyRust yet: move it to the top level",
                lineno=stmt.lineno,
                col=stmt.col_offset,
                filename=self.parsed.filename,
            )

        if isinstance(stmt, (ast.Global, ast.Nonlocal)):
            raise PyRustError(
                "global/nonlocal is not supported by PyRust yet: pass values as function arguments instead",
                lineno=stmt.lineno,
                col=stmt.col_offset,
                filename=self.parsed.filename,
            )

        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            # Function-level imports: the module is already linked at crate
            # level, so the statement itself is a no-op.
            return

        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, (str, type(Ellipsis))):
                return
            if isinstance(stmt.value, ast.NamedExpr):
                self._emit_named_assign(stmt.value, scope, plan, declared)
                return
            self.emit(f"{self._expr(stmt.value, scope, plan)};")
            return

        raise PyRustError(
            f"{type(stmt).__name__} statements are not supported by PyRust yet",
            lineno=getattr(stmt, "lineno", None),
            col=getattr(stmt, "col_offset", None),
            filename=self.parsed.filename,
        )

    def _emit_byref_for(self, stmt: ast.For, scope, plan, declared, current_class, expected_ret) -> None:
        """`for x in xs:` whose body modifies x in place: iterate by reference."""
        it, tgt = stmt.iter, stmt.target
        pre: list[str] = []
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "enumerate":
            place = self._place(it.args[0], scope, plan, pre)
            iter_code = f"{place}.iter_mut().enumerate().map(|(__i, __v)| (__i as i64, __v))"
            idx, var = tgt.elts[0], tgt.elts[1]
            pattern = f"({self._pattern_target(idx)}, {var.id})"
        elif isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "values":
            place = self._place(it.func.value, scope, plan, pre)
            iter_code = f"{place}.values_mut()"
            var = tgt
            pattern = var.id
        elif isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "items":
            place = self._place(it.func.value, scope, plan, pre)
            iter_code = f"{place}.items_mut()"
            k, var = tgt.elts[0], tgt.elts[1]
            pattern = f"({self._pattern_target(k)}, {var.id})"
        else:
            place = self._place(it, scope, plan, pre)
            iter_code = f"{place}.iter_mut()"
            var = tgt
            pattern = var.id
        for line in pre:
            self.emit(line)
        else_var = self._begin_loop_else(stmt.orelse)
        if else_var is not None:
            self.emit("{")
            self.indent += 1
            self.emit(f"let mut {else_var} = true;")
        self.emit(f"for {pattern} in {iter_code} {{")
        self.indent += 1
        inner_declared = set(declared) | {n.id for n in ast.walk(tgt) if isinstance(n, ast.Name)}
        prev_refs = self._ref_names
        self._ref_names = set(prev_refs) | {var.id}
        try:
            self._emit_block(stmt.body, scope, plan, inner_declared, current_class, expected_ret)
        finally:
            self._ref_names = prev_refs
        self.indent -= 1
        self.emit("}")
        self._end_loop_else(stmt.orelse, else_var, scope, plan, declared, current_class, expected_ret)

    def _begin_loop_else(self, orelse: list[ast.stmt]) -> str | None:
        """Return flag var name if orelse non-empty (emits nothing)."""
        if not orelse:
            self._else_stack.append(None)
            return None
        self._loop_counter += 1
        var = f"__py_else_{self._loop_counter}"
        self._else_stack.append(var)
        return var

    def _end_loop_else(
        self,
        orelse: list[ast.stmt],
        else_var: str | None,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        declared: set[str],
        current_class: str | None,
        expected_ret: PyType | None = None,
    ) -> None:
        popped = self._else_stack.pop() if self._else_stack else None
        assert popped == else_var, "loop-else stack imbalance"
        if else_var is None:
            return
        self.emit(f"if {else_var} {{")
        self.indent += 1
        self._emit_block(orelse, scope, plan, set(declared), current_class, expected_ret)
        self.indent -= 1
        self.emit("}")
        self.indent -= 1
        self.emit("}")

    def _emit_named_assign(
        self,
        node: ast.NamedExpr,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        declared: set[str],
    ) -> None:
        # Walrus targets are always `let mut`: the assignment may run
        # conditionally (if/while tests) or repeatedly (while loops).
        name = node.target.id
        val_code = self._expr(node.value, scope, plan)
        if name not in declared:
            vtype = scope.var_types.get(name, ANY)
            type_ann = f": {vtype.to_rust_type()}" if vtype != UNKNOWN else ""
            self.emit(f"let mut {name}{type_ann} = {val_code};")
            declared.add(name)
        else:
            self.emit(f"{name} = {val_code};")

    def _extract_walrus(
        self,
        test: ast.expr,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        declared: set[str],
    ) -> tuple[list[str], ast.expr]:
        """Hoist `:=` out of an if/while test into plain assignments.

        Returns (assignment lines, rewritten test). The rewrite mutates the
        test in place (replacing NamedExpr with its target Name) so that
        while-loop conditions re-assign the outer variable every iteration
        instead of shadowing it inside a block.
        """
        extractor = _WalrusExtractor()
        new_test = extractor.visit(test)
        lines: list[str] = []
        for node, replacement in extractor.replacements:
            name = node.target.id
            scope.expr_types[id(replacement)] = scope.var_types.get(name, ANY)
            if name not in declared:
                vtype = scope.var_types.get(name, ANY)
                type_ann = f": {vtype.to_rust_type()}" if vtype != UNKNOWN else ""
                lines.append(f"let mut {name}{type_ann} = Default::default();")
                declared.add(name)
        for node, _replacement in extractor.replacements:
            lines.append(f"{node.target.id} = {self._expr(node.value, scope, plan)};")
        return lines, new_test

    def _emit_raise(self, stmt: ast.Raise, scope: ScopeInfo, plan: OwnershipPlan) -> None:
        exc = stmt.exc
        if exc is None:
            raise PyRustError(
                "bare `raise` (re-raise) is not supported by PyRust yet",
                lineno=stmt.lineno,
                col=stmt.col_offset,
                filename=self.parsed.filename,
            )
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and not exc.keywords:
            ename = exc.func.id
            self._check_builtin_exception(ename, stmt)
            if not exc.args:
                self.emit(f'py_raise("{ename}".to_string());')
            elif len(exc.args) == 1:
                # str(KeyError(k)) is repr(k); other exceptions use str(arg).
                if ename == "KeyError":
                    arg_code = f"py_repr(&{self._expr(exc.args[0], scope, plan)})"
                else:
                    arg_code = self._print_arg(exc.args[0], scope, plan)
                self.emit(f'py_raise(format!("{ename}: {{}}", {arg_code}));')
            else:
                tup = ast.Tuple(elts=list(exc.args), ctx=ast.Load())
                scope.expr_types[id(tup)] = TupleType(
                    tuple(scope.expr_types.get(id(a), UNKNOWN) for a in exc.args)
                )
                self.emit(f'py_raise(format!("{ename}: {{}}", {self._print_arg(tup, scope, plan)}));')
            return
        if isinstance(exc, ast.Name):
            self._check_builtin_exception(exc.id, stmt)
            self.emit(f'py_raise("{exc.id}".to_string());')
            return
        raise PyRustError(
            "only `raise ErrorType(message)` is supported by PyRust",
            lineno=stmt.lineno,
            col=stmt.col_offset,
            filename=self.parsed.filename,
        )

    _BUILTIN_EXCEPTIONS = {
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
        "ZeroDivisionError", "ArithmeticError", "AssertionError", "AttributeError",
        "NotImplementedError", "OverflowError", "LookupError", "NameError", "OSError",
        "FileNotFoundError", "PermissionError", "StopIteration", "SystemError",
        "UnicodeError", "EOFError", "RecursionError", "MemoryError", "TimeoutError",
    }

    def _check_builtin_exception(self, name: str, node: ast.AST) -> None:
        # A user-defined exception class would print differently (and may
        # carry custom __str__); only builtin exceptions are native.
        if name not in self._BUILTIN_EXCEPTIONS or name in self.env.class_types:
            raise PyRustError(
                f"raising '{name}' is not supported natively (only builtin exception types are)",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )

    @staticmethod
    def _immutable(t: PyType) -> bool:
        if t in (INT, FLOAT, BOOL, STR, NONE):
            return True
        if isinstance(t, TupleType):
            return all(_RustGenerator._immutable(e) for e in t.elements)
        return False

    def _emit_delete(self, stmt: ast.Delete, scope: ScopeInfo, plan: OwnershipPlan) -> None:
        for target in stmt.targets:
            if isinstance(target, ast.Subscript):
                base_t = scope.expr_types.get(id(target.value), UNKNOWN)
                if self._has_subscript(target.value) or self._root_of(target) in self._ref_names:
                    pre: list[str] = []
                    base = self._place(target.value, scope, plan, pre)
                    for line in pre:
                        self.emit(line)
                else:
                    base = self._recv_no_clone(target.value, scope, plan)
                if isinstance(base_t, DictType):
                    key = self._expr(target.slice, scope, plan, expected_type=base_t.key)
                    self.emit(f"{base}.del_key(&{key});")
                    continue
                if isinstance(base_t, ListType) and not isinstance(target.slice, ast.Slice):
                    idx = self._expr(target.slice, scope, plan, expected_type=INT)
                    self.emit(f"py_list_del(&mut {base}, {idx});")
                    continue
                if isinstance(base_t, ListType) and isinstance(target.slice, ast.Slice):
                    if target.slice.step is not None:
                        raise PyRustError(
                            "extended slice deletion (del a[::2]) is not supported by PyRust yet",
                            lineno=getattr(target, "lineno", stmt.lineno),
                            col=getattr(target, "col_offset", stmt.col_offset),
                            filename=self.parsed.filename,
                        )
                    low = f"Some({self._expr(target.slice.lower, scope, plan)})" if target.slice.lower else "None"
                    high = f"Some({self._expr(target.slice.upper, scope, plan)})" if target.slice.upper else "None"
                    elem_rt = base_t.elem.to_rust_type()
                    self.emit(f"py_splice(&mut {base}, {low}, {high}, Vec::<{elem_rt}>::new());")
                    continue
            raise PyRustError(
                "only `del dict[key]`, `del list[i]` and `del list[i:j]` are supported by PyRust",
                lineno=getattr(target, "lineno", stmt.lineno),
                col=getattr(target, "col_offset", stmt.col_offset),
                filename=self.parsed.filename,
            )

    def _emit_assignment(
        self,
        target: ast.expr,
        value: ast.expr,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        declared: set[str],
    ) -> None:
        if isinstance(target, ast.Name):
            name = target.id
            var_type = scope.var_types.get(name, ANY)
            val_code = self._expr(value, scope, plan, expected_type=var_type)
            if name not in declared:
                mut_kw = "mut " if name in plan.mutable_vars else ""
                type_ann = f": {var_type.to_rust_type()}" if var_type != UNKNOWN else ""
                self.emit(f"let {mut_kw}{name}{type_ann} = {val_code};")
                declared.add(name)
            else:
                self.emit(f"{name} = {val_code};")
            return

        if isinstance(target, (ast.Tuple, ast.List)):
            vt = scope.expr_types.get(id(value), UNKNOWN)
            simple = all(isinstance(e, ast.Name) for e in target.elts) and isinstance(vt, TupleType) \
                and len(vt.elements) == len(target.elts)
            if not simple:
                self._emit_unpack(target, value, vt, scope, plan, declared)
                return
            names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
            val_code = self._expr(value, scope, plan)
            if all(n not in declared for n in names):
                pat = ", ".join(
                    f"{'mut ' if n in plan.mutable_vars else ''}{n}" for n in names
                )
                self.emit(f"let ({pat}) = {val_code};")
                declared.update(names)
            else:
                for n in names:
                    if n not in declared:
                        nt = scope.var_types.get(n, ANY).to_rust_type()
                        self.emit(f"let mut {n}: {nt} = Default::default();")
                        declared.add(n)
                pat = ", ".join(names)
                self.emit(f"({pat}) = {val_code};")
            return

        if isinstance(target, ast.Attribute):
            attr_t = scope.expr_types.get(id(target), None)
            if self._has_subscript(target.value) or self._root_of(target) in self._ref_names:
                base_t = scope.expr_types.get(id(target.value))
                if isinstance(base_t, ClassType):
                    latest = self.env.class_types.get(base_t.name, base_t)
                    attr_t = latest.fields.get(target.attr, attr_t)
                if attr_t is None:
                    attr_t = UNKNOWN
                self._loop_counter += 1
                vtmp = f"__pv{self._loop_counter}"
                self.emit(f"let {vtmp} = {self._expr(value, scope, plan, expected_type=attr_t)};")
                pre: list[str] = []
                place = self._place(target, scope, plan, pre)
                for line in pre:
                    self.emit(line)
                self.emit(f"{place} = {vtmp};")
                return
            lhs = self._attribute_lvalue(target, scope, plan)
            if (
                attr_t is None
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                cls_t = scope.var_types.get("self")
                if isinstance(cls_t, ClassType):
                    attr_t = cls_t.fields.get(target.attr)
            rhs = self._expr(value, scope, plan, expected_type=attr_t)
            self.emit(f"{lhs} = {rhs};")
            return

        if isinstance(target, ast.Subscript):
            base_t = scope.expr_types.get(id(target.value), UNKNOWN)
            if base_t in (ANY, UNKNOWN):
                raise PyRustError(
                    "item assignment on a dynamically-typed value is not supported natively",
                    lineno=getattr(target, "lineno", None),
                    col=getattr(target, "col_offset", None),
                    filename=self.parsed.filename,
                )
            if self._has_subscript(target.value) or self._root_of(target) in self._ref_names:
                # Nested place: evaluate value, then indices, then write in place.
                self._loop_counter += 1
                vtmp = f"__pv{self._loop_counter}"
                if isinstance(base_t, DictType):
                    self.emit(f"let {vtmp} = {self._expr(value, scope, plan, expected_type=base_t.val)};")
                elif isinstance(base_t, ListType) and not isinstance(target.slice, ast.Slice):
                    self.emit(f"let {vtmp} = {self._expr(value, scope, plan, expected_type=base_t.elem)};")
                elif isinstance(base_t, ListType):
                    self.emit(f"let {vtmp} = {self._expr(value, scope, plan, expected_type=base_t)};")
                else:
                    raise PyRustError(
                        "this item assignment is not supported natively",
                        lineno=getattr(target, "lineno", None),
                        col=getattr(target, "col_offset", None),
                        filename=self.parsed.filename,
                    )
                pre: list[str] = []
                base_place = self._place(target.value, scope, plan, pre)
                for line in pre:
                    self.emit(line)
                if isinstance(base_t, DictType):
                    k_code = self._expr(target.slice, scope, plan, expected_type=base_t.key)
                    self.emit(f"{base_place}.insert({k_code}, {vtmp});")
                elif isinstance(target.slice, ast.Slice):
                    if target.slice.step is not None:
                        raise PyRustError(
                            "extended slice assignment (a[::2] = ...) is not supported by PyRust yet",
                            lineno=getattr(target, "lineno", None),
                            col=getattr(target, "col_offset", None),
                            filename=self.parsed.filename,
                        )
                    low = f"Some({self._expr(target.slice.lower, scope, plan, expected_type=INT)})" if target.slice.lower else "None"
                    high = f"Some({self._expr(target.slice.upper, scope, plan, expected_type=INT)})" if target.slice.upper else "None"
                    self.emit(f"py_splice(&mut {base_place}, {low}, {high}, {vtmp});")
                else:
                    idx = self._expr(target.slice, scope, plan, expected_type=INT)
                    self.emit(f"py_set_index(&mut {base_place}, {idx}, {vtmp});")
                return
            base_code = self._recv_no_clone(target.value, scope, plan)
            if isinstance(base_t, DictType):
                k_code = self._expr(target.slice, scope, plan, expected_type=base_t.key)
                v_code = self._expr(value, scope, plan, expected_type=base_t.val)
                self.emit(f"{base_code}.insert({k_code}, {v_code});")
            elif isinstance(base_t, ListType) and isinstance(target.slice, ast.Slice):
                if target.slice.step is not None:
                    raise PyRustError(
                        "extended slice assignment (a[::2] = ...) is not supported by PyRust yet",
                        lineno=getattr(target, "lineno", None),
                        col=getattr(target, "col_offset", None),
                        filename=self.parsed.filename,
                    )
                low = f"Some({self._expr(target.slice.lower, scope, plan)})" if target.slice.lower else "None"
                high = f"Some({self._expr(target.slice.upper, scope, plan)})" if target.slice.upper else "None"
                elem_t = base_t.elem
                v_code = self._expr(value, scope, plan, expected_type=ListType(elem_t))
                self.emit(f"py_splice(&mut {base_code}, {low}, {high}, {v_code});")
            elif base_t == STR or isinstance(base_t, TupleType):
                raise PyRustError(
                    "item assignment on strings/tuples is not supported (they are immutable)",
                    lineno=getattr(target, "lineno", None),
                    col=getattr(target, "col_offset", None),
                    filename=self.parsed.filename,
                )
            else:
                idx_code = self._expr(target.slice, scope, plan)
                elem_t = base_t.elem if isinstance(base_t, ListType) else None
                v_code = self._expr(value, scope, plan, expected_type=elem_t)
                self.emit(f"py_set_index(&mut {base_code}, {idx_code}, {v_code});")
            return

    def _emit_unpack(
        self,
        target: ast.Tuple | ast.List,
        value: ast.expr,
        vt: PyType,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        declared: set[str],
    ) -> None:
        """`a, b[i] = value`: evaluate the right side once, then assign left to right."""
        n = len(target.elts)

        def err(msg: str) -> PyRustError:
            return PyRustError(msg, lineno=getattr(target, "lineno", None),
                               col=getattr(target, "col_offset", None), filename=self.parsed.filename)

        if any(isinstance(e, ast.Starred) for e in target.elts):
            raise err("starred unpacking (a, *b = ...) is not supported by PyRust yet")
        self._loop_counter += 1
        tmp = f"__unpack{self._loop_counter}"
        self.emit(f"let {tmp} = {self._expr(value, scope, plan)};")
        if isinstance(vt, TupleType):
            if len(vt.elements) != n:
                raise err(f"cannot unpack {len(vt.elements)} values into {n} targets")
            items = [(f"{tmp}.{i}", et) for i, et in enumerate(vt.elements)]
        elif isinstance(vt, ListType):
            self.emit(f"py_unpack_check({tmp}.len(), {n});")
            items = [(f"{tmp}[{i}]", vt.elem) for i in range(n)]
        else:
            raise err(f"unpacking a {vt.to_rust_type()} value is not supported by PyRust yet")
        for elt, (code, et) in zip(target.elts, items):
            ref = ast.Name(id=code if et.is_copy else f"{code}.clone()", ctx=ast.Load())
            scope.expr_types[id(ref)] = et
            self._emit_assignment(elt, ref, scope, plan, declared)

    def _attribute_lvalue(self, node: ast.Attribute, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return f"{self.self_var_name}.{node.attr}"
        return f"{self._recv_no_clone(node.value, scope, plan)}.{node.attr}"

    def _recv_no_clone(self, node: ast.expr, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        """Receiver for method calls / &mut bases: never clone (borrows)."""
        code = self._expr(node, scope, plan)
        # _expr clones non-Copy values (e.g. self.d.clone()); for a method
        # receiver that would mutate a temporary. Strip one trailing clone.
        if code.endswith(".clone()"):
            return code[: -len(".clone()")]
        return code

    def _pattern_target(self, target: ast.expr, plan: OwnershipPlan | None = None) -> str:
        if isinstance(target, ast.Name):
            if plan and target.id in plan.mutable_vars:
                return f"mut {target.id}"
            return target.id
        if isinstance(target, (ast.Tuple, ast.List)):
            return "(" + ", ".join(self._pattern_target(e, plan) for e in target.elts) + ")"
        return "_"

    def _boolop_expr(
        self,
        node: ast.BoolOp,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        result_type: PyType,
    ) -> str:
        """`a or b` -> `if truthy(a) { a } else { b }` (value semantics)."""
        if result_type == BOOL:
            op_str = " || " if isinstance(node.op, ast.Or) else " && "
            return "(" + op_str.join(self._cond_expr(v, scope, plan) for v in node.values) + ")"

        def emit_value(v: ast.expr) -> str:
            return self._expr(v, scope, plan, expected_type=result_type)

        def emit_chain(values: list[ast.expr]) -> str:
            first = values[0]
            cond = self._cond_expr(first, scope, plan)
            first_code = emit_value(first)
            if len(values) == 2:
                last_code = emit_value(values[1])
            else:
                last_code = emit_chain(values[1:])
            if isinstance(node.op, ast.Or):
                return f"if {cond} {{ {first_code} }} else {{ {last_code} }}"
            return f"if {cond} {{ {last_code} }} else {{ {first_code} }}"

        return emit_chain(list(node.values))

    def _comp_iter_chain(
        self,
        gen: ast.comprehension,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        iter_code = self._for_iter_expr(gen.iter, scope, plan)
        target_pat = self._pattern_target(gen.target)
        it = scope.expr_types.get(id(gen.iter), UNKNOWN)
        if isinstance(it, (ListType, SetType)):
            iter_elem: PyType = it.elem
        elif isinstance(it, DictType):
            iter_elem = it.key
        else:
            iter_elem = ANY
        chain = iter_code
        for cond in gen.ifs:
            cond_code = self._cond_expr(cond, scope, plan)
            if iter_elem.is_copy:
                chain += f".filter(|&{target_pat}| {cond_code})"
            else:
                chain += f".filter(|{target_pat}| {{ let {target_pat} = {target_pat}.clone(); {cond_code} }})"
        return chain

    def _check_no_walrus(self, node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.NamedExpr):
                raise PyRustError(
                    "the walrus operator inside a comprehension is not supported natively",
                    lineno=getattr(sub, "lineno", None),
                    col=getattr(sub, "col_offset", None),
                    filename=self.parsed.filename,
                )

    def _check_single_generator(self, node: ast.expr, kind: str) -> ast.comprehension:
        self._check_no_walrus(node)
        gens = getattr(node, "generators", [])
        if len(gens) != 1:
            raise PyRustError(
                f"{kind} with multiple generators is not supported by PyRust yet",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )
        return gens[0]

    def _dictcomp_expr(
        self,
        node: ast.DictComp,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        actual_type: PyType,
    ) -> str:
        gen = self._check_single_generator(node, "dict comprehensions")
        chain = self._comp_iter_chain(gen, scope, plan)
        k_t = actual_type.key if isinstance(actual_type, DictType) else None
        v_t = actual_type.val if isinstance(actual_type, DictType) else None
        k_code = self._expr(node.key, scope, plan, expected_type=k_t)
        v_code = self._expr(node.value, scope, plan, expected_type=v_t)
        target_pat = self._pattern_target(gen.target)
        chain += f".map(|{target_pat}| ({k_code}, {v_code})).collect::<PyDict<_, _>>()"
        return chain

    def _setcomp_expr(
        self,
        node: ast.SetComp,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        actual_type: PyType,
    ) -> str:
        gen = self._check_single_generator(node, "set comprehensions")
        chain = self._comp_iter_chain(gen, scope, plan)
        elem_t = actual_type.elem if isinstance(actual_type, SetType) else None
        elt_code = self._expr(node.elt, scope, plan, expected_type=elem_t)
        target_pat = self._pattern_target(gen.target)
        chain += f".map(|{target_pat}| {elt_code}).collect::<HashSet<_>>()"
        return chain

    def _formatted_value(
        self,
        part: ast.FormattedValue,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        val_t: PyType,
    ) -> tuple[str, str]:
        if part.conversion == 97:  # !a
            raise PyRustError(
                "f-string !a conversion is not supported by PyRust yet",
                lineno=getattr(part, "lineno", None),
                col=getattr(part, "col_offset", None),
                filename=self.parsed.filename,
            )
        spec_node = part.format_spec
        spec: str | None
        if spec_node is None or (isinstance(spec_node, ast.JoinedStr) and not spec_node.values):
            spec = None
        elif isinstance(spec_node, ast.JoinedStr) and len(spec_node.values) == 1 and isinstance(
            spec_node.values[0], ast.Constant
        ) and isinstance(spec_node.values[0].value, str):
            spec = spec_node.values[0].value
        else:
            raise PyRustError(
                "nested f-string format specs (e.g. f'{x:{w}}') are not supported by PyRust yet",
                lineno=getattr(spec_node, "lineno", None),
                col=getattr(spec_node, "col_offset", None),
                filename=self.parsed.filename,
            )
        if part.conversion in (114, 115):  # !r / !s: format the resulting str
            self._check_printable(val_t, part.value)
            fn = "py_repr" if part.conversion == 114 else "py_str"
            base = f"{fn}(&{self._expr(part.value, scope, plan)})"
            if spec is None:
                return "{}", base
            return "{}", f"py_format(&{base}, {self._rust_str_lit(spec)})"
        if spec is None:
            return "{}", self._print_arg(part.value, scope, plan)
        return "{}", self._format_with_spec(part.value, spec, scope, plan)

    @staticmethod
    def _rust_str_lit(s: str) -> str:
        return '"' + _RustGenerator._rust_string_escape(s) + '"'

    def _format_with_spec(self, node: ast.expr, spec: str, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        """`format(value, spec)` with Python's format-spec mini-language."""
        t = scope.expr_types.get(id(node), UNKNOWN)
        if t not in (INT, FLOAT, BOOL, STR, ANY):
            raise PyRustError(
                f"format spec '{spec}' on a {t.to_rust_type()} value is not supported by PyRust yet",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )
        return f"py_format(&{self._expr(node, scope, plan)}, {self._rust_str_lit(spec)})"

    def _str_format_call(self, node: ast.Call, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        """`"{} {x}".format(a, x=1)` -> `format!("{}", a, x=1)`-style Rust."""
        fmt_node = node.func.value
        if not (isinstance(fmt_node, ast.Constant) and isinstance(fmt_node.value, str)):
            raise PyRustError(
                "only constant format strings are supported in str.format()",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )
        fmt = fmt_node.value
        kw_map: dict[str, ast.expr] = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise PyRustError(
                    "** unpacking in str.format() is not supported by PyRust yet",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            kw_map[kw.arg] = kw.value
        out_parts: list[str] = []
        out_args: list[str] = []
        auto_idx = 0
        saw_auto = False
        saw_manual = False
        i = 0
        n = len(fmt)
        literal: list[str] = []

        def flush_literal() -> None:
            if literal:
                out_parts.append("".join(literal))
                literal.clear()

        while i < n:
            ch = fmt[i]
            if ch == "{":
                if i + 1 < n and fmt[i + 1] == "{":
                    literal.append("{{")
                    i += 2
                    continue
                j = fmt.find("}", i + 1)
                if j < 0:
                    raise PyRustError(
                        "unmatched '{' in format string",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                field = fmt[i + 1 : j]
                name, _, spec = field.partition(":")
                conv: str | None = None
                if "!" in name:
                    name, _, conv = name.partition("!")
                if conv not in (None, "", "s", "r"):
                    raise PyRustError(
                        f"conversion '!{conv}' is not supported in str.format()",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                if name == "":
                    saw_auto = True
                    if saw_manual:
                        raise PyRustError(
                            "cannot mix automatic and manual field numbering in str.format()",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    if auto_idx >= len(node.args):
                        raise PyRustError(
                            "not enough arguments for str.format()",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    arg_node = node.args[auto_idx]
                    auto_idx += 1
                elif name.isdigit():
                    saw_manual = True
                    if saw_auto:
                        raise PyRustError(
                            "cannot mix automatic and manual field numbering in str.format()",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    idx = int(name)
                    if idx >= len(node.args):
                        raise PyRustError(
                            "not enough arguments for str.format()",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    arg_node = node.args[idx]
                else:
                    if name not in kw_map:
                        raise PyRustError(
                            f"missing keyword argument '{name}' in str.format()",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    arg_node = kw_map[name]
                arg_t = scope.expr_types.get(id(arg_node), UNKNOWN)
                flush_literal()
                out_parts.append("{}")
                if conv in ("r", "s"):
                    self._check_printable(arg_t, arg_node)
                    fn = "py_repr" if conv == "r" else "py_str"
                    base = f"{fn}(&{self._expr(arg_node, scope, plan)})"
                    out_args.append(f"py_format(&{base}, {self._rust_str_lit(spec)})" if spec else base)
                elif spec:
                    if "{" in spec:
                        raise PyRustError(
                            "nested replacement fields in format specs are not supported by PyRust yet",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    out_args.append(self._format_with_spec(arg_node, spec, scope, plan))
                else:
                    out_args.append(self._print_arg(arg_node, scope, plan))
                i = j + 1
            elif ch == "}":
                if i + 1 < n and fmt[i + 1] == "}":
                    literal.append("}}")
                    i += 2
                    continue
                raise PyRustError(
                    "unmatched '}' in format string",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            else:
                literal.append(ch)
                i += 1
        flush_literal()
        fmt_str = "".join(out_parts)
        if out_args:
            return f'format!("{fmt_str}", {", ".join(out_args)})'
        return f'"{fmt_str}".to_string()'

    def _percent_format(
        self, fmt_node: ast.expr, args_node: ast.expr, node: ast.BinOp, scope: ScopeInfo, plan: OwnershipPlan
    ) -> str:
        """`"hi %s" % name` -> `format!("hi {}", name)`."""
        if not (isinstance(fmt_node, ast.Constant) and isinstance(fmt_node.value, str)):
            raise PyRustError(
                "only constant format strings are supported in % formatting",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )
        fmt = fmt_node.value
        if isinstance(args_node, ast.Tuple):
            arg_nodes = list(args_node.elts)
        else:
            single_t = scope.expr_types.get(id(args_node), UNKNOWN)
            if isinstance(single_t, (TupleType, DictType)) or single_t in (ANY, UNKNOWN):
                # A tuple (or mapping) on the right supplies several arguments.
                raise PyRustError(
                    "% formatting with a non-literal tuple/mapping/dynamic operand is not supported by PyRust yet",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            arg_nodes = [args_node]
        out_parts: list[str] = []
        out_args: list[str] = []
        i = 0
        k = 0
        n = len(fmt)
        while i < n:
            ch = fmt[i]
            if ch != "%":
                out_parts.append("{{" if ch == "{" else ("}}" if ch == "}" else ch))
                i += 1
                continue
            if i + 1 < n and fmt[i + 1] == "%":
                out_parts.append("%")
                i += 2
                continue
            j = i + 1
            while j < n and fmt[j] in "#0- +":
                j += 1
            while j < n and fmt[j].isdigit():
                j += 1
            if j < n and fmt[j] == ".":
                j += 1
                while j < n and fmt[j].isdigit():
                    j += 1
            if j >= n:
                raise PyRustError(
                    "incomplete % format specifier",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            conv = fmt[j]
            if conv == "(":
                raise PyRustError(
                    "%(name)s mapping keys are not supported in % formatting",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            if k >= len(arg_nodes):
                raise PyRustError(
                    "not enough arguments for % formatting",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            arg_node = arg_nodes[k]
            k += 1
            arg_t = scope.expr_types.get(id(arg_node), UNKNOWN)
            if conv not in ("s", "d", "i", "u", "r", "f", "F", "e", "E", "g", "G", "x", "X", "o"):
                raise PyRustError(
                    f"%{conv} is not supported in % formatting",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            out_parts.append("{}")
            out_args.append(self._percent_piece(fmt[i + 1 : j], conv, arg_node, arg_t, node, scope, plan))
            i = j + 1
        fmt_str = "".join(out_parts)
        if k != len(arg_nodes):
            raise PyRustError(
                "too many arguments for % formatting",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )
        return f'format!("{fmt_str}", {", ".join(out_args)})'

    def _percent_piece(
        self,
        body: str,
        conv: str,
        arg_node: ast.expr,
        arg_t: PyType,
        node: ast.AST,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        """One `%`-conversion, expressed through Python's format-spec engine."""
        m = re.match(r"^([-+ 0#]*)(\d*)(?:\.(\d+))?$", body)

        def err() -> PyRustError:
            return PyRustError(
                f"format '%{body}{conv}' is not supported by PyRust yet",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )

        if not m:
            raise err()
        flags, width, prec = m.group(1), m.group(2), m.group(3)
        left = "-" in flags
        if conv in ("s", "r"):
            self._check_printable(arg_t, arg_node)
            fn = "py_repr" if conv == "r" else "py_str"
            spec = ("<" if left else ">") + width + (f".{prec}" if prec is not None else "")
            base = f"{fn}(&{self._expr(arg_node, scope, plan)})"
            return f"py_format(&{base}, {self._rust_str_lit(spec)})" if spec not in ("<", ">") else base
        sign = "+" if "+" in flags else (" " if " " in flags else "")
        align = "<" if left else ""
        zero = "0" if ("0" in flags and not left) else ""
        alt = "#" if "#" in flags else ""
        if conv in ("d", "i", "u", "x", "X", "o"):
            if prec is not None:
                raise err()
            if arg_t == INT:
                val = self._expr(arg_node, scope, plan)
            elif arg_t == BOOL:
                val = f"({self._expr(arg_node, scope, plan)} as i64)"
            elif arg_t == FLOAT and conv in ("d", "i", "u"):
                val = f"py_int_of_float({self._expr(arg_node, scope, plan)})"
            else:
                raise err()
            kind = "d" if conv in ("d", "i", "u") else conv
            spec = f"{align}{sign}{alt}{zero}{width}{kind}"
            return f"py_format(&{val}, {self._rust_str_lit(spec)})"
        # float conversions
        if arg_t in (INT, BOOL):
            val = f"({self._expr(arg_node, scope, plan)} as f64)"
        elif arg_t == FLOAT:
            val = self._expr(arg_node, scope, plan)
        else:
            raise err()
        p = f".{prec}" if prec is not None else ""
        spec = f"{align}{sign}{alt}{zero}{width}{p}{conv}"
        return f"py_format(&{val}, {self._rust_str_lit(spec)})"

    def _for_iter_expr(self, iter_node: ast.expr, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name):
            fname = iter_node.func.id
            if fname == "range":
                return self._range_iter(iter_node, scope, plan)
            if fname == "enumerate" and len(iter_node.args) == 1:
                it = self._for_iter_expr(iter_node.args[0], scope, plan)
                return f"{it}.enumerate().map(|(i, v)| (i as i64, v))"
        iter_t = scope.expr_types.get(id(iter_node), UNKNOWN)
        self._check_ordered_iteration(iter_node, scope)
        if isinstance(iter_t, ClassType) and iter_t.name == "PyFile":
            iter_code = self._expr(iter_node, scope, plan)
            return f"({iter_code}).lines().into_iter()"
        iter_code = self._expr(iter_node, scope, plan)
        if iter_t == STR:
            return f"{iter_code}.chars().map(|c| c.to_string()).collect::<Vec<String>>().into_iter()"
        if isinstance(iter_t, DictType):
            return f"{iter_code}.keys().into_iter()"
        if iter_t == ANY:
            return f"py_value_iter(&{iter_code}).into_iter()"
        if isinstance(iter_t, (ListType, TupleType)) or iter_t == UNKNOWN:
            return f"{iter_code}.iter().cloned()"
        raise PyRustError(
            f"iterating over a {iter_t.to_rust_type()} value is not supported by PyRust yet",
            lineno=getattr(iter_node, "lineno", None),
            col=getattr(iter_node, "col_offset", None),
            filename=self.parsed.filename,
        )

    def _range_iter(self, node: ast.Call, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        """A Rust iterator over `range(...)` with Python semantics."""
        for a in node.args:
            if scope.expr_types.get(id(a), UNKNOWN) not in (INT, BOOL):
                raise PyRustError(
                    "range() arguments must be ints",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
        args = [self._expr(a, scope, plan, expected_type=INT) for a in node.args]
        if node.keywords or not 1 <= len(args) <= 3:
            raise PyRustError(
                "range() takes 1 to 3 positional arguments",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )
        if len(args) == 1:
            return f"(0..{args[0]})"
        if len(args) == 2:
            return f"({args[0]}..{args[1]})"
        step = node.args[2]
        if isinstance(step, ast.Constant) and type(step.value) is int and step.value > 0:
            return f"({args[0]}..{args[1]}).step_by({step.value})"
        return f"py_range_iter({args[0]}, {args[1]}, {args[2]})"

    def _check_ordered_iteration(self, node: ast.expr, scope: ScopeInfo) -> None:
        """Iterating a set observes CPython's hash-table order, which native
        code cannot reproduce; refuse it (sorted(), len(), `in` are fine)."""
        t = scope.expr_types.get(id(node), UNKNOWN)
        if isinstance(t, SetType):
            raise PyRustError(
                "iterating over a set is not supported natively (CPython's set order cannot be "
                "reproduced); iterate over sorted(the_set) instead",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )

    def _sum_call(self, node: ast.Call, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
        inner = self._expr(node.args[0], scope, plan)
        if isinstance(at, ListType):
            elem, seq = at.elem, f"&{inner}"
        elif isinstance(at, SetType) and at.elem in (INT, BOOL):
            elem, seq = at.elem, f"&{inner}.iter().cloned().collect::<Vec<_>>()"
        elif isinstance(at, DictType):
            elem, seq = at.key, f"&{inner}.keys()"
        else:
            raise PyRustError(
                f"sum() of a {at.to_rust_type()} value is not supported by PyRust yet",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )
        start_t = scope.expr_types.get(id(node.args[1]), UNKNOWN) if len(node.args) == 2 else INT
        start = self._expr(node.args[1], scope, plan) if len(node.args) == 2 else None
        if elem == BOOL:
            seq = f"&{seq[1:]}.iter().map(|b| *b as i64).collect::<Vec<i64>>()"
            elem = INT
        if elem in (INT, UNKNOWN) and start_t in (INT, BOOL):
            total = f"py_sum_ints({seq})"
            return total if start is None else f"({total} + {self._expr(node.args[1], scope, plan, expected_type=INT)})"
        if elem == FLOAT and start is None:
            return f"py_sum_floats({seq})"
        if elem in (FLOAT, INT) and start_t == FLOAT:
            if elem == INT:
                seq = f"&{seq[1:]}.iter().map(|v| *v as f64).collect::<Vec<f64>>()"
            return f"py_sum_floats_start({start}, {seq})"
        if elem == ANY:
            total = f"py_sum_values({seq})"
            return total if start is None else f"({total} + {self._expr(node.args[1], scope, plan, expected_type=ANY)})"
        raise PyRustError(
            "this sum() combination is not supported by PyRust yet",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _cond_expr(self, node: ast.expr, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        t = scope.expr_types.get(id(node), UNKNOWN)
        code = self._expr(node, scope, plan)
        if t == ANY:
            return f"{code}.is_truthy()"
        if t == INT:
            return f"({code} != 0)"
        if t == FLOAT:
            return f"({code} != 0.0)"
        if t == STR or isinstance(t, (ListType, DictType, SetType)):
            return f"(!{code}.is_empty())"
        return code

    def _expr(
        self,
        node: ast.expr,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        expected_type: PyType | None = None,
    ) -> str:
        if expected_type == UNKNOWN:
            expected_type = ANY  # an unrefined slot is a PyValue in Rust
        if expected_type == ANY and isinstance(node, ast.Constant) and node.value is None:
            return "PyValue::None"
        actual_type = scope.expr_types.get(id(node), UNKNOWN)
        raw = self._expr_raw(node, scope, plan, actual_type)
        if expected_type == ANY and actual_type not in (ANY, UNKNOWN):
            if not self._pyvalue_convertible(actual_type):
                raise PyRustError(
                    f"a {actual_type.to_rust_type()} value cannot be stored in a dynamically-typed slot natively",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            return f"PyValue::from({raw})"
        if expected_type == FLOAT and actual_type == INT:
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                return f"{float(node.value)}"
            return f"({raw} as f64)"
        if expected_type == FLOAT and actual_type in (ANY, UNKNOWN):
            return f"((({raw})).as_f64())"
        if expected_type == INT and actual_type == FLOAT:
            # Python never truncates a float silently where an int is needed.
            raise PyRustError(
                "a float value is used where an int is required",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )
        if expected_type == INT and actual_type == BOOL:
            return f"({raw} as i64)"
        if expected_type == INT and actual_type in (ANY, UNKNOWN):
            return f"((({raw})).as_i64())"
        if expected_type == BOOL and actual_type in (ANY, UNKNOWN):
            return f"((({raw})).is_truthy())"
        return raw

    @staticmethod
    def _pyvalue_convertible(t: PyType) -> bool:
        if t in (INT, FLOAT, BOOL, STR, NONE, ANY, UNKNOWN):
            return True
        if isinstance(t, ListType):
            return _RustGenerator._pyvalue_convertible(t.elem)
        if isinstance(t, DictType):
            return t.key == STR and _RustGenerator._pyvalue_convertible(t.val)
        return False

    def _expr_raw(
        self,
        node: ast.expr,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        actual_type: PyType,
    ) -> str:
        if isinstance(node, ast.NamedExpr):
            # Nested walrus (e.g. inside a comparison): evaluate in a block.
            val_code = self._expr(node.value, scope, plan)
            vtype = scope.var_types.get(node.target.id, ANY)
            type_ann = f": {vtype.to_rust_type()}" if vtype != UNKNOWN else ""
            return f"{{ let mut {node.target.id}{type_ann} = {val_code}; {node.target.id} }}"

        if isinstance(node, ast.Constant):
            val = node.value
            if val is None:
                return "PyValue::None" if actual_type == ANY else "()"
            if isinstance(val, bool):
                return "true" if val else "false"
            if isinstance(val, int):
                return str(val)
            if isinstance(val, float):
                if _pymath.isnan(val):
                    return "f64::NAN"
                if _pymath.isinf(val):
                    return "f64::INFINITY" if val > 0 else "f64::NEG_INFINITY"
                s = repr(val)
                s = s if ("." in s or "e" in s or "E" in s) else f"{s}.0"
                # Suffix pins the type so method calls on literals (e.g. math
                # expansions like `(16.0f64).sqrt()`) never hit E0689.
                return f"{s}f64"
            if isinstance(val, str):
                escaped = (
                    val.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                )
                return f'"{escaped}".to_string()'

        if isinstance(node, ast.Name):
            if node.id == "self":
                return self.self_var_name
            if (
                self.parsed.from_imports.get(node.id) == "math"
                and node.id in MATH_CONSTS
            ):
                return MATH_CONST_RUST[node.id]
            if self.parsed.from_imports.get(node.id) == "sys" and node.id == "argv":
                return "std::env::args().collect::<Vec<String>>()"
            src_mod = self.parsed.from_imports.get(node.id)
            if src_mod is not None and f"{src_mod}.{node.id}" in self.env.imported_consts:
                return f"crate::{src_mod}::{node.id}"
            if (
                self.in_function
                and node.id not in self._fn_locals
                and node.id not in self.module_consts
                and node.id in self.env.module_scope.var_types
            ):
                raise PyRustError(
                    f"module-level variable '{node.id}' is not visible inside functions in PyRust yet: "
                    "move it into the function, pass it as an argument, or make it an UPPER_CASE constant",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            if node.id in self._ref_names:
                # `&mut T` binding: borrow where a borrow is enough, else copy the value.
                return node.id if id(node) in plan.borrow_expr_ids else f"{node.id}.clone()"
            if node.id in plan.clone_expr_ids or id(node) in plan.clone_expr_ids:
                return f"{node.id}.clone()"
            return node.id

        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "math"
                and node.attr in MATH_CONSTS
            ):
                return MATH_CONST_RUST[node.attr]
            recv_any_t = scope.expr_types.get(id(node.value), UNKNOWN)
            if recv_any_t == ANY and not (
                isinstance(node.value, ast.Name) and node.value.id in self.parsed.imports
            ):
                raise PyRustError(
                    f"attribute '.{node.attr}' on a dynamically-typed value is not supported natively",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                if node.attr == "argv":
                    return "std::env::args().collect::<Vec<String>>()"
                if node.attr == "platform":
                    return '(if std::env::consts::FAMILY == "windows" { "nt" } else { "posix" }).to_string()'
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in self.parsed.imports
                and node.value.id not in scope.var_types
                and node.value.id not in self.env.module_scope.var_types
            ):
                if f"{node.value.id}.{node.attr}" not in self.env.imported_consts:
                    raise PyRustError(
                        f"module attribute '{node.value.id}.{node.attr}' is not supported natively "
                        "(only functions, classes and numeric constants of imported modules are)",
                        lineno=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                        filename=self.parsed.filename,
                    )
                return f"crate::{node.value.id}::{node.attr}"
            recv_t = scope.expr_types.get(id(node.value), UNKNOWN)
            if isinstance(recv_t, ClassType):
                latest = self.env.class_types.get(recv_t.name, recv_t)
                if node.attr not in latest.fields:
                    if node.attr in latest.methods:
                        raise PyRustError(
                            f"'{recv_t.name}.{node.attr}' is a method: call it with parentheses",
                            lineno=getattr(node, "lineno", None),
                            col=getattr(node, "col_offset", None),
                            filename=self.parsed.filename,
                        )
                    raise PyRustError(
                        f"type '{recv_t.name}' has no field '{node.attr}'",
                        lineno=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                        filename=self.parsed.filename,
                    )
            recv = self._expr(node.value, scope, plan)
            # Use final field type for clone decision: inference expr_types
            # for self.attr can be stale (e.g. INT in an early fixed-point
            # round) while the struct field finalized to PyValue.
            field_t = actual_type
            if isinstance(recv_t, ClassType):
                latest = self.env.class_types.get(recv_t.name)
                if latest is not None and node.attr in latest.fields:
                    field_t = latest.fields[node.attr]
            if not field_t.is_copy:
                # Avoid double .clone().clone() when recv already cloned.
                if recv.endswith(".clone()"):
                    # recv is e.g. `self.clone()`? No, recv is `self`, not field.
                    # Field clone still needed.
                    return f"{recv}.{node.attr}.clone()"
                return f"{recv}.{node.attr}.clone()"
            return f"{recv}.{node.attr}"

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return f"!({self._cond_expr(node.operand, scope, plan)})"
            ot = scope.expr_types.get(id(node.operand), UNKNOWN)
            op_val = self._expr(node.operand, scope, plan, expected_type=INT if ot == BOOL else None)
            if isinstance(node.op, ast.USub):
                return f"(-{op_val})"
            if isinstance(node.op, ast.UAdd):
                return f"{op_val}"
            if isinstance(node.op, ast.Invert) and ot in (INT, BOOL):
                return f"(!{op_val})"

        if isinstance(node, ast.BinOp):
            return self._binop_expr(node, scope, plan, actual_type)

        if isinstance(node, ast.BoolOp):
            return self._boolop_expr(node, scope, plan, actual_type)

        if isinstance(node, ast.DictComp):
            return self._dictcomp_expr(node, scope, plan, actual_type)

        if isinstance(node, ast.SetComp):
            return self._setcomp_expr(node, scope, plan, actual_type)

        def _is_none_const(n: ast.expr) -> bool:
            return isinstance(n, ast.Constant) and n.value is None

        if isinstance(node, ast.Compare):
            parts: list[str] = []
            left_node = node.left
            for op, right_node in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    container_t = scope.expr_types.get(id(right_node), UNKNOWN)
                    if self._has_identity_object(container_t):
                        raise PyRustError(
                            "`in` over plain class instances (compared by identity in Python) is not supported natively",
                            lineno=getattr(node, "lineno", None),
                            col=getattr(node, "col_offset", None),
                            filename=self.parsed.filename,
                        )
                    c_code = self._expr(right_node, scope, plan)
                    l_code = self._expr(left_node, scope, plan)
                    if container_t == ANY:
                        check = f"py_value_contains(&{c_code}, &PyValue::from({l_code}))"
                    elif isinstance(container_t, DictType):
                        check = f"{c_code}.contains_key(&{l_code})"
                    elif container_t == STR:
                        check = f"{c_code}.contains(&*{l_code})"
                    elif isinstance(container_t, TupleType):
                        mems = " || ".join(
                            f"({l_code} == {c_code}.{i})" for i in range(len(container_t.elements))
                        )
                        check = f"({mems})" if mems else "false"
                    else:
                        check = f"{c_code}.contains(&{l_code})"
                    parts.append(f"!({check})" if isinstance(op, ast.NotIn) else check)
                else:
                    if isinstance(op, (ast.Is, ast.IsNot)) and not (
                        _is_none_const(left_node) or _is_none_const(right_node)
                    ):
                        raise PyRustError(
                            "identity comparison is only supported with None in PyRust yet "
                            "(use == to compare values)",
                            lineno=getattr(node, "lineno", None),
                            col=getattr(node, "col_offset", None),
                            filename=self.parsed.filename,
                        )
                    lt = scope.expr_types.get(id(left_node), UNKNOWN)
                    rt = scope.expr_types.get(id(right_node), UNKNOWN)
                    if self._has_identity_object(lt) or self._has_identity_object(rt):
                        raise PyRustError(
                            "comparing plain class instances (compared by identity in Python) is not supported natively",
                            lineno=getattr(node, "lineno", None),
                            col=getattr(node, "col_offset", None),
                            filename=self.parsed.filename,
                        )
                    if self._is_dict_view(left_node) or self._is_dict_view(right_node):
                        raise PyRustError(
                            "comparing dict views is not supported natively",
                            lineno=getattr(node, "lineno", None),
                            col=getattr(node, "col_offset", None),
                            filename=self.parsed.filename,
                        )
                    mixed = (lt in (INT, BOOL) and rt == FLOAT) or (lt == FLOAT and rt in (INT, BOOL))
                    if mixed and isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                        # Python compares int and float exactly (no rounding of the int).
                        flip = {ast.Lt: "gt", ast.LtE: "ge", ast.Gt: "lt", ast.GtE: "le", ast.Eq: "eq", ast.NotEq: "eq"}
                        same = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "eq"}
                        if lt == FLOAT:
                            i_node, f_node, name = right_node, left_node, flip[type(op)]
                        else:
                            i_node, f_node, name = left_node, right_node, same[type(op)]
                        i_code = self._expr(i_node, scope, plan, expected_type=INT)
                        f_code = self._expr(f_node, scope, plan)
                        call = f"py_{name}_if({i_code}, {f_code})"
                        parts.append(f"!{call}" if isinstance(op, ast.NotEq) else call)
                        left_node = right_node
                        continue
                    if _is_none_const(left_node) or _is_none_const(right_node):
                        target_t: PyType | None = ANY
                    elif lt == ANY or rt == ANY:
                        target_t = ANY
                    elif lt == FLOAT or rt == FLOAT:
                        target_t = FLOAT
                    else:
                        target_t = None
                    l_code = self._expr(left_node, scope, plan, expected_type=target_t)
                    r_code = self._expr(right_node, scope, plan, expected_type=target_t)
                    op_map = {
                        ast.Eq: "==",
                        ast.NotEq: "!=",
                        ast.Lt: "<",
                        ast.LtE: "<=",
                        ast.Gt: ">",
                        ast.GtE: ">=",
                        ast.Is: "==",
                        ast.IsNot: "!=",
                    }
                    sym = next((v for k, v in op_map.items() if isinstance(op, k)), "==")
                    parts.append(f"({l_code} {sym} {r_code})")
                left_node = right_node
            return " && ".join(parts)

        if isinstance(node, ast.IfExp):
            cond = self._cond_expr(node.test, scope, plan)
            b_code = self._expr(node.body, scope, plan, expected_type=actual_type)
            e_code = self._expr(node.orelse, scope, plan, expected_type=actual_type)
            return f"if {cond} {{ {b_code} }} else {{ {e_code} }}"

        if isinstance(node, ast.JoinedStr):
            fmt_parts: list[str] = []
            fmt_args: list[str] = []
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    escaped = (
                        part.value.replace("\\", "\\\\")
                        .replace('"', '\\"')
                        .replace("{", "{{")
                        .replace("}", "}}")
                    )
                    fmt_parts.append(escaped)
                elif isinstance(part, ast.FormattedValue):
                    val_t = scope.expr_types.get(id(part.value), UNKNOWN)
                    placeholder, arg_code = self._formatted_value(part, scope, plan, val_t)
                    fmt_parts.append(placeholder)
                    fmt_args.append(arg_code)
            fmt_str = "".join(fmt_parts)
            if fmt_args:
                return f'format!("{fmt_str}", {", ".join(fmt_args)})'
            return f'"{fmt_str}".to_string()'

        if isinstance(node, ast.List):
            elem_t = actual_type.elem if isinstance(actual_type, ListType) else None
            elts = [self._expr(e, scope, plan, expected_type=elem_t) for e in node.elts]
            return f"vec![{', '.join(elts)}]"

        if isinstance(node, ast.Set):
            elem_t = actual_type.elem if isinstance(actual_type, SetType) else None
            elts = [self._expr(e, scope, plan, expected_type=elem_t) for e in node.elts]
            return f"HashSet::from([{', '.join(elts)}])"

        if isinstance(node, ast.Dict):
            if not node.keys:
                return "PyDict::new()"
            k_t = actual_type.key if isinstance(actual_type, DictType) else None
            v_t = actual_type.val if isinstance(actual_type, DictType) else None
            pairs = [
                f"({self._expr(k, scope, plan, expected_type=k_t)}, {self._expr(v, scope, plan, expected_type=v_t)})"
                for k, v in zip(node.keys, node.values)
                if k is not None
            ]
            unpacks = [
                f"{self._expr(v, scope, plan)}.into_vec()"
                for k, v in zip(node.keys, node.values)
                if k is None
            ]
            segments = ([f"vec![{', '.join(pairs)}]"] if pairs else []) + unpacks
            if len(segments) == 1 and pairs:
                return f"PyDict::from_vec({segments[0]})"
            chain = segments[0] + ".into_iter()" if segments else "vec![].into_iter()"
            for seg in segments[1:]:
                chain += f".chain({seg}.into_iter())"
            return f"PyDict::from_vec({chain}.collect())"

        if isinstance(node, ast.Tuple):
            elts = [self._expr(e, scope, plan) for e in node.elts]
            return f"({', '.join(elts)})"

        if isinstance(node, ast.ListComp):
            self._check_no_walrus(node)
            gen = self._check_single_generator(node, "list comprehensions")
            target_pat = self._pattern_target(gen.target)
            iter_code = self._for_iter_expr(gen.iter, scope, plan)
            chain = iter_code
            elem_t = actual_type.elem if isinstance(actual_type, ListType) else ANY
            for cond in gen.ifs:
                cond_code = self._cond_expr(cond, scope, plan)
                if elem_t.is_copy:
                    chain += f".filter(|&{target_pat}| {cond_code})"
                else:
                    chain += f".filter(|{target_pat}| {{ let {target_pat} = {target_pat}.clone(); {cond_code} }})"
            elt_code = self._expr(node.elt, scope, plan)
            chain += f".map(|{target_pat}| {elt_code}).collect::<Vec<_>>()"
            return chain

        if isinstance(node, ast.Subscript):
            val_t = scope.expr_types.get(id(node.value), UNKNOWN)
            val_code = self._expr(node.value, scope, plan)
            if isinstance(node.slice, ast.Slice):
                low = f"Some({self._expr(node.slice.lower, scope, plan)})" if node.slice.lower else "None"
                high = f"Some({self._expr(node.slice.upper, scope, plan)})" if node.slice.upper else "None"
                step = self._expr(node.slice.step, scope, plan) if node.slice.step else "1"
                if val_t == STR:
                    return f"py_str_slice(&{val_code}, {low}, {high}, {step})"
                return f"py_slice(&{val_code}, {low}, {high}, {step})"
            idx_code = self._expr(node.slice, scope, plan)
            if val_t == ANY and not isinstance(node.slice, ast.Slice):
                return f"py_dyn_index(&{val_code}, &PyValue::from({idx_code}))"
            if self._is_dict_view(node.value):
                raise PyRustError(
                    "subscripting a dict view raises TypeError in Python",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            if val_t == STR:
                return f"py_str_index(&{val_code}, {idx_code})"
            if isinstance(val_t, DictType):
                key_t = scope.expr_types.get(id(node.slice), UNKNOWN)
                if key_t != val_t.key:
                    raise PyRustError(
                        "dict subscript with a key of a different/dynamic type is not supported natively",
                        lineno=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                        filename=self.parsed.filename,
                    )
                return f"{val_code}[&{idx_code}].clone()"
            if isinstance(val_t, TupleType):
                const_idx: int | None = None
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                    const_idx = node.slice.value
                elif (
                    isinstance(node.slice, ast.UnaryOp)
                    and isinstance(node.slice.op, ast.USub)
                    and isinstance(node.slice.operand, ast.Constant)
                    and isinstance(node.slice.operand.value, int)
                ):
                    const_idx = -node.slice.operand.value
                if const_idx is not None:
                    resolved = len(val_t.elements) + const_idx if const_idx < 0 else const_idx
                    return f"{val_code}.{resolved}"
            return f"py_index(&{val_code}, {idx_code})"

        if isinstance(node, ast.Call):
            return self._call_expr(node, scope, plan)

        raise PyRustError(
            f"{type(node).__name__} expressions are not supported by PyRust yet",
            lineno=getattr(node, "lineno", None),
            col=getattr(node, "col_offset", None),
            filename=self.parsed.filename,
        )

    def _binop_expr(
        self,
        node: ast.BinOp,
        scope: ScopeInfo,
        plan: OwnershipPlan,
        actual_type: PyType,
    ) -> str:
        lt = scope.expr_types.get(id(node.left), UNKNOWN)
        rt = scope.expr_types.get(id(node.right), UNKNOWN)

        if lt == ANY or rt == ANY:
            l_code = self._expr(node.left, scope, plan, expected_type=ANY)
            r_code = self._expr(node.right, scope, plan, expected_type=ANY)
            if isinstance(node.op, ast.Add):
                return f"({l_code} + {r_code})"
            if isinstance(node.op, ast.Sub):
                return f"({l_code} - {r_code})"
            if isinstance(node.op, ast.Mult):
                return f"({l_code} * {r_code})"
            if isinstance(node.op, ast.Div):
                return f"({l_code} / {r_code})"
            if isinstance(node.op, ast.Mod):
                return f"({l_code} % {r_code})"
            if isinstance(node.op, ast.FloorDiv):
                return f"py_val_floordiv({l_code}, {r_code})"
            if isinstance(node.op, ast.Pow):
                return f"py_val_pow({l_code}, {r_code})"

        if isinstance(node.op, ast.FloorDiv):
            if lt == FLOAT or rt == FLOAT:
                l_code = self._expr(node.left, scope, plan, expected_type=FLOAT)
                r_code = self._expr(node.right, scope, plan, expected_type=FLOAT)
                return f"py_floordiv_f64({l_code}, {r_code})"
            l_code = self._expr(node.left, scope, plan)
            r_code = self._expr(node.right, scope, plan)
            return f"py_floordiv({l_code}, {r_code})"

        if isinstance(node.op, ast.Mod):
            if lt == STR:
                return self._percent_format(node.left, node.right, node, scope, plan)
            if lt == INT and rt == INT:
                return f"py_mod({self._expr(node.left, scope, plan)}, {self._expr(node.right, scope, plan)})"
            l_code = self._expr(node.left, scope, plan, expected_type=FLOAT)
            r_code = self._expr(node.right, scope, plan, expected_type=FLOAT)
            return f"py_mod_f64({l_code}, {r_code})"

        if isinstance(node.op, ast.Pow):
            if (
                isinstance(node.left, ast.Constant)
                and isinstance(node.right, ast.Constant)
                and type(node.left.value) is int
                and type(node.right.value) is int
                and 0 <= node.right.value < 4096
                and not (-(2**63) <= node.left.value**node.right.value < 2**63)
            ):
                raise PyRustError(
                    "integer constant exceeds PyRust's 64-bit int range (Python bigints are not supported natively)",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            if lt == FLOAT or rt == FLOAT or _is_negative_expr(node.right):
                l_code = self._expr(node.left, scope, plan, expected_type=FLOAT)
                r_code = self._expr(node.right, scope, plan, expected_type=FLOAT)
                return f"py_pow_f64({l_code}, {r_code})"
            return f"py_pow({self._expr(node.left, scope, plan)}, {self._expr(node.right, scope, plan)})"

        if isinstance(node.op, ast.Div):
            if lt == INT and rt == INT:
                return f"py_truediv_i64({self._expr(node.left, scope, plan)}, {self._expr(node.right, scope, plan)})"
            l_code = self._expr(node.left, scope, plan, expected_type=FLOAT)
            r_code = self._expr(node.right, scope, plan, expected_type=FLOAT)
            return f"py_div_f64({l_code}, {r_code})"

        if isinstance(node.op, ast.Add) and lt == STR and rt == STR:
            return f'format!("{{}}{{}}", {self._expr(node.left, scope, plan)}, {self._expr(node.right, scope, plan)})'

        if isinstance(lt, SetType) and isinstance(rt, SetType):
            set_ops = {ast.BitOr: "union", ast.BitAnd: "intersection", ast.Sub: "difference", ast.BitXor: "symmetric_difference"}
            for op_cls, meth in set_ops.items():
                if isinstance(node.op, op_cls):
                    l_code = self._recv_no_clone(node.left, scope, plan)
                    r_code = self._recv_no_clone(node.right, scope, plan)
                    return f"{l_code}.{meth}(&{r_code}).cloned().collect::<HashSet<_>>()"

        if self._is_dict_view(node.left) or self._is_dict_view(node.right):
            raise PyRustError(
                "arithmetic on dict views is not supported natively",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )

        if isinstance(node.op, ast.Add) and isinstance(lt, ListType) and isinstance(rt, ListType):
            return f"py_list_concat(&{self._expr(node.left, scope, plan)}, &{self._expr(node.right, scope, plan)})"

        if isinstance(node.op, ast.Mult):
            if lt == STR and rt in (INT, BOOL):
                return f"py_str_repeat(&{self._expr(node.left, scope, plan)}, {self._expr(node.right, scope, plan, expected_type=INT)})"
            if lt in (INT, BOOL) and rt == STR:
                return f"py_str_repeat(&{self._expr(node.right, scope, plan)}, {self._expr(node.left, scope, plan, expected_type=INT)})"
            if isinstance(lt, ListType) and rt == INT:
                return f"py_list_repeat(&{self._expr(node.left, scope, plan)}, {self._expr(node.right, scope, plan)})"
            if lt == INT and isinstance(rt, ListType):
                return f"py_list_repeat(&{self._expr(node.right, scope, plan)}, {self._expr(node.left, scope, plan)})"

        if lt == FLOAT or rt == FLOAT:
            target_num: PyType | None = FLOAT
        elif actual_type in (INT, BOOL):
            target_num = actual_type
        else:
            target_num = None
        l_code = self._expr(node.left, scope, plan, expected_type=target_num)
        r_code = self._expr(node.right, scope, plan, expected_type=target_num)

        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.BitAnd: "&",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.LShift: "<<",
            ast.RShift: ">>",
        }
        if isinstance(node.op, ast.LShift) and lt == INT and rt == INT:
            return f"py_lshift({l_code}, {r_code})"
        if isinstance(node.op, ast.RShift) and lt == INT and rt == INT:
            return f"py_rshift({l_code}, {r_code})"
        for op_cls, sym in op_map.items():
            if isinstance(node.op, op_cls):
                return f"({l_code} {sym} {r_code})"
        raise PyRustError(
            f"{type(node.op).__name__} on {lt.to_rust_type()} and {rt.to_rust_type()} "
            "is not supported by PyRust yet",
            lineno=getattr(node, "lineno", None),
            col=getattr(node, "col_offset", None),
            filename=self.parsed.filename,
        )

    def _math_call(
        self,
        func_name: str,
        args: list[ast.expr],
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        """Emit a supported `math.*` call, or raise a helpful error."""
        float_arg = lambda i: (
            self._expr(args[i], scope, plan, expected_type=FLOAT)
            if i < len(args)
            else "0.0"
        )
        for a in args:
            if scope.expr_types.get(id(a), UNKNOWN) not in (INT, FLOAT, BOOL, ANY):
                raise PyRustError(
                    f"math.{func_name}() of a non-number is not supported by PyRust yet",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
        if func_name == "pow" and len(args) == 2:
            return f"py_math_pow({float_arg(0)}, {float_arg(1)})"
        if func_name == "atan2" and len(args) == 2:
            return f"f64::atan2({float_arg(0)}, {float_arg(1)})"
        if func_name == "hypot" and len(args) == 2:
            return f"py_math_hypot({float_arg(0)}, {float_arg(1)})"
        if func_name == "log" and len(args) == 2:
            return f"py_math_log2args({float_arg(0)}, {float_arg(1)})"
        if func_name in MATH_FLOAT_FUNCS and func_name not in ("pow", "atan2", "hypot") and len(args) == 1:
            return f'py_math1("{func_name}", {float_arg(0)})'
        if func_name in ("floor", "ceil") and len(args) == 1:
            if scope.expr_types.get(id(args[0]), UNKNOWN) in (INT, BOOL):
                return self._expr(args[0], scope, plan, expected_type=INT)
            return f"py_float_to_int(f64::{func_name}({float_arg(0)}))"
        raise PyRustError(
            f"math.{func_name} is not supported by PyRust yet "
            "(supported functions: sqrt, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, "
            "exp, log, log2, log10, pow, hypot, fabs, floor, ceil; "
            "constants: pi, e, tau)",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _open_call(
        self,
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        if not node.args:
            raise PyRustError(
                "open() needs a file path argument",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )
        path_code = self._expr(node.args[0], scope, plan, expected_type=STR)
        mode = "r"
        if len(node.args) > 1:
            mode_node = node.args[1]
            if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
                mode = mode_node.value
        if "b" in mode:
            raise PyRustError(
                f"binary file mode '{mode}' is not supported by PyRust yet (text mode only)",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )
        if "a" in mode:
            return f"PyFile::open_append(&{path_code})"
        if "w" in mode:
            return f"PyFile::open_write(&{path_code})"
        return f"PyFile::open_read(&{path_code})"

    def _pyfile_method_call(
        self,
        node: ast.Call,
        recv_t: ClassType,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        attr = node.func.attr
        recv_code = self._recv_no_clone(node.func.value, scope, plan)
        if attr == "read" and not node.args:
            return f"{recv_code}.read()"
        if attr in ("readlines", "lines") and not node.args:
            return f"{recv_code}.readlines()"
        if attr == "write" and len(node.args) == 1:
            s_code = self._expr(node.args[0], scope, plan, expected_type=STR)
            return f"{recv_code}.write(&{s_code})"
        if attr == "close" and not node.args:
            return f"{recv_code}.close()"
        raise PyRustError(
            f"PyFile.{attr} is not supported by PyRust yet "
            "(supported: read, readlines, lines, write, close)",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    @staticmethod
    def _rust_string_escape(s: str) -> str:
        return (
            s.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )

    @staticmethod
    def _wrap_pyvalue(actual: PyType) -> bool:
        """Whether a container prints via PyValue (Python repr) instead of Debug."""
        if isinstance(actual, ListType):
            return actual.elem in (INT, FLOAT, BOOL, STR, ANY)
        if isinstance(actual, DictType):
            return actual.key == STR and actual.val in (INT, FLOAT, BOOL, STR, ANY)
        return False

    @staticmethod
    def _is_simple_scalar(t: PyType) -> bool:
        return t in (INT, FLOAT, BOOL, STR)

    def _check_printable(self, t: PyType, node: ast.AST) -> None:
        """Refuse values whose str()/repr() native code cannot reproduce."""
        def bad(why: str) -> PyRustError:
            return PyRustError(
                f"printing {why} is not supported natively",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )

        if isinstance(t, ClassType):
            if t.name == "PyFile":
                raise bad("a file object")
            if t.name not in self._dataclass_names():
                raise bad(f"a '{t.name}' object (its default repr shows a memory address)")
            for ft in self.env.class_types.get(t.name, t).fields.values():
                self._check_printable(ft, node)
        elif isinstance(t, SetType):
            raise bad("a set (CPython's set order cannot be reproduced)")
        elif isinstance(t, ListType):
            self._check_printable(t.elem, node)
        elif isinstance(t, DictType):
            self._check_printable(t.key, node)
            self._check_printable(t.val, node)
        elif isinstance(t, TupleType):
            for e in t.elements:
                self._check_printable(e, node)
        if self._is_dict_view(node):
            raise bad("a dict view")

    def _has_identity_object(self, t: PyType) -> bool:
        """True if values of type t contain plain (non-dataclass) objects."""
        if isinstance(t, ClassType):
            return t.name == "PyFile" or t.name not in self._dataclass_names()
        if isinstance(t, (ListType, SetType)):
            return self._has_identity_object(t.elem)
        if isinstance(t, DictType):
            return self._has_identity_object(t.key) or self._has_identity_object(t.val)
        if isinstance(t, TupleType):
            return any(self._has_identity_object(e) for e in t.elements)
        return False

    def _dataclass_names(self) -> set[str]:
        from pyrust.inference import _is_dataclass

        return {c.name for c in self.parsed.classes if _is_dataclass(c)}

    @staticmethod
    def _is_dict_view(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("keys", "values", "items")
            and not node.args
        )

    def _format_placeholder(self, arg: ast.expr, scope: ScopeInfo) -> str:
        return "{}"

    def _print_placeholder(self, arg: ast.expr, scope: ScopeInfo) -> str:
        return "{}"

    def _print_call(self, node: ast.Call, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        sep_node: ast.expr | None = None
        end_node: ast.expr | None = None
        file_node: ast.expr | None = None
        flush = False

        def err(msg: str) -> PyRustError:
            return PyRustError(msg, lineno=node.lineno, col=node.col_offset, filename=self.parsed.filename)

        for kw in node.keywords:
            if kw.arg is None:
                raise err("** unpacking in print() is not supported by PyRust yet")
            if kw.arg == "sep":
                sep_node = kw.value
            elif kw.arg == "end":
                end_node = kw.value
            elif kw.arg == "file":
                file_node = kw.value
            elif kw.arg == "flush":
                if not isinstance(kw.value, ast.Constant):
                    raise err("print(flush=...) needs a constant in PyRust")
                flush = bool(kw.value.value)
            else:
                raise err(f"print({kw.arg}=...) is not supported by PyRust yet")
        to_stderr = False
        if file_node is not None:
            if isinstance(file_node, ast.Attribute) and isinstance(file_node.value, ast.Name) and \
                    file_node.value.id == "sys" and file_node.attr in ("stderr", "stdout"):
                to_stderr = file_node.attr == "stderr"
            elif isinstance(file_node, ast.Name) and self.parsed.from_imports.get(file_node.id) == "sys" and \
                    file_node.id in ("stderr", "stdout"):
                to_stderr = file_node.id == "stderr"
            else:
                raise err("print(file=...) is only supported with sys.stdout / sys.stderr")
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise err("print(*args) is not supported by PyRust yet")
        parts = [self._print_arg(a, scope, plan) for a in node.args]

        def str_arg(n: ast.expr | None, default: str) -> str:
            if n is None or (isinstance(n, ast.Constant) and n.value is None):
                return default
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                return self._rust_str_lit(n.value)
            if scope.expr_types.get(id(n), UNKNOWN) != STR:
                raise err("print() sep/end must be strings")
            return f"&{self._expr(n, scope, plan)}"

        sep = str_arg(sep_node, '" "')
        end = str_arg(end_node, '"\\n"')
        if not to_stderr and sep == '" "' and end == '"\\n"':
            code = f'py_println!("{" ".join(["{}"] * len(parts))}", {", ".join(parts)})' if parts else "py_println!()"
        else:
            pieces = ", ".join(f'format!("{{}}", {p})' for p in parts)
            content = f"[{pieces}].join({sep})" if len(parts) != 1 else f'format!("{{}}", {parts[0]})'
            if not parts:
                content = "String::new()"
            text = f'format!("{{}}{{}}", {content}, {end})'
            code = f"py_eprint(&{text})" if to_stderr else f"py_write_str(&{text})"
        if flush and not to_stderr:
            return f"{{ {code}; let _ = py_flush_stdout(); }}"
        return code

    def _print_arg(self, node: ast.expr, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        """Code implementing Display that renders `str(node)` like CPython."""
        actual = scope.expr_types.get(id(node), UNKNOWN)
        self._check_printable(actual, node)
        code = self._expr(node, scope, plan)
        if actual == FLOAT:
            return f"py_float_repr({code})"
        if actual == BOOL:
            return f'(if {code} {{ "True" }} else {{ "False" }})'
        if actual == NONE:
            return f'{{ let _ = {code}; "None" }}'
        if actual in (INT, STR, ANY, UNKNOWN):
            return code
        return f"py_str(&{code})"

    def _sys_call(
        self,
        attr: str,
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        if attr == "argv" and not node.args and not node.keywords:
            raise PyRustError(
                "sys.argv is a list, not a function: use sys.argv without ()",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )
        if attr == "exit":
            if not node.args:
                return "py_exit(0)"
            if len(node.args) == 1 and not node.keywords:
                at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                if isinstance(node.args[0], ast.Constant) and node.args[0].value is None:
                    return "py_exit(0)"
                code = self._expr(node.args[0], scope, plan)
                if at == STR:
                    return f'{{ py_eprint(&format!("{{}}\\n", {code})); py_exit(1) }}'
                if at in (INT, BOOL):
                    # POSIX keeps the low 8 bits, like CPython's exit status.
                    return f"py_exit((({code}) as i64 & 0xff) as i32)"
        raise PyRustError(
            f"sys.{attr} is not supported by PyRust yet (supported: argv, exit)",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _os_call(
        self,
        attr: str,
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        if attr == "getenv" and len(node.args) == 1:
            key = self._expr(node.args[0], scope, plan, expected_type=STR)
            return f"py_os_getenv(&{key})"
        if attr == "getenv" and len(node.args) == 2 and scope.expr_types.get(id(node.args[1]), UNKNOWN) == STR:
            key = self._expr(node.args[0], scope, plan, expected_type=STR)
            dflt = self._expr(node.args[1], scope, plan, expected_type=STR)
            return f"std::env::var(&{key}).unwrap_or({dflt})"
        if attr == "getcwd" and not node.args:
            return "py_os_getcwd()"
        if attr == "listdir" and len(node.args) <= 1:
            path = self._expr(node.args[0], scope, plan, expected_type=STR) if node.args else '".".to_string()'
            return f"py_os_listdir(&{path})"
        if attr in ("makedirs", "mkdir") and len(node.args) == 1:
            path = self._expr(node.args[0], scope, plan, expected_type=STR)
            exist_ok = False  # Python's default for os.makedirs
            for kw in node.keywords:
                if attr == "makedirs" and kw.arg == "exist_ok" and isinstance(kw.value, ast.Constant):
                    exist_ok = bool(kw.value.value)
                else:
                    raise PyRustError(
                        f"os.{attr}() with keyword '{kw.arg}' is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
            recursive = "true" if attr == "makedirs" else "false"
            return f"py_os_mkdir(&{path}, {recursive}, {str(exist_ok).lower()})"
        if attr == "remove" and len(node.args) == 1:
            path = self._expr(node.args[0], scope, plan, expected_type=STR)
            return f"py_os_remove(&{path})"
        raise PyRustError(
            f"os.{attr} is not supported by PyRust yet "
            "(supported: getenv, getcwd, listdir, makedirs, mkdir, remove)",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _time_call(
        self,
        attr: str,
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        if attr == "time" and not node.args:
            return "py_unix_time()"
        if attr == "perf_counter" and not node.args:
            return "py_perf_counter()"
        if attr == "sleep" and len(node.args) == 1:
            secs = self._expr(node.args[0], scope, plan, expected_type=FLOAT)
            return f"py_sleep({secs})"
        raise PyRustError(
            f"time.{attr} is not supported by PyRust yet (supported: time, perf_counter, sleep)",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _platform_call(
        self,
        attr: str,
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        if node.args or node.keywords:
            raise PyRustError(
                f"platform.{attr}() takes no arguments",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )
        if attr == "system":
            return 'match std::env::consts::OS { "windows" => "Windows", "macos" => "Darwin", "linux" => "Linux", other => other }.to_string()'
        if attr == "machine":
            # CPython reports the OS architecture name ("AMD64" on Windows),
            # not the Rust target triple component.
            return '(if std::env::consts::FAMILY == "windows" { if std::env::consts::ARCH == "x86_64" { "AMD64" } else { std::env::consts::ARCH } } else { std::env::consts::ARCH }).to_string()'
        if attr == "processor":
            return "std::env::consts::ARCH.to_string()"
        if attr == "architecture":
            return '((if cfg!(target_pointer_width = "64") { "64bit" } else { "32bit" }).to_string(), (if std::env::consts::FAMILY == "windows" { "WindowsPE".to_string() } else if std::env::consts::OS == "linux" { "ELF".to_string() } else { "".to_string() }))'
        if attr == "python_version":
            return '"3.11".to_string()'
        if attr == "python_implementation":
            return '"PyRust".to_string()'
        if attr == "node":
            return 'std::env::var("COMPUTERNAME").or_else(|_| std::env::var("HOSTNAME")).unwrap_or_default()'
        raise PyRustError(
            f"platform.{attr} is not supported by PyRust yet "
            "(supported: system, machine, processor, architecture, python_version, python_implementation, node)",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _seq_pyvalues(self, args: list[ast.expr], scope: ScopeInfo, plan: OwnershipPlan, node: ast.Call) -> str:
        if len(args) != 1:
            raise PyRustError(
                "expected exactly one list argument",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )
        at = scope.expr_types.get(id(args[0]), UNKNOWN)
        inner = self._expr(args[0], scope, plan)
        if isinstance(at, ListType) and at.elem in (INT, FLOAT, BOOL, STR):
            return f"{inner}.iter().cloned().map(PyValue::from).collect::<Vec<PyValue>>()"
        if isinstance(at, ListType) and at.elem == ANY:
            return f"{inner}.clone()"
        raise PyRustError(
            "statistics functions need a list of numbers",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _statistics_call(
        self,
        func_name: str,
        args: list[ast.expr],
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        if func_name in ("harmonic_mean", "stdev", "pstdev", "fmean", "geometric_mean"):
            raise PyRustError(
                f"statistics.{func_name} is not supported natively (CPython computes it with exact fractions)",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )
        if func_name == "mean" and len(args) == 1:
            at = scope.expr_types.get(id(args[0]), UNKNOWN)
            if isinstance(at, ListType) and at.elem == FLOAT:
                return f"PyValue::Float(py_mean_floats(&{self._expr(args[0], scope, plan)}))"
        if func_name in ("mean", "variance", "pvariance"):
            at = scope.expr_types.get(id(args[0]), UNKNOWN) if args else UNKNOWN
            if not (isinstance(at, ListType) and at.elem in (INT, BOOL)):
                raise PyRustError(
                    f"statistics.{func_name} of non-integer data is not supported natively "
                    "(CPython computes it with exact fractions)",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            return f"py_{func_name}(&{self._seq_pyvalues(args, scope, plan, node)})"
        if func_name in ("median", "median_low", "median_high"):
            return f"py_{func_name}(&{self._seq_pyvalues(args, scope, plan, node)})"
        if func_name == "mode" and len(args) == 1:
            if scope.expr_types.get(id(args[0]), UNKNOWN) == STR:
                inner = self._expr(args[0], scope, plan)
                return f"py_mode(&{inner}.chars().collect::<Vec<char>>())"
            return f"py_mode(&{self._expr(args[0], scope, plan)})"
        raise PyRustError(
            f"statistics.{func_name} is not supported by PyRust yet "
            "(supported: mean, median, median_low, median_high, mode, variance, pvariance, stdev, pstdev, harmonic_mean)",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _gc_call(
        self,
        func_name: str,
        args: list[ast.expr],
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        if func_name == "collect" and not args:
            return "py_gc_collect()"
        if func_name == "isenabled" and not args:
            return "py_gc_isenabled()"
        if func_name == "disable" and not args:
            return "py_gc_disable()"
        if func_name == "enable" and not args:
            return "py_gc_enable()"
        if func_name == "get_count" and not args:
            return "py_gc_get_count()"
        raise PyRustError(
            f"gc.{func_name} is not supported by PyRust yet "
            "(supported: collect, isenabled, disable, enable, get_count)",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _random_call(
        self,
        func_name: str,
        args: list[ast.expr],
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> str:
        """Emit a supported `random.*` call, or raise a helpful error."""
        if func_name == "random" and not args:
            return "py_random()"
        if func_name == "seed":
            if not args:
                return "py_seed()"
            return f"py_seed_int({self._expr(args[0], scope, plan, expected_type=INT)})"
        if func_name == "randint" and len(args) == 2:
            a0 = self._expr(args[0], scope, plan, expected_type=INT)
            a1 = self._expr(args[1], scope, plan, expected_type=INT)
            return f"py_randint({a0}, {a1})"
        if func_name == "uniform" and len(args) == 2:
            a0 = self._expr(args[0], scope, plan, expected_type=FLOAT)
            a1 = self._expr(args[1], scope, plan, expected_type=FLOAT)
            return f"py_uniform({a0}, {a1})"
        if func_name == "choice" and len(args) == 1:
            return f"py_choice(&{self._expr(args[0], scope, plan)})"
        if func_name == "shuffle" and len(args) == 1:
            return f"py_shuffle(&mut {self._expr(args[0], scope, plan)})"
        raise PyRustError(
            f"random.{func_name} is not supported by PyRust yet "
            "(supported functions: seed, random, randint, uniform, choice, shuffle)",
            lineno=node.lineno,
            col=node.col_offset,
            filename=self.parsed.filename,
        )

    def _build_call_args(
        self,
        ft: FuncType,
        key: str,
        node: ast.Call,
        scope: ScopeInfo,
        plan: OwnershipPlan,
    ) -> list[str]:
        """Map positional/default/keyword/vararg call-site args onto callee params."""
        param_names = self.env.func_param_names.get(key, [])
        defaults = self.env.func_defaults.get(key, [])
        pos_args = list(node.args)
        kw_map: dict[str, ast.expr] = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise PyRustError(
                    "** argument unpacking is not supported by PyRust yet",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            kw_map[kw.arg] = kw.value
        n_fixed = len(ft.params) - (1 if ft.has_vararg else 0)
        out: list[str] = []
        refs = self.alias.ref_params.get(self._alias_key(key), set())
        if refs:
            # A `&mut` argument must be the only access to its root in this call
            # (Python would see one shared object; Rust cannot lend it twice).
            ref_roots: list[str] = []
            other_roots: set[str] = set()
            for i, pname in enumerate(param_names[:n_fixed]):
                a = pos_args[i] if i < len(pos_args) else kw_map.get(pname)
                if a is None:
                    continue
                if pname in refs and self._root_of(a):
                    ref_roots.append(self._root_of(a))
                else:
                    other_roots |= {n.id for n in ast.walk(a) if isinstance(n, ast.Name)}
            dup = {r for r in ref_roots if ref_roots.count(r) > 1} | (set(ref_roots) & other_roots)
            if dup:
                raise PyRustError(
                    f"passing '{sorted(dup)[0]}' (or parts of it) to a call that modifies one of them in place "
                    "is not supported natively",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
        for i in range(n_fixed):
            pname = param_names[i] if i < len(param_names) else None
            ptype = ft.params[i] if i < len(ft.params) else None
            if pname in refs:
                arg_node = pos_args[i] if i < len(pos_args) else kw_map.get(pname)
                if arg_node is None:
                    di = i - (n_fixed - len(defaults))
                    out.append(f"&mut {self._default_code(defaults[di], ptype, node)}")
                elif isinstance(arg_node, (ast.Name, ast.Attribute, ast.Subscript)) and self._root_of(arg_node):
                    pre: list[str] = []
                    place = self._place(arg_node, scope, plan, pre)
                    out.append("{ " + " ".join(pre) + f" &mut {place} }}" if pre else f"&mut {place}")
                else:
                    out.append(f"&mut {self._expr(arg_node, scope, plan, expected_type=ptype)}")
                continue
            if i < len(pos_args) and pname is not None and pname in kw_map:
                raise PyRustError(
                    f"got multiple values for argument '{pname}'",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            if i < len(pos_args):
                out.append(self._expr(pos_args[i], scope, plan, expected_type=ptype))
            elif pname is not None and pname in kw_map:
                out.append(self._expr(kw_map[pname], scope, plan, expected_type=ptype))
            else:
                di = i - (n_fixed - len(defaults))
                if 0 <= di < len(defaults):
                    out.append(self._default_code(defaults[di], ptype, node))
                else:
                    raise PyRustError(
                        f"missing required argument '{pname or 'arg%d' % i}'",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
        known = set(param_names)
        if ft.has_vararg:
            elem_t = ft.params[-1].elem if ft.params and isinstance(ft.params[-1], ListType) else None
            surplus = pos_args[n_fixed:]
            for k in kw_map:
                if k not in known:
                    raise PyRustError(
                        f"unexpected keyword argument '{k}'",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
            out.append(f"vec![{', '.join(self._expr(a, scope, plan, expected_type=elem_t) for a in surplus)}]")
        else:
            if len(pos_args) > n_fixed:
                raise PyRustError(
                    f"too many positional arguments (expected {n_fixed})",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            for k in kw_map:
                if k not in known:
                    raise PyRustError(
                        f"unexpected keyword argument '{k}'",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
        return out

    def _str_method_call(
        self, node: ast.Call, attr: str, recv: str, scope: ScopeInfo, plan: OwnershipPlan
    ) -> str:
        args = node.args
        n = len(args)

        def err(msg: str | None = None) -> PyRustError:
            return PyRustError(
                msg or f"str.{attr}() with these arguments is not supported by PyRust yet",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )

        def s_arg(i: int) -> str:
            if scope.expr_types.get(id(args[i]), UNKNOWN) != STR:
                raise err(f"str.{attr}() argument {i + 1} must be a str")
            return self._expr(args[i], scope, plan, expected_type=STR)

        def i_arg(i: int) -> str:
            if scope.expr_types.get(id(args[i]), UNKNOWN) not in (INT, BOOL):
                raise err(f"str.{attr}() argument {i + 1} must be an int")
            return self._expr(args[i], scope, plan, expected_type=INT)

        def opt_chars(i: int) -> str:
            if n <= i or (isinstance(args[i], ast.Constant) and args[i].value is None):
                return "None"
            return f"Some(&{s_arg(i)})"

        if attr == "upper" and n == 0:
            return f"{recv}.to_uppercase()"
        if attr == "lower" and n == 0:
            return f"{recv}.to_lowercase()"
        if attr in ("strip", "lstrip", "rstrip") and n <= 1:
            return f"py_{attr}(&{recv}, {opt_chars(0)})"
        if attr == "capitalize" and n == 0:
            return f"py_capitalize(&{recv})"
        if attr in ("isdigit", "isnumeric", "isdecimal", "isalpha", "isalnum", "isspace") and n == 0:
            return f"py_str_{attr}(&{recv})"
        if attr in ("find", "index") and 1 <= n <= 3:
            start = f"Some({i_arg(1)})" if n >= 2 else "None"
            end = f"Some({i_arg(2)})" if n >= 3 else "None"
            if attr == "index":
                if n > 1:
                    raise err()
                return f"py_str_index_of(&{recv}, &{s_arg(0)})"
            return f"py_str_find(&{recv}, &{s_arg(0)}, {start}, {end})"
        if attr == "rfind" and n == 1:
            return f"py_str_rfind(&{recv}, &{s_arg(0)})"
        if attr == "count" and n == 1:
            return f"py_str_count(&{recv}, &{s_arg(0)})"
        if attr in ("removeprefix", "removesuffix") and n == 1:
            return f"py_{attr}(&{recv}, &{s_arg(0)})"
        if attr == "split" and n <= 2 and not node.keywords:
            maxsplit = i_arg(1) if n == 2 else "-1"
            if n == 0 or (isinstance(args[0], ast.Constant) and args[0].value is None):
                return f"py_split_ws_max(&{recv}, {maxsplit})"
            return f"py_split(&{recv}, &{s_arg(0)}, {maxsplit})"
        if attr == "splitlines" and n == 0:
            return f"py_splitlines(&{recv})"
        if attr == "replace" and 2 <= n <= 3:
            count = i_arg(2) if n == 3 else "-1"
            return f"py_str_replace(&{recv}, &{s_arg(0)}, &{s_arg(1)}, {count})"
        if attr == "join" and n == 1:
            jt = scope.expr_types.get(id(args[0]), UNKNOWN)
            self._check_ordered_iteration(args[0], scope)
            inner = self._expr(args[0], scope, plan)
            if jt == STR:
                return f"{inner}.chars().map(|c| c.to_string()).collect::<Vec<String>>().join(&{recv})"
            if isinstance(jt, ListType) and jt.elem == STR:
                return f"{inner}.join(&{recv})"
            raise err("str.join() needs a list of str")
        if attr in ("startswith", "endswith") and n == 1:
            if scope.expr_types.get(id(args[0]), UNKNOWN) != STR:
                raise err(f"str.{attr}() with a tuple is not supported by PyRust yet")
            method = "starts_with" if attr == "startswith" else "ends_with"
            return f"{recv}.{method}({s_arg(0)}.as_str())"
        raise err(f"method '{attr}' is not supported on str by PyRust yet")

    def _default_code(self, default: ast.expr, ptype: PyType | None, call: ast.AST) -> str:
        """A default argument value at a call site. Python evaluates defaults
        once, when the function is defined: only values for which evaluating
        per call is indistinguishable (constants, tuples of them) are native."""
        def simple(n: ast.expr) -> bool:
            if isinstance(n, ast.Constant):
                return True
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                return simple(n.operand)
            if isinstance(n, ast.Tuple):
                return all(simple(e) for e in n.elts)
            if isinstance(n, (ast.List, ast.Dict, ast.Set)) and not getattr(n, "elts", None) and not getattr(n, "keys", None):
                return True  # fresh empty container: sharing was ruled out by the alias analysis
            return False

        if not simple(default):
            raise PyRustError(
                "a default argument value that is not a constant is not supported natively "
                "(Python evaluates it once, at definition time)",
                lineno=getattr(default, "lineno", None),
                col=getattr(default, "col_offset", None),
                filename=self.parsed.filename,
            )
        tmp_scope = ScopeInfo()
        from pyrust.inference import _TypeInferer

        inf = _TypeInferer.__new__(_TypeInferer)
        inf.parsed = self.parsed
        inf.env = self.env
        try:
            inf._infer_expr(default, tmp_scope, None)
        except Exception:
            pass
        return self._expr(default, tmp_scope, OwnershipPlan(), expected_type=ptype)

    def _call_expr(self, node: ast.Call, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        if isinstance(node.func, ast.Name):
            fname = node.func.id

            if fname == "print":
                return self._print_call(node, scope, plan)

            if fname == "len" and len(node.args) == 1:
                lt = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                if lt == STR:
                    return f"py_str_len(&{self._expr(node.args[0], scope, plan)})"
                return f"({self._expr(node.args[0], scope, plan)}.len() as i64)"

            if fname == "repr" and len(node.args) == 1:
                self._check_printable(scope.expr_types.get(id(node.args[0]), UNKNOWN), node.args[0])
                return f"py_repr(&{self._expr(node.args[0], scope, plan)})"

            if fname == "str" and len(node.args) == 1:
                return f'format!("{{}}", {self._print_arg(node.args[0], scope, plan)})'
            if fname == "str" and not node.args:
                return '"".to_string()'

            if fname == "bool" and len(node.args) == 1:
                return self._cond_expr(node.args[0], scope, plan)
            if fname == "bool" and not node.args:
                return "false"

            if fname == "int" and len(node.args) == 2:
                raise PyRustError(
                    "int(x, base) is not supported by PyRust yet (base 10 only)",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            if fname == "int" and len(node.args) == 1:
                at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                inner = self._expr(node.args[0], scope, plan)
                if at == STR:
                    return f"py_parse_int(&{inner})"
                if at in (ANY, UNKNOWN):
                    return f"py_int_of_value(&{inner})"
                if at == FLOAT:
                    return f"py_int_of_float({inner})"
                if at == INT:
                    return inner
                if at == BOOL:
                    return f"({inner} as i64)"
                raise PyRustError(
                    f"int() of a {at.to_rust_type()} value is not supported by PyRust yet",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            if fname == "int" and not node.args:
                return "0"

            if fname == "float" and len(node.args) == 1:
                at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                inner = self._expr(node.args[0], scope, plan)
                if at == STR:
                    return f"py_parse_float(&{inner})"
                if at in (ANY, UNKNOWN):
                    return f"py_float_of_value(&{inner})"
                if at == FLOAT:
                    return inner
                if at in (INT, BOOL):
                    return f"({inner} as f64)"
                raise PyRustError(
                    f"float() of a {at.to_rust_type()} value is not supported by PyRust yet",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            if fname == "float" and not node.args:
                return "0.0"

            if fname == "abs" and len(node.args) == 1:
                at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                inner = self._expr(node.args[0], scope, plan)
                if at in (ANY, UNKNOWN):
                    return f"py_abs(&{inner})"
                if at == BOOL:
                    return f"({inner} as i64)"
                if at == INT:
                    return f"i64::abs({inner})"
                if at == FLOAT:
                    return f"f64::abs({inner})"
                raise PyRustError(
                    f"abs() of a {at.to_rust_type()} value is not supported by PyRust yet",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )

            if fname == "sum" and 1 <= len(node.args) <= 2 and not node.keywords:
                return self._sum_call(node, scope, plan)

            if fname == "sorted" and len(node.args) == 1:
                reverse = False
                for kw in node.keywords:
                    if kw.arg == "reverse" and isinstance(kw.value, ast.Constant):
                        reverse = bool(kw.value.value)
                    else:
                        raise PyRustError(
                            "sorted() with key= (or a non-constant reverse=) is not supported by PyRust yet",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                at_sorted = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                inner = self._expr(node.args[0], scope, plan)
                if at_sorted == STR:
                    items = f"{inner}.chars().map(|c| c.to_string()).collect::<Vec<String>>()"
                elif isinstance(at_sorted, SetType):
                    items = f"{inner}.iter().cloned().collect::<Vec<_>>()"
                elif isinstance(at_sorted, DictType):
                    items = f"{inner}.keys()"
                elif isinstance(at_sorted, ListType):
                    items = f"{inner}.to_vec()"
                else:
                    raise PyRustError(
                        f"sorted() of a {at_sorted.to_rust_type()} value is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                if reverse:
                    # CPython: reverse, stable sort, reverse (keeps ties in order).
                    return f"{{ let mut __v = {items}; __v.reverse(); py_sort(&mut __v); __v.reverse(); __v }}"
                return f"{{ let mut __v = {items}; py_sort(&mut __v); __v }}"

            if fname == "list":
                if not node.args:
                    return "Vec::new()"
                self._check_ordered_iteration(node.args[0], scope)
                if scope.expr_types.get(id(node.args[0]), UNKNOWN) == STR:
                    inner = self._expr(node.args[0], scope, plan)
                    return f"{inner}.chars().map(|c| c.to_string()).collect::<Vec<String>>()"
                if isinstance(scope.expr_types.get(id(node.args[0]), UNKNOWN), DictType):
                    return f"{self._expr(node.args[0], scope, plan)}.keys()"
                return f"{self._expr(node.args[0], scope, plan)}.iter().cloned().collect::<Vec<_>>()"

            if fname == "set":
                if not node.args:
                    return "HashSet::new()"
                at0s = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                if at0s == STR:
                    inner = self._expr(node.args[0], scope, plan)
                    return f"{inner}.chars().map(|c| c.to_string()).collect::<HashSet<String>>()"
                if isinstance(at0s, DictType):
                    inner = self._expr(node.args[0], scope, plan)
                    return f"{inner}.keys().into_iter().collect::<HashSet<_>>()"
                return f"{self._expr(node.args[0], scope, plan)}.iter().cloned().collect::<HashSet<_>>()"

            if fname == "dict":
                if not node.args and not node.keywords:
                    return "PyDict::new()"
                if node.args and not node.keywords and len(node.args) == 1:
                    inner_t = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                    inner = self._expr(node.args[0], scope, plan)
                    if isinstance(inner_t, DictType):
                        return f"{inner}.clone()"
                    if isinstance(inner_t, ListType) and isinstance(inner_t.elem, TupleType):
                        return f"PyDict::from_vec({inner})"
                if not node.args and node.keywords and all(kw.arg is not None for kw in node.keywords):
                    pairs = ", ".join(
                        f'("{kw.arg}".to_string(), PyValue::from({self._expr(kw.value, scope, plan)}))'
                        for kw in node.keywords
                    )
                    return f"PyDict::from_vec(vec![{pairs}])"
                raise PyRustError(
                            "only dict(), dict(mapping), dict(pairs) and dict(k=v, ...) are supported by PyRust",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )

            if fname == "enumerate" and 1 <= len(node.args) <= 2:
                it = self._for_iter_expr(node.args[0], scope, plan)
                if len(node.args) == 2:
                    start = self._expr(node.args[1], scope, plan, expected_type=INT)
                    return f"{it}.enumerate().map(|(i, v)| (i as i64 + {start}, v)).collect::<Vec<_>>()"
                return f"{it}.enumerate().map(|(i, v)| (i as i64, v)).collect::<Vec<_>>()"

            if fname == "divmod" and len(node.args) == 2:
                at = scope.expr_types.get(id(node), TupleType((INT, INT)))
                if isinstance(at, TupleType) and all(e == FLOAT for e in at.elements):
                    a0 = self._expr(node.args[0], scope, plan, expected_type=FLOAT)
                    a1 = self._expr(node.args[1], scope, plan, expected_type=FLOAT)
                    return f"py_divmod_f64({a0}, {a1})"
                a0 = self._expr(node.args[0], scope, plan, expected_type=INT)
                a1 = self._expr(node.args[1], scope, plan, expected_type=INT)
                return f"py_divmod({a0}, {a1})"

            if fname == "round" and 1 <= len(node.args) <= 2:
                at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                if at not in (INT, BOOL, FLOAT):
                    raise PyRustError(
                        f"round() of a {at.to_rust_type()} value is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                if len(node.args) == 2:
                    a1 = self._expr(node.args[1], scope, plan, expected_type=INT)
                    if at in (INT, BOOL):
                        return f"py_round_int({self._expr(node.args[0], scope, plan, expected_type=INT)}, {a1})"
                    return f"py_round_prec({self._expr(node.args[0], scope, plan)}, {a1})"
                if at in (INT, BOOL):
                    return self._expr(node.args[0], scope, plan, expected_type=INT)
                return f"py_round_bankers({self._expr(node.args[0], scope, plan)})"

            if fname in ("all", "any") and len(node.args) == 1:
                inner = self._expr(node.args[0], scope, plan)
                at = scope.expr_types.get(id(node.args[0]), ListType(BOOL))
                elem = at.elem if isinstance(at, ListType) else BOOL
                method = "all" if fname == "all" else "any"
                if elem == BOOL:
                    return f"{inner}.iter().copied().{method}(|v| v)"
                if elem == INT:
                    return f"{inner}.iter().copied().{method}(|v| v != 0)"
                if elem == FLOAT:
                    return f"{inner}.iter().copied().{method}(|v| v != 0.0)"
                if elem == STR or isinstance(elem, (ListType, DictType, SetType)):
                    return f"{inner}.iter().{method}(|v| !v.is_empty())"
                if elem == ANY:
                    return f"{inner}.iter().{method}(|v| v.is_truthy())"
                raise PyRustError(
                    f"{fname}() over {elem.to_rust_type()} is not supported by PyRust yet",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )

            if fname == "reversed" and len(node.args) == 1:
                at = scope.expr_types.get(id(node.args[0]), ListType(ANY))
                if not (isinstance(at, ListType) or at == STR):
                    raise PyRustError(
                        f"reversed() of a {at.to_rust_type()} value is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                inner = self._expr(node.args[0], scope, plan)
                if at == STR:
                    return f"{inner}.chars().rev().map(|c| c.to_string()).collect::<Vec<String>>()"
                return f"{inner}.iter().rev().cloned().collect::<Vec<_>>()"

            if fname == "ord" and len(node.args) == 1:
                inner = self._expr(node.args[0], scope, plan, expected_type=STR)
                return f"py_ord(&{inner})"

            if fname == "chr" and len(node.args) == 1:
                inner = self._expr(node.args[0], scope, plan, expected_type=INT)
                return f"py_chr({inner})"

            if fname in ("hex", "oct", "bin") and len(node.args) == 1:
                inner = self._expr(node.args[0], scope, plan, expected_type=INT)
                return f"py_{fname}({inner})"

            if fname == "zip" and len(node.args) in (2, 3) and not node.keywords:
                its = [self._for_iter_expr(a, scope, plan) for a in node.args]
                if len(its) == 2:
                    return f"{its[0]}.zip({its[1]}).collect::<Vec<_>>()"
                return f"{its[0]}.zip({its[1]}).zip({its[2]}).map(|((a, b), c)| (a, b, c)).collect::<Vec<_>>()"

            if fname in ("min", "max") and node.args:
                if node.keywords:
                    raise PyRustError(
                        f"{fname}(... with keyword arguments) is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                if len(node.args) == 1:
                    inner = self._expr(node.args[0], scope, plan)
                    at = scope.expr_types.get(id(node.args[0]), ListType(INT))
                    if at == STR:
                        return f"py_{fname}_str(&{inner})"
                    if isinstance(at, ListType):
                        return f"py_{fname}_of(&{inner})"
                    if isinstance(at, SetType):
                        return f"py_{fname}_of(&{inner}.iter().cloned().collect::<Vec<_>>())"
                    if isinstance(at, DictType):
                        return f"py_{fname}_of(&{inner}.keys())"
                    raise PyRustError(
                        f"{fname}() of a {at.to_rust_type()} value is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                res_t = scope.expr_types.get(id(node), UNKNOWN)
                codes = [self._expr(a, scope, plan, expected_type=res_t if res_t == ANY else None) for a in node.args]
                acc = codes[0]
                for c in codes[1:]:
                    acc = f"py_{fname}2({acc}, {c})"
                return acc

            if fname == "range":
                return f"{self._range_iter(node, scope, plan)}.collect::<Vec<i64>>()"

            if fname == "input":
                prompt = self._expr(node.args[0], scope, plan) if node.args else '""'
                return f"py_input(&{prompt})"

            if fname == "open":
                return self._open_call(node, scope, plan)

            if fname == "super":
                raise PyRustError(
                    "super() is not supported by PyRust yet: define needed behavior directly in the subclass",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )

            # `from X import name` aliases resolve to qualified module calls
            alias_mod = self.parsed.from_imports.get(fname)
            if alias_mod == "math":
                return self._math_call(fname, node.args, node, scope, plan)
            if alias_mod == "random":
                return self._random_call(fname, node.args, node, scope, plan)
            if alias_mod == "platform":
                return self._platform_call(fname, node, scope, plan)
            if alias_mod == "statistics":
                return self._statistics_call(fname, node.args, node, scope, plan)
            if alias_mod == "gc":
                return self._gc_call(fname, node.args, node, scope, plan)
            if alias_mod == "sys":
                return self._sys_call(fname, node, scope, plan)
            if alias_mod == "os":
                return self._os_call(fname, node, scope, plan)
            if alias_mod == "time":
                return self._time_call(fname, node, scope, plan)
            if alias_mod == "json":
                if fname == "dumps" and len(node.args) == 1:
                    return f"py_json_dumps(&{self._expr(node.args[0], scope, plan)})"
                if fname == "loads" and len(node.args) == 1:
                    inner = self._expr(node.args[0], scope, plan, expected_type=STR)
                    return f"py_json_loads(&{inner})"
                raise PyRustError(
                    f"json.{fname} is not supported by PyRust yet (supported: dumps, loads)",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            if alias_mod is not None:
                qual = f"{alias_mod}.{fname}"
                if qual in self.env.class_types:
                    cls_t = self.env.class_types[qual]
                    init_ft = cls_t.methods.get("__init__", FuncType())
                    args = self._build_call_args(init_ft, f"{qual}.__init__", node, scope, plan)
                    return f"crate::{alias_mod}::{fname}::new({', '.join(args)})"
                if qual in self.env.func_types:
                    ft = self.env.func_types[qual]
                    args = self._build_call_args(ft, qual, node, scope, plan)
                    return f"crate::{alias_mod}::{fname}({', '.join(args)})"
                # Unknown signature (e.g. single-file preview): emit untyped.
                args = [self._expr(a, scope, plan) for a in node.args]
                return f"crate::{alias_mod}::{fname}({', '.join(args)})"

            if fname in self.env.class_types:
                cls_t = self.env.class_types[fname]
                init_ft = cls_t.methods.get("__init__", FuncType())
                args = self._build_call_args(init_ft, f"{fname}.__init__", node, scope, plan)
                return f"{fname}::new({', '.join(args)})"

            if fname in self.env.func_types:
                ft = self.env.func_types[fname]
                args = self._build_call_args(ft, fname, node, scope, plan)
                call_name = "py_main" if fname == "main" else fname
                return f"{call_name}({', '.join(args)})"

            raise PyRustError(
                f"function '{fname}' is not defined or not supported by PyRust yet",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )

        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            recv_path: list[str] = []
            _probe: ast.expr = node.func.value
            while isinstance(_probe, ast.Attribute):
                recv_path.append(_probe.attr)
                _probe = _probe.value
            if isinstance(_probe, ast.Name) and _probe.id in self.parsed.imports and recv_path:
                dotted = ".".join([_probe.id] + list(reversed(recv_path)) + [attr])
                raise PyRustError(
                    f"dotted module path '{dotted}' is not supported by PyRust yet",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.parsed.imports:
                mod = node.func.value.id
                if mod == "math":
                    return self._math_call(attr, node.args, node, scope, plan)
                if mod == "random":
                    return self._random_call(attr, node.args, node, scope, plan)
                if mod == "platform":
                    return self._platform_call(attr, node, scope, plan)
                if mod == "statistics":
                    return self._statistics_call(attr, node.args, node, scope, plan)
                if mod == "gc":
                    return self._gc_call(attr, node.args, node, scope, plan)
                if mod == "sys":
                    return self._sys_call(attr, node, scope, plan)
                if mod == "os":
                    return self._os_call(attr, node, scope, plan)
                if mod == "time":
                    return self._time_call(attr, node, scope, plan)
                if mod == "json":
                    if attr == "dumps" and len(node.args) == 1:
                        return f"py_json_dumps(&{self._expr(node.args[0], scope, plan)})"
                    if attr == "loads" and len(node.args) == 1:
                        inner = self._expr(node.args[0], scope, plan, expected_type=STR)
                        return f"py_json_loads(&{inner})"
                    raise PyRustError(
                        f"json.{attr} is not supported by PyRust yet (supported: dumps, loads)",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )

                qual = f"{mod}.{attr}"
                if qual in self.env.class_types:
                    cls_t = self.env.class_types[qual]
                    init_ft = cls_t.methods.get("__init__", FuncType())
                    args = self._build_call_args(init_ft, f"{qual}.__init__", node, scope, plan)
                    return f"crate::{mod}::{attr}::new({', '.join(args)})"
                if qual in self.env.func_types:
                    ft = self.env.func_types[qual]
                    args = self._build_call_args(ft, qual, node, scope, plan)
                    return f"crate::{mod}::{attr}({', '.join(args)})"
                # Unknown signature (e.g. single-file preview): emit untyped.
                args = [self._expr(a, scope, plan) for a in node.args]
                return f"crate::{mod}::{attr}({', '.join(args)})"

            recv_t = scope.expr_types.get(id(node.func.value), UNKNOWN)
            if isinstance(recv_t, ClassType) and recv_t.name == "PyFile":
                return self._pyfile_method_call(node, recv_t, scope, plan)
            if recv_t == STR and attr == "format":
                return self._str_format_call(node, scope, plan)
            mutating = attr in MUTATING_METHODS or (
                isinstance(recv_t, ClassType)
                and attr in self.alias_mut_methods().get(recv_t.name, set())
            )
            if mutating and (self._has_subscript(node.func.value) or self._root_of(node.func.value) in self._ref_names):
                pre: list[str] = []
                place = self._place(node.func.value, scope, plan, pre)
                inner = ast.Call(func=ast.Attribute(value=ast.Name(id="__recv_place__", ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                 args=node.args, keywords=node.keywords)
                scope.expr_types[id(inner.func.value)] = recv_t
                scope.expr_types[id(inner)] = scope.expr_types.get(id(node), UNKNOWN)
                self._place_override = place
                try:
                    call_code = self._call_expr(inner, scope, plan)
                finally:
                    self._place_override = None
                return "{ " + " ".join(pre) + f" {call_code} }}"
            recv_code = self._recv_no_clone(node.func.value, scope, plan)
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "__recv_place__":
                recv_code = self._place_override
            args = [self._expr(a, scope, plan) for a in node.args]

            # Builtin-type methods must never silently drop keyword arguments
            # (e.g. `xs.sort(key=...)`): refuse anything not handled exactly.
            if node.keywords and not isinstance(recv_t, ClassType):
                sort_reverse_only = (
                    isinstance(recv_t, ListType)
                    and attr == "sort"
                    and not node.args
                    and len(node.keywords) == 1
                    and node.keywords[0].arg == "reverse"
                    and isinstance(node.keywords[0].value, ast.Constant)
                )
                if not sort_reverse_only:
                    kw_names = ", ".join(f"{kw.arg or '**'}=" for kw in node.keywords)
                    raise PyRustError(
                        f"method '{attr}' with keyword argument(s) {kw_names} is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )

            if isinstance(recv_t, ListType):
                if attr == "append" and args:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"{recv_code}.push({elem_code})"
                if attr == "insert" and len(args) == 2:
                    idx = self._expr(node.args[0], scope, plan, expected_type=INT)
                    elem_code = self._expr(node.args[1], scope, plan, expected_type=recv_t.elem)
                    return f"py_list_insert(&mut {recv_code}, {idx}, {elem_code})"
                if attr == "clear" and not args:
                    return f"{recv_code}.clear()"
                if attr == "pop" and not args:
                    return f"py_list_pop(&mut {recv_code})"
                if attr == "pop" and len(args) == 1:
                    idx = self._expr(node.args[0], scope, plan, expected_type=INT)
                    return f"py_list_pop_at(&mut {recv_code}, {idx})"
                if attr == "extend" and len(args) == 1:
                    et = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                    self._check_ordered_iteration(node.args[0], scope)
                    if not (isinstance(et, ListType) and et.elem == recv_t.elem):
                        raise PyRustError(
                            "list.extend() with a non-list or differently-typed argument is not supported by PyRust yet",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    # Evaluate the argument first: `a.extend(a)` doubles a.
                    return f"{{ let __ext = {args[0]}.clone(); {recv_code}.extend(__ext); }}"
                if attr == "remove" and len(args) == 1:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"py_list_remove(&mut {recv_code}, &{elem_code})"
                if attr in ("index", "count", "remove") and self._has_identity_object(recv_t.elem):
                    raise PyRustError(
                        f"list.{attr}() over plain class instances (compared by identity in Python) is not supported natively",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                if attr == "index" and len(args) == 1:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"py_list_index(&{recv_code}, &{elem_code})"
                if attr == "count" and len(args) == 1:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"py_list_count(&{recv_code}, &{elem_code})"
                if attr == "copy" and not args:
                    return f"{recv_code}.clone()"
                if attr == "sort" and not args and not node.keywords:
                    return f"py_sort(&mut {recv_code})"
                if attr == "sort" and len(node.keywords) == 1 and len(args) == 0:
                    kw = node.keywords[0]
                    if kw.arg == "reverse" and isinstance(kw.value, ast.Constant):
                        if kw.value.value:
                            # CPython: reverse, stable sort, reverse (ties keep their order).
                            return f"{{ {recv_code}.reverse(); py_sort(&mut {recv_code}); {recv_code}.reverse(); }}"
                        return f"py_sort(&mut {recv_code})"
                if attr == "reverse" and not args:
                    return f"{recv_code}.reverse()"

            if isinstance(recv_t, SetType):
                if attr == "add" and len(args) == 1:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"{{ {recv_code}.insert({elem_code}); }}"
                if attr == "discard" and len(args) == 1:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"{{ {recv_code}.remove(&{elem_code}); }}"
                if attr == "remove" and len(args) == 1:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"{{ let __k = {elem_code}; if !{recv_code}.remove(&__k) {{ py_raise(format!(\"KeyError: {{}}\", py_repr(&__k))); }} }}"
                if attr == "clear" and not args:
                    return f"{recv_code}.clear()"
                if attr == "copy" and not args:
                    return f"{recv_code}.clone()"

            if isinstance(recv_t, DictType):
                if attr == "get" and args:
                    key_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.key)
                    if len(args) == 1:
                        return f"{recv_code}.get(&{key_code}).cloned().map(PyValue::from).unwrap_or(PyValue::None)"
                    default_code = self._expr(node.args[1], scope, plan, expected_type=recv_t.val)
                    return f"{recv_code}.get(&{key_code}).cloned().unwrap_or({default_code})"
                if attr == "keys":
                    return f"{recv_code}.keys()"
                if attr == "values":
                    return f"{recv_code}.values()"
                if attr == "items":
                    return f"{recv_code}.items()"
                if attr == "pop" and args:
                    key_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.key)
                    if len(args) > 1:
                        dflt = self._expr(node.args[1], scope, plan, expected_type=recv_t.val)
                        return f"{recv_code}.remove(&{key_code}).unwrap_or({dflt})"
                    return f"{recv_code}.pop_key(&{key_code})"
                if attr == "setdefault" and len(args) == 2:
                    key_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.key)
                    val_code = self._expr(node.args[1], scope, plan, expected_type=recv_t.val)
                    return f"{recv_code}.entry({key_code}).or_insert({val_code}).clone()"
                if attr == "update" and len(args) == 1:
                    if not isinstance(scope.expr_types.get(id(node.args[0]), UNKNOWN), DictType):
                        raise PyRustError(
                            "dict.update() with a non-dict argument is not supported by PyRust yet",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    other = self._expr(node.args[0], scope, plan)
                    return f"{{ let __upd = {other}.items(); {recv_code}.extend(__upd); }}"
                if attr == "copy" and not args:
                    return f"{recv_code}.clone()"
                if attr == "clear" and not args:
                    return f"{recv_code}.clear()"

            if recv_t == STR:
                return self._str_method_call(node, attr, recv_code, scope, plan)

            if isinstance(recv_t, ClassType):
                cls_t = self.env.class_types.get(recv_t.name, recv_t)
                if attr not in cls_t.methods:
                    raise PyRustError(
                        f"method '{recv_t.name}.{attr}' is not defined in this PyRust class",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                m_ft = cls_t.methods[attr]
                typed_args = self._build_call_args(m_ft, f"{cls_t.name}.{attr}", node, scope, plan)
                return f"{recv_code}.{attr}({', '.join(typed_args)})"

            if recv_t == ANY and attr in ("get", "keys", "values", "items"):
                recv_code = self._expr(node.func.value, scope, plan)
                if attr == "get" and 1 <= len(node.args) <= 2:
                    key = self._expr(node.args[0], scope, plan)
                    dflt = self._expr(node.args[1], scope, plan) if len(node.args) == 2 else "PyValue::None"
                    return f"py_value_get(&{recv_code}, &PyValue::from({key}), {dflt})"
                if attr in ("keys", "values", "items") and not node.args:
                    return f"py_value_{attr}(&{recv_code})"

            raise PyRustError(
                f"method '{attr}' is not supported on {recv_t.to_rust_type()} by PyRust yet",
                lineno=node.lineno,
                col=node.col_offset,
                filename=self.parsed.filename,
            )

        return "Default::default()"


def generate_rust_module(
    parsed: ParsedModule,
    env: ModuleTypeEnv,
    ownership: dict[str, OwnershipPlan],
    is_entry: bool = True,
    all_modules: list[str] | None = None,
) -> str:
    return _RustGenerator(parsed, env, ownership, is_entry=is_entry, all_modules=all_modules).generate()


def transpile_source(source: str, filename: str = "<string>") -> str:
    parsed = parse_source(source, filename=filename)
    env = infer_module(parsed)
    ownership = analyze_ownership(parsed, env)
    return generate_rust_module(parsed, env, ownership, is_entry=True)
