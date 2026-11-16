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
from pyrust.ownership import OwnershipPlan, analyze_ownership
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
        for stmt in self.parsed.main_body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, (int, float, bool))
            ):
                name = stmt.targets[0].id
                vtype = scope.var_types.get(name, UNKNOWN)
                if (
                    name in referenced_in_funcs
                    and name not in plan.mutable_vars
                    and vtype in (INT, FLOAT, BOOL)
                ):
                    code = self._expr(stmt.value, scope, plan, expected_type=vtype)
                    consts.append((name, vtype, code))
                    self.module_consts.add(name)
        return consts

    def generate(self) -> str:
        self.emit("#![allow(dead_code, unused_imports, unused_variables, unused_mut, non_snake_case)]")
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
            self.indent += 1
            scope = self.env.module_scope
            plan = self.ownership.get("__main__", OwnershipPlan())
            declared = set(scope.params) | set(self.module_consts)
            self._emit_hoisted_declarations(scope, plan, declared)
            self._emit_block(self.parsed.main_body, scope, plan, declared, current_class=None)
            self.indent -= 1
            self.emit("}")

        return "\n".join(self.lines) + "\n"

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
        strs = [
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
        try:
            self._emit_class_init_body(cls_name, fn_node, cls_type)
        finally:
            self.in_function = prev_in_function

    def _emit_class_init_body(self, cls_name: str, fn_node: ast.FunctionDef, cls_type: ClassType) -> None:
        scope = self.env.class_method_scopes[(cls_name, "__init__")]
        plan = self.ownership.get(f"{cls_name}.__init__", OwnershipPlan())
        init_ft = cls_type.methods["__init__"]
        non_self = [a for a in _positional_params(fn_node) if a.arg != "self"]
        param_strs = self._def_param_strs(fn_node, init_ft, plan, skip_self=True)
        self.emit(f"pub fn new({', '.join(param_strs)}) -> Self {{")
        self.indent += 1

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

    def _emit_class_method(self, cls_name: str, fn_node: ast.FunctionDef, cls_type: ClassType) -> None:
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
        try:
            self._emit_class_method_body(cls_name, fn_node, cls_type)
        finally:
            self.in_function = prev_in_function

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
        declared = set(self._all_param_names(fn_node))
        self._emit_hoisted_declarations(scope, plan, declared)
        self._emit_block(fn_node.body, scope, plan, declared, current_class=cls_name, expected_ret=m_ft.ret)
        self.indent -= 1
        self.emit("}")

    def _emit_function(self, fn_node: ast.FunctionDef) -> None:
        self._check_fn_decorators(fn_node, f"function {fn_node.name}")
        prev_in_function, self.in_function = self.in_function, True
        try:
            self._emit_function_body(fn_node)
        finally:
            self.in_function = prev_in_function

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
            for target in stmt.targets:
                self._emit_assignment(target, stmt.value, scope, plan, declared)
            return

        if isinstance(stmt, ast.AnnAssign):
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

            lhs = self._expr(stmt.target, scope, plan)
            target_t = scope.expr_types.get(id(stmt.target), UNKNOWN)
            rhs = self._expr(stmt.value, scope, plan, target_t)
            op_map = {
                ast.Add: "+=",
                ast.Sub: "-=",
                ast.Mult: "*=",
                ast.Div: "/=",
            }
            for op_cls, op_sym in op_map.items():
                if isinstance(stmt.op, op_cls):
                    if isinstance(stmt.target, ast.Name) and scope.var_types.get(stmt.target.id) == ANY:
                        bin_op = op_sym[0]
                        self.emit(f"{lhs} = {lhs}.clone() {bin_op} {rhs};")
                    elif isinstance(stmt.op, ast.Add) and target_t == STR:
                        self.emit(f"{lhs}.push_str(&{rhs});")
                    else:
                        self.emit(f"{lhs} {op_sym} {rhs};")
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
            if else_var is None:
                self.emit(f"for {target_str} in {iter_str} {{")
            else:
                self.emit("{")
                self.indent += 1
                self.emit(f"let mut {else_var} = true;")
                self.emit(f"for {target_str} in {iter_str} {{")
            self.indent += 1
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
                msg_str = self._expr(stmt.msg, scope, plan)
                self.emit(f'assert!({cond_str}, "{{}}", {msg_str});')
            else:
                self.emit(f"assert!({cond_str});")
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
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            ename = exc.func.id
            if not exc.args:
                self.emit(f'panic!("{ename}");')
            else:
                parts = ", ".join(["{}"] * len(exc.args))
                arg_code = ", ".join(self._expr(a, scope, plan) for a in exc.args)
                self.emit(f'panic!("{ename}: {parts}", {arg_code});')
            return
        if isinstance(exc, ast.Name):
            self.emit(f'panic!("{exc.id}");')
            return
        raise PyRustError(
            "only `raise ErrorType(message)` is supported by PyRust",
            lineno=stmt.lineno,
            col=stmt.col_offset,
            filename=self.parsed.filename,
        )

    def _emit_delete(self, stmt: ast.Delete, scope: ScopeInfo, plan: OwnershipPlan) -> None:
        for target in stmt.targets:
            if isinstance(target, ast.Subscript):
                base_t = scope.expr_types.get(id(target.value), UNKNOWN)
                base = self._expr(target.value, scope, plan)
                if isinstance(base_t, DictType):
                    key = self._expr(target.slice, scope, plan, expected_type=base_t.key)
                    self.emit(f"let _ = {base}.remove(&{key});")
                    continue
                if isinstance(base_t, ListType) and not isinstance(target.slice, ast.Slice):
                    idx = self._expr(target.slice, scope, plan)
                    self.emit(f"let _ = {base}.remove(({idx}) as usize);")
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
            lhs = self._attribute_lvalue(target, scope, plan)
            attr_t = scope.expr_types.get(id(target), None)
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

    def _check_single_generator(self, node: ast.expr, kind: str) -> ast.comprehension:
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
        use_debug = part.conversion == 114 and val_t == STR  # !r on strings
        spec_node = part.format_spec
        if spec_node is None or (isinstance(spec_node, ast.JoinedStr) and not spec_node.values):
            rust_spec: str | None = ""
            use_commas = False
        elif isinstance(spec_node, ast.JoinedStr) and len(spec_node.values) == 1 and isinstance(
            spec_node.values[0], ast.Constant
        ):
            rust_spec, use_commas = self._translate_fmt_spec(
                spec_node.values[0].value, val_t, spec_node
            )
        else:
            raise PyRustError(
                "nested f-string format specs (e.g. f'{x:{w}}') are not supported by PyRust yet",
                lineno=getattr(spec_node, "lineno", None),
                col=getattr(spec_node, "col_offset", None),
                filename=self.parsed.filename,
            )
        arg_code = self._expr(part.value, scope, plan)
        if use_debug and rust_spec:
            raise PyRustError(
                "!r with a format spec is not supported by PyRust yet",
                lineno=getattr(part, "lineno", None),
                col=getattr(part, "col_offset", None),
                filename=self.parsed.filename,
            )
        if use_debug:
            return "{:?}", arg_code
        if use_commas:
            return "{}", f"py_format_commas({arg_code})"
        if not rust_spec:
            return "{}", self._print_arg(part.value, scope, plan)
        return "{:" + rust_spec + "}", arg_code

    def _translate_fmt_spec(
        self, spec: object, val_t: PyType, node: ast.AST
    ) -> tuple[str, bool]:
        """Translate a Python format spec to Rust. Returns (rust_spec, use_commas)."""
        if not isinstance(spec, str):
            raise PyRustError(
                "f-string format specs must be plain strings in PyRust",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )

        def err() -> PyRustError:
            return PyRustError(
                f"format spec '{spec}' is not supported by PyRust yet",
                lineno=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
                filename=self.parsed.filename,
            )

        m = re.match(
            r"^(?P<align>[<>^]?)(?P<sign>[+\- ]?)(?P<alt>#?)(?P<zero>0?)(?P<width>\d*)(?P<comma>,?)(?P<prec>\.\d+)?(?P<kind>[bcdoxXneEfFgGs%])?$",
            spec,
        )
        if not m:
            raise err()
        align, sign, alt, zero, width, comma, prec, kind = m.groups()
        if alt or sign == " " or align == "=" or comma not in ("", ","):
            raise err()
        if sign == "-":
            sign = ""
        use_commas = comma == ","
        if use_commas and (prec or (kind and kind not in ("d", ""))):
            raise err()
        if kind in (None, ""):
            if prec:
                # Bare precision on floats means significant digits in Python.
                raise err()
            rust_kind = ""
        elif kind in ("d", "s", "n"):
            rust_kind = ""
        elif kind in ("x", "X", "o", "b"):
            rust_kind = kind
        elif kind in ("e", "E"):
            rust_kind = (prec or ".6") + kind
        elif kind in ("f", "F"):
            # Rust has no F; {:.N} matches Python's f to 6 default places.
            rust_kind = prec or ".6"
        elif kind in ("g", "G", "%", "c"):
            raise err()
        else:
            raise err()
        rust_spec = f"{align}{sign}"
        if zero and width and not align:
            rust_spec += "0" + width
        else:
            rust_spec += width
        rust_spec += rust_kind
        return rust_spec, use_commas

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
                if spec:
                    rust_spec, use_commas = self._translate_fmt_spec(spec, arg_t, node)
                    if use_commas:
                        flush_literal()
                        out_parts.append("{}")
                        out_args.append(f"py_format_commas({self._expr(arg_node, scope, plan)})")
                    else:
                        flush_literal()
                        out_parts.append("{:" + rust_spec + "}")
                        out_args.append(self._expr(arg_node, scope, plan))
                elif conv == "r" and arg_t == STR:
                    flush_literal()
                    out_parts.append("{:?}")
                    out_args.append(self._expr(arg_node, scope, plan))
                else:
                    flush_literal()
                    out_parts.append("{}")
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
            # Reuse the {}-style spec translator: the % body is already
            # flags+width+precision; only the '-' flag needs remapping.
            if conv not in ("s", "d", "i", "u", "r", "f", "e", "x", "X", "o"):
                raise PyRustError(
                    f"%{conv} is not supported in % formatting",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            body = fmt[i + 1 : j]
            if body.startswith("-"):
                body = "<" + body[1:]
            kind = {"i": "d", "u": "d", "r": "s"}.get(conv, conv)
            try:
                rust_spec, use_commas = self._translate_fmt_spec(body + kind, arg_t, node)
            except PyRustError:
                raise PyRustError(
                    f"format spec '%{body}{conv}' is not supported in % formatting",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                    filename=self.parsed.filename,
                )
            ph = "{}" if not rust_spec else "{:" + rust_spec + "}"
            if conv == "r" and arg_t == STR:
                out_parts.append(ph[:-1] + "?}" if rust_spec else "{:?}")
                out_args.append(self._expr(arg_node, scope, plan))
            elif conv in ("d", "i", "u"):
                out_parts.append(ph)
                out_args.append(self._expr(arg_node, scope, plan, expected_type=INT))
            elif use_commas:
                out_parts.append("{}")
                out_args.append(f"py_format_commas({self._expr(arg_node, scope, plan)})")
            elif conv in ("f", "e"):
                # Precision belongs to the float itself, not a stringified copy.
                out_parts.append(ph)
                out_args.append(self._expr(arg_node, scope, plan))
            else:
                out_parts.append(ph)
                out_args.append(self._print_arg(arg_node, scope, plan))
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

    def _for_iter_expr(self, iter_node: ast.expr, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name):
            fname = iter_node.func.id
            if fname == "range":
                args = [self._expr(a, scope, plan) for a in iter_node.args]
                if len(args) == 1:
                    return f"(0..{args[0]})"
                if len(args) == 2:
                    return f"({args[0]}..{args[1]})"
                if len(args) == 3:
                    if _is_negative_expr(iter_node.args[2]):
                        return f"py_range({args[0]}, {args[1]}, {args[2]}).into_iter()"
                    return f"({args[0]}..{args[1]}).step_by({args[2]} as usize)"
            if fname == "enumerate" and len(iter_node.args) == 1:
                inner = self._expr(iter_node.args[0], scope, plan)
                return f"{inner}.iter().cloned().enumerate().map(|(i, v)| (i as i64, v))"
        iter_t = scope.expr_types.get(id(iter_node), UNKNOWN)
        if isinstance(iter_t, ClassType) and iter_t.name == "PyFile":
            iter_code = self._expr(iter_node, scope, plan)
            return f"({iter_code}).lines().into_iter()"
        iter_code = self._expr(iter_node, scope, plan)
        return f"{iter_code}.iter().cloned()"

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
        if expected_type == ANY and isinstance(node, ast.Constant) and node.value is None:
            return "PyValue::None"
        actual_type = scope.expr_types.get(id(node), UNKNOWN)
        raw = self._expr_raw(node, scope, plan, actual_type)
        if expected_type == ANY and actual_type != ANY:
            return f"PyValue::from({raw})"
        if expected_type == FLOAT and actual_type == INT:
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                return f"{float(node.value)}"
            return f"({raw} as f64)"
        if expected_type == FLOAT and actual_type in (ANY, UNKNOWN):
            return f"((({raw})).as_f64())"
        if expected_type == INT and actual_type == FLOAT:
            return f"({raw} as i64)"
        if expected_type == INT and actual_type == BOOL:
            return f"({raw} as i64)"
        if expected_type == INT and actual_type in (ANY, UNKNOWN):
            return f"((({raw})).as_i64())"
        if expected_type == BOOL and actual_type in (ANY, UNKNOWN):
            return f"((({raw})).is_truthy())"
        return raw

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
            if (
                self.in_function
                and node.id not in scope.var_types
                and node.id not in scope.params
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
                recv_any = self._expr(node.value, scope, plan)
                return f'py_value_get(&{recv_any}, &PyValue::from("{node.attr}".to_string()), PyValue::None)'
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
            op_val = self._expr(node.operand, scope, plan)
            if isinstance(node.op, ast.USub):
                return f"-{op_val}"
            if isinstance(node.op, ast.UAdd):
                return f"{op_val}"
            if isinstance(node.op, ast.Not):
                return f"!({self._cond_expr(node.operand, scope, plan)})"

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
            if val_t == STR:
                return f"py_str_index(&{val_code}, {idx_code})"
            if isinstance(val_t, DictType):
                key_t = scope.expr_types.get(id(node.slice), UNKNOWN)
                if key_t in (ANY, UNKNOWN):
                    return f"py_value_get(&PyValue::from({val_code}), &PyValue::from({idx_code}), PyValue::None)"
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

        if isinstance(node.op, ast.Add) and isinstance(lt, ListType) and isinstance(rt, ListType):
            return f"py_list_concat(&{self._expr(node.left, scope, plan)}, &{self._expr(node.right, scope, plan)})"

        if isinstance(node.op, ast.Mult):
            if lt == STR and rt == INT:
                return f"{self._expr(node.left, scope, plan)}.repeat({self._expr(node.right, scope, plan)} as usize)"
            if lt == INT and rt == STR:
                return f"{self._expr(node.right, scope, plan)}.repeat({self._expr(node.left, scope, plan)} as usize)"
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
        if func_name == "pow" and len(args) == 2:
            return f"py_pow_f64({float_arg(0)}, {float_arg(1)})"
        if func_name == "atan2" and len(args) == 2:
            return f"({float_arg(0)}).atan2({float_arg(1)})"
        if func_name == "hypot" and len(args) == 2:
            return f"({float_arg(0)}).hypot({float_arg(1)})"
        if func_name == "log" and len(args) == 2:
            return f'py_div_f64(py_math1("log", {float_arg(0)}), py_math1("log", {float_arg(1)}))'
        if func_name in MATH_FLOAT_FUNCS and func_name not in ("pow", "atan2", "hypot") and len(args) == 1:
            return f'py_math1("{func_name}", {float_arg(0)})'
        if func_name in ("floor", "ceil") and len(args) == 1:
            return f"py_float_to_int(({float_arg(0)}).{func_name}())"
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

    def _tuple_repr(self, prefix: str, tup: TupleType) -> tuple[str, list[str]]:
        """Build (fmt, args) rendering a tuple with Python repr semantics."""
        parts: list[str] = []
        args: list[str] = []
        for i, e in enumerate(tup.elements):
            acc = f"{prefix}.{i}"
            if e == FLOAT:
                parts.append("{}")
                args.append(f"py_float_str({acc})")
            elif e == BOOL:
                parts.append("{}")
                args.append(f"(if {acc} {{ \"True\" }} else {{ \"False\" }})")
            elif e == STR:
                parts.append("{}")
                args.append(f"py_repr_str(&{acc})")
            elif e in (INT, ANY):
                parts.append("{}")
                args.append(acc)
            else:
                parts.append("{:?}")
                args.append(acc)
        if len(tup.elements) == 1:
            return f"({', '.join(parts)},)", args
        return f"({', '.join(parts)})", args

    def _format_placeholder(self, arg: ast.expr, scope: ScopeInfo) -> str:
        at = scope.expr_types.get(id(arg), UNKNOWN)
        if isinstance(at, (ListType, DictType, SetType, TupleType, ClassType)):
            return "{:?}"
        return "{}"

    def _print_placeholder(self, arg: ast.expr, scope: ScopeInfo) -> str:
        at = scope.expr_types.get(id(arg), UNKNOWN)
        if self._wrap_pyvalue(at):
            return "{}"
        if isinstance(at, TupleType) or (
            isinstance(at, ListType)
            and isinstance(at.elem, TupleType)
            and all(self._is_simple_scalar(e) or e == ANY for e in at.elem.elements)
        ):
            return "{}"
        return self._format_placeholder(arg, scope)

    def _print_call(self, node: ast.Call, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        sep_node: ast.expr | None = None
        end_node: ast.expr | None = None
        for kw in node.keywords:
            if kw.arg is None:
                raise PyRustError(
                    "** unpacking in print() is not supported by PyRust yet",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
            if kw.arg == "sep":
                sep_node = kw.value
            elif kw.arg == "end":
                end_node = kw.value
            else:
                raise PyRustError(
                    f"print({kw.arg}=...) is not supported by PyRust yet (supported: sep, end)",
                    lineno=node.lineno,
                    col=node.col_offset,
                    filename=self.parsed.filename,
                )
        if sep_node is None and end_node is None:
            if not node.args:
                return "println!()"
            parts: list[str] = []
            arg_exprs: list[str] = []
            for a in node.args:
                parts.append(self._print_placeholder(a, scope))
                arg_exprs.append(self._print_arg(a, scope, plan))
            fmt = " ".join(parts)
            return f'println!("{fmt}", {", ".join(arg_exprs)})'
        sep_code = (
            self._expr(sep_node, scope, plan, expected_type=STR)
            if sep_node is not None
            else '" ".to_string()'
        )
        elems = ", ".join(
            f"format!(\"{self._print_placeholder(a, scope)}\", {self._print_arg(a, scope, plan)})"
            for a in node.args
        )
        content = f"vec![{elems}].join(&{sep_code})"
        if end_node is None:
            return f"println!(\"{{}}\", {content})"
        if isinstance(end_node, ast.Constant) and end_node.value == "":
            return f"print!(\"{{}}\", {content})"
        if isinstance(end_node, ast.Constant) and isinstance(end_node.value, str):
            return f"print!(\"{{}}{self._rust_string_escape(end_node.value)}\", {content})"
        end_code = self._expr(end_node, scope, plan, expected_type=STR)
        return f"print!(\"{{}}{{}}\", {content}, {end_code})"

    def _print_arg(self, node: ast.expr, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        """Render a print()/str()-style argument (Python repr for float/bool/containers)."""
        actual = scope.expr_types.get(id(node), UNKNOWN)
        if actual == FLOAT:
            return f"py_float_str({self._expr(node, scope, plan)})"
        if actual == BOOL:
            return f"(if {self._expr(node, scope, plan)} {{ \"True\" }} else {{ \"False\" }})"
        if self._wrap_pyvalue(actual):
            # _expr already inserts .clone() when ownership requires one.
            return f"PyValue::from({self._expr(node, scope, plan)})"
        if isinstance(actual, TupleType) and all(
            self._is_simple_scalar(e) or e == ANY for e in actual.elements
        ):
            fmt, args = self._tuple_repr(self._expr(node, scope, plan), actual)
            return f'format!("{fmt}", {", ".join(args)})'
        if (
            isinstance(actual, ListType)
            and isinstance(actual.elem, TupleType)
            and all(self._is_simple_scalar(e) or e == ANY for e in actual.elem.elements)
        ):
            raw = self._expr(node, scope, plan)
            fmt, inner = self._tuple_repr("t", actual.elem)
            map_arg = ", ".join(inner)
            return f'format!("[{{}}]", {raw}.iter().map(|t| format!("{fmt}", {map_arg})).collect::<Vec<_>>().join(", "))'
        return self._expr(node, scope, plan)

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
                return "std::process::exit(0)"
            if len(node.args) == 1 and not node.keywords:
                at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                code = self._expr(node.args[0], scope, plan)
                if at == STR:
                    return f'{{ eprintln!("{{}}", {code}); std::process::exit(1) }}'
                return f"std::process::exit(({code}) as i32)"
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
            return f"std::env::var(&{key}).unwrap_or_default()"
        if attr == "getcwd" and not node.args:
            return "std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default()"
        if attr == "listdir" and len(node.args) <= 1:
            path = self._expr(node.args[0], scope, plan, expected_type=STR) if node.args else '".".to_string()'
            return (
                f"std::fs::read_dir(&{path}).map(|rd| rd.filter_map(|e| e.ok().and_then(|d| d.file_name().into_string().ok()))"
                ".collect::<Vec<String>>()).unwrap_or_default()"
            )
        if attr in ("makedirs", "mkdir") and node.args:
            path = self._expr(node.args[0], scope, plan, expected_type=STR)
            exist_ok = True
            for kw in node.keywords:
                if kw.arg == "exist_ok" and isinstance(kw.value, ast.Constant):
                    exist_ok = bool(kw.value.value)
                elif kw.arg is None:
                    raise PyRustError(
                        "** unpacking is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                else:
                    raise PyRustError(
                        f"os.makedirs() got an unexpected keyword '{kw.arg}'",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
            if attr == "mkdir" and not exist_ok:
                return f"std::fs::create_dir(&{path}).expect(\"FileExistsError\")"
            if not exist_ok:
                return f"{{ let _p = {path}; if !std::path::Path::new(&_p).exists() {{ std::fs::create_dir_all(&_p).expect(\"OSError\"); }} }}"
            return f"let _ = std::fs::create_dir_all(&{path});"
        if attr == "remove" and len(node.args) == 1:
            path = self._expr(node.args[0], scope, plan, expected_type=STR)
            return f"let _ = std::fs::remove_file(&{path});"
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
            return f"std::thread::sleep(std::time::Duration::from_secs_f64({secs}))"
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
        if func_name in ("mean", "harmonic_mean", "variance", "pvariance", "stdev", "pstdev"):
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
        for i in range(n_fixed):
            pname = param_names[i] if i < len(param_names) else None
            ptype = ft.params[i] if i < len(ft.params) else None
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
                    out.append(self._expr(defaults[di], scope, plan, expected_type=ptype))
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

    def _call_expr(self, node: ast.Call, scope: ScopeInfo, plan: OwnershipPlan) -> str:
        if isinstance(node.func, ast.Name):
            fname = node.func.id

            if fname == "print":
                return self._print_call(node, scope, plan)

            if fname == "len" and len(node.args) == 1:
                return f"({self._expr(node.args[0], scope, plan)}.len() as i64)"

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
                    return f"{inner}.trim().parse::<i64>().unwrap_or(0)"
                if at in (ANY, UNKNOWN):
                    return f"({inner}.as_i64())"
                return f"({inner} as i64)"
            if fname == "int" and not node.args:
                return "0"

            if fname == "float" and len(node.args) == 1:
                at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                inner = self._expr(node.args[0], scope, plan)
                if at == STR:
                    return f"{inner}.trim().parse::<f64>().unwrap_or(0.0)"
                if at in (ANY, UNKNOWN):
                    return f"({inner}.as_f64())"
                return f"({inner} as f64)"
            if fname == "float" and not node.args:
                return "0.0"

            if fname == "abs" and len(node.args) == 1:
                at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                inner = self._expr(node.args[0], scope, plan)
                if at in (ANY, UNKNOWN):
                    return f"py_abs(&{inner})"
                return f"{inner}.abs()"

            if fname == "sum" and 1 <= len(node.args) <= 2:
                at = scope.expr_types.get(id(node.args[0]), ListType(INT))
                elem_t = at.elem if isinstance(at, ListType) else None
                inner = self._expr(node.args[0], scope, plan)
                dyn_sum = elem_t is None or elem_t == ANY or elem_t == UNKNOWN
                if dyn_sum:
                    total = f"{inner}.iter().cloned().fold(PyValue::Int(0), |a, b| a + b)"
                else:
                    elem_rust = elem_t.to_rust_type()  # type: ignore[union-attr]
                    total = f"{inner}.iter().copied().sum::<{elem_rust}>()"
                if len(node.args) == 2:
                    if dyn_sum:
                        total = f"({total} + PyValue::from({self._expr(node.args[1], scope, plan)}))"
                    else:
                        total = f"({total} + {self._expr(node.args[1], scope, plan, expected_type=elem_t)})"
                return total

            if fname == "sorted" and len(node.args) == 1:
                if node.keywords:
                    raise PyRustError(
                        "sorted() with keyword arguments is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                at_sorted = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                if at_sorted == STR:
                    inner = self._expr(node.args[0], scope, plan)
                    return f"py_sorted(&{inner}.chars().map(|c| c.to_string()).collect::<Vec<String>>())"
                if isinstance(at_sorted, SetType):
                    inner = self._expr(node.args[0], scope, plan)
                    return f"py_sorted(&{inner}.iter().cloned().collect::<Vec<_>>())"
                if isinstance(at_sorted, DictType):
                    inner = self._expr(node.args[0], scope, plan)
                    return f"py_sorted(&{inner}.keys())"
                return f"py_sorted(&{self._expr(node.args[0], scope, plan)})"

            if fname == "list":
                if not node.args:
                    return "Vec::new()"
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
                inner = self._expr(node.args[0], scope, plan)
                if len(node.args) == 2:
                    start = self._expr(node.args[1], scope, plan, expected_type=INT)
                    return f"{inner}.iter().cloned().enumerate().map(|(i, v)| (i as i64 + {start}, v)).collect::<Vec<_>>()"
                return f"{inner}.iter().cloned().enumerate().map(|(i, v)| (i as i64, v)).collect::<Vec<_>>()" 

            if fname == "divmod" and len(node.args) == 2:
                a0 = self._expr(node.args[0], scope, plan)
                a1 = self._expr(node.args[1], scope, plan)
                at = scope.expr_types.get(id(node), TupleType((INT, INT)))
                if isinstance(at, TupleType) and all(e == FLOAT for e in at.elements):
                    return f"(py_floordiv_f64(({a0} as f64), ({a1} as f64)), py_mod_f64(({a0} as f64), ({a1} as f64)))"
                return f"(py_floordiv({a0}, {a1}), py_mod({a0}, {a1}))"

            if fname == "round" and 1 <= len(node.args) <= 2:
                if len(node.args) == 2:
                    a0 = self._expr(node.args[0], scope, plan, expected_type=FLOAT)
                    a1 = self._expr(node.args[1], scope, plan, expected_type=INT)
                    return f"py_round_prec({a0}, {a1})"
                at = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                if at == INT:
                    return self._expr(node.args[0], scope, plan)
                return f"py_round_bankers({self._expr(node.args[0], scope, plan, expected_type=FLOAT)})"

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
                inner = self._expr(node.args[0], scope, plan)
                if at == STR:
                    return f"{inner}.chars().rev().map(|c| c.to_string()).collect::<Vec<String>>()"
                return f"{inner}.iter().rev().cloned().collect::<Vec<_>>()"

            if fname == "ord" and len(node.args) == 1:
                inner = self._expr(node.args[0], scope, plan, expected_type=STR)
                return f"({inner}.chars().next().expect(\"ord() of empty string\") as i64)"

            if fname == "chr" and len(node.args) == 1:
                inner = self._expr(node.args[0], scope, plan, expected_type=INT)
                return f"std::char::from_u32(({inner}) as u32).unwrap_or('\\0').to_string()"

            if fname in ("hex", "oct", "bin") and len(node.args) == 1:
                inner = self._expr(node.args[0], scope, plan, expected_type=INT)
                fmt = {"hex": "#x", "oct": "#o", "bin": "#b"}[fname]
                return f'format!("{{:{fmt}}}", {inner})'

            if fname == "zip" and len(node.args) == 2:
                a0 = self._expr(node.args[0], scope, plan)
                a1 = self._expr(node.args[1], scope, plan)
                return f"{a0}.iter().cloned().zip({a1}.iter().cloned()).collect::<Vec<_>>()"

            if fname in ("min", "max") and node.args:
                if node.keywords:
                    raise PyRustError(
                        f"{fname}(... with keyword arguments) is not supported by PyRust yet",
                        lineno=node.lineno,
                        col=node.col_offset,
                        filename=self.parsed.filename,
                    )
                if len(node.args) == 1:
                    op = "min_by" if fname == "min" else "max_by"
                    inner = self._expr(node.args[0], scope, plan)
                    at = scope.expr_types.get(id(node.args[0]), ListType(INT))
                    if at == STR:
                        cmp = "a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)"
                        return f"{inner}.chars().{op}(|a, b| {cmp}).unwrap_or_default().to_string()"
                    elem = at.elem if isinstance(at, ListType) else INT
                    cmp = "a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)"
                    if elem.is_copy:
                        return f"*{inner}.iter().{op}(|a, b| {cmp}).unwrap()"
                    return f"{inner}.iter().{op}(|a, b| {cmp}).cloned().unwrap()"
                if len(node.args) == 2:
                    a0 = self._expr(node.args[0], scope, plan)
                    a1 = self._expr(node.args[1], scope, plan)
                    return f"{a0}.{fname}({a1})"

            if fname == "range":
                args = [self._expr(a, scope, plan) for a in node.args]
                if len(args) == 1:
                    return f"(0..{args[0]}).collect::<Vec<i64>>()"
                if len(args) == 2:
                    return f"({args[0]}..{args[1]}).collect::<Vec<i64>>()"
                if len(args) == 3:
                    if _is_negative_expr(node.args[2]):
                        return f"py_range({args[0]}, {args[1]}, {args[2]})"
                    return f"({args[0]}..{args[1]}).step_by({args[2]} as usize).collect::<Vec<i64>>()"

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
            recv_code = self._recv_no_clone(node.func.value, scope, plan)
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
                    return f"{recv_code}.insert(({idx}) as usize, {elem_code})"
                if attr == "clear" and not args:
                    return f"{recv_code}.clear()"
                if attr == "pop" and not args:
                    return f"{recv_code}.pop().unwrap_or_default()"
                if attr == "pop" and len(args) == 1:
                    idx = self._expr(node.args[0], scope, plan, expected_type=INT)
                    return f"{recv_code}.remove(({idx}) as usize)"
                if attr == "extend" and args:
                    return f"{recv_code}.extend({args[0]}.iter().cloned())"
                if attr == "remove" and len(args) == 1:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"py_list_remove(&mut {recv_code}, &{elem_code})"
                if attr == "index" and len(args) == 1:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"{recv_code}.iter().position(|v| v == &{elem_code}).expect(\"ValueError: list.index(x): x not in list\") as i64"
                if attr == "count" and len(args) == 1:
                    elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                    return f"py_list_count(&{recv_code}, &{elem_code})"
                if attr == "copy" and not args:
                    return f"{recv_code}.clone()"
                if attr == "sort" and not args and not node.keywords:
                    return f"{recv_code}.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))"
                if attr == "sort" and len(node.keywords) == 1 and len(args) == 0:
                    kw = node.keywords[0]
                    if kw.arg == "reverse" and isinstance(kw.value, ast.Constant):
                        if kw.value.value:
                            return f"{{ {recv_code}.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); {recv_code}.reverse(); }}"
                        return f"{recv_code}.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))"
                if attr == "reverse" and not args:
                    return f"{recv_code}.reverse()"

            if isinstance(recv_t, SetType) and attr == "add" and args:
                elem_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.elem)
                return f"{recv_code}.insert({elem_code})"

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
                    return f"{recv_code}.remove(&{key_code}).expect(\"KeyError\")"
                if attr == "setdefault" and len(args) == 2:
                    key_code = self._expr(node.args[0], scope, plan, expected_type=recv_t.key)
                    val_code = self._expr(node.args[1], scope, plan, expected_type=recv_t.val)
                    return f"{recv_code}.entry({key_code}).or_insert({val_code}).clone()"
                if attr == "update" and len(args) == 1:
                    other = self._expr(node.args[0], scope, plan)
                    return f"{recv_code}.extend({other}.iter().map(|(k, v)| (k.clone(), v.clone())))"
                if attr == "copy" and not args:
                    return f"{recv_code}.clone()"
                if attr == "clear" and not args:
                    return f"{recv_code}.clear()"

            if recv_t == STR:
                if attr == "upper":
                    return f"{recv_code}.to_uppercase()"
                if attr == "lower":
                    return f"{recv_code}.to_lowercase()"
                if attr == "strip":
                    return f"{recv_code}.trim().to_string()"
                if attr == "lstrip":
                    return f"{recv_code}.trim_start().to_string()"
                if attr == "rstrip":
                    return f"{recv_code}.trim_end().to_string()"
                if attr == "capitalize" and not args:
                    return f"{{ let mut _c = {recv_code}.chars(); match _c.next() {{ None => String::new(), Some(_f) => _f.to_uppercase().collect::<String>() + _c.as_str() }} }}"
                if attr in ("isdigit", "isnumeric"):
                    return f"{recv_code}.chars().all(|c| c.is_ascii_digit())"
                if attr == "isalpha":
                    return f"{recv_code}.chars().all(|c| c.is_ascii_alphabetic())"
                if attr == "isspace":
                    return f"{recv_code}.chars().all(|c| c.is_whitespace())"
                if attr == "find" and 1 <= len(args) <= 2:
                    pat = self._expr(node.args[0], scope, plan, expected_type=STR)
                    if len(args) == 2:
                        start = self._expr(node.args[1], scope, plan, expected_type=INT)
                        return (
                            f"{{ let _st = ({start} as i64).max(0); {recv_code}.get(_st as usize..)"
                            f".and_then(|_s| _s.find(&*{pat}).map(|_i| _i as i64 + _st)).unwrap_or(-1) }}"
                        )
                    return f"{recv_code}.find(&*{pat}).map(|i| i as i64).unwrap_or(-1)"
                if attr == "count" and len(args) == 1:
                    pat = self._expr(node.args[0], scope, plan, expected_type=STR)
                    return f"{recv_code}.matches(&*{pat}).count() as i64"
                if attr == "removeprefix" and len(args) == 1:
                    pre = self._expr(node.args[0], scope, plan, expected_type=STR)
                    return f"{{ let _s = {recv_code}; _s.strip_prefix(&*{pre}).unwrap_or(&_s).to_string() }}"
                if attr == "removesuffix" and len(args) == 1:
                    suf = self._expr(node.args[0], scope, plan, expected_type=STR)
                    return f"{{ let _s = {recv_code}; _s.strip_suffix(&*{suf}).unwrap_or(&_s).to_string() }}"
                if attr == "split" and not args:
                    return f"py_split_whitespace(&{recv_code})"
                if attr == "split" and len(args) <= 2:
                    sep = self._expr(node.args[0], scope, plan, expected_type=STR)
                    if len(args) == 2:
                        n = node.args[1]
                        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
                            return f"{recv_code}.splitn(({n.value} + 1) as usize, &*{sep}).map(|s| s.to_string()).collect::<Vec<String>>()"
                        raise PyRustError(
                            "str.split(sep, maxsplit) needs a constant maxsplit in PyRust",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    return f"{recv_code}.split(&*{sep}).map(|s| s.to_string()).collect::<Vec<String>>()"
                if attr == "replace" and len(args) == 2:
                    return f"{recv_code}.replace(&*{args[0]}, &*{args[1]})"
                if attr == "join" and len(args) == 1:
                    jt = scope.expr_types.get(id(node.args[0]), UNKNOWN)
                    if jt == STR:
                        return f"{args[0]}.chars().map(|c| c.to_string()).collect::<Vec<String>>().join(&*{recv_code})"
                    return f"{args[0]}.join(&*{recv_code})"
                if attr == "startswith" and len(args) == 1:
                    if scope.expr_types.get(id(node.args[0]), UNKNOWN) != STR:
                        raise PyRustError(
                            "str.startswith() with a tuple is not supported by PyRust yet",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    return f"{recv_code}.starts_with(&*{args[0]})"
                if attr == "endswith" and len(args) == 1:
                    if scope.expr_types.get(id(node.args[0]), UNKNOWN) != STR:
                        raise PyRustError(
                            "str.endswith() with a tuple is not supported by PyRust yet",
                            lineno=node.lineno,
                            col=node.col_offset,
                            filename=self.parsed.filename,
                        )
                    return f"{recv_code}.ends_with(&*{args[0]})"

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
