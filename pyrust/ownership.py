from __future__ import annotations

import ast
from dataclasses import dataclass, field

from pyrust.inference import ModuleTypeEnv, ScopeInfo
from pyrust.parser import ParsedModule
from pyrust.types import UNKNOWN


@dataclass
class OwnershipPlan:
    mutable_vars: set[str] = field(default_factory=set)
    clone_expr_ids: set[int] = field(default_factory=set)
    borrow_params: dict[str, list[bool]] = field(default_factory=dict)
    borrow_expr_ids: set[int] = field(default_factory=set)


MUTATING_METHODS = {
    "append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse",
    "add", "discard", "update", "setdefault", "popitem", "write", "close",
}


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def mutating_methods(parsed: ParsedModule) -> dict[str, set[str]]:
    """class name -> methods that (transitively) mutate `self`."""
    result: dict[str, set[str]] = {}
    for cls in parsed.classes:
        methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
        direct: set[str] = set()
        calls: dict[str, set[str]] = {}
        for name, fn in methods.items():
            calls[name] = set()
            for node in ast.walk(fn):
                targets: list[ast.expr] = []
                if isinstance(node, (ast.Assign, ast.Delete)):
                    targets = list(node.targets)
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    targets = [node.target]
                for t in targets:
                    if isinstance(t, (ast.Attribute, ast.Subscript)) and _root_name(t) == "self":
                        direct.add(name)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    recv = node.func.value
                    if node.func.attr in MUTATING_METHODS and isinstance(recv, (ast.Attribute, ast.Subscript)) \
                            and _root_name(recv) == "self":
                        direct.add(name)
                    if isinstance(recv, ast.Name) and recv.id == "self":
                        calls[name].add(node.func.attr)
                    elif _root_name(recv) == "self":
                        # self.field.method(...): mutating if that method mutates its receiver;
                        # treated conservatively as a mutation of self.
                        direct.add(name)
        mut = set(direct)
        changed = True
        while changed:
            changed = False
            for name, callees in calls.items():
                if name not in mut and callees & mut:
                    mut.add(name)
                    changed = True
        result[cls.name] = mut
    # single inheritance: inherited mutating methods
    by_name = {c.name: c for c in parsed.classes}
    for cls in parsed.classes:
        seen = set()
        base = cls
        while base.bases and isinstance(base.bases[0], ast.Name) and base.bases[0].id in by_name \
                and base.bases[0].id not in seen:
            seen.add(base.bases[0].id)
            base = by_name[base.bases[0].id]
            own = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
            result[cls.name] |= {m for m in result.get(base.name, set()) if m not in own}
    return result


def _mutated_roots(stmts: list[ast.stmt], scope: ScopeInfo, mut_methods: dict[str, set[str]]) -> set[str]:
    """Names whose value is modified in place (needs `let mut` / `&mut self`)."""
    roots: set[str] = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            targets: list[ast.expr] = []
            if isinstance(node, (ast.Assign, ast.Delete)):
                targets = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    roots.add(node.target.id)
            for t in targets:
                for sub in ([t] if not isinstance(t, (ast.Tuple, ast.List)) else list(t.elts)):
                    if isinstance(sub, (ast.Attribute, ast.Subscript)):
                        r = _root_name(sub)
                        if r:
                            roots.add(r)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                recv = node.func.value
                r = _root_name(recv)
                if r is None:
                    continue
                if node.func.attr in MUTATING_METHODS:
                    roots.add(r)
                    continue
                rt = scope.expr_types.get(id(recv))
                cls_name = getattr(rt, "name", None)
                if cls_name and node.func.attr in mut_methods.get(cls_name, set()):
                    roots.add(r)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    node.func.attr == "shuffle" and node.args:
                r = _root_name(node.args[0])
                if r:
                    roots.add(r)
    return roots


def _analyze_scope_stmts(
    stmts: list[ast.stmt],
    scope: ScopeInfo,
    parsed: ParsedModule | None = None,
) -> OwnershipPlan:
    mutable_vars = set(scope.mutated_vars) | set(scope.hoisted_vars)
    for var_name, count in scope.assignment_counts.items():
        if count > 1:
            mutable_vars.add(var_name)
    if parsed is not None:
        mutable_vars |= _mutated_roots(stmts, scope, mutating_methods(parsed))

    # Track loads of each non-Copy variable: (node, in_loop, is_borrow_only)
    loads_by_var: dict[str, list[tuple[ast.Name, bool, bool]]] = {}

    def record_load(node: ast.Name, in_loop: bool, is_borrow: bool) -> None:
        # Prefer per-occurrence type (correct for isolated comprehensions
        # like `[r.size for r in recs]` where outer `r` differs from inner).
        vtype = scope.expr_types.get(id(node), scope.var_types.get(node.id, UNKNOWN))
        if not vtype.is_copy:
            loads_by_var.setdefault(node.id, []).append((node, in_loop, is_borrow))

    def walk_target(target: ast.expr, in_loop: bool) -> None:
        if isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name):
                record_load(target.value, in_loop, is_borrow=True)
            else:
                walk_expr(target.value, in_loop)
            walk_expr(target.slice, in_loop)
        elif isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name):
                record_load(target.value, in_loop, is_borrow=True)
            else:
                walk_expr(target.value, in_loop)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                walk_target(elt, in_loop)

    def walk_expr(expr: ast.expr | None, in_loop: bool) -> None:
        if expr is None:
            return
        if isinstance(expr, ast.Name) and isinstance(expr.ctx, ast.Load):
            record_load(expr, in_loop, is_borrow=False)
            return

        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Attribute):
                if isinstance(expr.func.value, ast.Name):
                    record_load(expr.func.value, in_loop, is_borrow=True)
                else:
                    walk_expr(expr.func.value, in_loop)
            else:
                walk_expr(expr.func, in_loop)
            # random.shuffle(seq) takes &mut and random.choice(seq) takes &:
            # their arguments are borrows, never moves (a .clone() on them
            # would shuffle/read a throwaway copy).
            borrow_args = _random_borrow_call(expr, parsed) is not None
            if borrow_args:
                for arg in expr.args:
                    if isinstance(arg, ast.Name) and isinstance(arg.ctx, ast.Load):
                        record_load(arg, in_loop, is_borrow=True)
                    else:
                        walk_expr(arg, in_loop)
                for kw in expr.keywords:
                    walk_expr(kw.value, in_loop)
                return
            for arg in expr.args:
                walk_expr(arg, in_loop)
            for kw in expr.keywords:
                walk_expr(kw.value, in_loop)
            return

        if isinstance(expr, ast.Attribute):
            if isinstance(expr.value, ast.Name):
                record_load(expr.value, in_loop, is_borrow=True)
            else:
                walk_expr(expr.value, in_loop)
            return

        if isinstance(expr, ast.Subscript):
            if isinstance(expr.value, ast.Name):
                record_load(expr.value, in_loop, is_borrow=True)
            else:
                walk_expr(expr.value, in_loop)
            walk_expr(expr.slice, in_loop)
            return

        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp)):
            for gen in expr.generators:
                walk_expr(gen.iter, in_loop)
                for cond in gen.ifs:
                    walk_expr(cond, True)
            if isinstance(expr, ast.DictComp):
                walk_expr(expr.key, True)
                walk_expr(expr.value, True)
            else:
                walk_expr(expr.elt, True)
            return

        for child in ast.iter_child_nodes(expr):
            if isinstance(child, ast.expr):
                walk_expr(child, in_loop)

    def walk_stmts(block: list[ast.stmt], in_loop: bool) -> None:
        for s in block:
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    walk_target(t, in_loop)
                walk_expr(s.value, in_loop)
            elif isinstance(s, ast.AnnAssign):
                walk_target(s.target, in_loop)
                walk_expr(s.value, in_loop)
            elif isinstance(s, ast.AugAssign):
                walk_target(s.target, in_loop)
                walk_expr(s.value, in_loop)
            elif isinstance(s, ast.Return):
                walk_expr(s.value, in_loop)
            elif isinstance(s, ast.Expr):
                walk_expr(s.value, in_loop)
            elif isinstance(s, ast.Assert):
                walk_expr(s.test, in_loop)
                walk_expr(s.msg, in_loop)
            elif isinstance(s, ast.For):
                walk_expr(s.iter, in_loop)
                walk_stmts(s.body, True)
                walk_stmts(s.orelse, True)
            elif isinstance(s, ast.While):
                walk_expr(s.test, True)
                walk_stmts(s.body, True)
                walk_stmts(s.orelse, True)
            elif isinstance(s, ast.If):
                walk_expr(s.test, in_loop)
                walk_stmts(s.body, in_loop)
                walk_stmts(s.orelse, in_loop)
            elif isinstance(s, ast.Try):
                walk_stmts(s.body, in_loop)
                for h in s.handlers:
                    walk_stmts(h.body, in_loop)
                walk_stmts(s.orelse, in_loop)
                walk_stmts(s.finalbody, in_loop)
            elif isinstance(s, ast.With):
                for item in s.items:
                    walk_expr(item.context_expr, in_loop)
                    if isinstance(item.optional_vars, ast.Name):
                        pass
                    elif item.optional_vars is not None:
                        walk_target(item.optional_vars, in_loop)
                walk_stmts(s.body, in_loop)

    walk_stmts(stmts, in_loop=False)

    clone_expr_ids: set[int] = set()
    borrow_expr_ids = {id(node) for occ in loads_by_var.values() for node, _, is_borrow in occ if is_borrow}
    for var_name, occurrences in loads_by_var.items():
        total = len(occurrences)
        for idx, (node, in_loop, is_borrow) in enumerate(occurrences):
            if is_borrow:
                continue
            # Clone any value-moving use that precedes a later use (value OR borrow) or occurs in a loop
            if idx < total - 1 or in_loop:
                clone_expr_ids.add(id(node))

    return OwnershipPlan(mutable_vars=mutable_vars, clone_expr_ids=clone_expr_ids, borrow_expr_ids=borrow_expr_ids)


def _random_borrow_call(expr: ast.Call, parsed: ParsedModule | None) -> str | None:
    """Return the random function name if this call only borrows its arguments."""
    if isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name):
        if expr.func.value.id == "random" and expr.func.attr in ("shuffle", "choice"):
            return expr.func.attr
    if parsed is not None and isinstance(expr.func, ast.Name):
        if parsed.from_imports.get(expr.func.id) == "random" and expr.func.id in ("shuffle", "choice"):
            return expr.func.id
    return None


def analyze_ownership(parsed: ParsedModule, env: ModuleTypeEnv) -> dict[str, OwnershipPlan]:
    plans: dict[str, OwnershipPlan] = {
        "__main__": _analyze_scope_stmts(parsed.main_body, env.module_scope, parsed)
    }
    for fn in parsed.functions:
        plans[fn.name] = _analyze_scope_stmts(fn.body, env.func_scopes[fn.name], parsed)
    for cls in parsed.classes:
        for item in cls.body:
            if isinstance(item, ast.FunctionDef):
                key = f"{cls.name}.{item.name}"
                scope = env.class_method_scopes[(cls.name, item.name)]
                plans[key] = _analyze_scope_stmts(item.body, scope, parsed)
    # Inherited methods (e.g. ExtendedAnalyzer.report from Analyzer.report):
    # env holds a specialized scope, but no plan was created above.
    from pyrust.inference import _find_method_node_in

    class_nodes = {c.name: c for c in parsed.classes}
    for (cls_name, mname), scope in list(env.class_method_scopes.items()):
        key = f"{cls_name}.{mname}"
        if key in plans:
            continue
        found = _find_method_node_in(class_nodes, cls_name, mname)
        if found is None:
            continue
        _, base_item = found
        plans[key] = _analyze_scope_stmts(base_item.body, scope, parsed)
    return plans
