"""Definite-assignment check for native code.

Native code declares every Python local up front (with a default value), so a
read of a variable that Python would find *unbound* (UnboundLocalError /
NameError) would silently see that default instead. This pass proves that
every read of a local happens after an assignment on every path, and refuses
the program otherwise (it then runs on CPython, which raises properly).

At module level it also enforces definition order: calling a function (or
reading a constant) before the statement that defines it has run raises
NameError in Python.
"""

from __future__ import annotations

import ast

from pyrust.errors import PyRustError

_TERMINATED = None  # state after return/raise/break/continue: unreachable


def _assigned_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for e in target.elts:
            out += _assigned_names(e.value if isinstance(e, ast.Starred) else e)
        return out
    return []


def scope_locals(stmts: list[ast.stmt]) -> set[str]:
    """Names bound anywhere in this scope (not inside nested scopes)."""
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                names.add(node.name)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # Comprehension targets are local to the comprehension, but a
            # walrus inside one binds in the enclosing scope.
            for sub in ast.walk(node):
                if isinstance(sub, ast.NamedExpr):
                    names.add(sub.target.id)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        if isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for s in stmts:
        visit(s)
    return names


class _Checker:
    def __init__(self, filename: str, locals_: set[str], on_call=None) -> None:
        self.filename = filename
        self.locals = locals_
        self.on_call = on_call  # (call_node, state) -> None, module level only
        self.break_states: list[list] = []

    def fail(self, node: ast.AST, name: str) -> None:
        raise PyRustError(
            f"'{name}' may be used before it is assigned on some path (Python would raise "
            "NameError/UnboundLocalError there); this is not supported natively",
            lineno=getattr(node, "lineno", None),
            col=getattr(node, "col_offset", None),
            filename=self.filename,
        )

    # -- expressions ------------------------------------------------------
    def expr(self, node: ast.AST | None, state: set[str]) -> None:
        if node is None:
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            inner = set(state)
            for gen in node.generators:
                self.expr(gen.iter, inner)
                inner |= set(_assigned_names(gen.target))
                for cond in gen.ifs:
                    self.expr(cond, inner)
            if isinstance(node, ast.DictComp):
                self.expr(node.key, inner)
                self.expr(node.value, inner)
            else:
                self.expr(node.elt, inner)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.NamedExpr):
            self.expr(node.value, state)
            state.add(node.target.id)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in self.locals and node.id not in state:
                self.fail(node, node.id)
            return
        if isinstance(node, (ast.BoolOp,)):
            # Only the first operand is always evaluated; walrus in later
            # operands does not definitely assign.
            self.expr(node.values[0], state)
            for v in node.values[1:]:
                self.expr(v, set(state))
            return
        if isinstance(node, ast.IfExp):
            self.expr(node.test, state)
            a, b = set(state), set(state)
            self.expr(node.body, a)
            self.expr(node.orelse, b)
            state |= a & b
            return
        if isinstance(node, ast.Call) and self.on_call is not None:
            self.on_call(node, state)
        for child in ast.iter_child_nodes(node):
            self.expr(child, state)

    def target(self, node: ast.expr, state: set[str]) -> None:
        """Evaluate the reads inside an assignment target, then bind names."""
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            self.expr(node.value, state)
            if isinstance(node, ast.Subscript):
                self.expr(node.slice, state)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for e in node.elts:
                self.target(e.value if isinstance(e, ast.Starred) else e, state)
        elif isinstance(node, ast.Name):
            state.add(node.id)

    # -- statements -------------------------------------------------------
    def block(self, stmts: list[ast.stmt], state):
        for s in stmts:
            if state is _TERMINATED:
                return _TERMINATED
            state = self.stmt(s, state)
        return state

    @staticmethod
    def merge(a, b):
        if a is _TERMINATED:
            return b
        if b is _TERMINATED:
            return a
        return a & b

    def stmt(self, s: ast.stmt, state: set[str]):
        if isinstance(s, ast.Assign):
            self.expr(s.value, state)
            for t in s.targets:
                self.target(t, state)
            return state
        if isinstance(s, ast.AnnAssign):
            if s.value is not None:
                self.expr(s.value, state)
                self.target(s.target, state)
            return state
        if isinstance(s, ast.AugAssign):
            if isinstance(s.target, ast.Name):
                self.expr(ast.copy_location(ast.Name(id=s.target.id, ctx=ast.Load()), s.target), state)
            self.expr(s.value, state)
            self.target(s.target, state)
            return state
        if isinstance(s, (ast.Expr,)):
            self.expr(s.value, state)
            return state
        if isinstance(s, ast.Return):
            self.expr(s.value, state)
            return _TERMINATED
        if isinstance(s, ast.Raise):
            self.expr(s.exc, state)
            return _TERMINATED
        if isinstance(s, ast.Assert):
            self.expr(s.test, state)
            self.expr(s.msg, set(state))
            return state
        if isinstance(s, ast.Delete):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    if t.id in self.locals and t.id not in state:
                        self.fail(t, t.id)
                    state.discard(t.id)
                else:
                    self.target(t, set(state))
            return state
        if isinstance(s, ast.If):
            self.expr(s.test, state)
            a = self.block(s.body, set(state))
            b = self.block(s.orelse, set(state))
            return self.merge(a, b)
        if isinstance(s, ast.While):
            self.expr(s.test, state)
            infinite = isinstance(s.test, ast.Constant) and bool(s.test.value)
            self.break_states.append([])
            self.block(s.body, set(state))
            breaks = self.break_states.pop()
            if infinite:
                # Only reachable through `break`.
                out = _TERMINATED
                for b in breaks:
                    out = self.merge(out, b)
                return out
            after = self.block(s.orelse, set(state))
            for b in breaks:
                after = self.merge(after, b)
            return after
        if isinstance(s, ast.For):
            self.expr(s.iter, state)
            body_state = set(state)
            self.target(s.target, body_state)
            self.break_states.append([])
            self.block(s.body, body_state)
            breaks = self.break_states.pop()
            # The body may run zero times: nothing it binds is definite.
            after = self.block(s.orelse, set(state))
            for b in breaks:
                after = self.merge(after, b)
            return after
        if isinstance(s, (ast.Break,)):
            if self.break_states:
                self.break_states[-1].append(set(state))
            return _TERMINATED
        if isinstance(s, ast.Continue):
            return _TERMINATED
        if isinstance(s, ast.With):
            for item in s.items:
                self.expr(item.context_expr, state)
                if item.optional_vars is not None:
                    self.target(item.optional_vars, state)
            return self.block(s.body, state)
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            state.add(s.name)
            return state
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            for alias in s.names:
                state.add((alias.asname or alias.name).split(".")[0])
            return state
        if isinstance(s, (ast.Pass, ast.Global, ast.Nonlocal)):
            return state
        # Other statements are rejected by codegen anyway; evaluate reads.
        for child in ast.iter_child_nodes(s):
            if isinstance(child, ast.expr):
                self.expr(child, state)
        return state


def check_function(fn: ast.FunctionDef, filename: str) -> None:
    params = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    if fn.args.vararg:
        params.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        params.add(fn.args.kwarg.arg)
    locals_ = scope_locals(fn.body) | params
    _Checker(filename, locals_).block(fn.body, set(params))


def _global_reads(fn: ast.FunctionDef) -> set[str]:
    params = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    local = scope_locals(fn.body) | params
    return {
        n.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in local
    }


def check_module(tree: ast.Module, filename: str) -> None:
    """Module-level order: reads, and calls to functions whose bodies read
    module-level names, must happen after those names are bound."""
    module_names = scope_locals(tree.body)
    funcs: dict[str, ast.FunctionDef] = {}
    methods_reads: dict[str, set[str]] = {}
    for s in ast.walk(tree):
        if isinstance(s, ast.FunctionDef) and s in tree.body:
            funcs[s.name] = s
    for s in tree.body:
        if isinstance(s, ast.ClassDef):
            reads: set[str] = set()
            for item in s.body:
                if isinstance(item, ast.FunctionDef):
                    reads |= _global_reads(item)
            methods_reads[s.name] = reads

    reach_cache: dict[str, set[str]] = {}

    def needed(name: str) -> set[str]:
        """Module-level names a call to `name` (transitively) reads."""
        if name in reach_cache:
            return reach_cache[name]
        reach_cache[name] = set()
        direct = _global_reads(funcs[name]) if name in funcs else methods_reads.get(name, set())
        out = {n for n in direct if n in module_names}
        for n in list(out):
            if n in funcs or n in methods_reads:
                out |= needed(n)
        reach_cache[name] = out
        return out

    checker: _Checker

    def on_call(node: ast.Call, state: set[str]) -> None:
        if isinstance(node.func, ast.Name) and (node.func.id in funcs or node.func.id in methods_reads):
            for n in needed(node.func.id):
                if n not in state:
                    checker.fail(node, n)

    checker = _Checker(filename, module_names, on_call=on_call)
    checker.block(tree.body, set())
