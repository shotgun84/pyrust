"""Alias analysis: when may native value semantics stand in for Python's
reference semantics?

Generated Rust stores lists, dicts, sets and objects *by value*: assigning,
passing or storing one copies it. Python shares the object instead. The two
only differ when an object reachable through two names (or two slots) is
modified through one and later observed through the other. This module
proves the absence of such situations, or refuses the program (which then
runs on CPython).

It is a forward abstract interpretation over each scope:

* abstract objects are allocation sites (split into the most recent object
  and a summary of older ones, so `row = []` in a loop is not confused with
  the rows stored earlier), function parameters, and parameter contents;
* the state maps names to the objects they may point to, and objects to the
  objects they may contain; an object stored into containers more than once
  is marked as multiply referenced;
* precise liveness (computed backwards) decides which names may still
  observe an object after a modification.

A modification is accepted only if no *other* live name can reach the
modified object and the object is not multiply referenced. Function
summaries (which parameters a function modifies, which flow into its result
or into `self`) make calls precise; modified parameters are passed as
`&mut` by the code generator, and loops whose target is modified in place
iterate by reference (`iter_mut`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from pyrust.errors import PyRustError
from pyrust.inference import ModuleTypeEnv, ScopeInfo, _find_method_node_in, _positional_params
from pyrust.parser import ParsedModule
from pyrust.types import ANY, UNKNOWN, ClassType, DictType, ListType, PyType, SetType, TupleType

MUTATORS = {
    "append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse",
    "add", "discard", "update", "setdefault", "popitem",
}
COPYING_BUILTINS = {"list", "sorted", "reversed", "set", "dict", "tuple", "enumerate", "zip", "frozenset"}
ELEMENT_BUILTINS = {"max", "min"}
NATIVE_STDLIB = {"math", "random", "json", "sys", "os", "time", "statistics", "platform", "gc", "typing", "dataclasses"}


def is_mutable(t: PyType | None) -> bool:
    if isinstance(t, (ListType, DictType, SetType)):
        return True
    if isinstance(t, ClassType):
        return t.name != "PyFile"  # files are shared handles natively too
    if isinstance(t, TupleType):
        return any(is_mutable(e) for e in t.elements)
    return False


def has_mutable_parts(t: PyType | None) -> bool:
    """Values of this type can contain (references to) mutable objects."""
    if isinstance(t, (ListType, SetType)):
        return is_mutable(t.elem)
    if isinstance(t, DictType):
        return is_mutable(t.val)
    if isinstance(t, TupleType):
        return any(is_mutable(e) for e in t.elements)
    if isinstance(t, ClassType):
        return t.name != "PyFile"
    return False


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


def _uses(node: ast.AST | None) -> set[str]:
    """Names read by `node` (comprehension variables are local to it)."""
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id} if isinstance(node.ctx, ast.Load) else set()
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        out: set[str] = set()
        bound: set[str] = set()
        for gen in node.generators:
            out |= _uses(gen.iter) - bound
            bound |= _target_defs(gen.target)
            for c in gen.ifs:
                out |= _uses(c) - bound
        parts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        for part in parts:
            out |= _uses(part) - bound
        return out
    out = set()
    for child in ast.iter_child_nodes(node):
        out |= _uses(child)
    return out


def _target_uses(t: ast.expr) -> set[str]:
    """Names read by an assignment target (bases of subscripts/attributes)."""
    if isinstance(t, (ast.Subscript, ast.Attribute)):
        out = _uses(t.value)
        root = t
        while isinstance(root, (ast.Subscript, ast.Attribute)):
            root = root.value
        if isinstance(root, ast.Name):
            out.add(root.id)
        if isinstance(t, ast.Subscript):
            out |= _uses(t.slice)
        return out
    if isinstance(t, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for e in t.elts:
            out |= _target_uses(e)
        return out
    return set()


def _target_defs(t: ast.expr) -> set[str]:
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for e in t.elts:
            out |= _target_defs(e.value if isinstance(e, ast.Starred) else e)
        return out
    return set()


class Liveness:
    """live_after[id(stmt)] for every statement (backward dataflow)."""

    def __init__(self) -> None:
        self.after: dict[int, frozenset[str]] = {}
        self._loops: list[tuple[frozenset[str], frozenset[str]]] = []  # (break target, continue target)

    def block(self, stmts: list[ast.stmt], live: frozenset[str]) -> frozenset[str]:
        for s in reversed(stmts):
            live = self.stmt(s, live)
        return live

    def stmt(self, s: ast.stmt, live: frozenset[str]) -> frozenset[str]:
        prev = self.after.get(id(s))
        self.after[id(s)] = live if prev is None else (prev | live)
        if isinstance(s, ast.Assign):
            defs: set[str] = set()
            uses = _uses(s.value)
            for t in s.targets:
                defs |= _target_defs(t)
                uses |= _target_uses(t)
            return frozenset((live - defs) | uses)
        if isinstance(s, ast.AnnAssign):
            if s.value is None:
                return live
            return frozenset((live - _target_defs(s.target)) | _uses(s.value) | _target_uses(s.target))
        if isinstance(s, ast.AugAssign):
            uses = _uses(s.value) | _target_uses(s.target) | _target_defs(s.target)
            return frozenset(live | uses)
        if isinstance(s, (ast.Expr, ast.Assert)):
            return frozenset(live | _uses(s))
        if isinstance(s, ast.Return):
            return frozenset(_uses(s.value))
        if isinstance(s, ast.Raise):
            return frozenset(_uses(s.exc))
        if isinstance(s, ast.Delete):
            uses: set[str] = set()
            for t in s.targets:
                uses |= _target_uses(t)
            return frozenset(live | uses)
        if isinstance(s, ast.If):
            a = self.block(s.body, live)
            b = self.block(s.orelse, live)
            return frozenset(a | b | _uses(s.test))
        if isinstance(s, (ast.While, ast.For)):
            head_uses = _uses(s.test) if isinstance(s, ast.While) else _uses(s.iter)
            after_loop = self.block(s.orelse, live) if s.orelse else live
            # Fixpoint: the loop head is live-in of the body and of itself.
            head = frozenset(after_loop | head_uses)
            for _ in range(50):
                self._loops.append((live, head))
                body_live = head
                if isinstance(s, ast.For):
                    body_in = self.block(s.body, body_live)
                    new_head = frozenset((body_in - _target_defs(s.target)) | after_loop | head_uses)
                else:
                    body_in = self.block(s.body, body_live)
                    new_head = frozenset(body_in | after_loop | head_uses)
                self._loops.pop()
                if new_head == head:
                    break
                head = new_head
            return head
        if isinstance(s, ast.Break):
            return self._loops[-1][0] if self._loops else live
        if isinstance(s, ast.Continue):
            return self._loops[-1][1] if self._loops else live
        if isinstance(s, ast.With):
            body_in = self.block(s.body, live)
            uses: set[str] = set()
            defs: set[str] = set()
            for item in s.items:
                uses |= _uses(item.context_expr)
                if item.optional_vars is not None:
                    defs |= _target_defs(item.optional_vars)
            return frozenset((body_in - defs) | uses)
        return frozenset(live | _uses(s))


# ---------------------------------------------------------------------------
# Abstract state
# ---------------------------------------------------------------------------

Obj = tuple  # ("site", id, "new"|"old") | ("param", name) | ("pchild", name)


@dataclass
class State:
    env: dict[str, frozenset] = field(default_factory=dict)
    heap: dict[Obj, frozenset] = field(default_factory=dict)
    multi: frozenset = frozenset()

    def copy(self) -> "State":
        return State(dict(self.env), dict(self.heap), self.multi)

    def join(self, other: "State") -> "State":
        env = dict(self.env)
        for k, v in other.env.items():
            env[k] = env.get(k, frozenset()) | v
        heap = dict(self.heap)
        for k, v in other.heap.items():
            heap[k] = heap.get(k, frozenset()) | v
        return State(env, heap, self.multi | other.multi)

    def same(self, other: "State") -> bool:
        return self.env == other.env and self.heap == other.heap and self.multi == other.multi

    def children(self, objs) -> frozenset:
        out: set = set()
        for o in objs:
            out |= self.heap.get(o, frozenset())
        return frozenset(out)

    def reach(self, objs) -> frozenset:
        seen: set = set(objs)
        stack = list(objs)
        while stack:
            o = stack.pop()
            for c in self.heap.get(o, ()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return frozenset(seen)

    def referenced(self, o: Obj) -> bool:
        return any(o in kids for kids in self.heap.values())


@dataclass
class Summary:
    writes: set[str] = field(default_factory=set)  # params modified in place
    ret_from: set[str] = field(default_factory=set)  # params whose objects may be returned
    stores: set[tuple[str, str]] = field(default_factory=set)  # (src param, dst param): src stored into dst
    fresh_into: set[str] = field(default_factory=set)  # params that receive objects created by the callee
    fresh_multi: bool = False  # a created object was stored in more than one place
    ret_shared: set[str] = field(default_factory=set)  # params also holding (part of) the returned object
    rebinds: set[str] = field(default_factory=set)

    def key(self):
        return (frozenset(self.writes), frozenset(self.ret_from), frozenset(self.stores),
                frozenset(self.fresh_into), self.fresh_multi, frozenset(self.ret_shared))


@dataclass
class AliasInfo:
    ref_params: dict[str, set[str]] = field(default_factory=dict)  # func key -> params passed as &mut
    byref_loops: set[int] = field(default_factory=set)  # id(ast.For) iterating by reference


class _Conflict(Exception):
    pass


class _ScopeAnalyzer:
    def __init__(self, owner: "AliasAnalyzer", key: str, scope: ScopeInfo, fn: ast.FunctionDef | None,
                 params: list[str], filename: str) -> None:
        self.owner = owner
        self.key = key
        self.scope = scope
        self.fn = fn
        self.params = params
        self.filename = filename
        self.summary = Summary()
        self.live = Liveness()
        self.byref: dict[int, tuple[str, str]] = {}  # loop id -> (loop var, iterated root)
        self.byref_used: set[int] = set()
        self.families: list[tuple[str, str]] = []  # active by-ref loops (var, root)
        self.cur_stmt: ast.stmt | None = None
        self.break_states: list[list[State]] = []
        self.cont_states: list[list[State]] = []

    # ---- helpers ------------------------------------------------------
    def t(self, node: ast.AST) -> PyType:
        return self.scope.expr_types.get(id(node), UNKNOWN)

    def fail(self, node: ast.AST | None, msg: str) -> None:
        raise PyRustError(
            msg + " (Python shares the object; native code would modify a copy)",
            lineno=getattr(node, "lineno", None),
            col=getattr(node, "col_offset", None),
            filename=self.filename,
        )

    def alloc(self, st: State, node) -> Obj:
        site = node if isinstance(node, tuple) else id(node)
        new = ("site", site, "new")
        old = ("site", site, "old")
        if any(new in v for v in st.env.values()) or new in st.heap or st.referenced(new):
            # The previous object from this site becomes part of the "old" summary.
            def ren(objs):
                return frozenset(old if o == new else o for o in objs)

            st.env = {k: ren(v) for k, v in st.env.items()}
            heap: dict = {}
            for k, v in st.heap.items():
                k2 = old if k == new else k
                heap[k2] = heap.get(k2, frozenset()) | ren(v)
            st.heap = heap
            if new in st.multi:
                st.multi = (st.multi - {new}) | {old}
        st.heap[new] = frozenset()
        return new

    def store(self, st: State, containers, objs, node: ast.AST, existing: bool = True) -> None:
        """Make `containers` hold `objs`."""
        objs = frozenset(objs)
        if not objs or not containers:
            return
        for o in objs:
            if existing and st.referenced(o):
                st.multi = st.multi | {o}
        for c in containers:
            st.heap[c] = st.heap.get(c, frozenset()) | objs
        # params' contents stored into another param: summary
        for o in objs:
            for c in containers:
                src = self._param_of(o)
                dst = self._param_of(c)
                if src and dst and src != dst:
                    self.summary.stores.add((src, dst))
                elif dst and src is None:
                    self.summary.fresh_into.add(dst)
                    if o in st.multi:
                        self.summary.fresh_multi = True

    @staticmethod
    def _param_of(o: Obj) -> str | None:
        if o[0] in ("param", "pchild"):
            return o[1]
        return None

    def live_now(self) -> frozenset[str]:
        s = self.cur_stmt
        if s is None:
            return frozenset()
        uses = _uses(s)
        if isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = s.targets if isinstance(s, (ast.Assign, ast.Delete)) else [s.target]
            for t in targets:
                uses |= _target_uses(t)
        return self.live.after.get(id(s), frozenset()) | uses

    def write(self, st: State, objs, writer: str | None, node: ast.AST, deep: bool = False) -> None:
        """An in-place modification of `objs` performed through `writer`."""
        objs = frozenset(objs)
        if deep:
            objs = st.reach(objs)
        if not objs:
            return
        for o in objs:
            p = self._param_of(o)
            if p is not None:
                self.summary.writes.add(p)
        bad = objs & st.multi
        if bad:
            self.fail(node, "modifying an object that is stored in more than one place is not supported natively")
        family = {writer} if writer else set()
        for var, root in self.families:
            if writer in (var, root):
                family |= {var, root}
        for o in objs:
            p = self._param_of(o)
            if p is not None and p not in family:
                # The caller sees this object through the parameter; only
                # writes through the parameter itself (passed as &mut) reach it.
                self.fail(
                    node,
                    f"modifying parameter '{p}' (or part of it) through another name "
                    f"('{writer or 'an expression'}') is not supported natively",
                )
        for name in self.live_now():
            if name in family or name not in st.env:
                continue
            if st.reach(st.env[name]) & objs:
                self.fail(
                    node,
                    f"'{name}' and '{writer or 'this expression'}' refer to the same object here, and it is "
                    "modified in place; this is not supported natively",
                )

    # ---- expressions ----------------------------------------------------
    def root(self, node: ast.expr) -> str | None:
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def ev(self, node: ast.AST | None, st: State) -> frozenset:
        """Objects the value of `node` may be (and perform effects)."""
        if node is None:
            return frozenset()
        mut = is_mutable(self.t(node)) or isinstance(node, (ast.Name, ast.Subscript, ast.Attribute))
        if isinstance(node, ast.Name):
            return st.env.get(node.id, frozenset())
        if isinstance(node, ast.Constant):
            return frozenset()
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            kids: set = set()
            dup: set = set()
            for e in node.elts:
                eo = self.ev(e.value if isinstance(e, ast.Starred) else e, st)
                dup |= kids & eo  # the same object in two elements: [row, row]
                kids |= eo
            if not (is_mutable(self.t(node)) or kids):
                return frozenset()
            o = self.alloc(st, node)
            self.store(st, [o], kids, node)
            if dup:
                st.multi = st.multi | dup
            return frozenset({o})
        if isinstance(node, ast.Dict):
            kids = set()
            for k, v in zip(node.keys, node.values):
                self.ev(k, st)
                vo = self.ev(v, st)
                kids |= st.children(vo) if k is None else vo
            o = self.alloc(st, node)
            self.store(st, [o], kids, node)
            return frozenset({o})
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            inner = st  # comprehension effects apply to the state
            saved = {}
            elt_objs: set = set()
            for _ in range(2):  # the element may come from the previous iteration
                for gen in node.generators:
                    it = self.ev(gen.iter, inner)
                    for name in _target_defs(gen.target):
                        saved.setdefault(name, inner.env.get(name))
                        inner.env[name] = inner.children(it) if has_mutable_parts(self.t(gen.iter)) else frozenset()
                    for c in gen.ifs:
                        self.ev(c, inner)
                if isinstance(node, ast.DictComp):
                    self.ev(node.key, inner)
                    elt_objs |= self.ev(node.value, inner)
                else:
                    elt_objs |= self.ev(node.elt, inner)
            for name, old in saved.items():
                if old is None:
                    inner.env.pop(name, None)
                else:
                    inner.env[name] = old
            o = self.alloc(st, node)
            self.store(st, [o], elt_objs, node)
            # An element that is not created afresh by the element expression
            # is the same object on every iteration (or shared with its source).
            elt_node = node.value if isinstance(node, ast.DictComp) else node.elt
            fresh_sites = {id(n) for n in ast.walk(elt_node)}
            shared = {e for e in elt_objs if not (e[0] == "site" and e[1] in fresh_sites)}
            if shared:
                st.multi = st.multi | frozenset(shared)
            return frozenset({o})
        if isinstance(node, ast.Subscript):
            base = self.ev(node.value, st)
            if isinstance(node.slice, ast.Slice):
                for part in (node.slice.lower, node.slice.upper, node.slice.step):
                    self.ev(part, st)
                if not is_mutable(self.t(node)):
                    return frozenset()
                o = self.alloc(st, node)
                kids = st.children(base)
                self.store(st, [o], kids, node)
                if kids:
                    st.multi = st.multi | kids  # shallow copy: elements now in two lists
                return frozenset({o})
            self.ev(node.slice, st)
            return st.children(base) if mut else frozenset()
        if isinstance(node, ast.Attribute):
            base = self.ev(node.value, st)
            return st.children(base) if mut else frozenset()
        if isinstance(node, ast.BinOp):
            lo = self.ev(node.left, st)
            ro = self.ev(node.right, st)
            if not is_mutable(self.t(node)):
                return frozenset()
            o = self.alloc(st, node)
            if isinstance(node.op, ast.Mult):
                kids = st.children(lo) | st.children(ro)
                self.store(st, [o], kids, node)
                if kids:
                    st.multi = st.multi | kids  # [x] * n: x referenced n times
            else:
                kids = st.children(lo) | st.children(ro)
                self.store(st, [o], kids, node)
                if kids and isinstance(node.op, ast.Add):
                    st.multi = st.multi | kids
            return frozenset({o})
        if isinstance(node, ast.BoolOp):
            out: set = set()
            for v in node.values:
                out |= self.ev(v, st)
            return frozenset(out)
        if isinstance(node, ast.IfExp):
            self.ev(node.test, st)
            return self.ev(node.body, st) | self.ev(node.orelse, st)
        if isinstance(node, ast.NamedExpr):
            v = self.ev(node.value, st)
            st.env[node.target.id] = v
            return v
        if isinstance(node, ast.Call):
            return self.call(node, st)
        if isinstance(node, (ast.Compare, ast.UnaryOp, ast.JoinedStr, ast.FormattedValue)):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.ev(child, st)
            return frozenset()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.ev(child, st)
        return frozenset()

    def call(self, node: ast.Call, st: State) -> frozenset:
        func = node.func
        res_mut = is_mutable(self.t(node))
        # Method calls on builtin containers.
        if isinstance(func, ast.Attribute):
            recv_t = self.t(func.value)
            attr = func.attr
            if isinstance(func.value, ast.Name) and func.value.id in self.owner.parsed.imports:
                if func.value.id not in NATIVE_STDLIB:
                    self._reject_foreign_mutable_args(node)
                args = [self.ev(a, st) for a in node.args]
                for kw in node.keywords:
                    self.ev(kw.value, st)
                if func.value.id == "random" and attr == "shuffle" and node.args:
                    self.write(st, args[0], self.root(node.args[0]), node)
                if func.value.id == "random" and attr == "choice" and node.args:
                    return st.children(args[0])
                return frozenset()
            if isinstance(recv_t, (ListType, DictType, SetType)):
                recv = self.ev(func.value, st)
                args = [self.ev(a, st) for a in node.args]
                for kw in node.keywords:
                    self.ev(kw.value, st)
                writer = self.root(func.value)
                if attr in ("append", "add"):
                    self.write(st, recv, writer, node)
                    if args:
                        self.store(st, recv, args[0], node)
                    return frozenset()
                if attr == "insert":
                    self.write(st, recv, writer, node)
                    if len(args) > 1:
                        self.store(st, recv, args[1], node)
                    return frozenset()
                if attr == "extend" or attr == "update":
                    self.write(st, recv, writer, node)
                    if args:
                        kids = st.children(args[0])
                        self.store(st, recv, kids, node)
                        if kids:
                            st.multi = st.multi | kids
                    return frozenset()
                if attr == "setdefault":
                    self.write(st, recv, writer, node)
                    if len(args) > 1:
                        self.store(st, recv, args[1], node)
                    return st.children(recv)
                if attr in ("pop", "popitem"):
                    self.write(st, recv, writer, node)
                    return st.children(recv)
                if attr in ("remove", "clear", "sort", "reverse", "discard"):
                    self.write(st, recv, writer, node)
                    return frozenset()
                if attr == "copy":
                    o = self.alloc(st, node)
                    kids = st.children(recv)
                    self.store(st, [o], kids, node)
                    if kids:
                        st.multi = st.multi | kids
                    return frozenset({o})
                if attr in ("get", "values", "items", "keys"):
                    return st.children(recv)
                return frozenset()
            if isinstance(recv_t, ClassType) and recv_t.name != "PyFile":
                recv = self.ev(func.value, st)
                found = self.owner.method_node(recv_t.name, attr)
                if found is None:
                    for a in node.args:
                        self.ev(a, st)
                    return frozenset()
                def_cls, m_node = found
                return self.apply_summary(
                    node, st, f"{def_cls}.{attr}", m_node, node.args, node.keywords,
                    self_objs=recv, self_root=self.root(func.value), skip_self=True,
                )
            # str methods etc.
            self.ev(func.value, st)
            for a in node.args:
                self.ev(a, st)
            for kw in node.keywords:
                self.ev(kw.value, st)
            return frozenset()
        if isinstance(func, ast.Name):
            fname = func.id
            src_mod = self.owner.parsed.from_imports.get(fname)
            if src_mod is not None and src_mod not in NATIVE_STDLIB:
                self._reject_foreign_mutable_args(node)
            if fname in self.owner.funcs:
                return self.apply_summary(node, st, fname, self.owner.funcs[fname], node.args, node.keywords)
            if fname in self.owner.classes:
                obj = self.alloc(st, node)
                found = self.owner.method_node(fname, "__init__")
                if found is not None:
                    def_cls, init = found
                    self.apply_summary(
                        node, st, f"{def_cls}.__init__", init, node.args, node.keywords,
                        self_objs=frozenset({obj}), self_root=None, skip_self=True, alloc_result=False,
                    )
                else:
                    # dataclass synthetic __init__: stores every argument.
                    for a in list(node.args) + [kw.value for kw in node.keywords]:
                        self.store(st, [obj], self.ev(a, st), node)
                return frozenset({obj})
            args = [self.ev(a, st) for a in node.args]
            for kw in node.keywords:
                self.ev(kw.value, st)
            if fname in COPYING_BUILTINS and res_mut:
                o = self.alloc(st, node)
                kids: set = set()
                for a in args:
                    kids |= st.children(a)
                self.store(st, [o], kids, node)
                if kids:
                    st.multi = st.multi | kids
                return frozenset({o})
            if fname in ELEMENT_BUILTINS and args:
                if len(args) == 1:
                    return st.children(args[0])
                out: set = set()
                for a in args:
                    out |= a
                return frozenset(out)
            return frozenset()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.ev(child, st)
        return frozenset()

    def _reject_foreign_mutable_args(self, node: ast.Call) -> None:
        for a in list(node.args) + [kw.value for kw in node.keywords]:
            if is_mutable(self.t(a)):
                self.fail(node, "passing a list/dict/object to a function from another module is not supported natively yet")

    def apply_summary(self, node, st, key, fn_node, args, keywords, self_objs=None, self_root=None,
                      skip_self=False, alloc_result=True) -> frozenset:
        params = [a.arg for a in _positional_params(fn_node)]
        if skip_self and params and params[0] == "self":
            params = params[1:]
        bound: dict[str, frozenset] = {}
        roots: dict[str, str | None] = {}
        for p, a in zip(params, args):
            bound[p] = self.ev(a, st)
            roots[p] = self.root(a)
        for kw in keywords:
            v = self.ev(kw.value, st)
            if kw.arg:
                bound[kw.arg] = v
                roots[kw.arg] = self.root(kw.value)
        summ = self.owner.summary(key)
        # Mutable default values are shared across calls in Python.
        defaults = fn_node.args.defaults
        for p, d in zip(params[len(params) - len(defaults):], defaults):
            if p not in bound and (is_mutable_literal(d)) and (p in summ.writes or p in summ.ret_from
                                                               or any(p == s for s, _ in summ.stores)):
                self.fail(node, f"the mutable default value of parameter '{p}' is shared between calls")
        if self_objs is not None:
            bound["self"] = self_objs
            roots["self"] = self_root
        for p in summ.writes:
            if p in bound:
                self.write(st, bound[p], roots.get(p), node, deep=True)
        for src, dst in summ.stores:
            if src in bound and dst in bound:
                self.store(st, st.reach(bound[dst]) if dst != "self" else bound[dst], st.reach(bound[src]), node)
        for dst in summ.fresh_into:
            if dst in bound and bound[dst]:
                child = self.alloc(st, ("fresh", id(node), dst))
                self.store(st, bound[dst], {child}, node, existing=False)
                if summ.fresh_multi:
                    st.multi = st.multi | {child}
        out: set = set()
        for p in summ.ret_from:
            if p in bound:
                # The result is (or contains) the caller's object: shared.
                shared = st.reach(bound[p])
                st.multi = st.multi | shared
                out |= shared
        if alloc_result and is_mutable(self.t(node)):
            o = self.alloc(st, node)
            out.add(o)
            # A new container may hold the caller's objects (`return [x]`).
            held = set()
            for p in summ.ret_from:
                if p in bound:
                    held |= st.reach(bound[p])
            if held:
                st.heap[o] = st.heap.get(o, frozenset()) | frozenset(held)
            if summ.ret_shared:
                # The callee also stored the returned object into an argument
                # (e.g. a factory method appending to self): shared.
                st.multi = st.multi | {o}
                for p in summ.ret_shared:
                    if p in bound:
                        st.multi = st.multi | st.children(bound[p])
        return frozenset(out)

    # ---- statements -----------------------------------------------------
    def block(self, stmts: list[ast.stmt], st: State):
        for s in stmts:
            if st is None:
                return None
            st = self.stmt(s, st)
        return st

    def assign_to(self, target: ast.expr, objs: frozenset, st: State, node: ast.AST) -> None:
        if isinstance(target, ast.Name):
            vt = self.scope.var_types.get(target.id, UNKNOWN) if self.scope else UNKNOWN
            st.env[target.id] = objs if (is_mutable(vt) or vt in (UNKNOWN, ANY)) else frozenset()
            if self.fn is not None and target.id in self.params:
                self.summary.rebinds.add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            kids = st.children(objs) | objs
            for e in target.elts:
                self.assign_to(e.value if isinstance(e, ast.Starred) else e, kids, st, node)
            return
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            container = self.ev(target.value, st)
            if isinstance(target, ast.Subscript):
                self.ev(target.slice, st)
            self.write(st, container, self.root(target), node)
            self.store(st, container, objs, node)

    def stmt(self, s: ast.stmt, st: State):
        prev, self.cur_stmt = self.cur_stmt, s
        try:
            return self._stmt(s, st)
        finally:
            self.cur_stmt = prev

    def _stmt(self, s: ast.stmt, st: State):
        if isinstance(s, ast.Assign):
            objs = self.ev(s.value, st)
            for t in s.targets:
                self.assign_to(t, objs, st, s)
            return st
        if isinstance(s, ast.AnnAssign):
            if s.value is not None:
                self.assign_to(s.target, self.ev(s.value, st), st, s)
            return st
        if isinstance(s, ast.AugAssign):
            v = self.ev(s.value, st)
            tt = self.t(s.target)
            if isinstance(s.target, ast.Name):
                if is_mutable(tt):
                    tgt = st.env.get(s.target.id, frozenset())
                    self.write(st, tgt, s.target.id, s)
                    kids = st.children(v)
                    self.store(st, tgt, kids, s)
                    if kids:
                        st.multi = st.multi | kids
            else:
                container = self.ev(s.target.value, st)
                if isinstance(s.target, ast.Subscript):
                    self.ev(s.target.slice, st)
                self.write(st, container, self.root(s.target), s)
                if is_mutable(tt):
                    self.write(st, st.children(container), self.root(s.target), s)
            return st
        if isinstance(s, ast.Expr):
            self.ev(s.value, st)
            return st
        if isinstance(s, ast.Return):
            objs = self.ev(s.value, st)
            if s.value is not None:
                reach = st.reach(objs)
                for o in reach:
                    p = self._param_of(o)
                    if p is not None:
                        self.summary.ret_from.add(p)
                for p in self.params:
                    held = st.reach(st.env.get(p, frozenset())) - {("param", p), ("pchild", p)}
                    if held & reach:
                        self.summary.ret_shared.add(p)
            return None
        if isinstance(s, ast.Raise):
            self.ev(s.exc, st)
            return None
        if isinstance(s, ast.Assert):
            self.ev(s.test, st)
            self.ev(s.msg, st)
            return st
        if isinstance(s, ast.Delete):
            for t in s.targets:
                if isinstance(t, (ast.Subscript, ast.Attribute)):
                    container = self.ev(t.value, st)
                    if isinstance(t, ast.Subscript):
                        self.ev(t.slice, st)
                    self.write(st, container, self.root(t), s)
                elif isinstance(t, ast.Name):
                    st.env.pop(t.id, None)
            return st
        if isinstance(s, ast.If):
            self.ev(s.test, st)
            a = self.block(s.body, st.copy())
            b = self.block(s.orelse, st.copy())
            if a is None:
                return b
            if b is None:
                return a
            return a.join(b)
        if isinstance(s, ast.While):
            return self.loop(s, st)
        if isinstance(s, ast.For):
            return self.loop(s, st)
        if isinstance(s, ast.Break):
            if self.break_states:
                self.break_states[-1].append(st.copy())
            return None
        if isinstance(s, ast.Continue):
            if self.cont_states:
                self.cont_states[-1].append(st.copy())
            return None
        if isinstance(s, ast.With):
            for item in s.items:
                self.ev(item.context_expr, st)
                if item.optional_vars is not None:
                    self.assign_to(item.optional_vars, frozenset(), st, s)
            return self.block(s.body, st)
        for child in ast.iter_child_nodes(s):
            if isinstance(child, ast.expr):
                self.ev(child, st)
        return st

    def _byref_candidate(self, s: ast.For) -> tuple[str, str] | None:
        """(loop var, iterated root) when `for v in root` / `enumerate(root)` /
        `root.values()` / `root.items()` could iterate by reference."""
        it = s.iter
        tgt = s.target
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "enumerate" \
                and len(it.args) == 1 and isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2:
            it, tgt = it.args[0], tgt.elts[1]
        elif isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and not it.args and \
                it.func.attr in ("values", "items"):
            if it.func.attr == "items":
                if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2):
                    return None
                tgt = tgt.elts[1]
            it = it.func.value
            if not isinstance(self.t(it), DictType):
                return None
        if not isinstance(tgt, ast.Name):
            return None
        tgt_t = self.scope.var_types.get(tgt.id, UNKNOWN)
        place = it
        while isinstance(place, ast.Attribute):
            place = place.value
        if not isinstance(place, ast.Name):
            return None
        if not (isinstance(self.t(it), (ListType, DictType)) and is_mutable(tgt_t)):
            return None
        return tgt.id, place.id

    def loop(self, s, st: State):
        cand = self._byref_candidate(s) if isinstance(s, ast.For) else None
        src_node = None
        if isinstance(s, ast.While):
            self.ev(s.test, st)
        else:
            src_node = s.iter
            if isinstance(s.iter, ast.Call) and isinstance(s.iter.func, ast.Name) and \
                    s.iter.func.id == "enumerate" and s.iter.args:
                # enumerate() only pairs indices with the elements of its argument.
                src_node = s.iter.args[0]
                for extra in s.iter.args[1:]:
                    self.ev(extra, st)
                it_objs = self.ev(src_node, st)
            else:
                it_objs = self.ev(s.iter, st)
        state = st.copy()
        breaks_all: list[State] = []
        for _ in range(30):
            self.break_states.append([])
            self.cont_states.append([])
            body_st = state.copy()
            if isinstance(s, ast.For):
                elems = body_st.children(it_objs) if has_mutable_parts(self.t(src_node)) else frozenset()
                if isinstance(s.target, ast.Name):
                    body_st.env[s.target.id] = elems
                else:
                    self.assign_to(s.target, elems, body_st, s)
            if cand is not None:
                self.families.append(cand)
            writes_before = set(self.summary.writes)
            try:
                out = self.block(s.body, body_st)
            except PyRustError:
                if cand is None:
                    raise
                out = None
                self.families.pop() if self.families and self.families[-1] == cand else None
                cand = None
                raise
            if cand is not None:
                self.families.pop()
            breaks = self.break_states.pop()
            conts = self.cont_states.pop()
            new = state.copy()
            for x in ([out] if out is not None else []) + conts:
                new = new.join(x)
            breaks_all = breaks
            if new.same(state):
                break
            state = new
        if cand is not None and self._loop_writes_target(s, cand[0]):
            self._check_byref(s, cand)
            self.byref_used.add(id(s))
        after = state
        if isinstance(s, (ast.For, ast.While)) and s.orelse:
            after = self.block(s.orelse, after.copy())
        if isinstance(s, ast.While) and isinstance(s.test, ast.Constant) and s.test.value:
            after = None
        for b in breaks_all:
            after = b if after is None else after.join(b)
        return after

    def _loop_writes_target(self, s: ast.For, var: str) -> bool:
        """Is the loop variable modified in place in the body?"""
        for node in ast.walk(ast.Module(body=s.body, type_ignores=[])):
            if isinstance(node, ast.For) and node is not s:
                inner = node.iter
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "enumerate" \
                        and inner.args:
                    inner = inner.args[0]
                elif isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and \
                        inner.func.attr in ("values", "items"):
                    inner = inner.func.value
                tgt = node.target
                if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2:
                    tgt = tgt.elts[1]
                if self.root(inner) == var and isinstance(tgt, ast.Name) and self._loop_writes_target(node, tgt.id):
                    return True
            targets: list[ast.expr] = []
            if isinstance(node, (ast.Assign, ast.Delete)):
                targets = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                if isinstance(t, (ast.Subscript, ast.Attribute)) and self.root(t) == var:
                    return True
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and self.root(node.func.value) == var:
                    rt = self.t(node.func.value)
                    if node.func.attr in MUTATORS or (
                        isinstance(rt, ClassType) and node.func.attr in self.owner.mut_methods.get(rt.name, set())
                    ):
                        return True
                for i, a in enumerate(node.args):
                    if isinstance(a, ast.Name) and a.id == var:
                        callee = self.owner.callee_key(node, self.scope)
                        if callee:
                            fn = self.owner.fn_node(callee)
                            ps = [p.arg for p in _positional_params(fn)] if fn else []
                            if isinstance(node.func, ast.Attribute) and ps and ps[0] == "self":
                                ps = ps[1:]
                            if i < len(ps) and ps[i] in self.owner.summary(callee).writes:
                                return True
        return False

    def _check_byref(self, s: ast.For, cand: tuple[str, str]) -> None:
        var, root = cand
        for node in ast.walk(ast.Module(body=s.body, type_ignores=[])):
            if isinstance(node, ast.Name) and node.id == root:
                self.fail(node, f"modifying the loop variable '{var}' while also using '{root}' in the loop body "
                                "is not supported natively")
            if isinstance(node, ast.Name) and node.id == var and isinstance(node.ctx, ast.Store):
                self.fail(node, f"reassigning the loop variable '{var}' after modifying it is not supported natively")


def is_mutable_literal(node: ast.expr) -> bool:
    return isinstance(node, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp, ast.Call))


class AliasAnalyzer:
    def __init__(self, parsed: ParsedModule, env: ModuleTypeEnv) -> None:
        from pyrust.ownership import mutating_methods

        self.parsed = parsed
        self.env = env
        self.funcs = {f.name: f for f in parsed.functions}
        self.classes = {c.name: c for c in parsed.classes}
        self.summaries: dict[str, Summary] = {}
        self.mut_methods = mutating_methods(parsed)
        self.info = AliasInfo()

    def method_node(self, cls: str, name: str):
        return _find_method_node_in(self.classes, cls, name)

    def fn_node(self, key: str):
        if "." in key:
            cls, m = key.split(".", 1)
            found = self.method_node(cls, m)
            return found[1] if found else None
        return self.funcs.get(key)

    def callee_key(self, call: ast.Call, scope: ScopeInfo) -> str | None:
        if isinstance(call.func, ast.Name) and call.func.id in self.funcs:
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            rt = scope.expr_types.get(id(call.func.value))
            if isinstance(rt, ClassType) and rt.name in self.classes:
                found = self.method_node(rt.name, call.func.attr)
                if found:
                    return f"{found[0]}.{call.func.attr}"
        return None

    def summary(self, key: str) -> Summary:
        return self.summaries.setdefault(key, Summary())

    def _scope_for(self, key: str) -> ScopeInfo | None:
        if "." in key:
            cls, m = key.split(".", 1)
            return self.env.class_method_scopes.get((cls, m))
        return self.env.func_scopes.get(key)

    def analyze_function(self, key: str, fn: ast.FunctionDef) -> _ScopeAnalyzer:
        scope = self._scope_for(key)
        params = [a.arg for a in _positional_params(fn)]
        if fn.args.vararg:
            params.append(fn.args.vararg.arg)
        sa = _ScopeAnalyzer(self, key, scope, fn, params, self.parsed.filename)
        st = State()
        for p in params:
            pt = scope.var_types.get(p, UNKNOWN) if scope else UNKNOWN
            if is_mutable(pt) or p == "self":
                st.env[p] = frozenset({("param", p)})
                st.heap[("param", p)] = frozenset({("pchild", p)})
                st.heap[("pchild", p)] = frozenset({("pchild", p)})
        sa.live.block(fn.body, frozenset())
        sa.block(fn.body, st)
        return sa

    def run(self, include_main: bool = True) -> AliasInfo:
        keys: list[tuple[str, ast.FunctionDef]] = [(f.name, f) for f in self.parsed.functions]
        for cls in self.parsed.classes:
            for item in cls.body:
                if isinstance(item, ast.FunctionDef):
                    keys.append((f"{cls.name}.{item.name}", item))
        results: dict[str, _ScopeAnalyzer] = {}
        for _ in range(20):
            changed = False
            for key, fn in keys:
                if self._scope_for(key) is None:
                    continue
                before = self.summary(key).key()
                sa = self.analyze_function(key, fn)
                self.summaries[key] = sa.summary
                results[key] = sa
                if sa.summary.key() != before:
                    changed = True
            if not changed:
                break
        if include_main:
            sa = _ScopeAnalyzer(self, "__main__", self.env.module_scope, None, [], self.parsed.filename)
            sa.live.block(self.parsed.main_body, frozenset())
            sa.block(self.parsed.main_body, State())
            results["__main__"] = sa
        for key, res in results.items():
            refs = set()
            fn = self.fn_node(key) if key != "__main__" else None
            if fn is not None:
                for p in res.summary.writes:
                    if p == "self":
                        continue
                    if p in res.summary.rebinds:
                        raise PyRustError(
                            f"parameter '{p}' is both modified in place and reassigned in '{key}'; "
                            "this is not supported natively",
                            lineno=fn.lineno,
                            col=fn.col_offset,
                            filename=self.parsed.filename,
                        )
                    refs.add(p)
                if fn.args.vararg and fn.args.vararg.arg in refs:
                    raise PyRustError(
                        f"modifying the *{fn.args.vararg.arg} tuple's contents in place is not supported natively",
                        lineno=fn.lineno,
                        col=fn.col_offset,
                        filename=self.parsed.filename,
                    )
            if refs:
                self.info.ref_params[key] = refs
            self.info.byref_loops |= res.byref_used
        return self.info


def analyze_aliasing(parsed: ParsedModule, env: ModuleTypeEnv, include_main: bool = True) -> AliasInfo:
    return AliasAnalyzer(parsed, env).run(include_main)
