from __future__ import annotations

import ast
from dataclasses import dataclass, field

from pyrust.parser import ParsedModule
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
    type_from_annotation_name,
    unify_container_elem,
    unify_types,
    arith_result_type,
)

MUTATING_METHODS = {
    "append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse",
    "add", "update", "setdefault",
}


def _positional_params(fn_node: ast.FunctionDef) -> list[ast.arg]:
    """All positional parameters (including `/`-only), in order."""
    return list(fn_node.args.posonlyargs) + list(fn_node.args.args)


def _find_method_node_in(
    class_nodes: dict[str, ast.ClassDef], cls_name: str, attr: str, _seen: set[str] | None = None
) -> tuple[str, ast.FunctionDef] | None:
    """Find a method definition up the same-file inheritance chain."""
    node = class_nodes.get(cls_name)
    if node is None:
        return None
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == attr:
            return (cls_name, item)
    seen = _seen or set()
    if cls_name in seen or not node.bases or len(node.bases) != 1:
        return None
    seen.add(cls_name)
    base = node.bases[0]
    if isinstance(base, ast.Name) and base.id in class_nodes:
        return _find_method_node_in(class_nodes, base.id, attr, seen)
    return None


def _is_dataclass(cls_node: ast.ClassDef) -> bool:
    for d in cls_node.decorator_list:
        if isinstance(d, ast.Name) and d.id == "dataclass":
            return True
        if isinstance(d, ast.Attribute) and d.attr == "dataclass":
            return True
    return False


def _is_dataclass_field_call(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == "field")
        or (isinstance(node.func, ast.Attribute) and node.func.attr == "field")
    )


def _check_callable_supported(fn_node: ast.FunctionDef, filename: str) -> None:
    from pyrust.errors import PyRustError

    if fn_node.args.kwarg is not None:
        raise PyRustError(
            "**kwargs is not supported by PyRust yet: use explicit parameters",
            lineno=fn_node.lineno,
            col=fn_node.col_offset,
            filename=filename,
        )
    if fn_node.args.kwonlyargs:
        raise PyRustError(
            "keyword-only arguments are not supported by PyRust yet: use regular parameters",
            lineno=fn_node.lineno,
            col=fn_node.col_offset,
            filename=filename,
        )


def _pyfile_type() -> ClassType:
    return ClassType(
        name="PyFile",
        fields={},
        methods={
            "read": FuncType(params=[], ret=STR),
            "readlines": FuncType(params=[], ret=ListType(STR)),
            "lines": FuncType(params=[], ret=ListType(STR)),
            "write": FuncType(params=[STR], ret=NONE),
            "close": FuncType(params=[], ret=NONE),
        },
    )

# Supported `math` module surface: float-returning functions, int-returning
# functions, and float constants.
MATH_FLOAT_FUNCS = {
    "sqrt", "sin", "cos", "tan", "exp", "ln", "log", "log2", "log10", "abs", "fabs", "pow",
    "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "hypot",
}
MATH_INT_FUNCS = {"floor", "ceil", "factorial", "gcd"}
MATH_CONSTS = {"pi", "e", "tau"}
MATH_CONST_RUST = {
    "pi": "std::f64::consts::PI",
    "e": "std::f64::consts::E",
    "tau": "std::f64::consts::TAU",
}


@dataclass
class ScopeInfo:
    var_types: dict[str, PyType] = field(default_factory=dict)
    hoisted_vars: set[str] = field(default_factory=set)
    mutated_vars: set[str] = field(default_factory=set)
    first_assigned_at_top: set[str] = field(default_factory=set)
    assignment_counts: dict[str, int] = field(default_factory=dict)
    expr_types: dict[int, PyType] = field(default_factory=dict)
    params: list[str] = field(default_factory=list)


@dataclass
class ModuleTypeEnv:
    module_scope: ScopeInfo = field(default_factory=ScopeInfo)
    func_types: dict[str, FuncType] = field(default_factory=dict)
    func_scopes: dict[str, ScopeInfo] = field(default_factory=dict)
    class_types: dict[str, ClassType] = field(default_factory=dict)
    class_method_scopes: dict[tuple[str, str], ScopeInfo] = field(default_factory=dict)
    # Call-site support: parameter names and default value expressions,
    # keyed by function name (or "Cls.method" / "mod.func" for qualified).
    func_param_names: dict[str, list[str]] = field(default_factory=dict)
    func_defaults: dict[str, list[ast.expr]] = field(default_factory=dict)


def parse_annotation(node: ast.expr | None, class_types: dict[str, ClassType] | None = None) -> PyType:
    if node is None:
        return UNKNOWN
    if isinstance(node, ast.Constant) and node.value is None:
        return NONE
    if isinstance(node, ast.Name):
        if class_types and node.id in class_types:
            return class_types[node.id]
        return type_from_annotation_name(node.id)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        base = node.value.id.lower()
        if base in ("list", "sequence", "iterable"):
            return ListType(parse_annotation(node.slice, class_types))
        if base == "set":
            return SetType(parse_annotation(node.slice, class_types))
        if base in ("dict", "mapping"):
            if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                return DictType(
                    parse_annotation(node.slice.elts[0], class_types),
                    parse_annotation(node.slice.elts[1], class_types),
                )
            return DictType(UNKNOWN, UNKNOWN)
        if base == "tuple":
            if isinstance(node.slice, ast.Tuple):
                return TupleType(tuple(parse_annotation(e, class_types) for e in node.slice.elts))
            return TupleType((parse_annotation(node.slice, class_types),))
    return UNKNOWN


# Python identifiers that are reserved words in Rust (and cannot all be raw
# identifiers): renamed consistently before analysis.
RUST_RESERVED = {
    "abstract", "become", "box", "const", "crate", "do", "dyn", "enum", "extern", "final", "fn",
    "gen", "impl", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub",
    "ref", "Self", "static", "struct", "super", "trait", "try", "type", "typeof", "unsafe", "unsized",
    "use", "virtual", "where", "union", "main_",
}


def rename_rust_reserved(parsed: ParsedModule) -> None:
    """Rename identifiers that Rust reserves (`type`, `match`, `move`, ...).

    Records `parsed.rust_renames` (new -> original) so Python-visible text
    (e.g. a dataclass repr) can still use the original spelling.
    """
    if getattr(parsed, "rust_renames", None) is not None:
        return
    renames: dict[str, str] = {}

    def fix(name: str) -> str:
        if name in RUST_RESERVED:
            new = f"py_{name}"
            renames[new] = name
            return new
        return name

    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.Name):
            node.id = fix(node.id)
        elif isinstance(node, ast.arg):
            node.arg = fix(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.name = fix(node.name)
        elif isinstance(node, ast.Attribute):
            node.attr = fix(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            node.arg = fix(node.arg)
    parsed.from_imports = {fix(k): v for k, v in parsed.from_imports.items()}
    parsed.rust_renames = renames  # type: ignore[attr-defined]


class _TypeInferer:
    def __init__(self, parsed: ParsedModule) -> None:
        rename_rust_reserved(parsed)
        self.parsed = parsed
        self.env = ModuleTypeEnv()
        self.func_nodes: dict[str, ast.FunctionDef] = {fn.name: fn for fn in parsed.functions}
        self.class_nodes: dict[str, ast.ClassDef] = {cls.name: cls for cls in parsed.classes}

        # Seed classes
        for cls in parsed.classes:
            methods: dict[str, FuncType] = {}
            for item in cls.body:
                if isinstance(item, ast.FunctionDef):
                    _check_callable_supported(item, parsed.filename)
                    pos = _positional_params(item)
                    non_self_args = [a for a in pos if a.arg != "self"]
                    p_types = [parse_annotation(a.annotation) for a in non_self_args]
                    has_vararg = item.args.vararg is not None
                    if has_vararg:
                        p_types.append(ListType(UNKNOWN))
                    ret_type = NONE if item.name == "__init__" else parse_annotation(item.returns)
                    methods[item.name] = FuncType(params=p_types, ret=ret_type, has_vararg=has_vararg)
                    scope = ScopeInfo(params=[a.arg for a in pos])
                    if item.args.vararg is not None:
                        scope.params.append(item.args.vararg.arg)
                    for a, pt in zip(non_self_args, p_types):
                        scope.var_types[a.arg] = pt
                        scope.first_assigned_at_top.add(a.arg)
                    if item.args.vararg is not None:
                        scope.var_types[item.args.vararg.arg] = ListType(UNKNOWN)
                        scope.first_assigned_at_top.add(item.args.vararg.arg)
                    self.env.class_method_scopes[(cls.name, item.name)] = scope
                    qkey = f"{cls.name}.{item.name}"
                    self.env.func_param_names[qkey] = [a.arg for a in non_self_args]
                    if has_vararg and item.args.vararg is not None:
                        self.env.func_param_names[qkey].append(item.args.vararg.arg)
                    self.env.func_defaults[qkey] = list(item.args.defaults)
            self.env.class_types[cls.name] = ClassType(name=cls.name, fields={}, methods=methods)

        # Seed functions
        for fn in parsed.functions:
            _check_callable_supported(fn, parsed.filename)
            pos = _positional_params(fn)
            p_types = [parse_annotation(a.annotation, self.env.class_types) for a in pos]
            has_vararg = fn.args.vararg is not None
            if has_vararg:
                p_types.append(ListType(UNKNOWN))
            ret_type = parse_annotation(fn.returns, self.env.class_types)
            self.env.func_types[fn.name] = FuncType(params=p_types, ret=ret_type, has_vararg=has_vararg)
            scope = ScopeInfo(params=[a.arg for a in pos])
            if fn.args.vararg is not None:
                scope.params.append(fn.args.vararg.arg)
            for a, pt in zip(pos, p_types):
                scope.var_types[a.arg] = pt
                scope.first_assigned_at_top.add(a.arg)
            if fn.args.vararg is not None:
                scope.var_types[fn.args.vararg.arg] = ListType(UNKNOWN)
                scope.first_assigned_at_top.add(fn.args.vararg.arg)
            self.env.func_scopes[fn.name] = scope
            self.env.func_param_names[fn.name] = [a.arg for a in pos]
            if has_vararg and fn.args.vararg is not None:
                self.env.func_param_names[fn.name].append(fn.args.vararg.arg)
            self.env.func_defaults[fn.name] = list(fn.args.defaults)

    def run(self) -> ModuleTypeEnv:
        # First pass: structural assignment / hoisting scan
        self._scan_assignments_in_body(self.parsed.main_body, self.env.module_scope)
        for fn in self.parsed.functions:
            self._scan_assignments_in_body(fn.body, self.env.func_scopes[fn.name])
        for cls in self.parsed.classes:
            for item in cls.body:
                if isinstance(item, ast.FunctionDef):
                    scope = self.env.class_method_scopes[(cls.name, item.name)]
                    self._scan_assignments_in_body(item.body, scope)

        # Fixed-point type propagation across module, functions, and class methods (4 rounds)
        for _ in range(4):
            self._infer_stmts(self.parsed.main_body, self.env.module_scope, current_class=None)
            for cls in self.parsed.classes:
                self._infer_class(cls)
            for fn in self.parsed.functions:
                self._infer_function(fn)
            self._infer_stmts(self.parsed.main_body, self.env.module_scope, current_class=None)

        return self.env

    def _finalize_unknown_params(self) -> None:
        """Default never-constrained parameters to dynamic PyValue.

        If no call site, annotation, default, or body constraint ever pinned a
        parameter down (e.g. dead code), there is no static type to give it.
        Dynamic is the only honest answer — and it keeps the generated code
        compiling instead of failing on `UNKNOWN`.
        """
        for fn in self.parsed.functions:
            ft = self.env.func_types[fn.name]
            if any(p == UNKNOWN for p in ft.params):
                scope = self.env.func_scopes[fn.name]
                for a in _positional_params(fn):
                    if scope.var_types.get(a.arg, UNKNOWN) == UNKNOWN:
                        scope.var_types[a.arg] = ANY
                if fn.args.vararg is not None and not isinstance(
                    scope.var_types.get(fn.args.vararg.arg), ListType
                ):
                    scope.var_types[fn.args.vararg.arg] = ListType(ANY)
                self._infer_function(fn)
        for cls in self.parsed.classes:
            touched = False
            for item in cls.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                scope = self.env.class_method_scopes[(cls.name, item.name)]
                for a in [x for x in _positional_params(item) if x.arg != "self"]:
                    if scope.var_types.get(a.arg, UNKNOWN) == UNKNOWN:
                        scope.var_types[a.arg] = ANY
                        touched = True
                if item.args.vararg is not None and not isinstance(
                    scope.var_types.get(item.args.vararg.arg), ListType
                ):
                    scope.var_types[item.args.vararg.arg] = ListType(ANY)
                    touched = True
            if touched:
                self._infer_class(cls)

    def _scan_assignments_in_body(self, stmts: list[ast.stmt], scope: ScopeInfo) -> None:
        """Scan top-level vs nested assignments to detect hoisted variables and mutations."""
        counts: dict[str, int] = {p: 1 for p in scope.params}
        top_defined: set[str] = set(scope.params)

        def record_target(target: ast.expr, is_top: bool, is_for_target: bool = False) -> None:
            if isinstance(target, ast.Name):
                name = target.id
                counts[name] = counts.get(name, 0) + 1
                if is_top and not is_for_target:
                    top_defined.add(name)
                elif name not in top_defined and not is_for_target:
                    scope.hoisted_vars.add(name)
                    scope.mutated_vars.add(name)
                if counts[name] > 1 and not is_for_target:
                    scope.mutated_vars.add(name)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    record_target(elt, is_top, is_for_target=is_for_target)
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                scope.mutated_vars.add(target.value.id)
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                scope.mutated_vars.add(target.value.id)

        def walk_block(block: list[ast.stmt], is_top: bool) -> None:
            for s in block:
                for node in ast.walk(s):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.attr in MUTATING_METHODS
                    ):
                        scope.mutated_vars.add(node.func.value.id)
                    # random.shuffle(seq) mutates its sequence argument in place
                    if isinstance(node, ast.Call) and node.args:
                        shuffled: ast.expr | None = None
                        if (
                            isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "random"
                            and node.func.attr == "shuffle"
                        ):
                            shuffled = node.args[0]
                        elif (
                            isinstance(node.func, ast.Name)
                            and self.parsed.from_imports.get(node.func.id) == "random"
                            and node.func.id == "shuffle"
                        ):
                            shuffled = node.args[0]
                        if isinstance(shuffled, ast.Name):
                            scope.mutated_vars.add(shuffled.id)
                if isinstance(s, ast.Assign):
                    for t in s.targets:
                        record_target(t, is_top)
                elif isinstance(s, ast.AnnAssign):
                    record_target(s.target, is_top)
                elif isinstance(s, ast.AugAssign):
                    if isinstance(s.target, ast.Name):
                        scope.mutated_vars.add(s.target.id)
                        record_target(s.target, is_top)
                    elif isinstance(s.target, ast.Subscript) and isinstance(s.target.value, ast.Name):
                        scope.mutated_vars.add(s.target.value.id)
                elif isinstance(s, ast.For):
                    record_target(s.target, is_top=False, is_for_target=True)
                    walk_block(s.body, is_top=False)
                    walk_block(s.orelse, is_top=False)
                elif isinstance(s, ast.While):
                    walk_block(s.body, is_top=False)
                    walk_block(s.orelse, is_top=False)
                elif isinstance(s, ast.If):
                    walk_block(s.body, is_top=False)
                    walk_block(s.orelse, is_top=False)
                elif isinstance(s, ast.Try):
                    walk_block(s.body, is_top=False)
                    for h in s.handlers:
                        walk_block(h.body, is_top=False)
                    walk_block(s.orelse, is_top=False)
                    walk_block(s.finalbody, is_top=False)
                elif isinstance(s, ast.With):
                    for item in s.items:
                        if (
                            isinstance(item.context_expr, ast.Call)
                            and isinstance(item.context_expr.func, ast.Name)
                            and item.context_expr.func.id == "open"
                            and isinstance(item.optional_vars, ast.Name)
                        ):
                            # Top-level with-targets are declared by the with
                            # statement itself; nested ones hoist normally.
                            record_target(item.optional_vars, is_top)
                    walk_block(s.body, is_top=False)
                elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
                    func = s.value.func
                    if (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.attr in MUTATING_METHODS
                    ):
                        scope.mutated_vars.add(func.value.id)

        walk_block(stmts, is_top=True)
        scope.first_assigned_at_top = top_defined
        scope.assignment_counts = counts

    def _default_type_for(self, pos: list[ast.arg], defaults: list[ast.expr], idx: int) -> PyType:
        """Type of a parameter's default value (defaults align to trailing params)."""
        n_def = len(defaults)
        if n_def and idx >= len(pos) - n_def:
            return self._infer_expr(defaults[idx - (len(pos) - n_def)], self.env.module_scope, None)
        return UNKNOWN

    def _infer_function(self, fn: ast.FunctionDef) -> None:
        scope = self.env.func_scopes[fn.name]
        ft = self.env.func_types[fn.name]
        pos = _positional_params(fn)

        # Seed untyped parameters from usage constraints inside the body if still UNKNOWN
        seeded_params = list(ft.params)
        while len(seeded_params) < len(pos) + (1 if fn.args.vararg else 0):
            seeded_params.append(UNKNOWN)
        for idx, arg in enumerate(pos):
            cur = unify_types(seeded_params[idx], scope.var_types.get(arg.arg, UNKNOWN))
            cur = unify_types(cur, self._default_type_for(pos, fn.args.defaults, idx))
            if cur == UNKNOWN:
                cur = self._guess_param_from_body(arg.arg, fn.body)
            seeded_params[idx] = cur
            scope.var_types[arg.arg] = cur
        if fn.args.vararg is not None:
            vararg_t = scope.var_types.get(fn.args.vararg.arg, ListType(UNKNOWN))
            seeded_params[len(pos)] = unify_types(
                seeded_params[len(pos)],
                vararg_t if isinstance(vararg_t, ListType) else ListType(UNKNOWN),
            )
            scope.var_types[fn.args.vararg.arg] = seeded_params[len(pos)]

        ret_types: list[PyType] = []
        self._infer_stmts(fn.body, scope, current_class=None, ret_collector=ret_types)

        # Recompute the return type from the annotation plus the body on every
        # pass: seeding from a stale ft.ret would let an old ANY/UNKNOWN poison
        # newly inferable returns (unify(ANY, T) == ANY).
        inferred_ret = parse_annotation(fn.returns, self.env.class_types)
        for rt in ret_types:
            inferred_ret = unify_types(inferred_ret, rt)
        if inferred_ret == UNKNOWN and not ret_types:
            inferred_ret = NONE

        updated_params = [
            unify_types(seeded_params[i], scope.var_types.get(a.arg, UNKNOWN))
            for i, a in enumerate(pos)
        ]
        if fn.args.vararg is not None:
            updated_params.append(scope.var_types.get(fn.args.vararg.arg, ListType(UNKNOWN)))
        self.env.func_types[fn.name] = FuncType(
            params=updated_params, ret=inferred_ret, has_vararg=fn.args.vararg is not None
        )
        self.env.func_defaults[fn.name] = list(fn.args.defaults)
        self.env.func_param_names[fn.name] = [a.arg for a in pos]
        if fn.args.vararg is not None:
            self.env.func_param_names[fn.name].append(fn.args.vararg.arg)

    def _find_method_node(self, cls_name: str, attr: str) -> tuple[str, ast.FunctionDef] | None:
        return _find_method_node_in(self.class_nodes, cls_name, attr)

    def _merge_base_class(
        self, cls: ast.ClassDef, fields: dict[str, PyType], methods: dict[str, FuncType]
    ) -> None:
        """Single inheritance: copy base fields/methods in (derived wins).

        Fields always merge (data layout flows). Method *bodies* can only be
        re-emitted for same-file bases, so cross-module bases contribute
        fields; inherited method calls there raise a clear error at codegen.
        """
        if not cls.bases:
            return
        if len(cls.bases) > 1:
            from pyrust.errors import PyRustError

            raise PyRustError(
                "Multiple inheritance is not supported by PyRust yet: use a single base class",
                lineno=cls.lineno,
                col=cls.col_offset,
                filename=self.parsed.filename,
            )
        base = cls.bases[0]
        if isinstance(base, ast.Name):
            alias_mod = self.parsed.from_imports.get(base.id)
            base_name: str | None = f"{alias_mod}.{base.id}" if alias_mod else base.id
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            base_name = f"{base.value.id}.{base.attr}"
        else:
            base_name = None
        if base_name is None or base_name in ("object", "ABC") or base_name.endswith(".ABC"):
            return
        base_t = self.env.class_types.get(base_name)
        if base_t is None:
            from pyrust.errors import PyRustError

            raise PyRustError(
                f"base class '{base_name}' is not defined in this PyRust project",
                lineno=cls.lineno,
                col=cls.col_offset,
                filename=self.parsed.filename,
            )
        for fname, ftype in base_t.fields.items():
            fields.setdefault(fname, ftype)
        if "." not in base_name and base_name in self.class_nodes:
            for mname, mtype in base_t.methods.items():
                methods.setdefault(mname, mtype)
            # Keep a pointer so codegen can find inherited method bodies.
            self.env.func_param_names[f"{cls.name}.__bases__"] = [base_name]

    def _infer_class(self, cls: ast.ClassDef) -> None:
        cls_type = self.env.class_types[cls.name]
        fields = dict(cls_type.fields)
        methods = dict(cls_type.methods)
        self._merge_base_class(cls, fields, methods)
        self.env.class_types[cls.name] = ClassType(cls.name, fields, methods)

        is_dc = _is_dataclass(cls)
        dc_defaults: list[ast.expr] = []
        if is_dc:
            from pyrust.errors import PyRustError

            for item in cls.body:
                target: ast.expr | None = None
                ann: ast.expr | None = None
                dflt: ast.expr | None = None
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    # Skip ClassVar[...] (class-level, not instance fields).
                    if isinstance(item.annotation, ast.Subscript) and isinstance(
                        item.annotation.value, ast.Name
                    ):
                        if item.annotation.value.id == "ClassVar":
                            continue
                    target, ann, dflt = item.target, item.annotation, item.value
                elif (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                ):
                    target, dflt = item.targets[0], item.value
                if target is None:
                    continue
                if dflt is not None and _is_dataclass_field_call(dflt):
                    raise PyRustError(
                        "dataclass field(default_factory=...) is not supported by PyRust yet",
                        lineno=item.lineno,
                        col=item.col_offset,
                        filename=self.parsed.filename,
                    )
                ftype = parse_annotation(ann, self.env.class_types) if ann else UNKNOWN
                if dflt is not None:
                    # An explicit annotation is authoritative; otherwise the
                    # default value's type stands on its own.
                    dflt_t = self._infer_expr(dflt, self.env.module_scope, None)
                    ftype = ftype if ftype != UNKNOWN else dflt_t
                    dc_defaults.append(dflt)
                fields[target.id] = unify_container_elem(fields.get(target.id, UNKNOWN), ftype)
            self.env.class_types[cls.name] = ClassType(cls.name, fields, dict(methods))

        for item in cls.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            scope = self.env.class_method_scopes[(cls.name, item.name)]
            scope.var_types["self"] = ClassType(cls.name, fields, methods)
            pos = _positional_params(item)
            non_self_args = [a for a in pos if a.arg != "self"]
            cur_ft = methods[item.name]
            cur_params = list(cur_ft.params)
            while len(cur_params) < len(non_self_args) + (1 if item.args.vararg else 0):
                cur_params.append(UNKNOWN)
            for idx, a in enumerate(non_self_args):
                pt = unify_types(cur_params[idx], scope.var_types.get(a.arg, UNKNOWN))
                pt = unify_types(pt, self._default_type_for(pos, item.args.defaults, pos.index(a)))
                if pt == UNKNOWN:
                    pt = self._guess_param_from_body(a.arg, item.body)
                scope.var_types[a.arg] = pt
            if item.args.vararg is not None:
                vararg_t = scope.var_types.get(item.args.vararg.arg, ListType(UNKNOWN))
                scope.var_types[item.args.vararg.arg] = (
                    vararg_t if isinstance(vararg_t, ListType) else ListType(UNKNOWN)
                )

            ret_types: list[PyType] = []
            self._infer_stmts(item.body, scope, current_class=cls.name, ret_collector=ret_types)
            # Refresh fields discovered while walking method body
            fields = dict(self.env.class_types[cls.name].fields)

            inferred_ret = (
                NONE
                if item.name == "__init__"
                else parse_annotation(item.returns, self.env.class_types)
            )
            if item.name != "__init__":
                for rt in ret_types:
                    inferred_ret = unify_types(inferred_ret, rt)
                if inferred_ret == UNKNOWN and not ret_types:
                    inferred_ret = NONE

            updated_params = [
                unify_types(cur_params[i], scope.var_types.get(a.arg, UNKNOWN))
                for i, a in enumerate(non_self_args)
            ]
            has_vararg = item.args.vararg is not None
            if has_vararg:
                updated_params.append(scope.var_types.get(item.args.vararg.arg, ListType(UNKNOWN)))
            methods[item.name] = FuncType(params=updated_params, ret=inferred_ret, has_vararg=has_vararg)
            self.env.class_types[cls.name] = ClassType(cls.name, fields, methods)
            qkey = f"{cls.name}.{item.name}"
            self.env.func_param_names[qkey] = [a.arg for a in non_self_args]
            if has_vararg:
                self.env.func_param_names[qkey].append(item.args.vararg.arg)
            self.env.func_defaults[qkey] = list(item.args.defaults)

        has_own_init = any(
            isinstance(item, ast.FunctionDef) and item.name == "__init__" for item in cls.body
        )
        own_names = {
            item.name for item in cls.body if isinstance(item, ast.FunctionDef)
        }
        if is_dc and not has_own_init and fields:
            # Synthesize __init__ from dataclass fields (defaults = trailing).
            ordered = list(fields.keys())
            init_params = [fields[fname] for fname in ordered]
            init_ft = FuncType(params=init_params, ret=NONE)
            methods["__init__"] = init_ft
            self.env.class_types[cls.name] = ClassType(cls.name, dict(fields), dict(methods))
            self.env.class_method_scopes[(cls.name, "__init__")] = ScopeInfo(
                params=list(ordered),
                var_types=dict(zip(ordered, init_params)),
                first_assigned_at_top=set(ordered),
            )
            self.env.func_param_names[f"{cls.name}.__init__"] = list(ordered)
            self.env.func_defaults[f"{cls.name}.__init__"] = list(dc_defaults)

        # Re-infer inherited method bodies in the derived context so their
        # signatures specialize (e.g. get() -> i64 when Child.x is i64).
        final_cls = self.env.class_types.get(cls.name, ClassType(cls.name, fields, methods))
        for mname in list(methods.keys()):
            if mname in own_names:
                continue
            found = _find_method_node_in(self.class_nodes, cls.name, mname)
            if found is None:
                continue
            _, base_item = found
            key = (cls.name, mname)
            scope = self.env.class_method_scopes.get(key)
            if scope is None:
                pos = [a.arg for a in _positional_params(base_item) if a.arg != "self"]
                var_types: dict[str, PyType] = {"self": final_cls}
                for i, pname in enumerate(pos):
                    var_types[pname] = methods[mname].params[i] if i < len(methods[mname].params) else UNKNOWN
                scope = ScopeInfo(
                    params=["self"] + pos,
                    var_types=var_types,
                    first_assigned_at_top=set(["self"] + pos),
                )
                self.env.class_method_scopes[key] = scope
                # Populate mutated/hoisted info for inherited bodies (e.g.
                # `lines.append` must mark `lines` mutable for `let mut`).
                self._scan_assignments_in_body(base_item.body, scope)
            scope.var_types["self"] = self.env.class_types[cls.name]
            ret_types: list[PyType] = []
            self._infer_stmts(base_item.body, scope, current_class=cls.name, ret_collector=ret_types)
            if mname == "__init__":
                inferred_ret = NONE
            else:
                inferred_ret = parse_annotation(base_item.returns, self.env.class_types)
                for rt in ret_types:
                    inferred_ret = unify_types(inferred_ret, rt)
                if inferred_ret == UNKNOWN and not ret_types:
                    inferred_ret = NONE
            pos_args = [a for a in _positional_params(base_item) if a.arg != "self"]
            updated = [
                unify_types(
                    methods[mname].params[i] if i < len(methods[mname].params) else UNKNOWN,
                    scope.var_types.get(a.arg, UNKNOWN),
                )
                for i, a in enumerate(pos_args)
            ]
            has_va = base_item.args.vararg is not None
            if has_va:
                updated.append(scope.var_types.get(base_item.args.vararg.arg, ListType(UNKNOWN)))
            methods[mname] = FuncType(params=updated, ret=inferred_ret, has_vararg=has_va)
            # Refresh fields: _assign_target writes went straight to env.
            fields = dict(self.env.class_types[cls.name].fields)
            self.env.class_types[cls.name] = ClassType(cls.name, fields, dict(methods))
            self.env.func_param_names[f"{cls.name}.{mname}"] = [a.arg for a in pos_args]
            if has_va and base_item.args.vararg is not None:
                self.env.func_param_names[f"{cls.name}.{mname}"].append(base_item.args.vararg.arg)
            self.env.func_defaults[f"{cls.name}.{mname}"] = list(base_item.args.defaults)

    @staticmethod
    def _comp_target_names(target: ast.expr) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            out: list[str] = []
            for elt in target.elts:
                out.extend(_TypeInferer._comp_target_names(elt))
            return out
        return []

    @staticmethod
    def _isolate_comp_targets(
        generators: list[ast.comprehension], scope: ScopeInfo
    ) -> dict[str, PyType | None]:
        """Save outer types and clear for comprehension targets (Py3 comps don't leak)."""
        saved: dict[str, PyType | None] = {}
        for gen in generators:
            for name in _TypeInferer._comp_target_names(gen.target):
                if name not in saved:
                    saved[name] = scope.var_types.get(name)
                    # Remove so inner `_assign_target` unifies from UNKNOWN,
                    # not from the outer variable of the same name.
                    scope.var_types.pop(name, None)
        return saved

    @staticmethod
    def _restore_comp_targets(saved: dict[str, PyType | None], scope: ScopeInfo) -> None:
        for name, old in saved.items():
            if old is None:
                scope.var_types.pop(name, None)
            else:
                scope.var_types[name] = old

    def _guess_param_from_body(self, param_name: str, body: list[ast.stmt]) -> PyType:
        """Heuristic constraint seeding when a function parameter is used with numeric/collection ops."""
        guessed: PyType = UNKNOWN
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                if any(isinstance(op, ast.Name) and op.id == param_name for op in operands):
                    for other in operands:
                        if isinstance(other, ast.Constant):
                            if isinstance(other.value, bool):
                                guessed = unify_types(guessed, BOOL)
                            elif isinstance(other.value, int):
                                guessed = unify_types(guessed, INT)
                            elif isinstance(other.value, float):
                                guessed = unify_types(guessed, FLOAT)
                            elif isinstance(other.value, str):
                                guessed = unify_types(guessed, STR)
            elif isinstance(node, ast.BinOp):
                for side, other in ((node.left, node.right), (node.right, node.left)):
                    if isinstance(side, ast.Name) and side.id == param_name:
                        if isinstance(other, ast.Constant):
                            if isinstance(other.value, int):
                                guessed = unify_types(guessed, INT)
                            elif isinstance(other.value, float):
                                guessed = unify_types(guessed, FLOAT)
                            elif isinstance(other.value, str) and isinstance(node.op, ast.Add):
                                guessed = unify_types(guessed, STR)
        return guessed

    def _infer_stmts(
        self,
        stmts: list[ast.stmt],
        scope: ScopeInfo,
        current_class: str | None,
        ret_collector: list[PyType] | None = None,
    ) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                val_type = self._infer_expr(stmt.value, scope, current_class)
                for target in stmt.targets:
                    if isinstance(target, ast.Subscript):
                        # Record base/index expression types for codegen.
                        self._infer_expr(target, scope, current_class)
                    elif isinstance(target, ast.Attribute):
                        self._infer_expr(target.value, scope, current_class)
                    self._assign_target(target, val_type, scope, current_class)
            elif isinstance(stmt, ast.AnnAssign):
                ann_type = parse_annotation(stmt.annotation, self.env.class_types)
                val_type = (
                    self._infer_expr(stmt.value, scope, current_class)
                    if stmt.value is not None
                    else UNKNOWN
                )
                merged = unify_types(ann_type, val_type)
                if isinstance(stmt.target, ast.Subscript):
                    self._infer_expr(stmt.target, scope, current_class)
                self._assign_target(stmt.target, merged, scope, current_class)
            elif isinstance(stmt, ast.AugAssign):
                target_type = self._infer_expr(stmt.target, scope, current_class)
                rhs_type = self._infer_expr(stmt.value, scope, current_class)
                res_type = self._infer_binop_type(stmt.op, target_type, rhs_type)
                self._assign_target(stmt.target, res_type, scope, current_class)
            elif isinstance(stmt, ast.Return):
                rt = self._infer_expr(stmt.value, scope, current_class) if stmt.value is not None else NONE
                if ret_collector is not None:
                    ret_collector.append(rt)
            elif isinstance(stmt, ast.For):
                iter_type = self._infer_expr(stmt.iter, scope, current_class)
                elem_type = self._iterable_element_type(iter_type)
                self._assign_target(stmt.target, elem_type, scope, current_class)
                self._infer_stmts(stmt.body, scope, current_class, ret_collector)
                self._infer_stmts(stmt.orelse, scope, current_class, ret_collector)
            elif isinstance(stmt, ast.While):
                self._infer_expr(stmt.test, scope, current_class)
                self._infer_stmts(stmt.body, scope, current_class, ret_collector)
                self._infer_stmts(stmt.orelse, scope, current_class, ret_collector)
            elif isinstance(stmt, ast.If):
                self._infer_expr(stmt.test, scope, current_class)
                self._infer_stmts(stmt.body, scope, current_class, ret_collector)
                self._infer_stmts(stmt.orelse, scope, current_class, ret_collector)
            elif isinstance(stmt, ast.Try):
                self._infer_stmts(stmt.body, scope, current_class, ret_collector)
                for h in stmt.handlers:
                    if h.name:
                        scope.var_types[h.name] = STR
                    self._infer_stmts(h.body, scope, current_class, ret_collector)
                self._infer_stmts(stmt.orelse, scope, current_class, ret_collector)
                self._infer_stmts(stmt.finalbody, scope, current_class, ret_collector)
            elif isinstance(stmt, ast.Expr):
                self._infer_expr(stmt.value, scope, current_class)
            elif isinstance(stmt, ast.Assert):
                self._infer_expr(stmt.test, scope, current_class)
                if stmt.msg:
                    self._infer_expr(stmt.msg, scope, current_class)
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    if (
                        isinstance(item.context_expr, ast.Call)
                        and isinstance(item.context_expr.func, ast.Name)
                        and item.context_expr.func.id == "open"
                        and isinstance(item.optional_vars, ast.Name)
                    ):
                        scope.var_types[item.optional_vars.id] = _pyfile_type()
                self._infer_stmts(stmt.body, scope, current_class, ret_collector)
            elif isinstance(stmt, ast.Delete):
                for target in stmt.targets:
                    if isinstance(target, ast.Subscript):
                        self._infer_expr(target, scope, current_class)
            elif isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Call):
                    for a in stmt.exc.args:
                        self._infer_expr(a, scope, current_class)
                elif stmt.exc is not None and not isinstance(stmt.exc, ast.Name):
                    self._infer_expr(stmt.exc, scope, current_class)

    def _assign_target(
        self,
        target: ast.expr,
        val_type: PyType,
        scope: ScopeInfo,
        current_class: str | None,
    ) -> None:
        if isinstance(target, ast.Name):
            existing = scope.var_types.get(target.id, UNKNOWN)
            scope.var_types[target.id] = unify_types(existing, val_type)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(val_type, TupleType) and len(val_type.elements) == len(target.elts):
                for elt, et in zip(target.elts, val_type.elements):
                    self._assign_target(elt, et, scope, current_class)
            elif isinstance(val_type, ListType):
                for elt in target.elts:
                    self._assign_target(elt, val_type.elem, scope, current_class)
            else:
                for elt in target.elts:
                    self._assign_target(elt, ANY, scope, current_class)
        elif (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and current_class is not None
        ):
            cls_type = self.env.class_types[current_class]
            fields = dict(cls_type.fields)
            fields[target.attr] = unify_container_elem(fields.get(target.attr, UNKNOWN), val_type)
            self.env.class_types[current_class] = ClassType(
                cls_type.name, fields, dict(cls_type.methods)
            )
            scope.var_types["self"] = self.env.class_types[current_class]
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            # Dynamic field creation on instances (p.extra = 1).
            recv_name = target.value.id
            recv_t = scope.var_types.get(recv_name, self.env.module_scope.var_types.get(recv_name, UNKNOWN))
            if isinstance(recv_t, ClassType):
                latest = self.env.class_types.get(recv_t.name, recv_t)
                fields = dict(latest.fields)
                fields[target.attr] = unify_container_elem(fields.get(target.attr, UNKNOWN), val_type)
                self.env.class_types[latest.name] = ClassType(
                    latest.name, fields, dict(latest.methods)
                )
                scope.var_types[recv_name] = self.env.class_types[latest.name]
                scope.mutated_vars.add(recv_name)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            container_name = target.value.id
            cur_container = scope.var_types.get(container_name, UNKNOWN)
            idx_type = self._infer_expr(target.slice, scope, current_class)
            if isinstance(cur_container, DictType):
                scope.var_types[container_name] = DictType(
                    unify_container_elem(cur_container.key, idx_type),
                    unify_container_elem(cur_container.val, val_type),
                )
            elif isinstance(cur_container, ListType):
                if isinstance(target.slice, ast.Slice) and isinstance(val_type, ListType):
                    # Slice assignment replaces a run of elements with a list.
                    elem_t = val_type.elem
                else:
                    elem_t = val_type
                scope.var_types[container_name] = ListType(unify_container_elem(cur_container.elem, elem_t))
        elif (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
        ):
            # Subscript assignment through an attribute (self.d[k] = v).
            recv_name = target.value.value.id
            recv_t = scope.var_types.get(recv_name, self.env.module_scope.var_types.get(recv_name, UNKNOWN))
            if isinstance(recv_t, ClassType):
                latest = self.env.class_types.get(recv_t.name, recv_t)
                cur = latest.fields.get(target.value.attr, UNKNOWN)
                idx_type = self._infer_expr(target.slice, scope, current_class)
                if isinstance(cur, DictType) or cur == UNKNOWN:
                    new_t = DictType(
                        unify_container_elem(cur.key if isinstance(cur, DictType) else UNKNOWN, idx_type),
                        unify_container_elem(cur.val if isinstance(cur, DictType) else UNKNOWN, val_type),
                    )
                elif isinstance(cur, ListType):
                    if isinstance(target.slice, ast.Slice) and isinstance(val_type, ListType):
                        elem_t = val_type.elem
                    else:
                        elem_t = val_type
                    new_t = ListType(unify_container_elem(cur.elem, elem_t))
                else:
                    new_t = cur
                fields = dict(latest.fields)
                fields[target.value.attr] = new_t
                self.env.class_types[latest.name] = ClassType(
                    latest.name, fields, dict(latest.methods)
                )
                scope.var_types[recv_name] = self.env.class_types[latest.name]
                scope.mutated_vars.add(recv_name)

    def _refine_receiver(self, recv: ast.expr, updated: PyType, scope: ScopeInfo) -> None:
        """Record a container's refined type after an in-place method call
        (`x.append(1)` on a variable or on an object field)."""
        if isinstance(recv, ast.Name):
            scope.var_types[recv.id] = updated
        elif isinstance(recv, ast.Attribute) and isinstance(recv.value, ast.Name):
            obj_t = scope.var_types.get(recv.value.id, self.env.module_scope.var_types.get(recv.value.id, UNKNOWN))
            if isinstance(obj_t, ClassType):
                latest = self.env.class_types.get(obj_t.name, obj_t)
                fields = dict(latest.fields)
                fields[recv.attr] = unify_container_elem(fields.get(recv.attr, UNKNOWN), updated)
                self.env.class_types[latest.name] = ClassType(latest.name, fields, dict(latest.methods))
                scope.var_types[recv.value.id] = self.env.class_types[latest.name]

    def _iterable_element_type(self, iter_type: PyType) -> PyType:
        if isinstance(iter_type, ListType):
            return iter_type.elem
        if isinstance(iter_type, SetType):
            return iter_type.elem
        if isinstance(iter_type, DictType):
            return iter_type.key
        if isinstance(iter_type, ClassType) and iter_type.name == "PyFile":
            # for line in f: iterates text lines.
            return STR
        if isinstance(iter_type, TupleType):
            elem: PyType = UNKNOWN
            for e in iter_type.elements:
                elem = unify_container_elem(elem, e)
            return elem
        if iter_type == STR:
            return STR
        if iter_type == UNKNOWN:
            return UNKNOWN  # still being inferred (e.g. a recursive call's result)
        return ANY

    def _infer_binop_type(self, op: ast.operator, left: PyType, right: PyType) -> PyType:
        if left == ANY or right == ANY:
            return ANY
        arith = arith_result_type(type(op).__name__, left, right)
        if arith is not None:
            return arith
        if isinstance(op, ast.Div):
            return FLOAT
        if isinstance(op, ast.FloorDiv):
            if left == FLOAT or right == FLOAT:
                return FLOAT
            return INT
        if isinstance(op, ast.Add):
            if left == STR and right == STR:
                return STR
            if isinstance(left, ListType) and isinstance(right, ListType):
                return unify_types(left, right)
        if isinstance(op, ast.Mult):
            if (left == STR and right == INT) or (left == INT and right == STR):
                return STR
            if isinstance(left, ListType) and right == INT:
                return left
            if left == INT and isinstance(right, ListType):
                return right
        if isinstance(op, ast.Mod) and left == STR:
            return STR
        if isinstance(left, SetType) and isinstance(right, SetType) and isinstance(
            op, (ast.BitOr, ast.BitAnd, ast.Sub, ast.BitXor)
        ):
            return SetType(unify_container_elem(left.elem, right.elem))
        if left == UNKNOWN or right == UNKNOWN:
            return unify_types(left, right)
        return ANY

    def _infer_expr(self, node: ast.expr, scope: ScopeInfo, current_class: str | None) -> PyType:
        result = self._infer_expr_inner(node, scope, current_class)
        scope.expr_types[id(node)] = result
        return result

    def _infer_expr_inner(
        self,
        node: ast.expr,
        scope: ScopeInfo,
        current_class: str | None,
    ) -> PyType:
        if isinstance(node, ast.Constant):
            val = node.value
            if val is None:
                return NONE
            if isinstance(val, bool):
                return BOOL
            if isinstance(val, int):
                return INT
            if isinstance(val, float):
                return FLOAT
            if isinstance(val, str):
                return STR
            return ANY

        if isinstance(node, ast.Name):
            if node.id in scope.var_types:
                return scope.var_types[node.id]
            if node.id in self.env.module_scope.var_types:
                return self.env.module_scope.var_types[node.id]
            alias_mod = self.parsed.from_imports.get(node.id)
            if alias_mod == "math" and node.id in MATH_CONSTS:
                return FLOAT
            if alias_mod == "sys" and node.id == "argv":
                return ListType(STR)
            return UNKNOWN

        if isinstance(node, ast.UnaryOp):
            operand_type = self._infer_expr(node.operand, scope, current_class)
            if isinstance(node.op, ast.Not):
                return BOOL
            if operand_type == BOOL:
                return INT  # -True == -1, ~True == -2
            return operand_type

        if isinstance(node, ast.BinOp):
            left = self._infer_expr(node.left, scope, current_class)
            right = self._infer_expr(node.right, scope, current_class)
            return self._infer_binop_type(node.op, left, right)

        if isinstance(node, ast.BoolOp):
            # `or`/`and` yield an operand value (e.g. `user or "anon"`), so the
            # type is the unification of all operands, not bool.
            result: PyType = UNKNOWN
            for v in node.values:
                result = unify_types(result, self._infer_expr(v, scope, current_class))
            return result if result != UNKNOWN else BOOL

        if isinstance(node, ast.Compare):
            self._infer_expr(node.left, scope, current_class)
            for c in node.comparators:
                self._infer_expr(c, scope, current_class)
            return BOOL

        if isinstance(node, ast.IfExp):
            self._infer_expr(node.test, scope, current_class)
            body_t = self._infer_expr(node.body, scope, current_class)
            else_t = self._infer_expr(node.orelse, scope, current_class)
            return unify_types(body_t, else_t)

        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    self._infer_expr(v.value, scope, current_class)
            return STR

        if isinstance(node, ast.List):
            elem_t: PyType = UNKNOWN
            for elt in node.elts:
                elem_t = unify_container_elem(elem_t, self._infer_expr(elt, scope, current_class))
            return ListType(elem_t)

        if isinstance(node, ast.Set):
            elem_t = UNKNOWN
            for elt in node.elts:
                elem_t = unify_container_elem(elem_t, self._infer_expr(elt, scope, current_class))
            return SetType(elem_t)

        if isinstance(node, ast.Dict):
            k_t: PyType = UNKNOWN
            v_t: PyType = UNKNOWN
            for k, v in zip(node.keys, node.values):
                if k is None:
                    vt = self._infer_expr(v, scope, current_class)
                    if isinstance(vt, DictType):
                        k_t = unify_container_elem(k_t, vt.key)
                        v_t = unify_container_elem(v_t, vt.val)
                    else:
                        v_t = unify_container_elem(v_t, vt)
                else:
                    k_t = unify_container_elem(k_t, self._infer_expr(k, scope, current_class))
                    v_t = unify_container_elem(v_t, self._infer_expr(v, scope, current_class))
            return DictType(k_t, v_t)

        if isinstance(node, ast.Tuple):
            return TupleType(tuple(self._infer_expr(e, scope, current_class) for e in node.elts))

        if isinstance(node, ast.ListComp):
            saved = self._isolate_comp_targets(node.generators, scope)
            try:
                for gen in node.generators:
                    iter_t = self._infer_expr(gen.iter, scope, current_class)
                    self._assign_target(gen.target, self._iterable_element_type(iter_t), scope, current_class)
                    for cond in gen.ifs:
                        self._infer_expr(cond, scope, current_class)
                elt_t = self._infer_expr(node.elt, scope, current_class)
            finally:
                self._restore_comp_targets(saved, scope)
            return ListType(elt_t)

        if isinstance(node, (ast.DictComp, ast.SetComp)):
            saved = self._isolate_comp_targets(node.generators, scope)
            try:
                for gen in node.generators:
                    iter_t = self._infer_expr(gen.iter, scope, current_class)
                    self._assign_target(gen.target, self._iterable_element_type(iter_t), scope, current_class)
                    for cond in gen.ifs:
                        self._infer_expr(cond, scope, current_class)
                if isinstance(node, ast.DictComp):
                    k_t = self._infer_expr(node.key, scope, current_class)
                    v_t = self._infer_expr(node.value, scope, current_class)
                    result: PyType = DictType(k_t, v_t)
                else:
                    assert isinstance(node, ast.SetComp)
                    elt_t = self._infer_expr(node.elt, scope, current_class)
                    result = SetType(elt_t)
            finally:
                self._restore_comp_targets(saved, scope)
            return result

        if isinstance(node, ast.Subscript):
            val_t = self._infer_expr(node.value, scope, current_class)
            if isinstance(node.slice, ast.Slice):
                if node.slice.lower:
                    self._infer_expr(node.slice.lower, scope, current_class)
                if node.slice.upper:
                    self._infer_expr(node.slice.upper, scope, current_class)
                if node.slice.step:
                    self._infer_expr(node.slice.step, scope, current_class)
                return val_t
            idx_t = self._infer_expr(node.slice, scope, current_class)
            if isinstance(val_t, ListType):
                return val_t.elem
            if isinstance(val_t, DictType):
                return val_t.val
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
                    if 0 <= resolved < len(val_t.elements):
                        return val_t.elements[resolved]
            if val_t == STR and idx_t == INT:
                return STR
            if val_t == UNKNOWN:
                return UNKNOWN
            return ANY

        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and (
                    node.value.id == "math"
                    or self.parsed.from_imports.get(node.value.id) == "math"
                )
                and node.attr in MATH_CONSTS
            ):
                return FLOAT
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                if node.attr == "argv":
                    return ListType(STR)
                if node.attr == "platform":
                    return STR
            obj_t = self._infer_expr(node.value, scope, current_class)
            if isinstance(obj_t, ClassType):
                latest = self.env.class_types.get(obj_t.name, obj_t)
                if node.attr in latest.fields:
                    return latest.fields[node.attr]
                # Missing (possibly not-yet-discovered) attribute: stay
                # unknown so later information can still refine the type.
                return UNKNOWN
            if obj_t == UNKNOWN:
                return UNKNOWN
            return ANY

        if isinstance(node, ast.NamedExpr):
            val_t = self._infer_expr(node.value, scope, current_class)
            if isinstance(node.target, ast.Name):
                self._assign_target(node.target, val_t, scope, current_class)
                scope.mutated_vars.add(node.target.id)
            return val_t

        if isinstance(node, ast.Call):
            return self._infer_call(node, scope, current_class)

        return ANY

    def _infer_call(
        self,
        node: ast.Call,
        scope: ScopeInfo,
        current_class: str | None,
    ) -> PyType:
        arg_types = [self._infer_expr(a, scope, current_class) for a in node.args]

        if isinstance(node.func, ast.Name):
            fname = node.func.id
            # Built-ins
            if fname == "int":
                return INT
            if fname == "float":
                return FLOAT
            if fname == "str":
                return STR
            if fname == "bool":
                return BOOL
            if fname == "len":
                return INT
            if fname == "range":
                return ListType(INT)
            if fname == "abs":
                if not arg_types:
                    return INT
                return INT if arg_types[0] == BOOL else arg_types[0]
            if fname == "sum":
                if not arg_types:
                    return INT
                elem = self._iterable_element_type(arg_types[0])
                if len(arg_types) >= 2:
                    start = arg_types[1]
                    if start == FLOAT and elem in (INT, FLOAT, BOOL):
                        return FLOAT
                    if start in (INT, BOOL) and elem in (INT, BOOL):
                        return INT
                    return ANY
                if elem in (INT, BOOL, UNKNOWN):
                    return INT
                # sum([]) is the int 0 even for a list of floats.
                return ANY
            if fname in ("min", "max"):
                if not arg_types:
                    return INT
                first = arg_types[0]
                if len(arg_types) == 1:
                    elem = self._iterable_element_type(first)
                    return elem if elem != UNKNOWN else INT
                acc: PyType = UNKNOWN
                for at in arg_types:
                    acc = unify_types(acc, at)
                return acc if acc != UNKNOWN else INT
            if fname == "sorted":
                if arg_types:
                    return ListType(self._iterable_element_type(arg_types[0]))
                return ListType(UNKNOWN)
            if fname == "list":
                if arg_types:
                    return ListType(self._iterable_element_type(arg_types[0]))
                return ListType(UNKNOWN)
            if fname == "set":
                if arg_types:
                    return SetType(self._iterable_element_type(arg_types[0]))
                return SetType(UNKNOWN)
            if fname == "dict":
                if node.keywords and all(k.arg is not None for k in node.keywords):
                    for kw in node.keywords:
                        self._infer_expr(kw.value, scope, current_class)
                    return DictType(STR, ANY)
                if len(node.args) == 1:
                    at = arg_types[0] if arg_types else UNKNOWN
                    if isinstance(at, DictType):
                        return at
                    if isinstance(at, ListType) and isinstance(at.elem, TupleType) and len(at.elem.elements) == 2:
                        return DictType(at.elem.elements[0], at.elem.elements[1])
                if not node.args and not node.keywords:
                    return DictType(UNKNOWN, UNKNOWN)
                return ANY
            if fname == "enumerate":
                elem = self._iterable_element_type(arg_types[0]) if arg_types else ANY
                return ListType(TupleType((INT, elem)))
            if fname == "zip":
                elems = tuple(self._iterable_element_type(at) for at in arg_types)
                return ListType(TupleType(elems))
            if fname == "input":
                return STR
            if fname == "print":
                return NONE
            if fname == "divmod" and len(arg_types) == 2:
                if arg_types[0] == FLOAT or arg_types[1] == FLOAT:
                    return TupleType((FLOAT, FLOAT))
                return TupleType((INT, INT))
            if fname == "round":
                if len(arg_types) == 2:
                    return INT if arg_types[0] in (INT, BOOL) else FLOAT
                return INT
            if fname in ("all", "any"):
                return BOOL
            if fname == "reversed" and arg_types:
                return ListType(self._iterable_element_type(arg_types[0]))
            if fname == "ord":
                return INT
            if fname in ("chr", "hex", "oct", "bin"):
                return STR
            if fname == "open":
                return _pyfile_type()

            # `from X import name` aliases: resolve to the source module
            alias_mod = self.parsed.from_imports.get(fname)
            if alias_mod == "math":
                if fname in ("floor", "ceil"):
                    return INT
                if fname in MATH_FLOAT_FUNCS:
                    return FLOAT
            elif alias_mod == "random":
                return self._infer_random_call(fname, arg_types)
            elif alias_mod == "json":
                if fname == "dumps":
                    return STR
                if fname == "loads":
                    return ANY
            elif alias_mod == "platform":
                if fname in (
                    "system", "machine", "processor", "architecture",
                    "python_version", "python_implementation", "node",
                ):
                    return STR
            elif alias_mod == "statistics":
                return self._infer_statistics_call(fname, arg_types)
            elif alias_mod == "gc":
                if fname == "collect":
                    return INT
                if fname == "isenabled":
                    return BOOL
                if fname in ("disable", "enable"):
                    return NONE
                if fname == "get_count":
                    return TupleType((INT, INT, INT))
            elif alias_mod == "sys":
                if fname == "exit":
                    return NONE
            elif alias_mod == "os":
                if fname == "getenv":
                    return STR if len(arg_types) == 2 and arg_types[1] == STR else ANY
                if fname == "getcwd":
                    return STR
                if fname == "listdir":
                    return ListType(STR)
                if fname in ("makedirs", "mkdir", "remove"):
                    return NONE
            elif alias_mod == "time":
                if fname in ("time", "perf_counter"):
                    return FLOAT
                if fname == "sleep":
                    return NONE
            elif alias_mod is not None:
                qual = f"{alias_mod}.{fname}"
                if qual in self.env.class_types:
                    return self.env.class_types[qual]
                if qual in self.env.func_types:
                    return self.env.func_types[qual].ret

            # Class constructor call
            if fname in self.env.class_types:
                cls_type = self.env.class_types[fname]
                if "__init__" in cls_type.methods:
                    init_ft = cls_type.methods["__init__"]
                    init_node = next(
                        (
                            m
                            for m in self.class_nodes[fname].body
                            if isinstance(m, ast.FunctionDef) and m.name == "__init__"
                        ),
                        None,
                    )
                    if init_node is not None:
                        init_scope = self.env.class_method_scopes[(fname, "__init__")]
                        non_self = [a for a in _positional_params(init_node) if a.arg != "self"]
                        new_params = self._propagate_into_callable(
                            non_self,
                            init_node.args.vararg.arg if init_node.args.vararg else None,
                            init_scope,
                            list(init_ft.params),
                            arg_types,
                        )
                        methods = dict(cls_type.methods)
                        methods["__init__"] = FuncType(params=new_params, ret=NONE)
                        self.env.class_types[fname] = ClassType(cls_type.name, dict(cls_type.fields), methods)
                        self._infer_class(self.class_nodes[fname])
                return self.env.class_types[fname]

            # User-defined function call
            if fname in self.env.func_types:
                ft = self.env.func_types[fname]
                fn_scope = self.env.func_scopes[fname]
                fn_node = self.func_nodes[fname]
                pos_defs = _positional_params(fn_node)
                new_params = self._propagate_into_callable(
                    pos_defs,
                    fn_node.args.vararg.arg if fn_node.args.vararg else None,
                    fn_scope,
                    list(ft.params),
                    arg_types,
                )
                self.env.func_types[fname] = FuncType(
                    params=new_params, ret=ft.ret, has_vararg=ft.has_vararg
                )
                # A container argument refined inside the callee (dst.append(1))
                # is the caller's object: refine the caller's variable too.
                for a, pt in zip(node.args, new_params):
                    if isinstance(a, ast.Name) and isinstance(pt, (ListType, DictType, SetType)):
                        cur = scope.var_types.get(a.id)
                        if isinstance(cur, type(pt)):
                            scope.var_types[a.id] = unify_container_elem(cur, pt)
                return self.env.func_types[fname].ret

        if isinstance(node.func, ast.Attribute):
            recv_type = self._infer_expr(node.func.value, scope, current_class)
            attr = node.func.attr

            # List methods
            if isinstance(recv_type, ListType):
                if attr == "append" and arg_types:
                    updated = ListType(unify_container_elem(recv_type.elem, arg_types[0]))
                    self._refine_receiver(node.func.value, updated, scope)
                    return NONE
                if attr == "extend" and arg_types:
                    it = self._iterable_element_type(arg_types[0])
                    updated = ListType(unify_container_elem(recv_type.elem, it))
                    self._refine_receiver(node.func.value, updated, scope)
                    return NONE
                if attr == "pop":
                    return recv_type.elem
                if attr in ("sort", "reverse", "clear", "remove"):
                    return NONE
                if attr in ("index", "count"):
                    return INT
                if attr == "copy":
                    return recv_type

            # Set methods
            if isinstance(recv_type, SetType):
                if attr == "add" and arg_types:
                    updated = SetType(unify_container_elem(recv_type.elem, arg_types[0]))
                    self._refine_receiver(node.func.value, updated, scope)
                    return NONE
                if attr == "contains":
                    return BOOL

            # Dict methods
            if isinstance(recv_type, DictType):
                if attr == "get" and arg_types:
                    if len(arg_types) == 1:
                        # No default: Python returns None on missing keys.
                        return ANY
                    return unify_container_elem(recv_type.val, arg_types[1])
                if attr == "keys":
                    return ListType(recv_type.key)
                if attr == "values":
                    return ListType(recv_type.val)
                if attr == "items":
                    return ListType(TupleType((recv_type.key, recv_type.val)))
                if attr == "setdefault" and len(arg_types) == 2:
                    updated = DictType(
                        unify_container_elem(recv_type.key, arg_types[0]),
                        unify_container_elem(recv_type.val, arg_types[1]),
                    )
                    self._refine_receiver(node.func.value, updated, scope)
                    return updated.val
                if attr == "update" and arg_types:
                    other = arg_types[0]
                    if isinstance(other, DictType):
                        updated = DictType(
                            unify_container_elem(recv_type.key, other.key),
                            unify_container_elem(recv_type.val, other.val),
                        )
                        self._refine_receiver(node.func.value, updated, scope)
                    return NONE

            # String methods
            if recv_type == STR:
                if attr in ("upper", "lower", "strip", "lstrip", "rstrip", "replace", "join", "format", "capitalize", "removeprefix", "removesuffix"):
                    return STR
                if attr in ("split", "splitlines"):
                    return ListType(STR)
                if attr in ("startswith", "endswith", "isdigit", "isalpha", "isspace", "isnumeric", "isdecimal", "isalnum"):
                    return BOOL
                if attr in ("find", "rfind", "count", "index"):
                    return INT

            # User class method call
            if isinstance(recv_type, ClassType):
                if isinstance(node.func.value, ast.Name):
                    scope.mutated_vars.add(node.func.value.id)
                latest_cls = self.env.class_types.get(recv_type.name, recv_type)
                if attr in latest_cls.methods:
                    m_ft = latest_cls.methods[attr]
                    defining = self._find_method_node(latest_cls.name, attr)
                    if defining is None:
                        # Synthetic runtime types (e.g. PyFile) carry signatures
                        # but have no analyzable body: return the declared type.
                        return m_ft.ret
                    def_cls, m_node = defining
                    m_scope = self.env.class_method_scopes.get(
                        (latest_cls.name, attr),
                        self.env.class_method_scopes.get((def_cls, attr)),
                    )
                    if m_scope is None:
                        return m_ft.ret
                    non_self = [a for a in _positional_params(m_node) if a.arg != "self"]
                    new_params = self._propagate_into_callable(
                        non_self,
                        m_node.args.vararg.arg if m_node.args.vararg else None,
                        m_scope,
                        list(m_ft.params),
                        arg_types,
                    )
                    methods = dict(latest_cls.methods)
                    methods[attr] = FuncType(params=new_params, ret=m_ft.ret, has_vararg=m_ft.has_vararg)
                    self.env.class_types[latest_cls.name] = ClassType(
                        latest_cls.name, dict(latest_cls.fields), methods
                    )
                    return methods[attr].ret

            # Math, random, json, sys, os, time, or imported module calls
            if isinstance(node.func.value, ast.Name):
                mod_name = node.func.value.id
                if mod_name == "math":
                    if attr in ("floor", "ceil", "factorial", "gcd"):
                        return INT
                    return FLOAT
                if mod_name == "random":
                    return self._infer_random_call(attr, arg_types)
                if mod_name == "sys":
                    if attr == "argv":
                        return ListType(STR)
                    if attr == "exit":
                        return NONE
                if mod_name == "os":
                    if attr == "getenv":
                        return STR if len(arg_types) == 2 and arg_types[1] == STR else ANY
                    if attr == "getcwd":
                        return STR
                    if attr == "listdir":
                        return ListType(STR)
                    if attr in ("makedirs", "mkdir", "remove"):
                        return NONE
                if mod_name == "time":
                    if attr in ("time", "perf_counter"):
                        return FLOAT
                    if attr == "sleep":
                        return NONE
                if mod_name == "json":
                    if attr == "dumps":
                        return STR
                    if attr == "loads":
                        return ANY
                if mod_name == "platform":
                    if attr in (
                        "system", "machine", "processor", "architecture",
                        "python_version", "python_implementation", "node",
                    ):
                        return STR
                if mod_name == "statistics":
                    return self._infer_statistics_call(attr, arg_types)
                if mod_name == "gc":
                    if attr == "collect":
                        return INT
                    if attr == "isenabled":
                        return BOOL
                    if attr in ("disable", "enable"):
                        return NONE
                    if attr == "get_count":
                        return TupleType((INT, INT, INT))
                qual = f"{mod_name}.{attr}"
                if qual in self.env.func_types:
                    return self.env.func_types[qual].ret

        return ANY

    @staticmethod
    def _propagate_into_callable(
        pos_params: list[ast.arg],
        vararg_name: str | None,
        scope: ScopeInfo,
        current_params: list[PyType],
        arg_types: list[PyType],
    ) -> list[PyType]:
        """Unify call-site argument types into callee parameter types (in place)."""
        new_params = list(current_params)
        while len(new_params) < len(pos_params) + (1 if vararg_name else 0):
            new_params.append(UNKNOWN)
        for i, (arg_def, at) in enumerate(zip(pos_params, arg_types)):
            if at is None:
                continue
            merged = unify_types(new_params[i], at)
            new_params[i] = merged
            scope.var_types[arg_def.arg] = merged
        if vararg_name is not None and len(arg_types) > len(pos_params):
            elem = new_params[-1].elem if new_params and isinstance(new_params[-1], ListType) else UNKNOWN
            for at in arg_types[len(pos_params):]:
                if at is not None:
                    elem = unify_container_elem(elem, at)
            new_params[-1] = ListType(elem)
            scope.var_types[vararg_name] = ListType(elem)
        return new_params

    @staticmethod
    def _infer_statistics_call(func_name: str, arg_types: list[PyType]) -> PyType:
        if func_name in ("mean", "variance", "pvariance"):
            return ANY  # int when the exact result is integral, else float
        if func_name in ("harmonic_mean", "stdev", "pstdev"):
            return FLOAT
        if func_name == "mode" and arg_types and isinstance(arg_types[0], ListType):
            elem = arg_types[0].elem
            return elem if elem != UNKNOWN else ANY
        if func_name in ("median", "median_low", "median_high", "mode"):
            return ANY
        return ANY

    @staticmethod
    def _infer_random_call(func_name: str, arg_types: list[PyType]) -> PyType:
        if func_name in ("seed", "shuffle"):
            return NONE
        if func_name == "randint":
            return INT
        if func_name in ("random", "uniform"):
            return FLOAT
        if func_name == "choice" and arg_types:
            seq_t = arg_types[0]
            if isinstance(seq_t, (ListType, SetType)):
                return seq_t.elem
            if seq_t == STR:
                return STR
            if isinstance(seq_t, TupleType) and seq_t.elements:
                elem: PyType = UNKNOWN
                for e in seq_t.elements:
                    elem = unify_container_elem(elem, e)
                return elem
        return ANY


def infer_module(parsed: ParsedModule) -> ModuleTypeEnv:
    inf = _TypeInferer(parsed)
    env = inf.run()
    inf._finalize_unknown_params()
    return env
