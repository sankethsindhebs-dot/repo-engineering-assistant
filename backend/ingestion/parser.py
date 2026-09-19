"""Extract declarations, bindings and reference sites without importing target code."""

import ast
from collections import defaultdict
from dataclasses import dataclass, field

from backend.ingestion.models import Entity, Reference, Relationship, stable_id


@dataclass
class Binding:
    line: int
    target_id: str | None = None
    import_site: str | None = None
    unsafe: bool = False


@dataclass
class Scope:
    entity: Entity
    parent: str | None
    bindings: dict[str, list[Binding]] = field(default_factory=lambda: defaultdict(list))
    wildcard: bool = False


@dataclass
class Site:
    reference: Reference
    scope_id: str
    expression: ast.expr | None = None
    import_module: str = ""
    import_name: str | None = None
    level: int = 0
    bind_root: bool = False
    blocked_reason: str = ""


class Extractor(ast.NodeVisitor):
    def __init__(self, module: Entity):
        self.entities = {module.id: module}
        self.scopes = {module.id: Scope(module, None)}
        self.scope = self.scopes[module.id]
        self.sites: list[Site] = []
        self.edges: list[Relationship] = []
        self.unsafe_targets: set[str] = set()
        self.attribute_mutations: list[tuple[str, ast.Attribute]] = []
        self.module = module
        self.conditional = 0
        self.blocked_reason = ""
        self.occurrences: dict[tuple[str, str, str], int] = defaultdict(int)

    def record(self, kind: str, node: ast.AST, owner: str | None = None,
               expression: ast.expr | None = None, suffix: str = "") -> Site:
        text = ast.get_source_segment(self.module.source, node) or ast.unparse(node)
        owner = owner or self.scope.entity.id
        reference = Reference(
            id=stable_id("reference-v1", owner, kind, node.lineno, node.col_offset,
                         node.end_lineno, node.end_col_offset, suffix),
            owner_id=owner, kind=kind, path=self.module.path, expression=text,
            start_line=node.lineno, end_line=node.end_lineno or node.lineno,
            start_column=node.col_offset, end_column=node.end_col_offset or 0,
        )
        site = Site(reference, self.scope.entity.id, expression, blocked_reason=self.blocked_reason)
        self.sites.append(site)
        return site

    def bind_unknown(self, name: str, line: int) -> None:
        self.scope.bindings[name].append(Binding(line, unsafe=True))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bind_unknown(node.id, node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        self.record("CALLS", node, expression=node.func)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.attribute_mutations.append((self.scope.entity.id, node))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            site = self.record("IMPORTS", alias, suffix=alias.name)
            site.import_module = alias.name
            site.bind_root = alias.asname is None and "." in alias.name
            name = alias.asname or alias.name.split(".")[0]
            self.scope.bindings[name].append(Binding(node.lineno, import_site=site.reference.id,
                                                     unsafe=bool(self.conditional)))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            site = self.record("IMPORTS", node, suffix=f"{alias.name}:{alias.lineno}:{alias.col_offset}")
            site.import_module, site.import_name, site.level = node.module or "", alias.name, node.level
            if alias.name == "*":
                self.scope.wildcard = True
            else:
                self.scope.bindings[alias.asname or alias.name].append(
                    Binding(node.lineno, import_site=site.reference.id, unsafe=bool(self.conditional))
                )

    def definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        parent = self.scope
        kind = "Class" if isinstance(node, ast.ClassDef) else (
            "Method" if parent.entity.kind == "Class" else "Function"
        )
        key = (parent.entity.id, kind, node.name)
        occurrence = self.occurrences[key]
        self.occurrences[key] += 1
        qualifier = ".<locals>." if parent.entity.kind in {"Function", "Method"} else "."
        qualified_name = parent.entity.qualified_name + qualifier + node.name
        start = min([node.lineno] + [item.lineno for item in node.decorator_list])
        end = node.end_lineno or node.lineno
        lines = self.module.source.split("\n")
        source = "\n".join(lines[start - 1:end])
        if end < len(lines):
            source += "\n"
        entity = Entity(
            stable_id("entity-v1", parent.entity.id, kind, node.name, occurrence),
            self.module.repository_id, self.module.path, kind, node.name, qualified_name,
            start, end, self.module.source_hash, source, self.module.module_name,
        )
        self.entities[entity.id] = entity
        parent.bindings[node.name].append(Binding(end, entity.id, unsafe=bool(self.conditional)))
        self.edges.append(Relationship(stable_id("contains-v1", parent.entity.id, entity.id),
                                       parent.entity.id, entity.id, "CONTAINS"))
        if node.decorator_list or getattr(node, "keywords", []) or getattr(node, "type_params", []):
            self.unsafe_targets.add(entity.id)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                self.record("INHERITS", base, entity.id, base)
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword.value)
        else:
            for value in node.args.defaults + [v for v in node.args.kw_defaults if v is not None]:
                self.visit(value)
            for arg in ast.walk(node.args):
                if isinstance(arg, ast.arg) and arg.annotation:
                    self.visit_blocked(arg.annotation, "annotation_context")
            if node.returns:
                self.visit_blocked(node.returns, "annotation_context")
        for parameter in getattr(node, "type_params", []):
            self.visit_blocked(parameter, "type_parameter_scope")
        child = Scope(entity, parent.entity.id)
        self.scopes[entity.id] = child
        old_conditional = self.conditional
        self.scope, self.conditional = child, 0
        for parameter in getattr(node, "type_params", []):
            self.bind_unknown(parameter.name, node.lineno)
        if not isinstance(node, ast.ClassDef):
            for arg in ast.walk(node.args):
                if isinstance(arg, ast.arg):
                    self.bind_unknown(arg.arg, node.lineno)
        for statement in node.body:
            self.visit(statement)
        self.scope, self.conditional = parent, old_conditional

    visit_FunctionDef = definition
    visit_AsyncFunctionDef = definition
    visit_ClassDef = definition

    def visit_blocked(self, node: ast.AST, reason: str) -> None:
        previous = self.blocked_reason
        self.blocked_reason = reason
        self.visit(node)
        self.blocked_reason = previous

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # Preserve every call but do not pretend these implicit scopes are analysed.
        self.visit_blocked_children(node, "lambda_scope")

    def visit_blocked_children(self, node: ast.AST, reason: str) -> None:
        previous = self.blocked_reason
        self.blocked_reason = reason
        self.generic_visit(node)
        self.blocked_reason = previous

    def comprehension(self, node: ast.AST) -> None:
        self.visit_blocked_children(node, "comprehension_scope")

    visit_ListComp = comprehension
    visit_SetComp = comprehension
    visit_DictComp = comprehension
    visit_GeneratorExp = comprehension

    def conditional_block(self, node: ast.AST) -> None:
        self.conditional += 1
        self.generic_visit(node)
        self.conditional -= 1

    visit_If = conditional_block
    visit_For = conditional_block
    visit_AsyncFor = conditional_block
    visit_While = conditional_block
    visit_Try = conditional_block
    visit_TryStar = conditional_block
    visit_With = conditional_block
    visit_AsyncWith = conditional_block
    visit_Match = conditional_block

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bind_unknown(node.name, node.lineno)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global | ast.Nonlocal) -> None:
        # A declaration can change bindings outside this scope; invalidate all ancestors.
        scope = self.scope
        while scope:
            for name in node.names:
                scope.bindings[name].append(Binding(node.lineno, unsafe=True))
            scope = self.scopes.get(scope.parent)

    visit_Nonlocal = visit_Global

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.bind_unknown(node.name, node.lineno)
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.bind_unknown(node.rest, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        self.visit_blocked(node.annotation, "annotation_context")
        if node.value:
            self.visit(node.value)

    def visit_TypeAlias(self, node: ast.AST) -> None:
        self.visit_blocked_children(node, "type_alias_scope")
