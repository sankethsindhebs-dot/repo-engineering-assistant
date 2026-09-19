"""Resolve only unique lexical definitions and explicit imports of indexed Python code."""

import ast
from dataclasses import replace

from backend.ingestion.models import Reference, Relationship, stable_id
from backend.ingestion.parser import Extractor, Scope, Site


class Resolver:
    def __init__(self, extractors: list[Extractor]):
        self.entities = {key: value for item in extractors for key, value in item.entities.items()}
        self.scopes = {key: value for item in extractors for key, value in item.scopes.items()}
        self.sites = {site.reference.id: site for item in extractors for site in item.sites}
        self.modules = {item.module.module_name: item.module.id for item in extractors
                        if not item.module.module_name.startswith("<file:")}
        self.unsafe = set().union(*(item.unsafe_targets for item in extractors))
        self.mutated_bindings: set[tuple[str, str]] = set()
        mutations = set()
        # Identify receivers from declared imports before applying invalidations,
        # so file visitation order cannot change the result. Runtime objects do
        # not acquire module identity merely by sharing an attribute name.
        for item in extractors:
            for scope_id, node in item.attribute_mutations:
                target, _ = self.expression(node.value, scope_id, node.lineno)
                if target and self.entities[target].kind == "Module":
                    module_name = self.entities[target].module_name
                else:
                    # An explicit import also identifies its namespace prefixes,
                    # even when the package has no indexed __init__.py node.
                    module_name = self.explicit_import_path(node.value, scope_id, node.lineno, prefixes=True)
                if module_name is not None:
                    mutations.add((module_name, node.attr))
        self.mutated_bindings = mutations

    def import_target(self, site: Site, seen: frozenset[str]) -> tuple[str | None, str]:
        if site.reference.id in seen:
            return None, "cyclic_import_or_alias"
        seen = seen | {site.reference.id}
        module = site.import_module
        if site.level:
            owner = self.entities[site.scope_id]
            if not owner.module_name or owner.module_name.startswith("<file:"):
                return None, "relative_import_without_package"
            package = owner.module_name.split(".")
            if not owner.path.endswith("/__init__.py") and owner.path != "__init__.py":
                package = package[:-1]
            if site.level > len(package):
                return None, "relative_import_outside_source_root"
            prefix = package[:len(package) - site.level + 1]
            module = ".".join(prefix + ([module] if module else []))
        if site.import_name == "*":
            return None, "wildcard_import"
        if site.import_name is None:
            return self.modules.get(module), "local_module" if module in self.modules else "external_or_unindexed"
        module_id = self.modules.get(module)
        if (module, site.import_name) in self.mutated_bindings:
            return None, "attribute_rebinding_present"
        # Explicit attributes take priority over a same-named submodule. Ambiguity
        # is preserved rather than falling through to a plausible module name.
        if module_id and (site.import_name in self.scopes[module_id].bindings or self.scopes[module_id].wildcard):
            return self.binding(self.scopes[module_id], site.import_name, None, seen)
        submodule = ".".join(filter(None, [module, site.import_name]))
        if submodule in self.modules:
            return self.modules[submodule], "local_submodule"
        return None, "external_or_unindexed"

    def binding(self, scope: Scope, name: str, line: int | None,
                seen: frozenset[str]) -> tuple[str | None, str]:
        values = scope.bindings.get(name, [])
        if scope.wildcard:
            return None, "wildcard_scope"
        if scope.entity.kind == "Module" and (scope.entity.module_name, name) in self.mutated_bindings:
            return None, "attribute_rebinding_present"
        if len(values) != 1 or values[0].unsafe:
            return None, "ambiguous_or_runtime_binding"
        value = values[0]
        if line is not None and value.line >= line:
            return None, "binding_not_yet_available"
        if value.import_site:
            site = self.sites[value.import_site]
            if site.bind_root:
                root = site.import_module.split(".")[0]
                return self.modules.get(root), "local_import_root" if root in self.modules else "namespace_root_unindexed"
            return self.import_target(site, seen)
        if value.target_id in self.unsafe:
            return None, "decorated_or_metaprogrammed_definition"
        return value.target_id, "unique_lexical_definition"

    def lookup(self, scope_id: str, name: str, line: int) -> tuple[str | None, str]:
        scope = self.scopes[scope_id]
        deferred = False
        first = True
        while scope:
            if first or scope.entity.kind != "Class":
                if name in scope.bindings or scope.wildcard:
                    return self.binding(scope, name, None if deferred else line, frozenset())
            if scope.entity.kind in {"Function", "Method"}:
                deferred = True
            scope = self.scopes.get(scope.parent)
            first = False
        return None, "unknown_or_external_name"

    def expression(self, node: ast.expr, scope_id: str, line: int) -> tuple[str | None, str]:
        if isinstance(node, ast.Name):
            return self.lookup(scope_id, node.id, line)
        if isinstance(node, ast.Attribute):
            imported = self.modules.get(self.explicit_import_path(node, scope_id, line))
            if imported:
                return imported, "explicit_local_submodule_import"
            target, reason = self.expression(node.value, scope_id, line)
            if target and self.entities[target].kind == "Module":
                scope = self.scopes[target]
                if node.attr in scope.bindings or scope.wildcard:
                    return self.binding(scope, node.attr, None, frozenset())
                # Do not infer package submodule attributes without a binding.
                return None, "module_attribute_unbound"
            return None, "dynamic_attribute_or_dispatch" if target else reason
        return None, "runtime_expression"

    def explicit_import_path(self, node: ast.expr, scope_id: str, line: int,
                             prefixes: bool = False) -> str | None:
        """Identify an explicit dotted import, or its prefixes for mutation tracking."""
        parts = []
        root = node
        while isinstance(root, ast.Attribute):
            parts.insert(0, root.attr)
            root = root.value
        if not isinstance(root, ast.Name):
            return None
        parts.insert(0, root.id)
        module_name = ".".join(parts)
        scope = self.scopes[scope_id]
        deferred, first = False, True
        while scope:
            if first or scope.entity.kind != "Class":
                if scope.wildcard:
                    return None
                if root.id in scope.bindings:
                    if scope.entity.kind == "Module" and (scope.entity.module_name, root.id) in self.mutated_bindings:
                        return None
                    values = scope.bindings[root.id]
                    if len(values) != 1 or values[0].unsafe:
                        return None
                    binding = values[0]
                    if not deferred and binding.line >= line:
                        return None
                    imported = self.sites.get(binding.import_site)
                    if imported and imported.bind_root and imported.import_module in self.modules:
                        matches = imported.import_module == module_name or (
                            prefixes and imported.import_module.startswith(module_name + ".")
                        )
                        mutated = any((".".join(parts[:index]), part) in self.mutated_bindings
                                      for index, part in enumerate(parts[1:], 1))
                        if matches and not mutated:
                            return module_name
                    return None
            deferred = deferred or scope.entity.kind in {"Function", "Method"}
            first = False
            scope = self.scopes.get(scope.parent)
        return None

    def resolve(self) -> tuple[list[Reference], list[Relationship]]:
        references, edges = [], []
        for site in self.sites.values():
            if site.blocked_reason:
                target, reason = None, site.blocked_reason
            elif site.reference.kind == "IMPORTS":
                target, reason = self.import_target(site, frozenset())
            else:
                target, reason = self.expression(site.expression, site.scope_id, site.reference.start_line)
                allowed = {"Class"} if site.reference.kind == "INHERITS" else {"Function", "Class"}
                if target and self.entities[target].kind not in allowed:
                    target, reason = None, "target_kind_not_statically_callable"
            reference = replace(site.reference, target_id=target, reason=reason)
            references.append(reference)
            if target:
                edges.append(Relationship(stable_id("reference-edge-v1", reference.id, target),
                                          reference.owner_id, target, reference.kind, reference.id))
        return references, edges
