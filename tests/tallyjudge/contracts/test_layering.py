import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[3] / "tallyjudge"
ALLOWED = {
    "sdk": {"sdk"},
    "engine": {"sdk", "engine"},
    "registry": {"sdk", "registry"},
    "plugins": {"sdk", "plugins"},
}
RUN_DRIVERS = {"optimization"}
PACKS = {"metrics", "synthesis", "optimization"}
ADAPTERS = {
    "models",
    "embeddings",
    "storage",
    "cache",
    "tracing",
    "adapters",
    "versioning",
}
DOMAIN_FIELDS = (
    "user_input",
    "retrieved_contexts",
    "retrieved_context_ids",
    "reference_contexts",
    "reference_context_ids",
    "chunk_id",
    "testset_size",
    "holdout",
)


def _imported(path: Path) -> set[tuple[str, ...]]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        for name in names:
            parts = tuple(name.split("."))
            if parts[0] == "tallyjudge":
                modules.add(parts[1:])
    return modules


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_imports_only_the_layers_below_it(layer):
    for path in (PACKAGE / layer).rglob("*.py"):
        allowed = set(ALLOWED[layer])
        if layer == "plugins" and path.relative_to(PACKAGE).parts[1] in RUN_DRIVERS:
            allowed.add("engine")
        reached = {module[0] for module in _imported(path) if module}
        assert reached <= allowed, f"{path.relative_to(PACKAGE)} imports {reached}"


def test_packs_do_not_import_adapters():
    for pack in PACKS:
        for path in (PACKAGE / "plugins" / pack).rglob("*.py"):
            reached = {
                module[1]
                for module in _imported(path)
                if len(module) > 1 and module[0] == "plugins"
            }
            assert not reached & ADAPTERS, f"{path.relative_to(PACKAGE)}: {reached}"


@pytest.mark.parametrize("layer", ["sdk", "engine"])
def test_kernel_names_no_domain_fields(layer):
    for path in (PACKAGE / layer).rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        found = [field for field in DOMAIN_FIELDS if field in source]
        assert not found, f"{path.relative_to(PACKAGE)} names {found}"
