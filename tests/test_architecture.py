"""Enforce the package layering described in docs/architecture/overview.md."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "aeris"
COMPONENTS = {"agents", "environments", "data", "experiments"}
ALLOWED = {
    "domain": set(),
    "contracts": {"domain"},
    **{name: {"domain", "contracts"} for name in COMPONENTS},
    "analysis": {"domain", "contracts", "agents", "environments"},
    "visualization": {"domain", "contracts", "agents", "environments", "analysis"},
    "workflows": {"domain", "contracts", "analysis", "visualization", *COMPONENTS},
    "cli": {"domain", "data", "workflows"},
}


def imports(path: Path) -> set[str]:
    """Every module imported anywhere in a file, including lazy and TYPE_CHECKING imports."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def package_files() -> list[tuple[str, Path]]:
    return [
        (path.relative_to(SRC).parts[0], path) for path in SRC.rglob("*.py") if path.parent != SRC
    ]


def test_every_package_is_covered():
    assert {package for package, _ in package_files()} == set(ALLOWED)


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_packages_import_only_lower_layers(package):
    violations = [
        f"{path.relative_to(SRC)} imports aeris.{module.split('.')[1]}"
        for owner, path in package_files()
        if owner == package
        for module in imports(path)
        if module.startswith("aeris.") and module.split(".")[1] not in ALLOWED[package] | {package}
    ]
    assert not violations, "\n".join(violations)


def test_optional_dependencies_stay_where_they_belong():
    allowed = {
        "numpy": lambda package, path: package not in {"domain", "contracts", "cli"},
        "torch": lambda package, path: (
            package == "agents" or path.name in {"four_rooms_training.py", "_runtime.py"}
        ),
        "matplotlib": lambda package, path: package == "visualization",
    }
    violations = [
        f"{path.relative_to(SRC)} imports {module}"
        for package, path in package_files()
        for module in imports(path)
        if module.split(".")[0] in allowed and not allowed[module.split(".")[0]](package, path)
    ]
    assert not violations, "\n".join(violations)
