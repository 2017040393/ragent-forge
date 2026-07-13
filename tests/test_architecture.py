import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "ragent_forge"


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules


def _assert_layer_avoids(layer: str, forbidden_prefixes: tuple[str, ...]) -> None:
    violations: list[str] = []
    for path in sorted((PACKAGE_ROOT / layer).rglob("*.py")):
        for module in _imported_modules(path):
            if module.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)} -> {module}")
    assert violations == []


def test_core_does_not_depend_on_application_or_presentation_layers() -> None:
    _assert_layer_avoids(
        "core",
        (
            "ragent_forge.app",
            "ragent_forge.cli",
            "ragent_forge.config",
            "ragent_forge.tui",
        ),
    )


def test_application_does_not_depend_on_presentation_layers() -> None:
    _assert_layer_avoids(
        "app",
        (
            "ragent_forge.cli",
            "ragent_forge.tui",
        ),
    )
