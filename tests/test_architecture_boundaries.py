import ast
from pathlib import Path

import ragent_forge

PACKAGE_ROOT = Path(ragent_forge.__file__).parent
COMPATIBILITY_FACADES = {
    PACKAGE_ROOT / "app" / "composition.py",
    PACKAGE_ROOT / "app" / "storage.py",
    PACKAGE_ROOT / "app" / "workspace.py",
    PACKAGE_ROOT / "app" / "services" / "embedding_service.py",
    PACKAGE_ROOT / "app" / "services" / "generation_service.py",
    PACKAGE_ROOT / "app" / "services" / "retrieval_eval_service.py",
    PACKAGE_ROOT / "app" / "services" / "text_generation_client.py",
}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def test_application_and_core_do_not_import_infrastructure_adapters() -> None:
    violations: list[str] = []
    for layer in ("app", "core"):
        for path in (PACKAGE_ROOT / layer).rglob("*.py"):
            if path in COMPATIBILITY_FACADES:
                continue
            for module in _imports(path):
                if module == "ragent_forge.infrastructure" or module.startswith(
                    "ragent_forge.infrastructure."
                ):
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)} -> {module}")

    assert violations == []


def test_core_does_not_import_application_layer() -> None:
    violations = [
        f"{path.relative_to(PACKAGE_ROOT)} -> {module}"
        for path in (PACKAGE_ROOT / "core").rglob("*.py")
        for module in _imports(path)
        if module == "ragent_forge.app" or module.startswith("ragent_forge.app.")
    ]

    assert violations == []


def test_composition_root_owns_concrete_adapter_selection() -> None:
    composition_imports = _imports(PACKAGE_ROOT / "composition.py")
    assert "ragent_forge.infrastructure.http_client" in composition_imports
    assert "ragent_forge.infrastructure.local_workspace" in composition_imports
    assert "ragent_forge.infrastructure.providers.embedding" in composition_imports
    assert (
        "ragent_forge.infrastructure.providers.openai_generation"
        in composition_imports
    )


def test_legacy_import_paths_are_small_compatibility_facades() -> None:
    for path in COMPATIBILITY_FACADES:
        assert path.is_file()
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 80


def test_presentation_and_eval_responsibilities_have_focused_modules() -> None:
    expected_modules = (
        PACKAGE_ROOT / "cli" / "parser.py",
        PACKAGE_ROOT / "cli" / "handlers" / "workspace.py",
        PACKAGE_ROOT / "cli" / "handlers" / "chunks.py",
        PACKAGE_ROOT / "cli" / "handlers" / "config.py",
        PACKAGE_ROOT / "cli" / "handlers" / "traces.py",
        PACKAGE_ROOT / "cli" / "handlers" / "index.py",
        PACKAGE_ROOT / "cli" / "handlers" / "retrieval.py",
        PACKAGE_ROOT / "cli" / "handlers" / "evaluation.py",
        PACKAGE_ROOT / "tui" / "controllers" / "workers.py",
        PACKAGE_ROOT / "tui" / "controllers" / "session.py",
        PACKAGE_ROOT / "app" / "services" / "evaluation" / "contracts.py",
        PACKAGE_ROOT / "app" / "services" / "evaluation" / "cases.py",
        PACKAGE_ROOT / "app" / "services" / "evaluation" / "metrics.py",
        PACKAGE_ROOT / "app" / "services" / "evaluation" / "reporting.py",
        PACKAGE_ROOT / "app" / "services" / "evaluation" / "runner.py",
    )
    assert all(path.is_file() for path in expected_modules)

    cli_tree = ast.parse(
        (PACKAGE_ROOT / "cli" / "__init__.py").read_text(encoding="utf-8")
    )
    cli_functions = {
        node.name for node in cli_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert cli_functions == {"main"}
