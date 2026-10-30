import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "riftcount"
TESTS = ROOT / "tests"
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".arena-riftcount-tmp"}
PRIVATE = {"riftcount._cardvision", "riftcount._media"}


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def _imported(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            for alias in node.names:
                found.add(f"{node.module}.{alias.name}")
    return found


def test_private_modules_are_not_imported_outside_the_package() -> None:
    offenders: list[str] = []
    for path in _python_files(ROOT):
        if path.is_relative_to(PACKAGE) or path.is_relative_to(TESTS):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hit = sorted(name for name in _imported(tree) if name in PRIVATE or name.startswith("riftcount._"))
        if hit:
            offenders.append(f"{path.relative_to(ROOT)}: {', '.join(hit)}")
    assert offenders == []


def test_package_does_not_import_spike() -> None:
    offenders: list[str] = []
    for path in _python_files(PACKAGE):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported(tree):
            if name == "spike" or name.startswith("spike."):
                offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert offenders == []
