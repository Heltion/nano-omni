"""Verify bilingual documentation and its Python source inventories."""

import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DOCS = (ROOT / "docs/en/scripts.md", ROOT / "docs/zh/scripts.md")
CORE_DOC_DIRS = (ROOT / "docs/en/core", ROOT / "docs/zh/core")


def documentation_pairs() -> list[tuple[Path, Path]]:
    english = ROOT / "docs/en"
    chinese = ROOT / "docs/zh"
    names = {path.relative_to(english).as_posix() for path in english.rglob("*.md")} | {
        path.relative_to(chinese).as_posix() for path in chinese.rglob("*.md")
    }
    pairs = [(english / name, chinese / name) for name in sorted(names)]
    # Discover every existing source-note pair, including renamed and study
    # instances. A stale hard-coded instance list silently misses new notes.
    source_directories = {
        path.parent
        for name in ("SOURCES.md", "SOURCES.zh.md")
        for path in (ROOT / "perf").rglob(name)
    }
    pairs.extend(
        (directory / "SOURCES.md", directory / "SOURCES.zh.md")
        for directory in sorted(source_directories)
    )
    return pairs


def linked_python_files(document: Path) -> list[str]:
    links = re.findall(r"]\(([^)#]+\.py)\)", document.read_text(encoding="utf-8"))
    return [
        (document.parent / link).resolve().relative_to(ROOT).as_posix()
        for link in links
    ]


def verify_inventory(
    documents: tuple[tuple[Path, ...], tuple[Path, ...]], expected: set[str]
) -> None:
    inventories = [
        [link for document in language for link in linked_python_files(document)]
        for language in documents
    ]
    for language, inventory in zip(documents, inventories, strict=True):
        counts = Counter(inventory)
        missing = expected - counts.keys()
        unexpected = counts.keys() - expected
        repeated = {path for path, count in counts.items() if count != 1}
        if missing or unexpected or repeated:
            raise ValueError(
                f"Invalid Python inventory in {language}: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
                f"repeated={sorted(repeated)}"
            )
    if inventories[0] != inventories[1]:
        raise ValueError(f"English and Chinese inventories differ: {documents}")


def main() -> None:
    changed = {
        ROOT / path
        for path in subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True
        ).splitlines()
    }
    for english, chinese in documentation_pairs():
        if not english.is_file() or not chinese.is_file():
            raise FileNotFoundError(
                f"Missing bilingual counterpart: {english}, {chinese}"
            )
        english_link = Path(os.path.relpath(chinese, english.parent)).as_posix()
        chinese_link = Path(os.path.relpath(english, chinese.parent)).as_posix()
        if f"]({english_link})" not in english.read_text(encoding="utf-8"):
            raise ValueError(f"Missing Chinese link in {english}")
        if f"]({chinese_link})" not in chinese.read_text(encoding="utf-8"):
            raise ValueError(f"Missing English link in {chinese}")
        if (english in changed) != (chinese in changed):
            raise ValueError(
                f"Stage both language versions together: {english}, {chinese}"
            )

    scripts = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").rglob("*.py")
        if ROOT / "scripts/experiment" not in path.parents
    }
    core = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/nano_omni/core").rglob("*.py")
    }
    verify_inventory(((SCRIPT_DOCS[0],), (SCRIPT_DOCS[1],)), scripts)
    verify_inventory(
        (
            tuple(sorted(CORE_DOC_DIRS[0].rglob("*.md"))),
            tuple(sorted(CORE_DOC_DIRS[1].rglob("*.md"))),
        ),
        core,
    )


if __name__ == "__main__":
    main()
