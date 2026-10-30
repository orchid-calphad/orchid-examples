"""Repository hygiene checks for the examples repository.

Checks the files that git tracks (or every file outside a git checkout) for: no absolute local paths, no file over
50 MB, a total size under 100 MB, PDFs only inside a ``figures/`` folder, no image or document types, and the basic
rules of ``.zenodo.json`` and ``CITATION.cff``.

Usage::

    python tests/check_hygiene.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
mib = 2**20  # bytes per mebibyte, used for the size limits and the report
max_file_bytes = 50 * mib  # a single tracked file may not exceed 50 MB
max_total_bytes = 100 * mib  # all tracked files together may not exceed 100 MB
text_suffixes = (".py", ".sh", ".md", ".json", ".yml", ".yaml", ".txt", ".csv", ".cff", ".ipynb", ".tdb", ".TDB")
scan_exempt = {"tests/check_hygiene.py", "tests/test_repo_checks.py"}  # files that contain fake paths on purpose
image_suffixes = (".jpg", ".jpeg", ".tif", ".tiff", ".djvu", ".epub")
local_path_pattern = re.compile(r"(/Users/[A-Za-z]|/home/[a-z]|/private/tmp/|C:\\{1,2}Users|~/projects/|\.venvs/)")


def tracked_files(root: Path = ROOT) -> list[str]:
    """List the repository files to check.

    Args:
        root (Path): Repository root.

    Returns:
        list[str]: Posix paths relative to ``root``, from ``git ls-files``, or from a directory walk when the
        directory is not a git checkout.
    """
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True)
        files = out.stdout.split()
    except (OSError, subprocess.CalledProcessError):
        files = []
    if not files:
        files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    return files


def file_problems(rel: str, root: Path = ROOT) -> list[str]:
    """Check one file's name, size and, for text types, its contents for local absolute paths.

    Args:
        rel (str): Posix path relative to ``root``.
        root (Path): Repository root.

    Returns:
        list[str]: One message per problem found; empty when the file passes.
    """
    path = root / rel
    size = path.stat().st_size
    problems = []
    if size > max_file_bytes:
        problems.append(f"{rel}: {size / mib:.1f} MB > 50 MB")
    if rel.lower().endswith(".pdf") and "/figures/" not in "/" + rel:
        problems.append(f"{rel}: PDF outside a figures/ folder")
    if rel.lower().endswith(image_suffixes):
        problems.append(f"{rel}: image/document type not expected")
    if rel.endswith(text_suffixes) and rel not in scan_exempt:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if local_path_pattern.search(line):
                    problems.append(f"{rel}:{i}: local absolute path: {line.strip()[:120]}")
    return problems


def metadata_problems(root: Path = ROOT) -> list[str]:
    """Check the ``.zenodo.json`` and ``CITATION.cff`` rules (no grants, no funding key).

    Args:
        root (Path): Repository root.

    Returns:
        list[str]: One message per violated rule.
    """
    problems = []
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    if "grants" in zenodo:
        problems.append(".zenodo.json has 'grants'")
    if re.search(r"^funding:", (root / "CITATION.cff").read_text(encoding="utf-8"), re.M):
        problems.append("CITATION.cff has a 'funding' key")
    return problems


def run(root: Path = ROOT) -> int:
    """Run every check, print the summary and return the exit status.

    Args:
        root (Path): Repository root.

    Returns:
        int: 0 when every check passes, 1 otherwise.
    """
    files = tracked_files(root)
    problems: list[str] = []
    total = 0
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        total += path.stat().st_size
        problems.extend(file_problems(rel, root))
    if total > max_total_bytes:
        problems.append(f"repository {total / mib:.1f} MB > 100 MB")
    problems.extend(metadata_problems(root))
    print(f"{len(files)} files, {total / mib:.1f} MB")
    print("\n".join(problems) if problems else "hygiene ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(run())
