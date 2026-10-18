"""Repository file tree traversal, filtering, and text extraction."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator
from pathlib import Path

from repopilot.ingestion.models import SourceFile

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".xml": "xml",
    ".dockerfile": "dockerfile",
}

SPECIAL_NAMES: dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}

BINARY_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tiff", ".webp",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib",
    ".pyc", ".pyo", ".pyd", ".wasm",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".parquet", ".db", ".sqlite", ".sqlite3",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
}

DEFAULT_SKIP_PATTERNS: list[str] = [
    ".git",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".venv",
    "*.lock",
    "package-lock.json",
    "poetry.lock",
    "Cargo.lock",
    "yarn.lock",
    "pnpm-lock.yaml",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.pyc",
]


def detect_language(path: Path) -> str:
    """Detect language from file extension or special file name."""
    name_lower = path.name.lower()
    if name_lower in SPECIAL_NAMES:
        return SPECIAL_NAMES[name_lower]
    ext = path.suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, ext.lstrip(".") if ext else "text")


def is_pattern_match(name: str, rel_path: str, patterns: list[str]) -> bool:
    """Check if file/directory name or its relative path matches any pattern."""
    for pattern in patterns:
        pat = pattern.rstrip("/\\")
        if name == pat or fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_path, pat):
            return True
    return False


def is_binary_file(file_path: Path) -> bool:
    """Check if file is binary by extension or content inspection."""
    if file_path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with file_path.open("rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
    except OSError:
        return True
    return False


def walk_repo_files(
    repo_path: Path,
    *,
    max_file_size_bytes: int = 1_000_000,
    skip_patterns: list[str] | None = None,
) -> Iterator[SourceFile]:
    """Traverse a repository and yield valid text source files.

    Skips .git directories, binary files, lockfiles, vendored directories,
    and files exceeding max_file_size_bytes.

    Args:
        repo_path: Root path of the repository to walk.
        max_file_size_bytes: Maximum allowed file size in bytes.
        skip_patterns: Glob patterns or directory names to ignore.

    Yields:
        :class:`SourceFile` instances for discovered text files.
    """
    repo_path = Path(repo_path).resolve()
    patterns = skip_patterns if skip_patterns is not None else DEFAULT_SKIP_PATTERNS

    for root, dirnames, filenames in os.walk(repo_path, topdown=True):
        root_path = Path(root)
        try:
            rel_root = root_path.relative_to(repo_path)
            rel_root_posix = rel_root.as_posix()
        except ValueError:
            rel_root_posix = ""

        # Prune ignored directories in-place to avoid recursing into them
        pruned_dirs: list[str] = []
        for d in dirnames:
            rel_dir_posix = f"{rel_root_posix}/{d}".lstrip("/")
            if is_pattern_match(d, rel_dir_posix, patterns):
                pruned_dirs.append(d)
        for d in pruned_dirs:
            dirnames.remove(d)

        for filename in filenames:
            file_path = root_path / filename
            try:
                rel_file = file_path.relative_to(repo_path)
                rel_file_posix = rel_file.as_posix()
            except ValueError:
                rel_file_posix = filename

            # 1. Skip pattern check
            if is_pattern_match(filename, rel_file_posix, patterns):
                continue

            # 2. Size check
            try:
                st = file_path.stat()
                if st.st_size > max_file_size_bytes or st.st_size == 0:
                    continue
                size_bytes = st.st_size
            except OSError:
                continue

            # 3. Binary check
            if is_binary_file(file_path):
                continue

            # 4. Read UTF-8 text content
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            language = detect_language(file_path)
            yield SourceFile(
                path=file_path,
                relative_path=rel_file_posix,
                content=content,
                language=language,
                size_bytes=size_bytes,
            )
