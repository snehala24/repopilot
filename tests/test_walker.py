"""Unit tests for repository file walker skipping and language detection."""

from pathlib import Path

from repopilot.ingestion.walker import walk_repo_files


def test_file_walker_skips_and_language_detection(tmp_path: Path):
    """Verify walker skips ignored directories, lockfiles, binaries, and oversized files."""
    # 1. Valid source files
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "main.py").write_text("print('hello')", encoding="utf-8")
    (src_dir / "util.ts").write_text("export const x = 1;", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Repo", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM alpine", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("key: value", encoding="utf-8")

    # 2. Ignored directories
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding="utf-8")

    node_modules = tmp_path / "node_modules" / "some-pkg"
    node_modules.mkdir(parents=True)
    (node_modules / "index.js").write_text("module.exports = {};", encoding="utf-8")

    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    (vendor_dir / "lib.py").write_text("vendored", encoding="utf-8")

    # 3. Lockfiles and minified files
    (tmp_path / "poetry.lock").write_text("[package]", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    (src_dir / "bundle.min.js").write_text("var a=1;", encoding="utf-8")

    # 4. Binaries (extension and null byte content)
    (tmp_path / "icon.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00")
    (src_dir / "embedded_null.dat").write_bytes(b"hello \x00 world")

    # 5. Oversized file (> 500 bytes when limit is 500)
    huge_file = tmp_path / "huge.txt"
    huge_file.write_text("A" * 600, encoding="utf-8")

    # Walk with 500 bytes limit
    discovered = list(walk_repo_files(tmp_path, max_file_size_bytes=500))
    rel_paths = {f.relative_path: f.language for f in discovered}

    # Must contain only valid source files
    expected = {
        "src/main.py": "python",
        "src/util.ts": "typescript",
        "README.md": "markdown",
        "Dockerfile": "dockerfile",
        "config.yaml": "yaml",
    }
    assert rel_paths == expected

    # Verify content and size attributes
    main_file = next(f for f in discovered if f.relative_path == "src/main.py")
    assert main_file.content == "print('hello')"
    assert main_file.size_bytes == len("print('hello')")
