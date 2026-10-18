"""Git repository cloning and commit inspection via GitPython."""

from __future__ import annotations

from pathlib import Path

import git


def clone_repo(
    url: str,
    dest: Path,
    *,
    shallow: bool = True,
) -> git.Repo:
    """Clone a git repository to *dest*.

    Args:
        url: Git clone URL or local repository path.
        dest: Local destination directory (pathlib.Path).
        shallow: When True, perform a shallow clone (depth=1).

    Returns:
        The cloned or opened :class:`git.Repo`.
    """
    dest = Path(dest).resolve()
    if dest.exists() and (dest / ".git").exists():
        return git.Repo(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"depth": 1} if shallow else {}
    return git.Repo.clone_from(url, dest, **kwargs)


def get_repo_commit_sha(repo: git.Repo | Path) -> str:
    """Return the current HEAD commit SHA for a repository."""
    if isinstance(repo, Path):
        repo = git.Repo(repo)
    try:
        return repo.head.commit.hexsha
    except Exception:
        return ""
