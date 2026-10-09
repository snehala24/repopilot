"""Disk cache for LLM completions, stored under ``.cache/llm/``.

Cache keys are SHA-256 hashes over (model, system, prompt, params), so a
cache hit never costs an API call. Entries are JSON files written atomically
(Windows-safe via ``os.replace``).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .base import CompletionRequest


@dataclass(frozen=True)
class CacheEntry:
    """A previously completed generation stored on disk."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    created_at: float


class DiskCache:
    """Content-addressed completion cache under a directory."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._lock = threading.Lock()

    @staticmethod
    def cache_key(request: CompletionRequest) -> str:
        """Hash (model, system, prompt, params) into a stable cache key."""
        payload = {
            "model": request.model,
            "system": request.system,
            "prompt": request.prompt,
            "params": {
                "temperature": request.temperature,
                "max_output_tokens": request.max_output_tokens,
                "response_schema": request.response_schema,
            },
        }
        canonical = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, request: CompletionRequest) -> CacheEntry | None:
        """Return the cached entry for *request*, or ``None`` on any miss."""
        path = self._path(self.cache_key(request))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CacheEntry(**data)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError):
            # Corrupt/incompatible entry: treat as a miss and drop it.
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def set(self, request: CompletionRequest, entry: CacheEntry) -> None:
        """Atomically store *entry* for *request*."""
        key = self.cache_key(request)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(entry), ensure_ascii=False, indent=None)
        with self._lock:
            fd, tmp_name = tempfile.mkstemp(
                dir=self.cache_dir, prefix=".tmp-", suffix=".json"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                os.replace(tmp_name, self._path(key))
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
