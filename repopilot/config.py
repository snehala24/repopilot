"""Typed configuration for RepoPilot.

Secrets (API keys) are read from ``.env`` / environment variables using
``pydantic-settings``; non-secret settings are read from ``configs/*.yaml``
with Pydantic models. All paths are ``pathlib.Path`` objects and relative
paths resolve against the project root, so the project is Windows-friendly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
SETTINGS_YAML = "settings.yaml"
LLM_YAML = "llm.yaml"


class AppSettings(BaseModel):
    """High-level application settings from ``settings.yaml``."""

    name: str = "repopilot"
    version: str = "0.1.0"
    environment: str = "dev"
    log_level: str = "INFO"


class PathsSettings(BaseModel):
    """Filesystem locations; relative paths resolve against the project root."""

    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    llm_cache_dir: Path = Path(".cache/llm")

    @field_validator("data_dir", "cache_dir", "llm_cache_dir", mode="before")
    @classmethod
    def _resolve_relative(cls, value: Any) -> Any:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path


class BudgetSettings(BaseModel):
    """Per-run LLM budgets enforced by the gateway budget guard."""

    max_llm_calls: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)


class LLMTierSettings(BaseModel):
    """One LLM tier definition from ``llm.yaml``."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(gt=0)


class LLMSettings(BaseModel):
    """All configured LLM tiers (``fast``, ``strong``, ``judge``, ...)."""

    tiers: dict[str, LLMTierSettings]

    def tier(self, name: str) -> LLMTierSettings:
        """Return one tier by name, with a helpful error for typos."""
        try:
            return self.tiers[name]
        except KeyError:
            available = ", ".join(sorted(self.tiers)) or "<none>"
            raise KeyError(
                f"Unknown LLM tier {name!r}. Available tiers: {available}"
            ) from None


class Environment(BaseSettings):
    """Secrets from ``.env`` / environment variables. Never hard-code these."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr | None = None
    github_token: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None


class Settings(BaseModel):
    """Fully assembled configuration object."""

    app: AppSettings
    paths: PathsSettings
    budgets: BudgetSettings
    llm: LLMSettings
    env: Environment


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*, failing loudly on missing/invalid files."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing configuration file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}, got {type(data).__name__}"
        )
    return data


def load_settings(
    config_dir: Path | None = None,
    *,
    env_file: Path | None = None,
) -> Settings:
    """Load ``.env`` + YAML configuration into a typed :class:`Settings` object.

    Args:
        config_dir: Directory containing ``settings.yaml`` and ``llm.yaml``.
            Defaults to the project's ``configs/`` directory.
        env_file: Explicit ``.env`` path (mainly for tests). When ``None``,
            the project-root ``.env`` is used if present.
    """
    config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    app_data = _load_yaml(config_dir / SETTINGS_YAML)
    llm_data = _load_yaml(config_dir / LLM_YAML)

    env_kwargs: dict[str, Any] = {}
    if env_file is not None:
        env_kwargs["_env_file"] = str(env_file)

    return Settings(
        app=AppSettings(**app_data.get("app", {})),
        paths=PathsSettings(**app_data.get("paths", {})),
        budgets=BudgetSettings(**app_data.get("budgets", {})),
        llm=LLMSettings(**llm_data),
        env=Environment(**env_kwargs),
    )
