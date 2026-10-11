"""Application configuration.

Settings are read from environment variables (optionally from a ``.env`` file
in the project root). The LLM provider is switchable:

* ``LLM_PROVIDER=gemini``    - Google Gemini (used for testing / demo)
* ``LLM_PROVIDER=anthropic`` - Anthropic Claude (used for production)
* ``LLM_PROVIDER=offline``   - no LLM, deterministic reasoning only
* ``LLM_PROVIDER=auto``      - ``anthropic`` when ``APP_ENV=production``,
  otherwise ``gemini``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_PROVIDERS = ("auto", "gemini", "anthropic", "offline")


def load_dotenv(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Minimal, dependency-free ``.env`` loader.

    Lines look like ``KEY=value`` (``export`` prefix, quotes and ``#`` comments
    are supported). Existing environment variables win unless ``override``.
    """
    path = path or PROJECT_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return default


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    app_env: str = "development"
    llm_provider: str = "auto"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    llm_temperature: float = 0.1
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 4096
    debate_max_rounds: int = 3
    noise_threshold: float = 0.20
    tp_threshold: float = 0.70
    fp_threshold: float = 0.40
    auto_medium_risk_confidence: float = 0.85
    log_window_minutes: int = 180
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    var_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "var")

    @property
    def db_path(self) -> Path:
        return self.var_dir / "sentinelcrew.db"

    @property
    def reports_dir(self) -> Path:
        return self.var_dir / "reports"

    @property
    def model_path(self) -> Path:
        return self.var_dir / "models" / "triage_isolation_forest.json"

    def resolved_provider(self) -> str:
        """Return the concrete provider name (never ``auto``)."""
        provider = (self.llm_provider or "auto").strip().lower()
        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.llm_provider}'. Use one of {VALID_PROVIDERS}."
            )
        if provider == "auto":
            return "anthropic" if self.is_production else "gemini"
        return provider

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in ("production", "prod")

    def with_overrides(self, **kwargs: object) -> "Settings":
        return replace(self, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, use_dotenv: bool = True) -> "Settings":
        if env is None:
            if use_dotenv:
                load_dotenv()
            env = os.environ
        data_dir = Path(env.get("SENTINEL_DATA_DIR", str(PROJECT_ROOT / "data")))
        var_dir = Path(env.get("SENTINEL_VAR_DIR", str(PROJECT_ROOT / "var")))
        return cls(
            app_env=env.get("APP_ENV", "development"),
            llm_provider=env.get("LLM_PROVIDER", "auto"),
            gemini_api_key=env.get("GEMINI_API_KEY", env.get("GOOGLE_API_KEY", "")),
            gemini_model=env.get("GEMINI_MODEL", cls.gemini_model),
            anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
            anthropic_model=env.get("ANTHROPIC_MODEL", cls.anthropic_model),
            llm_temperature=_float(env, "LLM_TEMPERATURE", cls.llm_temperature),
            llm_timeout_s=_float(env, "LLM_TIMEOUT_S", cls.llm_timeout_s),
            llm_max_retries=_int(env, "LLM_MAX_RETRIES", cls.llm_max_retries),
            llm_max_tokens=_int(env, "LLM_MAX_TOKENS", cls.llm_max_tokens),
            debate_max_rounds=_int(env, "DEBATE_MAX_ROUNDS", cls.debate_max_rounds),
            noise_threshold=_float(env, "NOISE_THRESHOLD", cls.noise_threshold),
            tp_threshold=_float(env, "TP_THRESHOLD", cls.tp_threshold),
            fp_threshold=_float(env, "FP_THRESHOLD", cls.fp_threshold),
            auto_medium_risk_confidence=_float(
                env, "AUTO_MEDIUM_RISK_CONFIDENCE", cls.auto_medium_risk_confidence
            ),
            log_window_minutes=_int(env, "LOG_WINDOW_MINUTES", cls.log_window_minutes),
            data_dir=data_dir,
            var_dir=var_dir,
        )
