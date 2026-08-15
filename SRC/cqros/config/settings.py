"""CQROS process-wide configuration settings cache.

This module maintains a single cached ``Config`` instance for process-wide
access. Loading and validation are delegated entirely to ``ConfigLoader``.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from cqros.config.loader import ConfigLoader
from cqros.config.models import Config

_DEFAULT_CONFIG_PATH = Path("configs") / "default.toml"

_lock = Lock()
_settings: Config | None = None


def get_settings() -> Config:
    """Return the cached configuration, loading it on first access.

    When the cache is empty, configuration is loaded from
    ``configs/default.toml`` via ``ConfigLoader``.

    Returns:
        The process-wide ``Config`` instance.

    Raises:
        ConfigLoadError: If the default configuration file cannot be read
            or parsed.
        ConfigurationValidationError: If the loaded configuration fails
            validation.
    """
    global _settings
    cached = _settings
    if cached is not None:
        return cached
    with _lock:
        if _settings is None:
            _settings = ConfigLoader().load(_DEFAULT_CONFIG_PATH)
        return _settings


def reload_settings(path: Path | str) -> Config:
    """Reload configuration from ``path`` and replace the cache.

    Args:
        path: Filesystem path to a TOML configuration file.

    Returns:
        The newly loaded ``Config`` instance.

    Raises:
        ConfigLoadError: If the file cannot be read or parsed.
        ConfigurationValidationError: If validation fails.
    """
    global _settings
    config = ConfigLoader().load(path)
    with _lock:
        _settings = config
    return config


def set_settings(config: Config) -> None:
    """Replace the cached configuration with ``config``.

    Args:
        config: Configuration instance to install as the process-wide cache.
    """
    global _settings
    with _lock:
        _settings = config


def clear_settings() -> None:
    """Clear the cached configuration.

    The next call to ``get_settings`` reloads configuration from the default
    path.
    """
    global _settings
    with _lock:
        _settings = None
