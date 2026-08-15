"""Unit tests for CQROS process-wide configuration settings cache."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from cqros.config.loader import ConfigLoadError
from cqros.config.models import AppConfig, Config, Environment
from cqros.config.settings import (
    clear_settings,
    get_settings,
    reload_settings,
    set_settings,
)


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Generator[None]:
    """Ensure each test starts and ends with an empty settings cache."""
    clear_settings()
    yield
    clear_settings()


def test_set_settings_is_returned_by_get_settings() -> None:
    """set_settings installs a Config that get_settings returns."""
    config = Config(app=AppConfig(name="Injected", environment=Environment.TESTING))

    set_settings(config)

    assert get_settings() is config


def test_get_settings_returns_cached_instance() -> None:
    """Repeated get_settings calls return the same cached object."""
    config = Config()
    set_settings(config)

    assert get_settings() is get_settings()
    assert get_settings() is config


def test_clear_settings_removes_cached_instance() -> None:
    """clear_settings drops the cache so a new value can be installed."""
    set_settings(Config(app=AppConfig(name="Before")))
    clear_settings()
    replacement = Config(app=AppConfig(name="After"))
    set_settings(replacement)

    assert get_settings() is replacement
    assert get_settings().app.name == "After"


def test_reload_settings_loads_and_caches(tmp_path: Path) -> None:
    """reload_settings loads via ConfigLoader and updates the cache."""
    path = tmp_path / "custom.toml"
    path.write_text(
        """
[app]
name = "Reloaded"
environment = "paper"
""",
        encoding="utf-8",
    )

    config = reload_settings(path)

    assert config.app.name == "Reloaded"
    assert config.app.environment is Environment.PAPER
    assert get_settings() is config


def test_reload_settings_accepts_string_path(tmp_path: Path) -> None:
    """reload_settings accepts a string path."""
    path = tmp_path / "string_path.toml"
    path.write_text(
        """
[app]
name = "FromString"
""",
        encoding="utf-8",
    )

    config = reload_settings(str(path))

    assert config.app.name == "FromString"
    assert get_settings().app.name == "FromString"


def test_get_settings_lazy_loads_default_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_settings loads configs/default.toml on first access."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "default.toml").write_text(
        """
[app]
name = "DefaultLazy"
environment = "development"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = get_settings()

    assert config.app.name == "DefaultLazy"
    assert get_settings() is config


def test_reload_settings_missing_file_does_not_clear_cache(tmp_path: Path) -> None:
    """A failed reload leaves the previously cached Config intact."""
    existing = Config(app=AppConfig(name="Existing"))
    set_settings(existing)
    missing = tmp_path / "missing.toml"

    with pytest.raises(ConfigLoadError):
        reload_settings(missing)

    assert get_settings() is existing
