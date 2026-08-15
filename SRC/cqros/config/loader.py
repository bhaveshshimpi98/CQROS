"""CQROS configuration file loader.

This module reads TOML configuration files, builds immutable ``Config``
dataclasses via field introspection, and validates the result with
``ConfigValidator``.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast, get_args, get_origin, get_type_hints

from cqros.config.models import Config
from cqros.config.validator import ConfigValidator, ValidationResult


class ConfigLoadError(Exception):
    """Raised when a configuration file cannot be read or parsed as TOML."""


class ConfigurationValidationError(Exception):
    """Raised when a loaded configuration fails semantic validation.

    Attributes:
        result: Aggregated validation outcome containing all issues.
    """

    def __init__(self, message: str, result: ValidationResult) -> None:
        """Initialize the error with a message and validation result.

        Args:
            message: Human-readable summary of the validation failure.
            result: ``ValidationResult`` produced by ``ConfigValidator``.
        """
        super().__init__(message)
        self.result = result


class ConfigLoader:
    """Load and validate CQROS configuration from a TOML file.

    Nested configuration sections are constructed through dataclass
    introspection so newly added sections on ``Config`` are supported
    without updating loader field maps.

    Args:
        validator: Validator applied after the ``Config`` object is built.
            When omitted, a new ``ConfigValidator`` is created.
    """

    def __init__(self, validator: ConfigValidator | None = None) -> None:
        """Initialize the loader with an optional validator dependency.

        Args:
            validator: Validator used after configuration construction.
        """
        self._validator = ConfigValidator() if validator is None else validator

    def load(self, path: Path | str) -> Config:
        """Load, build, and validate configuration from a TOML file.

        Args:
            path: Filesystem path to a TOML configuration file.

        Returns:
            Validated immutable ``Config`` instance.

        Raises:
            ConfigLoadError: If the file cannot be read or is not valid TOML,
                or if TOML values cannot be mapped onto configuration fields.
            ConfigurationValidationError: If ``ConfigValidator`` reports
                one or more error-severity issues.
        """
        config_path = Path(path)
        raw = self._read_toml(config_path)
        config = _build_dataclass(Config, raw)
        result = self._validator.validate(config)
        if not result.valid:
            error_count = len(result.errors())
            raise ConfigurationValidationError(
                f"Configuration validation failed with {error_count} error(s)",
                result,
            )
        return config

    def _read_toml(self, path: Path) -> dict[str, Any]:
        """Read and parse a TOML file into a mapping.

        Args:
            path: Path to the TOML configuration file.

        Returns:
            Parsed TOML table as a dictionary.

        Raises:
            ConfigLoadError: On I/O failure or TOML decode errors.
        """
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        except FileNotFoundError as exc:
            raise ConfigLoadError(f"Configuration file not found: {path}") from exc
        except OSError as exc:
            raise ConfigLoadError(f"Failed to read configuration file: {path}: {exc}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigLoadError(f"Invalid TOML in configuration file: {path}: {exc}") from exc
        return data


def _build_dataclass[DataclassT](
    cls: type[DataclassT],
    raw: Mapping[str, Any],
) -> DataclassT:
    """Construct a dataclass from a mapping using field introspection.

    Only keys that match declared fields are consumed. Missing keys retain
    dataclass defaults, which keeps the loader forward-compatible when new
    optional sections are introduced.

    Args:
        cls: Dataclass type to instantiate.
        raw: Mapping of field names to raw TOML values.

    Returns:
        Immutable dataclass instance populated from ``raw``.

    Raises:
        ConfigLoadError: If a value cannot be coerced to its field type.
    """
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_info in fields(cast(Any, cls)):
        name = field_info.name
        if name not in raw:
            continue
        field_type = type_hints[name]
        kwargs[name] = _coerce_value(field_type, raw[name], name)
    return cls(**kwargs)


def _coerce_value(field_type: object, value: Any, field_name: str) -> Any:
    """Coerce a raw TOML value to the annotated configuration field type.

    Args:
        field_type: Resolved type annotation for the target field.
        value: Raw value parsed from TOML.
        field_name: Field name used in error messages.

    Returns:
        Value converted to ``field_type`` when conversion is required.

    Raises:
        ConfigLoadError: If ``value`` cannot be represented as ``field_type``.
    """
    origin = get_origin(field_type)

    if isinstance(field_type, type) and is_dataclass(field_type):
        if not isinstance(value, Mapping):
            raise ConfigLoadError(
                f"Expected a TOML table for field '{field_name}', " f"got {type(value).__name__}"
            )
        nested_raw = cast(Mapping[str, Any], value)
        return _build_dataclass(cast(type[Any], field_type), nested_raw)

    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return _coerce_enum(field_type, value, field_name)

    if origin is tuple:
        return _coerce_tuple(cast(object, field_type), value, field_name)

    if field_type is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)

    return value


def _coerce_enum(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    """Convert a raw value into an ``Enum`` member.

    Args:
        enum_type: Target enumeration type.
        value: Raw TOML value.
        field_name: Field name used in error messages.

    Returns:
        Matching enumeration member.

    Raises:
        ConfigLoadError: If ``value`` is not a valid member of ``enum_type``.
    """
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError(f"Invalid value for field '{field_name}': {value!r}") from exc


def _coerce_tuple(
    field_type: object,
    value: Any,
    field_name: str,
) -> tuple[Any, ...]:
    """Convert a TOML array into a typed tuple.

    Args:
        field_type: Tuple annotation such as ``tuple[str, ...]``.
        value: Raw TOML value, expected to be a sequence.
        field_name: Field name used in error messages.

    Returns:
        Tuple with elements coerced to the annotated item type.

    Raises:
        ConfigLoadError: If ``value`` is not a suitable sequence.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigLoadError(
            f"Expected a TOML array for field '{field_name}', " f"got {type(value).__name__}"
        )

    items = cast(Sequence[Any], value)
    args = get_args(field_type)

    if len(args) == 2 and args[1] is Ellipsis:
        item_type = args[0]
        return tuple(_coerce_value(item_type, item, field_name) for item in items)

    if len(args) == 0:
        return tuple(items)

    if len(args) != len(items):
        raise ConfigLoadError(
            f"Field '{field_name}' expected a tuple of length {len(args)}, "
            f"got length {len(items)}"
        )
    return tuple(
        _coerce_value(item_type, item, field_name)
        for item_type, item in zip(args, items, strict=True)
    )
