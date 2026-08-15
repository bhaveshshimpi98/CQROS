"""Unit tests for CQROS OMS ``OrderVerifier``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import inf, nan
from typing import cast

import polars as pl
import pytest

from cqros.oms import OrderSide, OrderStatus, OrderType, OrderVerifier
from cqros.oms.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    MERGED_ORDER_SCHEMA,
    REQUIRED_COLUMNS,
)
from cqros.oms.verification import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    OMSValidationError,
    VerificationReport,
)
from cqros.oms.verification import OrderVerifier as OrderVerifierFromPackage
from cqros.oms.verification.verifier import OrderVerifier as OrderVerifierFromModule
from cqros.processing.verification.interfaces import DataVerifier

_START = datetime(2024, 1, 1, tzinfo=UTC)
_INTERVAL = timedelta(hours=1)
_MODEL_NAME = "alpha-lgbm"
_MODEL_VERSION = "1.0.0"
_POLICY = "equal_weight"
_OPTIMIZER = "equal_weight"


def _open_times(count: int = 3) -> list[datetime]:
    """Build monotonically increasing UTC open_time values."""
    return [_START + (index * _INTERVAL) for index in range(count)]


def _order_frame(
    *,
    symbols: list[str | None] | None = None,
    timeframes: list[str | None] | None = None,
    open_times: list[datetime | None] | None = None,
    order_ids: list[str | None] | None = None,
    parent_order_ids: list[str | None] | None = None,
    model_names: list[str | None] | None = None,
    model_versions: list[str | None] | None = None,
    policies: list[str | None] | None = None,
    optimizers: list[str | None] | None = None,
    sides: list[str | None] | None = None,
    order_types: list[str | None] | None = None,
    quantities: list[float | None] | None = None,
    limit_prices: list[float | None] | None = None,
    stop_prices: list[float | None] | None = None,
    filled_quantities: list[float | None] | None = None,
    average_fill_prices: list[float | None] | None = None,
    statuses: list[str | None] | None = None,
    created_ats: list[datetime | None] | None = None,
    updated_ats: list[datetime | None] | None = None,
    column_order: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Build a canonical merged OMS order verification frame."""
    times = open_times if open_times is not None else _open_times()
    row_count = len(times)
    default_order_ids = [f"ord-{index}" for index in range(row_count)]
    default_parent_ids = [f"parent-{index}" for index in range(row_count)]
    default_sides = [
        OrderSide.BUY.value,
        OrderSide.SELL.value,
        OrderSide.BUY.value,
    ]
    while len(default_sides) < row_count:
        default_sides.append(OrderSide.BUY.value)
    default_types = [
        OrderType.MARKET.value,
        OrderType.LIMIT.value,
        OrderType.STOP_MARKET.value,
    ]
    while len(default_types) < row_count:
        default_types.append(OrderType.MARKET.value)
    default_statuses = [
        OrderStatus.PENDING.value,
        OrderStatus.SUBMITTED.value,
        OrderStatus.FILLED.value,
    ]
    while len(default_statuses) < row_count:
        default_statuses.append(OrderStatus.PENDING.value)
    default_quantities = [1.0, 2.0, 3.0]
    while len(default_quantities) < row_count:
        default_quantities.append(1.0)
    default_prices = [100.0, 200.0, 300.0]
    while len(default_prices) < row_count:
        default_prices.append(100.0)
    created = created_ats if created_ats is not None else times
    updated = updated_ats if updated_ats is not None else times
    data: dict[str, object] = {
        "symbol": symbols if symbols is not None else ["BTCUSDT"] * row_count,
        "timeframe": timeframes if timeframes is not None else ["1h"] * row_count,
        "open_time": times,
        "order_id": order_ids if order_ids is not None else default_order_ids,
        "parent_order_id": (
            parent_order_ids if parent_order_ids is not None else default_parent_ids
        ),
        "model_name": (model_names if model_names is not None else [_MODEL_NAME] * row_count),
        "model_version": (
            model_versions if model_versions is not None else [_MODEL_VERSION] * row_count
        ),
        "policy": policies if policies is not None else [_POLICY] * row_count,
        "optimizer": optimizers if optimizers is not None else [_OPTIMIZER] * row_count,
        "side": sides if sides is not None else default_sides[:row_count],
        "order_type": (order_types if order_types is not None else default_types[:row_count]),
        "quantity": (quantities if quantities is not None else default_quantities[:row_count]),
        "limit_price": (limit_prices if limit_prices is not None else default_prices[:row_count]),
        "stop_price": (stop_prices if stop_prices is not None else default_prices[:row_count]),
        "filled_quantity": (
            filled_quantities if filled_quantities is not None else [0.0] * row_count
        ),
        "average_fill_price": (
            average_fill_prices if average_fill_prices is not None else [0.0] * row_count
        ),
        "status": statuses if statuses is not None else default_statuses[:row_count],
        "created_at": created,
        "updated_at": updated,
    }
    order = column_order if column_order is not None else CANONICAL_COLUMN_ORDER
    frame = pl.DataFrame(data, schema=dict(COLUMN_DTYPES))
    return frame.select(list(order))


def _verifier() -> OrderVerifier:
    """Build an OrderVerifier instance."""
    return OrderVerifier()


def _assert_clean_pass(report: VerificationReport, *, rows: int) -> None:
    """Assert a fully passing report for ``rows`` checked."""
    assert report == VerificationReport(
        rows_checked=rows,
        duplicate_timestamp_rows=0,
        null_rows=0,
        nan_rows=0,
        invalid_timestamp_rows=0,
        invalid_numeric_rows=0,
        warnings=(),
        passed=True,
    )


def test_package_exports_order_verifier() -> None:
    """Package re-exports match the verification module symbol."""
    assert OrderVerifier is OrderVerifierFromModule
    assert OrderVerifierFromPackage is OrderVerifierFromModule


def test_order_verifier_satisfies_data_verifier_protocol() -> None:
    """OrderVerifier structurally satisfies DataVerifier."""
    assert isinstance(_verifier(), DataVerifier)


def test_successful_verification() -> None:
    """A clean sorted merged order frame passes verification."""
    report = _verifier().verify(_order_frame())
    _assert_clean_pass(report, rows=3)


def test_empty_frame_passes() -> None:
    """An empty schema-valid frame passes verification."""
    frame = pl.DataFrame(schema=MERGED_ORDER_SCHEMA).select(list(CANONICAL_COLUMN_ORDER))
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=0)


def test_canonical_schema_columns() -> None:
    """Successful verification inspects the canonical OMS Order column set."""
    frame = _order_frame()
    assert frame.columns == list(CANONICAL_COLUMN_ORDER)
    assert frame.columns == [
        "symbol",
        "timeframe",
        "open_time",
        "order_id",
        "parent_order_id",
        "model_name",
        "model_version",
        "policy",
        "optimizer",
        "side",
        "order_type",
        "quantity",
        "limit_price",
        "stop_price",
        "filled_quantity",
        "average_fill_price",
        "status",
        "created_at",
        "updated_at",
    ]
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_pass_report() -> None:
    """PASS report has zero defect counters and empty warnings."""
    report = _verifier().verify(_order_frame())
    assert report.passed is True
    assert report.warnings == ()
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0


def test_duplicate_primary_keys() -> None:
    """Duplicate (symbol, timeframe, open_time, order_id) keys fail verification."""
    times: list[datetime | None] = [_START, _START, _START + _INTERVAL]
    frame = _order_frame(
        open_times=times,
        order_ids=["ord-0", "ord-0", "ord-1"],
    )
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 1
    assert report.passed is False
    assert "Duplicate timestamps detected." in report.warnings


def test_duplicate_keys_allow_same_open_time_across_order_ids() -> None:
    """Identical open_time values with distinct order_ids are not duplicates."""
    frame = _order_frame(
        open_times=[_START, _START, _START + _INTERVAL],
        order_ids=["ord-0", "ord-1", "ord-2"],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_missing_columns() -> None:
    """Missing required columns raise OMSValidationError."""
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": [_START],
        }
    )
    with pytest.raises(OMSValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_REQUIRED_COLUMNS
    missing = cast(tuple[str, ...], dict(exc_info.value.details)["missing_columns"])
    assert "order_id" in missing
    assert "side" in missing
    assert "status" in missing
    assert "quantity" in missing
    assert set(REQUIRED_COLUMNS) - set(frame.columns) == set(missing)


def test_null_values() -> None:
    """NULL values in required non-nullable columns fail verification."""
    frame = _order_frame(
        statuses=[OrderStatus.PENDING.value, None, OrderStatus.FILLED.value],
    )
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_limit_price_passes_for_market_orders() -> None:
    """Market orders with NULL limit_price verify cleanly."""
    frame = _order_frame(
        order_types=[
            OrderType.MARKET.value,
            OrderType.MARKET.value,
            OrderType.MARKET.value,
        ],
        limit_prices=[None, None, None],
        stop_prices=[None, None, None],
        average_fill_prices=[None, None, None],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_null_stop_price_passes() -> None:
    """Orders with NULL stop_price verify cleanly."""
    frame = _order_frame(
        order_types=[
            OrderType.MARKET.value,
            OrderType.LIMIT.value,
            OrderType.LIMIT.value,
        ],
        limit_prices=[None, 200.0, 300.0],
        stop_prices=[None, None, None],
        average_fill_prices=[None, None, None],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_null_average_fill_price_passes() -> None:
    """Unfilled orders with NULL average_fill_price verify cleanly."""
    frame = _order_frame(
        statuses=[
            OrderStatus.PENDING.value,
            OrderStatus.SUBMITTED.value,
            OrderStatus.PENDING.value,
        ],
        average_fill_prices=[None, None, None],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_valid_oms_dataset_with_optional_nulls_passes() -> None:
    """Valid OMS frames with optional NULLs report zero nulls and PASS."""
    frame = _order_frame(
        order_types=[
            OrderType.MARKET.value,
            OrderType.MARKET.value,
            OrderType.MARKET.value,
        ],
        statuses=[
            OrderStatus.PENDING.value,
            OrderStatus.PENDING.value,
            OrderStatus.PENDING.value,
        ],
        limit_prices=[None, None, None],
        stop_prices=[None, None, None],
        filled_quantities=[0.0, 0.0, 0.0],
        average_fill_prices=[None, None, None],
    )
    report = _verifier().verify(frame)
    assert report.null_rows == 0
    assert report.warnings == ()
    assert report.passed is True
    _assert_clean_pass(report, rows=3)


def test_null_primary_key_rows() -> None:
    """NULL primary-key values are counted as null rows."""
    frame = _order_frame(order_ids=["ord-0", None, "ord-2"])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_quantity_still_fails() -> None:
    """NULL quantity in a required numeric column fails verification."""
    frame = _order_frame(quantities=[1.0, None, 3.0])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings


def test_null_timestamp_rows() -> None:
    """NULL open_time values count as both null and invalid timestamp rows."""
    frame = _order_frame(open_times=[_START, None, _START + 2 * _INTERVAL])
    report = _verifier().verify(frame)
    assert report.null_rows == 1
    assert report.invalid_timestamp_rows == 1
    assert report.passed is False
    assert "Rows containing NULL values." in report.warnings
    assert "Invalid timestamps detected." in report.warnings


def test_nan_numeric_rows() -> None:
    """NaN numeric values are counted and fail verification."""
    frame = _order_frame(
        quantities=[1.0, nan, 3.0],
        limit_prices=[100.0, 200.0, nan],
    )
    report = _verifier().verify(frame)
    assert report.nan_rows == 2
    assert report.passed is False
    assert "Rows containing NaN values." in report.warnings


def test_non_finite_quantity_rows() -> None:
    """Infinite quantity values are counted and fail verification."""
    frame = _order_frame(quantities=[1.0, inf, -inf])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 2
    assert report.passed is False
    assert "Invalid quantity values detected." in report.warnings


def test_non_finite_filled_quantity_rows() -> None:
    """Infinite filled_quantity values are counted and fail verification."""
    frame = _order_frame(filled_quantities=[0.0, inf, 0.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid filled_quantity values detected." in report.warnings


def test_non_finite_average_fill_price_rows() -> None:
    """Infinite average_fill_price values are counted and fail verification."""
    frame = _order_frame(average_fill_prices=[0.0, -inf, 0.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid average_fill_price values detected." in report.warnings


def test_non_finite_limit_price_rows() -> None:
    """Infinite limit_price values are counted and fail verification."""
    frame = _order_frame(limit_prices=[100.0, inf, 300.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid limit_price values detected." in report.warnings


def test_non_finite_stop_price_rows() -> None:
    """Infinite stop_price values are counted and fail verification."""
    frame = _order_frame(stop_prices=[100.0, -inf, 300.0])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid stop_price values detected." in report.warnings


def test_unsorted_timestamps() -> None:
    """Unsorted open_time fails without incrementing counters."""
    frame = _order_frame(
        open_times=[_START, _START + 2 * _INTERVAL, _START + _INTERVAL],
    )
    report = _verifier().verify(frame)
    assert report.duplicate_timestamp_rows == 0
    assert report.null_rows == 0
    assert report.nan_rows == 0
    assert report.invalid_timestamp_rows == 0
    assert report.invalid_numeric_rows == 0
    assert report.passed is False
    assert report.warnings == ("Frame is not sorted by open_time.",)


def test_invalid_order_side_values() -> None:
    """Side values outside OrderSide fail verification."""
    frame = _order_frame(
        sides=[OrderSide.BUY.value, "LONG", OrderSide.SELL.value],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid OrderSide values detected." in report.warnings


def test_invalid_order_type_values() -> None:
    """Order type values outside OrderType fail verification."""
    frame = _order_frame(
        order_types=[OrderType.MARKET.value, "IOC", OrderType.LIMIT.value],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid OrderType values detected." in report.warnings


def test_invalid_order_status_values() -> None:
    """Status values outside OrderStatus fail verification."""
    frame = _order_frame(
        statuses=[OrderStatus.PENDING.value, "OPEN", OrderStatus.FILLED.value],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Invalid OrderStatus values detected." in report.warnings


def test_invalid_enum_values_use_oms_enums() -> None:
    """Only canonical OMS enum values are accepted."""
    frame = _order_frame(
        sides=[OrderSide.BUY.value, OrderSide.SELL.value, OrderSide.BUY.value],
        order_types=[
            OrderType.MARKET.value,
            OrderType.LIMIT.value,
            OrderType.STOP_LIMIT.value,
        ],
        statuses=[
            OrderStatus.PENDING.value,
            OrderStatus.SUBMITTED.value,
            OrderStatus.CANCELLED.value,
        ],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)

    frame = _order_frame(
        sides=["buy", "sell", "buy"],
        order_types=["market", "limit", "stop_market"],
        statuses=["pending", "submitted", "filled"],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 3
    assert report.passed is False
    assert "Invalid OrderSide values detected." in report.warnings
    assert "Invalid OrderType values detected." in report.warnings
    assert "Invalid OrderStatus values detected." in report.warnings


def test_empty_order_id_values() -> None:
    """Empty order_id strings fail verification."""
    frame = _order_frame(order_ids=["ord-0", "", "ord-2"])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Empty order_id values detected." in report.warnings


def test_empty_parent_order_id_values() -> None:
    """Empty parent_order_id strings fail verification."""
    frame = _order_frame(parent_order_ids=["parent-0", "", "parent-2"])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Empty parent_order_id values detected." in report.warnings


def test_empty_side_values() -> None:
    """Empty side strings fail verification."""
    frame = _order_frame(sides=[OrderSide.BUY.value, "", OrderSide.SELL.value])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Empty side values detected." in report.warnings
    assert "Invalid OrderSide values detected." in report.warnings


def test_empty_order_type_values() -> None:
    """Empty order_type strings fail verification."""
    frame = _order_frame(order_types=[OrderType.MARKET.value, "", OrderType.LIMIT.value])
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Empty order_type values detected." in report.warnings
    assert "Invalid OrderType values detected." in report.warnings


def test_empty_status_values() -> None:
    """Empty status strings fail verification."""
    frame = _order_frame(
        statuses=[OrderStatus.PENDING.value, "", OrderStatus.FILLED.value],
    )
    report = _verifier().verify(frame)
    assert report.invalid_numeric_rows == 1
    assert report.passed is False
    assert "Empty status values detected." in report.warnings
    assert "Invalid OrderStatus values detected." in report.warnings


def test_incorrect_column_order() -> None:
    """Wrong column order fails verification with a deterministic warning."""
    reordered = (
        *CANONICAL_COLUMN_ORDER[1:],
        CANONICAL_COLUMN_ORDER[0],
    )
    frame = _order_frame(column_order=reordered)
    report = _verifier().verify(frame)
    assert report.passed is False
    assert report.warnings == ("Frame column order does not match canonical order.",)


def test_dtype_mismatch_open_time() -> None:
    """Wrong open_time dtype raises OMSValidationError schema mismatch."""
    frame = _order_frame().with_columns(pl.col("open_time").cast(pl.Int64))
    with pytest.raises(OMSValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert mismatched == ("open_time",)


def test_dtype_mismatch_quantity() -> None:
    """Non-Float64 quantity columns raise OMSValidationError."""
    frame = _order_frame().with_columns(pl.col("quantity").cast(pl.Float32))
    with pytest.raises(OMSValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "quantity" in mismatched


def test_dtype_mismatch_side_column() -> None:
    """Non-Utf8 side columns raise OMSValidationError."""
    frame = _order_frame().with_columns(pl.col("side").cast(pl.Categorical))
    with pytest.raises(OMSValidationError) as exc_info:
        _verifier().verify(frame)
    assert exc_info.value.error_code == ERROR_SCHEMA_MISMATCH
    mismatched = cast(
        tuple[str, ...],
        dict(exc_info.value.details)["mismatched_columns"],
    )
    assert "side" in mismatched


def test_does_not_enforce_lifecycle_or_fill_rules() -> None:
    """Verifier does not enforce fill math or lifecycle business rules."""
    frame = _order_frame(
        quantities=[1.0, 1.0, 1.0],
        filled_quantities=[5.0, 5.0, 5.0],
        statuses=[
            OrderStatus.PENDING.value,
            OrderStatus.PENDING.value,
            OrderStatus.PENDING.value,
        ],
        average_fill_prices=[999.0, 999.0, 999.0],
    )
    report = _verifier().verify(frame)
    _assert_clean_pass(report, rows=3)


def test_immutability() -> None:
    """verify does not mutate the input DataFrame."""
    frame = _order_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        sides=[OrderSide.BUY.value, None, "LONG"],
        statuses=[OrderStatus.PENDING.value, "", "OPEN"],
        quantities=[1.0, nan, inf],
        order_ids=["ord-0", "", "ord-0"],
    )
    original = frame.clone()
    report = _verifier().verify(frame)
    assert frame.equals(original)
    assert report.passed is False


def test_report_values_combined() -> None:
    """Combined failures populate report fields correctly."""
    reordered = (
        *CANONICAL_COLUMN_ORDER[1:],
        CANONICAL_COLUMN_ORDER[0],
    )
    frame = _order_frame(
        open_times=[_START + _INTERVAL, _START, _START],
        order_ids=["", "ord-1", "ord-1"],
        sides=[OrderSide.BUY.value, None, "LONG"],
        statuses=[OrderStatus.PENDING.value, "", "OPEN"],
        quantities=[1.0, nan, inf],
        filled_quantities=[0.0, 0.0, -inf],
        column_order=reordered,
    )
    report = _verifier().verify(frame)
    assert report.rows_checked == 3
    assert report.duplicate_timestamp_rows == 1
    assert report.null_rows == 1
    assert report.nan_rows == 1
    assert report.invalid_numeric_rows >= 1
    assert report.passed is False
    assert "Frame column order does not match canonical order." in report.warnings
    assert "Duplicate timestamps detected." in report.warnings
    assert "Rows containing NULL values." in report.warnings
    assert "Rows containing NaN values." in report.warnings
    assert "Empty order_id values detected." in report.warnings
    assert "Invalid OrderSide values detected." in report.warnings
    assert "Empty status values detected." in report.warnings
    assert "Invalid quantity values detected." in report.warnings
    assert "Invalid filled_quantity values detected." in report.warnings
    assert "Frame is not sorted by open_time." in report.warnings
    assert isinstance(report.warnings, tuple)


def test_order_verifier_exported_from_root_package() -> None:
    """OMS package re-exports OrderVerifier."""
    from cqros import oms as oms_package

    assert "OrderVerifier" in oms_package.__all__
    assert oms_package.OrderVerifier is OrderVerifierFromModule
