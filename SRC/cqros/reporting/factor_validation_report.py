"""CQROS factor validation researcher report builder.

Purpose:
    Provide a read-only reporting layer that loads existing factor validation
    parquet ledgers and emits researcher-facing CSV, Excel, and HTML reports.

Responsibilities:
    - Load every validation ledger for a selected manager via
      ``FactorValidationRepository``
    - Build a ranked one-row-per-factor summary table
    - Export validation summary, top factors, rejected factors, and HTML
    - Remain free of factor generation, validation math, selection, and
      repository mutation

Dependencies:
    ``html``, ``logging``, ``zipfile``, ``polars``, ``cqros.core``,
    ``cqros.factor_validation``, and ``cqros.reporting.exceptions``.

Public API:
    ``FactorValidationReporter``, ``SUMMARY_COLUMNS``, ``TOP_FACTOR_COUNT``,
    ``REPORT_FORMATS``, ``DEFAULT_OUTPUT_ROOT``.

Notes:
    This module never writes parquet ledgers and never modifies validation
    schema, engine, pipeline, or repository contracts. Excel workbooks are
    emitted as Office Open XML via the standard library so no third-party
    spreadsheet engine is required.
"""

from __future__ import annotations

import html
import logging
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal
from xml.sax.saxutils import escape as xml_escape

import polars as pl

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.types import Exchange, Market
from cqros.factor_validation import FactorValidationRepository, FactorValidationStatus
from cqros.reporting.exceptions import ReportingValidationError

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "FactorValidationReporter",
    "REPORT_FORMATS",
    "SUMMARY_COLUMNS",
    "TOP_FACTOR_COUNT",
]

_logger = logging.getLogger(__name__)

ReportFormat = Literal["html", "csv", "xlsx"]

REPORT_FORMATS: Final[tuple[ReportFormat, ...]] = ("html", "csv", "xlsx")

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports") / "factor_validation"

TOP_FACTOR_COUNT: Final[int] = 20

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "factor_category",
    "status",
    "information_coefficient",
    "rank_information_coefficient",
    "ic_information_ratio",
    "monotonicity_score",
    "quantile_spread",
    "turnover",
    "observations",
    "validation_start_time",
    "validation_end_time",
    "dataset_version",
    "label_version",
)

_REJECTED_EXPORT_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("factor_name", "factor_name"),
    ("status", "status"),
    ("observations", "observations"),
    ("information_coefficient", "IC"),
    ("rank_information_coefficient", "Rank IC"),
    ("ic_information_ratio", "ICIR"),
    ("monotonicity_score", "Monotonicity"),
)

_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "rank_information_coefficient",
    "information_coefficient",
    "ic_information_ratio",
    "monotonicity_score",
)

_METRIC_DISTRIBUTION_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("information_coefficient", "IC"),
    ("rank_information_coefficient", "Rank IC"),
    ("ic_information_ratio", "ICIR"),
    ("monotonicity_score", "Monotonicity"),
)

_ERROR_MANAGER: Final[str] = "REPORT-FACTOR-VALIDATION-001"
_ERROR_NO_PARTITIONS: Final[str] = "REPORT-FACTOR-VALIDATION-002"
_ERROR_OUTPUT_DIR: Final[str] = "REPORT-FACTOR-VALIDATION-003"
_ERROR_FORMAT: Final[str] = "REPORT-FACTOR-VALIDATION-004"

_SHEET_SUMMARY: Final[str] = "Summary"
_SHEET_TOP: Final[str] = "Top Factors"
_SHEET_REJECTED: Final[str] = "Rejected Factors"

_OOXML_PKG: Final[str] = "http://schemas.openxmlformats.org/package/2006"
_OOXML_DOC: Final[str] = "http://schemas.openxmlformats.org/officeDocument/2006"
_OOXML_SS: Final[str] = "application/vnd.openxmlformats-officedocument.spreadsheetml"

_CONTENT_TYPES_XML: Final[str] = "\n".join(
    [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<Types xmlns="{_OOXML_PKG}/content-types">',
        ('  <Default Extension="rels" ContentType=' f'"{_OOXML_PKG}/relationships+xml"/>'),
        '  <Default Extension="xml" ContentType="application/xml"/>',
        ('  <Override PartName="/xl/workbook.xml" ' f'ContentType="{_OOXML_SS}.sheet.main+xml"/>'),
        (
            '  <Override PartName="/xl/worksheets/sheet1.xml" '
            f'ContentType="{_OOXML_SS}.worksheet+xml"/>'
        ),
        (
            '  <Override PartName="/xl/worksheets/sheet2.xml" '
            f'ContentType="{_OOXML_SS}.worksheet+xml"/>'
        ),
        (
            '  <Override PartName="/xl/worksheets/sheet3.xml" '
            f'ContentType="{_OOXML_SS}.worksheet+xml"/>'
        ),
        ('  <Override PartName="/xl/styles.xml" ' f'ContentType="{_OOXML_SS}.styles+xml"/>'),
        "</Types>",
        "",
    ]
)

_ROOT_RELS_XML: Final[str] = "\n".join(
    [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<Relationships xmlns="{_OOXML_PKG}/relationships">',
        (
            '  <Relationship Id="rId1" '
            f'Type="{_OOXML_DOC}/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
        ),
        "</Relationships>",
        "",
    ]
)

_STYLES_XML: Final[str] = "\n".join(
    [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        ("<styleSheet " 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'),
        '  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>',
        '  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>',
        '  <borders count="1"><border/></borders>',
        '  <cellStyleXfs count="1"><xf/></cellStyleXfs>',
        '  <cellXfs count="1"><xf xfId="0"/></cellXfs>',
        "</styleSheet>",
        "",
    ]
)


class FactorValidationReporter:
    """Read-only reporter for factor validation researcher artifacts.

    Attributes:
        manager: Order manager whose ledgers are reported.
        output_dir: Destination directory for report files.
    """

    def __init__(
        self,
        repository: FactorValidationRepository,
        *,
        manager: str,
        output_dir: Path | None = None,
        exchange: Exchange = EXCHANGE_BINANCE,
        market: Market = MARKET_USDT_PERPETUAL,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize a read-only factor validation reporter.

        Args:
            repository: Factor validation repository used for discovery/load.
            manager: Order manager identity whose ledgers are reported.
            output_dir: Directory receiving report files. Defaults to
                ``reports/factor_validation/<manager>``.
            exchange: Exchange identity for partition discovery.
            market: Market segment for partition discovery.
            logger: Optional logger instance.

        Raises:
            ReportingValidationError: If ``manager`` is blank.
        """
        cleaned_manager = manager.strip()
        if not cleaned_manager:
            raise ReportingValidationError(
                "manager must be a non-empty string",
                error_code=_ERROR_MANAGER,
            )
        self._repository = repository
        self._manager = cleaned_manager
        self._exchange = exchange
        self._market = market
        self._logger = logger if logger is not None else _logger
        root = DEFAULT_OUTPUT_ROOT if output_dir is None else Path(output_dir)
        self._output_dir = root / self._manager
        self._raw: pl.DataFrame | None = None
        self._summary: pl.DataFrame | None = None
        self._generated_at: datetime | None = None

    @property
    def manager(self) -> str:
        """Return the manager identity being reported."""
        return self._manager

    @property
    def output_dir(self) -> Path:
        """Return the report output directory for this manager."""
        return self._output_dir

    def load(self) -> pl.DataFrame:
        """Load every factor validation ledger for the selected manager.

        Returns:
            Concatenated ledger frame for all discovered partitions.

        Raises:
            ReportingValidationError: If no partitions exist for the manager.
        """
        partitions = self._repository.discover(
            managers=(self._manager,),
            exchange=self._exchange,
            market=self._market,
        )
        if not partitions:
            raise ReportingValidationError(
                f"No factor validation partitions found for manager '{self._manager}'",
                error_code=_ERROR_NO_PARTITIONS,
                details={"manager": self._manager},
                recovery_suggestion=(
                    "Generate factor validation ledgers before running the report CLI."
                ),
            )

        frames: list[pl.DataFrame] = []
        for partition in partitions:
            frame = self._repository.load(
                manager=partition.manager,
                exchange=partition.exchange,
                market=partition.market,
                timeframe=partition.timeframe,
                year=partition.year,
            )
            frames.append(frame)
            self._logger.debug(
                "Loaded factor validation partition for reporting",
                extra={
                    "manager": partition.manager,
                    "timeframe": partition.timeframe,
                    "year": partition.year,
                    "rows": frame.height,
                },
            )

        combined = pl.concat(frames, how="vertical_relaxed")
        self._raw = combined
        self._summary = None
        self._generated_at = datetime.now(tz=UTC)
        self._logger.info(
            "Loaded factor validation ledgers for reporting",
            extra={
                "manager": self._manager,
                "partitions": len(partitions),
                "rows": combined.height,
            },
        )
        return combined

    def summarize(self) -> pl.DataFrame:
        """Build a ranked one-row-per-factor validation summary.

        Returns:
            Summary frame sorted by Rank IC, IC, ICIR, then monotonicity.
        """
        raw = self._ensure_loaded()
        if self._summary is None:
            self._summary = _build_summary(raw)
            self._logger.info(
                "Built factor validation summary",
                extra={"manager": self._manager, "factors": self._summary.height},
            )
        return self._summary

    def write_csv(self, output_dir: Path | None = None) -> Path:
        """Write summary, top-factor, and rejected-factor CSV exports.

        Args:
            output_dir: Optional override for the manager report directory.

        Returns:
            Directory containing the written CSV files.
        """
        destination = self._resolve_output_dir(output_dir)
        summary = self.summarize()
        top = _top_factors(summary)
        rejected = _rejected_factors(summary)

        summary_path = destination / "validation_summary.csv"
        top_path = destination / "top_factors.csv"
        rejected_path = destination / "rejected_factors.csv"
        summary.write_csv(summary_path)
        top.write_csv(top_path)
        rejected.write_csv(rejected_path)
        self._logger.info(
            "Wrote factor validation CSV reports",
            extra={
                "manager": self._manager,
                "output_dir": str(destination),
                "summary_rows": summary.height,
                "top_rows": top.height,
                "rejected_rows": rejected.height,
            },
        )
        return destination

    def write_excel(self, output_dir: Path | None = None) -> Path:
        """Write the Excel workbook with Summary, Top, and Rejected sheets.

        Args:
            output_dir: Optional override for the manager report directory.

        Returns:
            Path to ``validation_summary.xlsx``.
        """
        destination = self._resolve_output_dir(output_dir)
        summary = self.summarize()
        top = _top_factors(summary)
        rejected = _rejected_factors(summary)
        workbook_path = destination / "validation_summary.xlsx"
        _write_xlsx(
            workbook_path,
            {
                _SHEET_SUMMARY: summary,
                _SHEET_TOP: top,
                _SHEET_REJECTED: rejected,
            },
        )
        self._logger.info(
            "Wrote factor validation Excel report",
            extra={"manager": self._manager, "path": str(workbook_path)},
        )
        return workbook_path

    def write_html(self, output_dir: Path | None = None) -> Path:
        """Write the standalone HTML researcher report.

        Args:
            output_dir: Optional override for the manager report directory.

        Returns:
            Path to ``validation_report.html``.
        """
        destination = self._resolve_output_dir(output_dir)
        summary = self.summarize()
        generated_at = self._generated_at or datetime.now(tz=UTC)
        html_path = destination / "validation_report.html"
        html_path.write_text(
            _render_html(
                manager=self._manager,
                summary=summary,
                generated_at=generated_at,
            ),
            encoding="utf-8",
        )
        self._logger.info(
            "Wrote factor validation HTML report",
            extra={"manager": self._manager, "path": str(html_path)},
        )
        return html_path

    def write(
        self,
        *,
        formats: Sequence[ReportFormat] | None = None,
        output_dir: Path | None = None,
    ) -> Path:
        """Write the selected report formats for the loaded manager.

        Args:
            formats: Formats to emit. Defaults to html, csv, and xlsx.
            output_dir: Optional override for the manager report directory.

        Returns:
            Destination directory containing the written artifacts.

        Raises:
            ReportingValidationError: If an unsupported format is requested.
        """
        selected = tuple(REPORT_FORMATS if formats is None else formats)
        for item in selected:
            if item not in REPORT_FORMATS:
                raise ReportingValidationError(
                    f"Unsupported report format '{item}'",
                    error_code=_ERROR_FORMAT,
                    details={"format": item, "allowed": list(REPORT_FORMATS)},
                )
        destination = self._resolve_output_dir(output_dir)
        if "csv" in selected:
            self.write_csv(destination)
        if "xlsx" in selected:
            self.write_excel(destination)
        if "html" in selected:
            self.write_html(destination)
        return destination

    def _ensure_loaded(self) -> pl.DataFrame:
        """Return the cached ledger frame, loading it when necessary."""
        if self._raw is None:
            return self.load()
        return self._raw

    def _resolve_output_dir(self, output_dir: Path | None) -> Path:
        """Resolve and create the report destination directory."""
        destination = self._output_dir if output_dir is None else Path(output_dir)
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ReportingValidationError(
                f"Unable to create report output directory '{destination}'",
                error_code=_ERROR_OUTPUT_DIR,
                details={"output_dir": str(destination)},
            ) from exc
        return destination


def _build_summary(raw: pl.DataFrame) -> pl.DataFrame:
    """Collapse ledgers to one ranked row per factor identity."""
    if raw.is_empty():
        return raw.select(list(SUMMARY_COLUMNS)).clear()

    return (
        raw.sort("validation_time", descending=True, nulls_last=True)
        .unique(subset=["factor_name", "factor_version"], keep="first")
        .select(list(SUMMARY_COLUMNS))
        .sort(
            by=list(_SORT_COLUMNS),
            descending=[True, True, True, True],
            nulls_last=True,
        )
    )


def _top_factors(summary: pl.DataFrame) -> pl.DataFrame:
    """Return the top ranked factors for export."""
    return summary.head(TOP_FACTOR_COUNT)


def _rejected_factors(summary: pl.DataFrame) -> pl.DataFrame:
    """Return FAIL-status factors with researcher-facing column names."""
    rejected = summary.filter(pl.col("status") == FactorValidationStatus.FAIL.value)
    selected = rejected.select([source for source, _ in _REJECTED_EXPORT_COLUMNS])
    rename_map = {source: target for source, target in _REJECTED_EXPORT_COLUMNS}
    return selected.rename(rename_map)


def _mean_or_none(frame: pl.DataFrame, column: str) -> float | None:
    """Return the mean of ``column``, or ``None`` when undefined."""
    if frame.is_empty() or column not in frame.columns:
        return None
    series = frame.get_column(column).drop_nulls()
    if series.is_empty():
        return None
    value = series.mean()
    return _as_optional_float(value)


def _unique_joined(frame: pl.DataFrame, column: str) -> str:
    """Join sorted unique non-null string values for metadata display."""
    if frame.is_empty() or column not in frame.columns:
        return ""
    values = frame.get_column(column).drop_nulls().cast(pl.String).unique().sort().to_list()
    return ", ".join(str(value) for value in values)


def _format_optional_float(value: float | None, *, digits: int = 6) -> str:
    """Format an optional float for HTML display."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _status_counts(summary: pl.DataFrame) -> Mapping[str, int]:
    """Count PASS / FAIL / SKIPPED rows in the summary."""
    counts = {
        FactorValidationStatus.PASS.value: 0,
        FactorValidationStatus.FAIL.value: 0,
        FactorValidationStatus.SKIPPED.value: 0,
    }
    if summary.is_empty():
        return counts
    for row in summary.group_by("status").len().iter_rows(named=True):
        status = str(row["status"])
        if status in counts:
            counts[status] = int(row["len"])
    return counts


def _as_optional_float(value: object) -> float | None:
    """Convert a Polars scalar to ``float`` when present."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return None


def _distribution_rows(summary: pl.DataFrame, column: str) -> dict[str, str]:
    """Compute simple distribution statistics for one metric column."""
    empty = {
        "count": "0",
        "mean": "n/a",
        "std": "n/a",
        "min": "n/a",
        "p25": "n/a",
        "median": "n/a",
        "p75": "n/a",
        "max": "n/a",
    }
    if summary.is_empty() or column not in summary.columns:
        return empty
    series = summary.get_column(column).drop_nulls()
    if series.is_empty():
        return empty
    return {
        "count": str(series.len()),
        "mean": _format_optional_float(_as_optional_float(series.mean())),
        "std": _format_optional_float(_as_optional_float(series.std())),
        "min": _format_optional_float(_as_optional_float(series.min())),
        "p25": _format_optional_float(_as_optional_float(series.quantile(0.25))),
        "median": _format_optional_float(_as_optional_float(series.quantile(0.50))),
        "p75": _format_optional_float(_as_optional_float(series.quantile(0.75))),
        "max": _format_optional_float(_as_optional_float(series.max())),
    }


def _html_table(frame: pl.DataFrame, *, max_rows: int | None = None) -> str:
    """Render a Polars frame as a simple HTML table."""
    view = frame if max_rows is None else frame.head(max_rows)
    if view.is_empty():
        return "<p><em>No rows.</em></p>"
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in view.columns)
    body_rows: list[str] = []
    for row in view.iter_rows():
        cells = "".join(
            f"<td>{html.escape('' if value is None else str(value))}</td>" for value in row
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<table>\n<thead><tr>"
        + headers
        + "</tr></thead>\n<tbody>\n"
        + "\n".join(body_rows)
        + "\n</tbody>\n</table>"
    )


def _render_html(
    *,
    manager: str,
    summary: pl.DataFrame,
    generated_at: datetime,
) -> str:
    """Render the standalone HTML researcher report."""
    counts = _status_counts(summary)
    top = _top_factors(summary)
    rejected = _rejected_factors(summary)
    avg_ic = _mean_or_none(summary, "information_coefficient")
    avg_rank_ic = _mean_or_none(summary, "rank_information_coefficient")
    avg_icir = _mean_or_none(summary, "ic_information_ratio")
    avg_mono = _mean_or_none(summary, "monotonicity_score")
    dataset_version = _unique_joined(summary, "dataset_version")
    label_version = _unique_joined(summary, "label_version")

    distribution_sections: list[str] = []
    for column, label in _METRIC_DISTRIBUTION_COLUMNS:
        stats = _distribution_rows(summary, column)
        distribution_sections.append(
            f"<h3>{html.escape(label)}</h3>\n"
            "<table>\n"
            "<thead><tr>"
            "<th>count</th><th>mean</th><th>std</th><th>min</th>"
            "<th>p25</th><th>median</th><th>p75</th><th>max</th>"
            "</tr></thead>\n"
            "<tbody><tr>"
            f"<td>{html.escape(stats['count'])}</td>"
            f"<td>{html.escape(stats['mean'])}</td>"
            f"<td>{html.escape(stats['std'])}</td>"
            f"<td>{html.escape(stats['min'])}</td>"
            f"<td>{html.escape(stats['p25'])}</td>"
            f"<td>{html.escape(stats['median'])}</td>"
            f"<td>{html.escape(stats['p75'])}</td>"
            f"<td>{html.escape(stats['max'])}</td>"
            "</tr></tbody>\n</table>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Factor Validation Report — {html.escape(manager)}</title>
  <style>
    body {{ font-family: Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
    h1, h2, h3 {{ color: #111; }}
    table {{ border-collapse: collapse; margin: 1rem 0 2rem; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f3f3f3; }}
    .meta td {{ width: 50%; }}
  </style>
</head>
<body>
  <h1>Factor Validation Report</h1>

  <h2>1. Executive Summary</h2>
  <table class="meta">
    <tbody>
      <tr><th>Total Factors</th><td>{summary.height}</td></tr>
      <tr><th>PASS</th><td>{counts[FactorValidationStatus.PASS.value]}</td></tr>
      <tr><th>FAIL</th><td>{counts[FactorValidationStatus.FAIL.value]}</td></tr>
      <tr><th>SKIPPED</th><td>{counts[FactorValidationStatus.SKIPPED.value]}</td></tr>
      <tr><th>Average IC</th><td>{html.escape(_format_optional_float(avg_ic))}</td></tr>
      <tr><th>Average Rank IC</th><td>{html.escape(_format_optional_float(avg_rank_ic))}</td></tr>
      <tr><th>Average ICIR</th><td>{html.escape(_format_optional_float(avg_icir))}</td></tr>
      <tr><th>Average Monotonicity</th><td>{html.escape(_format_optional_float(avg_mono))}</td></tr>
    </tbody>
  </table>

  <h2>2. Top Factors</h2>
  {_html_table(top)}

  <h2>3. Rejected Factors</h2>
  {_html_table(rejected)}

  <h2>4. Metric Distributions</h2>
  {"\n  ".join(distribution_sections)}

  <h2>5. Status Breakdown</h2>
  <table class="meta">
    <tbody>
      <tr><th>PASS</th><td>{counts[FactorValidationStatus.PASS.value]}</td></tr>
      <tr><th>FAIL</th><td>{counts[FactorValidationStatus.FAIL.value]}</td></tr>
      <tr><th>SKIPPED</th><td>{counts[FactorValidationStatus.SKIPPED.value]}</td></tr>
    </tbody>
  </table>

  <h2>6. Dataset Metadata</h2>
  <table class="meta">
    <tbody>
      <tr><th>manager</th><td>{html.escape(manager)}</td></tr>
      <tr><th>generation timestamp</th><td>{html.escape(generated_at.isoformat())}</td></tr>
      <tr><th>dataset version</th><td>{html.escape(dataset_version)}</td></tr>
      <tr><th>label version</th><td>{html.escape(label_version)}</td></tr>
    </tbody>
  </table>
</body>
</html>
"""


def _col_letter(index: int) -> str:
    """Convert a zero-based column index to an Excel column letter."""
    remaining = index + 1
    letters: list[str] = []
    while remaining > 0:
        remaining, remainder = divmod(remaining - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _cell_xml(row_number: int, column_index: int, value: object) -> str:
    """Serialize one worksheet cell as Office Open XML."""
    ref = f"{_col_letter(column_index)}{row_number}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, int) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    if isinstance(value, float):
        if value != value:  # NaN
            return f'<c r="{ref}"/>'
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = xml_escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _worksheet_xml(frame: pl.DataFrame) -> str:
    """Serialize a DataFrame as an OOXML worksheet part."""
    rows_xml: list[str] = []
    header_cells = "".join(
        _cell_xml(1, index, column) for index, column in enumerate(frame.columns)
    )
    rows_xml.append(f'<row r="1">{header_cells}</row>')
    for row_offset, values in enumerate(frame.iter_rows(), start=2):
        cells = "".join(
            _cell_xml(row_offset, column_index, value) for column_index, value in enumerate(values)
        )
        rows_xml.append(f'<row r="{row_offset}">{cells}</row>')
    sheet_data = "\n".join(rows_xml)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
        f"<sheetData>\n{sheet_data}\n</sheetData>\n"
        "</worksheet>\n"
    )


def _write_xlsx(path: Path, sheets: Mapping[str, pl.DataFrame]) -> None:
    """Write a multi-sheet XLSX workbook using only the standard library."""
    sheet_items = list(sheets.items())
    if len(sheet_items) != 3:
        raise ReportingValidationError(
            "Excel workbook requires exactly three sheets",
            error_code=_ERROR_FORMAT,
            details={"sheet_count": len(sheet_items)},
        )

    workbook_sheets = "\n".join(
        (f'<sheet name="{xml_escape(name)}" sheetId="{index}" ' f'r:id="rId{index}"/>')
        for index, (name, _) in enumerate(sheet_items, start=1)
    )
    workbook_xml = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            (
                "<workbook "
                'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                f'xmlns:r="{_OOXML_DOC}/relationships">'
            ),
            f"<sheets>\n{workbook_sheets}\n</sheets>",
            "</workbook>",
            "",
        ]
    )
    relationship_lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<Relationships xmlns="{_OOXML_PKG}/relationships">',
    ]
    for index in range(1, len(sheet_items) + 1):
        relationship_lines.append(
            f'<Relationship Id="rId{index}" '
            f'Type="{_OOXML_DOC}/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    relationship_lines.extend(
        [
            (
                '<Relationship Id="rId4" '
                f'Type="{_OOXML_DOC}/relationships/styles" '
                'Target="styles.xml"/>'
            ),
            "</Relationships>",
            "",
        ]
    )
    workbook_rels = "\n".join(relationship_lines)

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        archive.writestr("_rels/.rels", _ROOT_RELS_XML)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", _STYLES_XML)
        for index, (_, frame) in enumerate(sheet_items, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(frame))
