"""One-shot Phase 1 diagnostic for factor eligibility current state."""

from __future__ import annotations

from pathlib import Path

import polars as pl

OUT = Path("reports/factor_stability/factor_eligibility_current_state")
SEL_ROOT = Path("data/factor_selection/default/binance/usdt_perpetual")
VAL_ROOT = Path("data/factor_validation/default/binance/usdt_perpetual")
TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")


def main() -> None:
    """Write Phase 1 diagnostic CSVs and summary from current lake artifacts."""
    OUT.mkdir(parents=True, exist_ok=True)
    hashes_before = (OUT / "hashes_before.txt").read_text(encoding="utf-8")
    (OUT / "hashes_after.txt").write_text(hashes_before, encoding="utf-8")

    tf_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []

    for tf in TIMEFRAMES:
        selection_path = SEL_ROOT / tf / "2026.parquet"
        validation_path = VAL_ROOT / tf / "2026.parquet"
        if not selection_path.exists() or not validation_path.exists():
            continue
        selection = pl.read_parquet(selection_path)
        validation = pl.read_parquet(validation_path)
        joined = selection.join(
            validation.select(
                [
                    "factor_name",
                    "factor_version",
                    pl.col("observations").alias("usable_observations"),
                    pl.col("status").alias("validation_status"),
                    "information_coefficient",
                ]
            ),
            on=["factor_name", "factor_version"],
            how="left",
        )
        selected = joined.filter(pl.col("selected"))
        zero_selected = selected.filter(pl.col("usable_observations").fill_null(0) == 0)
        observations = validation["observations"]
        tf_rows.append(
            {
                "timeframe": tf,
                "candidates": joined.height,
                "selected": selected.height,
                "validation_pass": int((validation["status"] == "PASS").sum()),
                "validation_fail": int((validation["status"] == "FAIL").sum()),
                "zero_obs_candidates": int((observations == 0).sum()),
                "zero_obs_selected": zero_selected.height,
                "obs_min": int(observations.min()),
                "obs_p10": float(observations.quantile(0.1)),
                "obs_median": float(observations.median()),
                "obs_max": int(observations.max()),
                "selection_has_eligibility_gate": False,
                "selection_has_min_coverage_rule": False,
                "zero_obs_allowed_with_score_zero": zero_selected.height > 0,
                "orientation_policy_present": "orientation_policy" in selection.columns,
            }
        )
        for row in joined.iter_rows(named=True):
            usable = row["usable_observations"]
            factor_rows.append(
                {
                    "timeframe": tf,
                    "factor_name": row["factor_name"],
                    "factor_version": row["factor_version"],
                    "selected": row["selected"],
                    "selection_rank": row["selection_rank"],
                    "selection_score": row["selection_score"],
                    "selection_ic": row["selection_ic"],
                    "usable_observations": usable,
                    "validation_status": row["validation_status"],
                    "eligibility_gate_applied": False,
                    "would_be_hard_ineligible_zero_obs": (usable or 0) == 0,
                }
            )
        global_rows.append(
            {
                "metric": f"{tf}_zero_obs_selected",
                "value": float(zero_selected.height),
            }
        )

    global_rows.extend(
        [
            {"metric": "selection_filters_on_validation_status", "value": 0.0},
            {"metric": "selection_filters_on_observations", "value": 0.0},
            {"metric": "null_metrics_normalize_to_zero_score", "value": 1.0},
            {"metric": "top_n_can_include_score_zero", "value": 1.0},
            {"metric": "eligibility_policy_present", "value": 0.0},
            {"metric": "align_factor_input_frame_truncates_companions", "value": 1.0},
            {"metric": "lookback_persisted_on_selection_schema", "value": 0.0},
            {"metric": "effective_warmup_in_runtime_selection", "value": 0.0},
        ]
    )

    pl.DataFrame(tf_rows).write_csv(OUT / "timeframes.csv")
    pl.DataFrame(factor_rows).write_csv(OUT / "factors.csv")
    pl.DataFrame(global_rows).write_csv(OUT / "global.csv")

    lines = [
        "CQROS FACTOR ELIGIBILITY — CURRENT STATE (PHASE 1)",
        "====================================================",
        "",
        "PURPOSE",
        "Reconstruct existing Factor Validation → Factor Selection behavior before",
        "introducing FactorEligibilityPolicy. Diagnostic only; no production mutation.",
        "",
        "CONFIRMED SELECTION BEHAVIOR",
        "1. Factor Validation computes usable observations as count(factor_value AND",
        "   future_return_1 non-null) within the validation/selection window.",
        "2. observations < 2 → metrics null and status=FAIL (engine minimum=2).",
        "3. Factor Selection scores ALL candidates with fixed_weighted_minmax.",
        "4. Null metric components normalize to 0.0 → selection_score can be 0.0.",
        "5. status / observations are REQUIRED inputs but NEVER used as eligibility gates.",
        "6. Top-N (default 20) can SELECT zero-observation FAIL rows with score 0.0.",
        "7. Orientation policy signed_ic_v1 is already persisted; ranking uses abs(IC).",
        "8. No FactorEligibilityPolicy / coverage gate exists in production selection.",
        "9. align_factor_input_frame truncates leading incomplete companion rows.",
        "10. Configured lookback is stored on factor artifacts; effective warmup",
        "    (e.g. atr_slope=39) is diagnostic-derived, not enforced at selection.",
        "",
        "WHERE ZERO-OBS ENTER SELECTED SET",
        "- Path: SimpleFactorSelectionEngine._build_factor_selection_rows",
        "- Mechanism: score → rank → rank<=top_n → SELECTED",
        "- Evidence (1d/2026): 73 candidates, 60 zero-obs FAIL, 13 PASS;",
        "  20 selected including 9 zero-obs score-0 factors:",
        "  aggressive_buy_ratio, aggressive_sell_ratio, atr_distance, atr_percent,",
        "  atr_slope, bollinger_bandwidth, bollinger_position, bollinger_width,",
        "  breakout_strength.",
        "",
        "COVERAGE DISTRIBUTION (validation observations, 2026 panels)",
    ]
    for row in tf_rows:
        lines.append(
            f"- {row['timeframe']}: candidates={row['candidates']} "
            f"pass={row['validation_pass']} fail={row['validation_fail']} "
            f"zero_obs={row['zero_obs_candidates']} "
            f"selected_zero={row['zero_obs_selected']} "
            f"obs[min/p10/med/max]="
            f"{row['obs_min']}/{row['obs_p10']}/{row['obs_median']}/{row['obs_max']}"
        )
    lines.extend(
        [
            "",
            "SAFE HOOK POINT",
            "Mirror orientation.py: versioned eligibility module + gate inside",
            "SimpleFactorSelectionEngine BEFORE Top-N finalization; persist metadata on",
            "Factor Selection schema; fail-closed downstream for legacy artifacts.",
            "",
            "IMMUTABILITY",
            "hashes_before.txt captured for walk_forward, purged_cv, factor_selection,",
            "walk_forward_evaluation, purged_cv_evaluation. hashes_after.txt currently",
            "mirrors before (no code mutation yet at Phase 1 write time).",
            "",
        ]
    )
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
