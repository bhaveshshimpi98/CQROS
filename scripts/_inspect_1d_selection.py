"""Inspect 1d factor selection results after eligibility policy injection."""
from __future__ import annotations

import polars as pl

df = pl.read_parquet("data/factor_selection/default/binance/usdt_perpetual/1d/2026.parquet")

sel = df.filter(pl.col("selected"))
not_sel = df.filter(~pl.col("selected"))

print(f"=== SELECTED ({sel.height}) ===")
for row in sel.sort("selection_rank").iter_rows(named=True):
    print(
        f"  rank={row['selection_rank']:2d}  {row['factor_name']:<35}"
        f"  dir={row['selected_direction']:+d}  elig={row['eligibility_status']}"
        f"  obs={row['usable_observations']}  policy={row['eligibility_policy']}"
    )

print()
print(f"=== NOT SELECTED ({not_sel.height}) - eligibility_status breakdown ===")
breakdown = not_sel.group_by("eligibility_status").len().sort("eligibility_status")
for row in breakdown.iter_rows(named=True):
    print(f"  {row['eligibility_status']:<40} count={row['len']}")

print()
zero_obs = not_sel.filter(pl.col("usable_observations") == 0)
print(f"=== ZERO-OBS INELIGIBLE (count={zero_obs.height}) ===")
for row in zero_obs.sort("factor_name").iter_rows(named=True):
    print(f"  {row['factor_name']:<35}  elig={row['eligibility_status']}")

print()
print(f"Total candidates: {df.height}")
print(f"Eligible selected: {sel.height}")
print(f"Ineligible (zero obs): {zero_obs.height}")
print(f"Policy: {sel['eligibility_policy'][0] if sel.height > 0 else 'N/A'}")
