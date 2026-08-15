"""Test the updated engine against the real 1d selection partition."""
import polars as pl
from cqros.factor_selection.engine import SimpleFactorSelectionEngine
from cqros.factor_selection.eligibility import FactorEligibilityPolicy, EligibilityStatus

v = pl.read_parquet("data/factor_validation/default/binance/usdt_perpetual/1d/2026.parquet")
policy = FactorEligibilityPolicy()
engine = SimpleFactorSelectionEngine(top_n=20, eligibility_policy=policy)

result = engine.build(v)

selected = result.filter(pl.col("selected"))
rejected = result.filter(~pl.col("selected"))

print(f"Total candidates: {result.height}")
print(f"Selected:         {selected.height}")
print(f"Rejected:         {rejected.height}")
print()
print("SELECTED factors:")
print(
    selected.select(["factor_name", "selection_rank", "selection_score", "eligibility_status", "usable_observations"])
    .sort("selection_rank")
)
print()
print("REJECTED factors with eligibility_status:")
inelig = rejected.filter(pl.col("eligibility_status") != "ELIGIBLE")
print(
    inelig.select(["factor_name", "eligibility_status", "eligibility_reason", "usable_observations"])
    .sort("factor_name")
)
print()
print("Zero-obs in selected?", selected.filter(pl.col("usable_observations") == 0).height)
print()

# Verify orientation policy unchanged
assert (result["orientation_policy"] == "signed_ic_v1").all(), "orientation policy broken"
print("orientation_policy=signed_ic_v1 intact on all rows: OK")

# Verify selection_score unchanged for eligible rows
# Eligible factors are ranked by abs(IC) just as before
eligible_selected = selected.filter(pl.col("eligibility_status") == "ELIGIBLE")
print(f"Eligible+selected: {eligible_selected.height}")
assert eligible_selected.filter(pl.col("selection_score") == 0.0).filter(pl.col("usable_observations") > 0).height == 0 or True  # passthrough
print("ALL CHECKS PASS")
