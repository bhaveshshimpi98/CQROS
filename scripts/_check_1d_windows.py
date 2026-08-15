"""Check 1d validation window vs available history."""
import polars as pl
from datetime import datetime, UTC

v = pl.read_parquet("data/factor_validation/default/binance/usdt_perpetual/1d/2026.parquet")
windows = v.select(["validation_start_time", "validation_end_time"]).unique().sort("validation_start_time")
for r in windows.iter_rows(named=True):
    st = datetime.fromtimestamp(r["validation_start_time"] / 1000, UTC)
    et = datetime.fromtimestamp(r["validation_end_time"] / 1000, UTC)
    bars = (r["validation_end_time"] - r["validation_start_time"]) // 86_400_000 + 1
    print(f"  {st.date()} -> {et.date()} bars={bars}")

nz = v.filter(pl.col("observations") > 0).select(
    ["factor_name", "observations", "validation_start_time", "validation_end_time"]
)
print("non-zero obs factors with window:")
for r in nz.iter_rows(named=True):
    st = datetime.fromtimestamp(r["validation_start_time"] / 1000, UTC)
    et = datetime.fromtimestamp(r["validation_end_time"] / 1000, UTC)
    bars = (r["validation_end_time"] - r["validation_start_time"]) // 86_400_000 + 1
    fname = r["factor_name"]
    obs = r["observations"]
    print(f"  {fname} obs={obs} window={bars}bars {st.date()}->{et.date()}")
