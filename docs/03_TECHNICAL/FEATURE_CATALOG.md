# CQROS Feature Catalog

Version: 1.0.0

Status: Active

---

# 1. Purpose

This document defines every engineered feature used by CQROS.

Each feature specification includes

- Definition
- Formula
- Required inputs
- Lookback period
- Update frequency
- Normalization
- Leakage constraints
- Intended predictive use

The catalog serves as the canonical reference for the
Feature Engineering Layer.

---

# 2. Feature Design Principles

Every feature must satisfy the following requirements.

- Deterministic
- Reproducible
- Timestamp aligned
- No look-ahead bias
- Version controlled
- Documented
- Unit tested

---

# 3. Feature Categories

CQROS organizes features into the following groups.

1. Price Action
2. Trend
3. Momentum
4. Volatility
5. Volume
6. Market Structure
7. Order Flow
8. Order Book
9. Derivatives
10. Cross-Asset
11. Statistical
12. Regime
13. On-Chain
14. Macro
15. Machine Learning Meta Features

---

# 4. Feature Template

Every feature follows this template.

Name

Description

Inputs

Formula

Lookback

Update Frequency

Normalization

Output Type

Typical Range

Leakage Risk

Dependencies

Version

---

# PRICE ACTION FEATURES

---

# PA001 Close Price

Description

Latest closing price.

Inputs

- Close

Formula

```
close
```

Lookback

1 candle

Update

Every candle close

Normalization

None

Output

Float

Leakage

None

---

# PA002 Open Price

Inputs

Open

Formula

```
open
```

---

# PA003 High Price

Inputs

High

Formula

```
high
```

---

# PA004 Low Price

Inputs

Low

Formula

```
low
```

---

# PA005 Candle Body

Formula

```
close - open
```

Measures buying or selling pressure.

---

# PA006 Absolute Candle Body

Formula

```
abs(close - open)
```

Removes directional bias.

---

# PA007 Upper Wick

Formula

```
high - max(open, close)
```

Measures rejection from highs.

---

# PA008 Lower Wick

Formula

```
min(open, close) - low
```

Measures rejection from lows.

---

# PA009 Total Range

Formula

```
high - low
```

Measures intrabar volatility.

---

# PA010 Body Ratio

Formula

```
abs(close-open)/(high-low)
```

Higher values indicate stronger directional candles.

---

# PA011 Upper Wick Ratio

Formula

```
upper_wick / total_range
```

---

# PA012 Lower Wick Ratio

Formula

```
lower_wick / total_range
```

---

# PA013 Gap Up

Formula

```
open > previous_high
```

Boolean feature.

---

# PA014 Gap Down

Formula

```
open < previous_low
```

Boolean feature.

---

# PA015 Typical Price

Formula

```
(high + low + close) / 3
```

---

# PA016 Median Price

Formula

```
(high + low) / 2
```

---

# PA017 Weighted Close

Formula

```
(high + low + 2*close)/4
```

---

# PA018 OHLC4

Formula

```
(open + high + low + close)/4
```

---

# PA019 HLC3

Formula

```
(high + low + close)/3
```

---

# PA020 Candle Direction

Output

```
1

0

-1
```

Formula

```
sign(close-open)
```

---

# PA021 Return

Formula

```
(close / previous_close) - 1
```

---

# PA022 Log Return

Formula

```
ln(close / previous_close)
```

Preferred for statistical modeling.

---

# PA023 Cumulative Return

Rolling cumulative returns over configurable windows.

Default windows

- 5
- 10
- 20
- 50

---

# PA024 Rolling Mean Price

Rolling average.

Configurable windows

- 5
- 10
- 20
- 50
- 100

---

# PA025 Rolling Median Price

Rolling median of close prices.

More robust to outliers.

---

# PA026 Rolling Maximum

Highest close over window.

---

# PA027 Rolling Minimum

Lowest close over window.

---

# PA028 Distance From Rolling High

Formula

```
(close - rolling_high)
/ rolling_high
```

---

# PA029 Distance From Rolling Low

Formula

```
(close - rolling_low)
/ rolling_low
```

---

# PA030 Price Percentile

Rolling percentile rank of current close within the
lookback window.

Useful for identifying relative price position.

---

# Common Lookback Windows

Unless otherwise specified, CQROS supports

- 5
- 10
- 20
- 50
- 100
- 200

The Feature Engine may compute multiple windows for the
same feature.

---

# Timestamp Alignment Rules

Every feature

- Uses only historical information
- Is computed after candle close
- Is timestamped with the close time of the source candle
- Must never access future observations

---

# Missing Data Policy

If insufficient history exists

- Output NULL during warm-up
- Do not forward-fill by default
- Preserve chronological integrity

---

# Feature Versioning

Every feature includes

- Feature ID
- Semantic version
- Change history
- Deprecation status

Breaking formula changes require a new major version.

---

# Summary

Price Action features form the foundation of CQROS.
They describe raw market behavior and provide the inputs
for higher-level trend, momentum, volatility, and machine
learning features.