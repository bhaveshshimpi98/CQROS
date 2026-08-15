# CQROS Database Schema

Version: 1.0.0

Status: Active

---

# 1. Purpose

This document defines the logical database schema for CQROS.

CQROS uses two complementary databases.

- DuckDB
- PostgreSQL

Each database serves a different purpose.

---

# 2. Database Responsibilities

## DuckDB

Analytical database.

Used for

- Historical market data
- Feature matrices
- Training datasets
- Backtests
- Research queries

Optimized for

- Large scans
- Aggregations
- OLAP workloads

---

## PostgreSQL

Operational database.

Used for

- Metadata
- Registries
- Audit logs
- Deployments
- Experiments
- Live state

Optimized for

- Transactions
- Consistency
- Small updates

---

# 3. Storage Layout

```
                CQROS

        ┌────────┴────────┐

        ▼                 ▼

     DuckDB         PostgreSQL

 Historical Data     Metadata

 Feature Data        Registry

 Backtests           Experiments

 Analytics           Deployments

 Research            Audit Logs
```

---

# 4. DuckDB Schema

Schemas

```
market

research

features

backtests

analytics
```

---

# 5. market.candles

Primary analytical table.

Columns

```
symbol

timeframe

open_time

close_time

open

high

low

close

volume

quote_volume

trade_count
```

Primary Key

```
(symbol, timeframe, open_time)
```

---

# 6. market.trades

Columns

```
trade_id

symbol

timestamp

price

quantity

side

buyer_maker
```

---

# 7. market.orderbook

Columns

```
symbol

timestamp

bid_levels

ask_levels

checksum
```

---

# 8. market.funding

Columns

```
symbol

funding_time

funding_rate

mark_price
```

---

# 9. market.open_interest

Columns

```
symbol

timestamp

open_interest

notional_value
```

---

# 10. research.datasets

Columns

```
dataset_id

version

created_at

rows

columns

checksum

lineage
```

---

# 11. features.feature_sets

Columns

```
feature_set_id

dataset_id

version

feature_count

created_at
```

---

# 12. features.feature_catalog

Columns

```
feature_name

version

dtype

description

source

enabled
```

---

# 13. analytics.factor_metrics

Columns

```
factor_name

horizon

ic

rank_ic

coverage

stability

created_at
```

---

# 14. analytics.model_metrics

Columns

```
model_id

accuracy

precision

recall

f1

auc

sharpe

sortino

created_at
```

---

# 15. backtests.runs

Columns

```
run_id

strategy

dataset

started_at

completed_at

status
```

---

# 16. backtests.trades

Columns

```
trade_id

run_id

symbol

entry

exit

pnl

fees
```

---

# 17. PostgreSQL Schemas

```
registry

metadata

operations

audit

security
```

---

# 18. registry.models

Columns

```
model_id

version

algorithm

status

created_at

approved_at
```

---

# 19. registry.datasets

Columns

```
dataset_id

version

owner

checksum

created_at
```

---

# 20. registry.features

Columns

```
feature_name

version

status

created_at
```

---

# 21. metadata.assets

Columns

```
symbol

exchange

base_asset

quote_asset

status

tick_size

step_size
```

---

# 22. metadata.exchanges

Columns

```
exchange_id

name

timezone

api_version
```

---

# 23. operations.deployments

Columns

```
deployment_id

version

environment

git_commit

status

created_at
```

---

# 24. operations.live_positions

Columns

```
position_id

symbol

quantity

average_price

unrealized_pnl

opened_at
```

---

# 25. operations.orders

Columns

```
order_id

broker_order_id

status

symbol

side

quantity

price

created_at
```

---

# 26. operations.signals

Columns

```
signal_id

strategy

confidence

direction

entry_price

timestamp
```

---

# 27. operations.portfolios

Columns

```
portfolio_id

equity

cash

margin

updated_at
```

---

# 28. audit.events

Columns

```
event_id

timestamp

actor

action

resource

details
```

---

# 29. audit.logs

Columns

```
log_id

level

service

message

timestamp
```

---

# 30. security.api_keys

Columns

```
provider

key_name

created_at

expires_at
```

Actual secret values are **never stored in plaintext**.

---

# 31. Relationships

```
Exchange

↓

Asset

↓

Market Data

↓

Dataset

↓

Feature Set

↓

Model

↓

Prediction

↓

Signal

↓

Order

↓

Position

↓

Portfolio
```

---

# 32. Indexes

Examples

```
candles

(symbol, timeframe, open_time)

trades

(symbol, timestamp)

signals

(timestamp)

orders

(status)

positions

(symbol)
```

---

# 33. Partitioning

Large analytical tables are partitioned by

```
symbol

↓

year

↓

month
```

Example

```
BTCUSDT/

2023/

01/

part-000.parquet
```

---

# 34. Compression

Preferred

```
Parquet

↓

ZSTD
```

DuckDB reads Parquet directly.

---

# 35. Retention Policy

Raw market data

Permanent

Feature datasets

Permanent

Temporary caches

Configurable

Logs

365 days

Audit logs

Permanent

---

# 36. Migrations

Every schema change

- Versioned
- Reviewed
- Backward compatible when possible

Migration scripts stored in

```
migrations/
```

---

# 37. Backup Strategy

PostgreSQL

Daily backup

DuckDB

Snapshot after dataset publication

Metadata

Continuous backup

---

# 38. Validation

Every table enforces

- Primary keys
- Foreign keys (PostgreSQL)
- Type validation
- Non-null constraints where required

---

# 39. Future Extensions

Planned additions

- Feature lineage tables
- Data quality metrics
- Model lineage graph
- Drift statistics
- Broker execution history
- Multi-exchange support

---

# 40. Summary

CQROS separates analytical and operational storage to
maximize performance, reliability, and maintainability.

DuckDB powers high-performance research and analytics,
while PostgreSQL manages transactional metadata,
registries, deployments, and operational state, providing
a scalable persistence architecture for institutional-grade
quantitative research and trading.