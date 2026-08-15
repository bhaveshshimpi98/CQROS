# CQROS Configuration Reference

Version: 1.0.0

Status: Active

---

# 1. Purpose

This document defines every configurable option used by
CQROS.

No production behavior should rely on hard-coded values.

Configuration must be

- Explicit
- Versioned
- Validated
- Documented
- Environment-aware

---

# 2. Configuration Sources

Configuration is loaded in the following priority order
(highest first):

```
Runtime Overrides

↓

Command Line

↓

Environment Variables

↓

Environment Configuration File

↓

Default Configuration
```

---

# 3. Configuration Files

```
configs/

default.toml

development.toml

testing.toml

paper.toml

production.toml

secrets.example.toml
```

Secrets must never be committed.

---

# 4. Environment Variables

Environment variables override file values.

Examples

```
CQROS_ENV

CQROS_LOG_LEVEL

CQROS_DATA_PATH

CQROS_CACHE_PATH

CQROS_EXCHANGE

CQROS_BROKER

CQROS_DATABASE_URL

CQROS_REDIS_URL
```

---

# 5. General Settings

| Key | Type | Default |
|------|------|---------|
| app.name | string | CQROS |
| app.version | string | 1.0.0 |
| app.environment | string | development |
| app.timezone | string | UTC |
| app.debug | bool | false |

---

# 6. Logging

| Key | Type | Default |
|------|------|---------|
| logging.level | string | INFO |
| logging.format | string | json |
| logging.console | bool | true |
| logging.file | bool | true |
| logging.directory | string | logs |

Allowed levels

- DEBUG
- INFO
- WARNING
- ERROR
- CRITICAL

---

# 7. Storage

| Key | Default |
|------|---------|
| storage.root | data |
| storage.raw | raw |
| storage.processed | processed |
| storage.features | features |
| storage.models | models |
| storage.reports | reports |

---

# 8. Database

DuckDB

| Key | Default |
|------|---------|
| duckdb.path | data/database.duckdb |

PostgreSQL

| Key | Default |
|------|---------|
| postgres.host | localhost |
| postgres.port | 5432 |
| postgres.database | cqros |

---

# 9. Cache

Optional

Redis

| Key | Default |
|------|---------|
| redis.enabled | false |
| redis.host | localhost |
| redis.port | 6379 |

---

# 10. Exchange

| Key | Default |
|------|---------|
| exchange.name | binance |
| exchange.market | usdt_perpetual |
| exchange.testnet | false |

---

# 11. Market Data

| Key | Default |
|------|---------|
| market.timeframes | 1m,5m,15m,1h,4h,1d |
| market.max_symbols | 200 |
| market.history_days | 3650 |

---

# 11a. Download Retention

Dataset-specific historical download defaults. Retention policy belongs in
configuration; downloaders must not hardcode exchange retention limits.

| Key | Default |
|------|---------|
| download.ohlcv_history_days | 3650 |
| download.funding_history_days | 3650 |
| download.futures_data_history_days | 30 |
| download.futures_data_safety_margin_days | 1 |

`futures_data_history_days` is shared by Open Interest, Taker Volume, Global
Long/Short, Top Trader Account Ratio, and Top Trader Position Ratio. Requested
Futures Data windows are clamped to
`futures_data_history_days - futures_data_safety_margin_days` before requests
are sent so start times stay strictly inside exchange retention.

---

# 12. Data Validation

| Key | Default |
|------|---------|
| validation.enabled | true |
| validation.strict | true |
| validation.allow_missing | false |

---

# 13. Dataset Builder

| Key | Default |
|------|---------|
| dataset.versioning | true |
| dataset.compression | zstd |
| dataset.chunk_size | 500000 |

---

# 14. Feature Engineering

| Key | Default |
|------|---------|
| features.parallel | true |
| features.max_workers | auto |
| features.store_intermediate | false |

---

# 15. Model Training

| Key | Default |
|------|---------|
| training.random_seed | 42 |
| training.parallel | true |
| training.save_checkpoints | true |

---

# 16. Hyperparameter Optimization

| Key | Default |
|------|---------|
| hpo.enabled | true |
| hpo.max_trials | 100 |
| hpo.timeout_minutes | 240 |

---

# 17. Strategy Engine

| Key | Default |
|------|---------|
| strategy.enabled | true |
| strategy.max_signals | 20 |
| strategy.parallel | true |

---

# 18. Portfolio

| Key | Default |
|------|---------|
| portfolio.base_currency | USDT |
| portfolio.max_positions | 10 |
| portfolio.allow_fractional | true |

---

# 19. Risk Management

| Key | Default |
|------|---------|
| risk.max_drawdown | 0.15 |
| risk.max_leverage | 3 |
| risk.max_position_size | 0.10 |
| risk.stop_loss_required | true |

---

# 20. Execution

| Key | Default |
|------|---------|
| execution.paper | true |
| execution.max_retries | 3 |
| execution.retry_delay_ms | 500 |

---

# 21. Backtesting

| Key | Default |
|------|---------|
| backtest.initial_capital | 10000 |
| backtest.commission | 0.0004 |
| backtest.slippage | 0.0002 |

---

# 22. Monitoring

| Key | Default |
|------|---------|
| monitoring.enabled | true |
| monitoring.metrics | true |
| monitoring.health | true |

---

# 23. Alerts

| Key | Default |
|------|---------|
| alerts.discord | false |
| alerts.email | false |
| alerts.slack | false |

---

# 24. Security

| Key | Default |
|------|---------|
| security.encrypt_secrets | true |
| security.audit_logging | true |

API credentials must only be supplied through
environment variables or a secure secrets provider.

---

# 25. Performance

| Key | Default |
|------|---------|
| performance.max_workers | auto |
| performance.batch_size | 1000 |

---

# 26. Validation Rules

Every configuration value must be validated for

- Type
- Range
- Required status
- Dependencies
- Allowed values

Startup fails if validation fails.

---

# 27. Configuration Versioning

Every configuration file includes

```
config_version
```

Breaking changes require a new version.

---

# 28. Environment Profiles

Supported profiles

- development
- testing
- paper
- production

Each profile inherits from `default.toml`
and overrides only necessary values.

---

# 29. Example Layout

```
configs/

default.toml

development.toml

testing.toml

paper.toml

production.toml

secrets.example.toml
```

---

# 30. Summary

CQROS configuration is centralized, validated,
versioned, and environment-aware.

Every runtime behavior is controlled through documented
configuration, ensuring reproducibility, portability,
and safe deployment across development, testing, paper
trading, and production environments.