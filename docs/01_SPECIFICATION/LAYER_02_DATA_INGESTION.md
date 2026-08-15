# Layer 02 – Data Ingestion Specification

**Layer ID:** L02

**Layer Name:** Data Ingestion

**Version:** 1.0.0

**Status:** Draft

**Dependencies**

- Layer 00 – Foundation
- Layer 01 – Exchange Connectivity

**Required By**

- Layer 03 – Storage
- Layer 04 – Validation

---

# 1. Purpose

The Data Ingestion layer is responsible for acquiring market data from supported exchanges and transforming it into a normalized internal representation suitable for downstream processing.

It provides reliable, fault-tolerant, observable, and repeatable ingestion pipelines.

No data enters CQROS without passing through this layer.

---

# 2. Responsibilities

This layer is responsible for

- Historical downloads
- Incremental downloads
- Real-time streaming
- Snapshot acquisition
- Symbol discovery
- Time synchronization
- Checkpointing
- Resume support
- Normalization
- Deduplication
- Ingestion scheduling
- Backfilling
- Gap detection

---

# 3. Out of Scope

Layer 02 does not perform

- Storage
- Feature engineering
- Validation
- Statistics
- Machine learning
- Trading
- Portfolio optimization
- Risk management

---

# 4. Supported Data Types

## Market Data

- OHLCV candles
- Trades
- Quotes
- Best Bid/Ask
- Order Book Snapshot
- Order Book Updates
- Ticker

---

## Derivatives Data

- Funding Rates
- Open Interest
- Mark Price
- Index Price
- Insurance Fund
- Premium Index

---

## Exchange Metadata

- Exchange information
- Symbols
- Filters
- Tick sizes
- Lot sizes
- Trading permissions
- Fee schedules

---

## Reference Data

- Trading calendar
- Maintenance windows
- Delisted symbols
- Listing dates

---

# 5. Data Sources

Supported sources

Exchange REST API

Exchange WebSocket

Bulk historical archives

Future

Third-party providers

Internal replay files

Simulation feeds

---

# 6. Architecture

```
Exchange Adapter

↓

REST / WebSocket

↓

Data Collector

↓

Normalizer

↓

Checkpoint Manager

↓

Gap Detector

↓

Output Queue

↓

Layer 03 Storage
```

---

# 7. Package Structure

```
src/cqros/ingestion/

interfaces.py

models.py

config.py

exceptions.py

service.py

scheduler.py

collectors/

normalizers/

streaming/

historical/

incremental/

backfill/

checkpoint/

gap/

metadata/

queue/

validators/

tests/
```

---

# 8. Collectors

Collectors retrieve raw information.

Collectors include

OHLCVCollector

TradeCollector

OrderBookCollector

FundingCollector

OpenInterestCollector

TickerCollector

MetadataCollector

Collectors never write directly to storage.

---

# 9. Historical Downloader

Responsibilities

Download historical ranges

Resume interrupted downloads

Partition requests

Respect rate limits

Validate completeness

Retry failed segments

Support multi-year downloads

---

# 10. Incremental Downloader

Used after historical synchronization.

Downloads only missing information.

Tracks

Latest timestamp

Latest sequence number

Checkpoint version

---

# 11. Streaming Engine

Responsibilities

Persistent WebSocket

Automatic reconnect

Message buffering

Sequence validation

Gap detection

Heartbeat

Recovery

---

# 12. Normalization

All exchanges are converted into CQROS canonical models.

Examples

Binance Kline

↓

Canonical Candle

Bybit Trade

↓

Canonical Trade

OKX Funding

↓

Canonical Funding

Higher layers never process vendor-specific models.

---

# 13. Canonical Models

Layer 02 produces

Candle

Trade

OrderBook

Ticker

FundingRate

OpenInterest

MarkPrice

ExchangeMetadata

SymbolMetadata

---

# 14. Checkpoint Manager

Stores

Latest timestamp

Latest sequence

Latest download

Download progress

Resume token

Checkpoint version

Allows interrupted jobs to continue safely.

---

# 15. Gap Detection

Detect

Missing candles

Missing trades

Missing order book updates

Timestamp discontinuities

Sequence gaps

Recovery automatically requests missing data.

---

# 16. Scheduling

Supports

Manual jobs

Cron jobs

Continuous synchronization

Priority queues

Parallel downloads

Backfill scheduling

---

# 17. Configuration

Configuration includes

Exchange

Symbols

Intervals

Workers

Retry count

Timeout

Chunk size

Checkpoint interval

Streaming buffer size

Heartbeat interval

Configuration validated before execution.

---

# 18. Output Contract

Output objects contain

Payload

Timestamp

Exchange

Market

Symbol

Metadata

Checksum

Version

Source

Reception timestamp

---

# 19. Error Handling

Errors include

ConnectionError

DownloadFailed

StreamInterrupted

SequenceGap

CheckpointCorruption

UnsupportedSymbol

MalformedMessage

Timeout

Errors are recoverable whenever possible.

---

# 20. Retry Strategy

Retry

Temporary network failures

HTTP 5xx

Timeouts

WebSocket disconnects

Do not retry

Authentication errors

Permission errors

Invalid symbols

Malformed requests

---

# 21. Logging

Log

Downloads

Reconnects

Retries

Progress

Bandwidth

Symbols

Queue depth

Failures

Never log

Credentials

Secrets

API keys

---

# 22. Validation

Verify

Timestamp ordering

Sequence ordering

Required fields

Schema

Data types

Precision

Duplicate messages

Validation failures are forwarded to Layer 04.

---

# 23. Performance Requirements

Historical download

Parallelized

Realtime latency

Less than 1 second

Reconnect

Less than 5 seconds

Memory usage

Bounded

Queue growth

Controlled

---

# 24. Thread Safety

Collectors

Independent

Queues

Concurrent-safe

Checkpoint manager

Atomic updates

Streaming engine

Thread-safe

---

# 25. Monitoring

Expose metrics

Downloaded records

Messages/sec

Reconnect count

Gap count

Retry count

Bandwidth

Queue size

Worker utilization

Checkpoint progress

---

# 26. Dependencies

Allowed

```
Ingestion

↓

Foundation

↓

Exchange Connectivity
```

Forbidden

```
Ingestion

↓

Storage

↓

Validation

↓

Features

↓

Portfolio
```

---

# 27. Testing Requirements

Coverage

100%

Tests

Historical download

Streaming

Reconnect

Gap detection

Checkpoint resume

Normalization

Retry logic

Concurrency

Performance smoke tests

Large dataset ingestion

---

# 28. Deliverables

```
ingestion/

interfaces.py

models.py

config.py

exceptions.py

service.py

scheduler.py

collectors/

historical/

streaming/

normalizers/

checkpoint/

gap/

queue/

validators/

tests/
```

---

# 29. Acceptance Criteria

Layer complete when

✓ Historical downloads succeed

✓ Streaming remains stable

✓ Checkpoints resume correctly

✓ Gap detection functions

✓ Canonical models generated

✓ Metadata captured

✓ Performance targets achieved

✓ Unit tests pass

✓ Integration tests pass

✓ Documentation complete

---

# 30. Future Extensions

Planned enhancements

- Multi-exchange synchronization
- Distributed ingestion workers
- Kafka integration
- Apache Arrow streaming
- Real-time compression
- Adaptive scheduling
- Market replay engine
- Tick-level recording
- Cross-exchange synchronization

These enhancements should integrate without changing the public interfaces of Layer 02.

---

# 31. Summary

Layer 02 provides the complete market data acquisition pipeline for CQROS.

It is responsible for reliably collecting, normalizing, checkpointing, and delivering market data from external exchanges while ensuring resilience, observability, and deterministic behavior. Every downstream layer depends on the correctness and completeness of the data produced here.