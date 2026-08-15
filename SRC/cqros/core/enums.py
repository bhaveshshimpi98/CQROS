"""CQROS shared enumerations.

Purpose:
    Provide stable, serializable string enumerations used across CQROS
    packages for configuration, persistence, and inter-layer contracts.

Responsibilities:
    - Define canonical domain enumerations for markets, orders, positions,
      signals, risk, models, trades, data, and pipelines
    - Expose string values suitable for configuration and serialization
    - Remain free of business logic and side effects

Dependencies:
    Python standard library only (``enum.StrEnum``).

Public API:
    The enumeration types listed in ``__all__``.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "MarketType",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "PositionSide",
    "SignalSide",
    "RiskLevel",
    "ModelType",
    "TradeStatus",
    "DataSource",
    "DatasetSplit",
    "PipelineStage",
]


class MarketType(StrEnum):
    """Tradeable market category.

    Attributes:
        SPOT: Immediate settlement spot market.
        PERPETUAL: Perpetual swap / perpetual futures market.
        FUTURES: Dated futures contract market.
        OPTIONS: Options contract market.
    """

    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURES = "futures"
    OPTIONS = "options"


class OrderSide(StrEnum):
    """Order direction relative to the base asset.

    Attributes:
        BUY: Acquire the base asset.
        SELL: Dispose of the base asset.
    """

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Supported order execution styles.

    Values follow the CQROS execution policy. Mandatory types are
    ``MARKET``, ``LIMIT``, ``STOP_MARKET``, and ``STOP_LIMIT``.

    Attributes:
        MARKET: Execute immediately at the prevailing market price.
        LIMIT: Execute at the limit price or better.
        STOP_MARKET: Trigger a market order when the stop price is reached.
        STOP_LIMIT: Trigger a limit order when the stop price is reached.
        TAKE_PROFIT: Trigger a market order to realize profit.
        TAKE_PROFIT_LIMIT: Trigger a limit order to realize profit.
        TRAILING_STOP: Stop that trails price by a configured distance.
    """

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(StrEnum):
    """Lifecycle state of an order.

    Attributes:
        CREATED: Order constructed locally and not yet validated.
        VALIDATED: Order passed local exchange and risk validation.
        SUBMITTED: Order sent to the exchange or broker gateway.
        ACCEPTED: Exchange acknowledged the order.
        PARTIALLY_FILLED: Order filled for a portion of the quantity.
        FILLED: Order fully filled.
        CANCELLED: Order cancelled before full fill.
        REJECTED: Order rejected by validation or the exchange.
        EXPIRED: Order expired according to time-in-force rules.
        ARCHIVED: Terminal order retained for audit history.
    """

    CREATED = "created"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class TimeInForce(StrEnum):
    """Order duration / time-in-force policy.

    Attributes:
        GTC: Good till cancelled.
        IOC: Immediate or cancel.
        FOK: Fill or kill.
    """

    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class PositionSide(StrEnum):
    """Directional state of a position.

    Attributes:
        LONG: Net long exposure.
        SHORT: Net short exposure.
        FLAT: No open exposure.
    """

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalSide(StrEnum):
    """Directional recommendation produced by a strategy or alpha model.

    Attributes:
        LONG: Recommend a long position or buy bias.
        SHORT: Recommend a short position or sell bias.
        FLAT: Recommend no directional exposure.
    """

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class RiskLevel(StrEnum):
    """Discrete risk severity classification.

    Attributes:
        LOW: Elevated risk is not indicated.
        MEDIUM: Moderate risk requiring heightened monitoring.
        HIGH: Significant risk requiring active controls.
        CRITICAL: Severe risk requiring immediate intervention.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelType(StrEnum):
    """Machine learning model category.

    Categories align with the CQROS model catalog groupings.

    Attributes:
        STATISTICAL: Classical statistical estimators.
        LINEAR: Linear and generalized linear models.
        TREE: Tree-based models excluding gradient boosting ensembles.
        GRADIENT_BOOSTING: Gradient-boosted tree models.
        DEEP_LEARNING: Neural network and deep sequence models.
        TIME_SERIES: Dedicated time-series forecasting models.
        PROBABILISTIC: Probabilistic and generative models.
        ENSEMBLE: Multi-model ensembles.
        META: Meta-learners and model-of-models.
    """

    STATISTICAL = "statistical"
    LINEAR = "linear"
    TREE = "tree"
    GRADIENT_BOOSTING = "gradient_boosting"
    DEEP_LEARNING = "deep_learning"
    TIME_SERIES = "time_series"
    PROBABILISTIC = "probabilistic"
    ENSEMBLE = "ensemble"
    META = "meta"


class TradeStatus(StrEnum):
    """Lifecycle state of a trade or position trade record.

    Attributes:
        PENDING: Trade awaiting activation or entry.
        OPEN: Trade is active with open exposure.
        PARTIALLY_CLOSED: Trade partially reduced but still open.
        CLOSED: Trade fully closed.
        CANCELLED: Trade cancelled before becoming active.
        FAILED: Trade failed during creation or execution.
    """

    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class DataSource(StrEnum):
    """Origin of market or research data.

    Attributes:
        EXCHANGE_REST: Exchange REST API.
        EXCHANGE_WEBSOCKET: Exchange WebSocket stream.
        HISTORICAL_ARCHIVE: Bulk historical archive.
        THIRD_PARTY: External third-party data provider.
        REPLAY: Internal deterministic replay feed.
        SIMULATION: Synthetic or simulated data feed.
    """

    EXCHANGE_REST = "exchange_rest"
    EXCHANGE_WEBSOCKET = "exchange_websocket"
    HISTORICAL_ARCHIVE = "historical_archive"
    THIRD_PARTY = "third_party"
    REPLAY = "replay"
    SIMULATION = "simulation"


class DatasetSplit(StrEnum):
    """Dataset partition used in research and model evaluation.

    Attributes:
        TRAIN: Training partition.
        VALIDATION: Validation / development partition.
        TEST: Held-out test partition.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class PipelineStage(StrEnum):
    """Named stage in the CQROS research and trading pipeline.

    Attributes:
        INGESTION: Historical and live data acquisition.
        STORAGE: Raw and derived dataset persistence.
        VALIDATION: Schema, integrity, and leakage validation.
        METADATA: Metadata and lineage recording.
        DATASET: Research dataset construction.
        FEATURES: Feature engineering.
        TARGETS: Target generation.
        STATISTICS: Statistical analysis and factor evaluation.
        REGIME: Market regime detection.
        TRAINING: Model training.
        EVALUATION: Model evaluation and selection.
        ALPHA: Alpha signal generation.
        PORTFOLIO: Portfolio construction.
        RISK: Risk measurement and limit checks.
        EXECUTION: Order routing and execution.
        BACKTESTING: Simulation and backtesting.
        MONITORING: Operational and research monitoring.
        DEPLOYMENT: Production deployment and promotion.
    """

    INGESTION = "ingestion"
    STORAGE = "storage"
    VALIDATION = "validation"
    METADATA = "metadata"
    DATASET = "dataset"
    FEATURES = "features"
    TARGETS = "targets"
    STATISTICS = "statistics"
    REGIME = "regime"
    TRAINING = "training"
    EVALUATION = "evaluation"
    ALPHA = "alpha"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    EXECUTION = "execution"
    BACKTESTING = "backtesting"
    MONITORING = "monitoring"
    DEPLOYMENT = "deployment"
