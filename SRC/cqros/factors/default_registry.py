"""CQROS production Factor Registry bootstrap.

Purpose:
    Build the canonical in-memory catalog of every production factor
    implementation using default production configuration.

Responsibilities:
    - Instantiate every production factor exactly once
    - Register all factors into a ``FactorRegistry``
    - Rely on registry validation for duplicate names and produced columns
    - Remain free of pipeline, CLI, storage, repository, and dataframe logic

Dependencies:
    Category factor packages under ``cqros.factors`` and
    ``cqros.factors.registry.FactorRegistry``.

Public API:
    ``build_default_registry``
"""

from __future__ import annotations

from cqros.factors.composite import (
    BreakoutConfirmationFactor,
    CrowdingFactor,
    FlowConfirmationFactor,
    FundingDivergenceFactor,
    LeveragedLongBuildUpFactor,
    LeveragedShortBuildUpFactor,
    LongSqueezeFactor,
    PositionBuildUpFactor,
    ShortSqueezeFactor,
    TrendConfirmationFactor,
)
from cqros.factors.funding import (
    BasisFactor,
    BasisMomentumFactor,
    BasisZScoreFactor,
    CarryFactor,
    FundingAccelerationFactor,
    FundingRateLevelFactor,
    FundingRateMomentumFactor,
    FundingRateZScoreFactor,
    FundingVolatilityFactor,
    PremiumIndexFactor,
)
from cqros.factors.interfaces import Factor
from cqros.factors.liquidation import (
    LeverageChangeFactor,
    LeveragePressureFactor,
    LiquidationImbalanceFactor,
    LiquidationIntensityFactor,
    LiquidationMomentumFactor,
    LiquidationSpikeFactor,
    LiquidationTrendFactor,
    LiquidationZScoreFactor,
    LongLiquidationPressureFactor,
    ShortLiquidationPressureFactor,
)
from cqros.factors.microstructure import (
    AggressiveBuyRatioFactor,
    AggressiveSellRatioFactor,
    BuySellImbalanceFactor,
    MicroPricePressureFactor,
    OrderFlowMomentumFactor,
    SignedVolumeFactor,
    TradeImbalanceFactor,
    TradeIntensityFactor,
    VWAPDistanceFactor,
    VWAPZScoreFactor,
)
from cqros.factors.open_interest import (
    OpenInterestAccelerationFactor,
    OpenInterestFundingDivergenceFactor,
    OpenInterestIntensityFactor,
    OpenInterestLevelFactor,
    OpenInterestMomentumFactor,
    OpenInterestPriceDivergenceFactor,
    OpenInterestTrendFactor,
    OpenInterestVolatilityFactor,
    OpenInterestVolumeRatioFactor,
    OpenInterestZScoreFactor,
)
from cqros.factors.price import (
    ATRDistanceFactor,
    ATRPercentFactor,
    ATRSlopeFactor,
    BollingerBandwidthFactor,
    BollingerPositionFactor,
    BollingerWidthFactor,
    BreakoutStrengthFactor,
    ChoppinessIndexFactor,
    CommodityChannelIndexFactor,
    DetrendedPriceOscillatorFactor,
    DistanceFromHighFactor,
    DistanceFromLowFactor,
    DonchianPositionFactor,
    EfficiencyRatioFactor,
    EMADistanceFactor,
    GarmanKlassVolatilityFactor,
    HistoricalVolatilityFactor,
    LinearRegressionR2Factor,
    MaximumDrawdownFactor,
    MeanReversionScoreFactor,
    MomentumFactor,
    MultiHorizonMomentumFactor,
    ParkinsonVolatilityFactor,
    PriceAccelerationFactor,
    PriceOscillatorFactor,
    PriceZScoreFactor,
    RateOfChangeFactor,
    RecoveryStrengthFactor,
    RegressionResidualFactor,
    RegressionResidualZScoreFactor,
    RollingReturnMeanFactor,
    RollingReturnMedianFactor,
    RSIFactor,
    SMADistanceFactor,
    StochasticDFactor,
    StochasticKFactor,
    TrendAngleFactor,
    TrendPersistenceFactor,
    TrendSlopeFactor,
    UlcerIndexFactor,
    WilliamsRFactor,
)
from cqros.factors.registry import FactorRegistry
from cqros.factors.relative import (
    BetaToBTCFactor,
    BetaToETHFactor,
    CorrelationBTCFactor,
    CorrelationETHFactor,
    RelativeMomentumBTCFactor,
    RelativeMomentumETHFactor,
    RelativeStrengthBTCFactor,
    RelativeStrengthETHFactor,
    RelativeVolatilityFactor,
    TrackingErrorFactor,
)
from cqros.factors.volume import (
    AccumulationDistributionFactor,
    ChaikinMoneyFlowFactor,
    EaseOfMovementFactor,
    MoneyFlowIndexFactor,
    OnBalanceVolumeFactor,
    PriceVolumeTrendFactor,
    RelativeVolumeFactor,
    VolumeRateOfChangeFactor,
    VolumeTrendFactor,
    VolumeZScoreFactor,
)

__all__ = ["build_default_registry"]


def build_default_registry() -> FactorRegistry:
    """Return a registry containing every production factor implementation.

    Instantiates each production factor with its default configuration,
    registers every factor exactly once, and validates uniqueness of factor
    names and produced columns through ``FactorRegistry``.

    Returns:
        Fully populated ``FactorRegistry`` containing the production catalog.
    """
    registry = FactorRegistry()
    registry.register_many(_default_production_factors())
    return registry


def _default_production_factors() -> tuple[Factor, ...]:
    """Instantiate every production factor with default configuration."""
    return (
        # Price
        ATRDistanceFactor(),
        ATRPercentFactor(),
        ATRSlopeFactor(),
        BollingerBandwidthFactor(),
        BollingerPositionFactor(),
        BollingerWidthFactor(),
        BreakoutStrengthFactor(),
        ChoppinessIndexFactor(),
        CommodityChannelIndexFactor(),
        DetrendedPriceOscillatorFactor(),
        DistanceFromHighFactor(),
        DistanceFromLowFactor(),
        DonchianPositionFactor(),
        EfficiencyRatioFactor(),
        EMADistanceFactor(),
        GarmanKlassVolatilityFactor(),
        HistoricalVolatilityFactor(),
        LinearRegressionR2Factor(),
        MaximumDrawdownFactor(),
        MeanReversionScoreFactor(),
        MomentumFactor(),
        MultiHorizonMomentumFactor(),
        ParkinsonVolatilityFactor(),
        PriceAccelerationFactor(),
        PriceOscillatorFactor(),
        PriceZScoreFactor(),
        RateOfChangeFactor(),
        RecoveryStrengthFactor(),
        RegressionResidualFactor(),
        RegressionResidualZScoreFactor(),
        RollingReturnMeanFactor(),
        RollingReturnMedianFactor(),
        RSIFactor(),
        SMADistanceFactor(),
        StochasticDFactor(),
        StochasticKFactor(),
        TrendAngleFactor(),
        TrendPersistenceFactor(),
        TrendSlopeFactor(),
        UlcerIndexFactor(),
        WilliamsRFactor(),
        # Volume
        AccumulationDistributionFactor(),
        ChaikinMoneyFlowFactor(),
        EaseOfMovementFactor(),
        MoneyFlowIndexFactor(),
        OnBalanceVolumeFactor(),
        PriceVolumeTrendFactor(),
        RelativeVolumeFactor(),
        VolumeRateOfChangeFactor(),
        VolumeTrendFactor(),
        VolumeZScoreFactor(),
        # Microstructure
        AggressiveBuyRatioFactor(),
        AggressiveSellRatioFactor(),
        BuySellImbalanceFactor(),
        MicroPricePressureFactor(),
        OrderFlowMomentumFactor(),
        SignedVolumeFactor(),
        TradeImbalanceFactor(),
        TradeIntensityFactor(),
        VWAPDistanceFactor(),
        VWAPZScoreFactor(),
        # Funding
        BasisFactor(),
        BasisMomentumFactor(),
        BasisZScoreFactor(),
        CarryFactor(),
        FundingAccelerationFactor(),
        FundingRateLevelFactor(),
        FundingRateMomentumFactor(),
        FundingRateZScoreFactor(),
        FundingVolatilityFactor(),
        PremiumIndexFactor(),
        # Open Interest
        OpenInterestAccelerationFactor(),
        OpenInterestFundingDivergenceFactor(),
        OpenInterestIntensityFactor(),
        OpenInterestLevelFactor(),
        OpenInterestMomentumFactor(),
        OpenInterestPriceDivergenceFactor(),
        OpenInterestTrendFactor(),
        OpenInterestVolatilityFactor(),
        OpenInterestVolumeRatioFactor(),
        OpenInterestZScoreFactor(),
        # Liquidation
        LeverageChangeFactor(),
        LeveragePressureFactor(),
        LiquidationImbalanceFactor(),
        LiquidationIntensityFactor(),
        LiquidationMomentumFactor(),
        LiquidationSpikeFactor(),
        LiquidationTrendFactor(),
        LiquidationZScoreFactor(),
        LongLiquidationPressureFactor(),
        ShortLiquidationPressureFactor(),
        # Relative
        BetaToBTCFactor(),
        BetaToETHFactor(),
        CorrelationBTCFactor(),
        CorrelationETHFactor(),
        RelativeMomentumBTCFactor(),
        RelativeMomentumETHFactor(),
        RelativeStrengthBTCFactor(),
        RelativeStrengthETHFactor(),
        RelativeVolatilityFactor(),
        TrackingErrorFactor(),
        # Composite
        BreakoutConfirmationFactor(),
        CrowdingFactor(),
        FlowConfirmationFactor(),
        FundingDivergenceFactor(),
        LeveragedLongBuildUpFactor(),
        LeveragedShortBuildUpFactor(),
        LongSqueezeFactor(),
        PositionBuildUpFactor(),
        ShortSqueezeFactor(),
        TrendConfirmationFactor(),
    )
