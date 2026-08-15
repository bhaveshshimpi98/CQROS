"""CQROS research package public API."""

from cqros.core.exceptions import ResearchError
from cqros.research.exceptions import TargetDefinitionError, TargetError
from cqros.research.experiment import (
    ExperimentDefinition,
    ExperimentResult,
    ResearchExperiment,
)
from cqros.research.factor_correlation import (
    FactorCorrelationAnalyzer,
    FactorCorrelationResult,
    find_highly_correlated,
)
from cqros.research.factor_decay import (
    DecayPoint,
    FactorDecayAnalyzer,
    FactorDecayResult,
)
from cqros.research.factor_stability import (
    FactorStabilityAnalyzer,
    FactorStabilityResult,
    StabilityWindow,
)
from cqros.research.information_coefficient import (
    InformationCoefficient,
    InformationCoefficientResult,
)
from cqros.research.quantile_analysis import (
    QuantileAnalysisResult,
    QuantileAnalyzer,
    QuantileStatistics,
)
from cqros.research.rank_ic import RankICResult, RankInformationCoefficient
from cqros.research.report import (
    CorrelationSummary,
    FactorSummary,
    FailedFactorSummary,
    LeaderboardSummary,
    OverallStatistics,
    ResearchReport,
    ResearchReportGenerator,
    SkippedFactorSummary,
    SymbolSummary,
    TimeframeSummary,
)
from cqros.research.runner import (
    AssetExperimentRecord,
    FactorLeaderboardEntry,
    FactorResearchRunner,
    FactorResearchRunnerConfig,
    FactorResearchRunResult,
    SkippedFactorEvaluation,
)
from cqros.research.signal_threshold_calibrator import (
    PredictionDistributionStatistics,
    SignalThresholdCalibrator,
    SymbolTimeframeCalibration,
    ThresholdCalibrationResult,
    ThresholdRecommendation,
)
from cqros.research.target import ForwardReturnTarget, TargetDefinition
from cqros.research.walk_forward import (
    WalkForwardResult,
    WalkForwardValidator,
    WalkForwardWindow,
)

__all__ = [
    "AssetExperimentRecord",
    "CorrelationSummary",
    "DecayPoint",
    "ExperimentDefinition",
    "ExperimentResult",
    "FactorCorrelationAnalyzer",
    "FactorCorrelationResult",
    "FactorDecayAnalyzer",
    "FactorDecayResult",
    "FactorLeaderboardEntry",
    "FactorResearchRunResult",
    "FactorResearchRunner",
    "FactorResearchRunnerConfig",
    "FactorStabilityAnalyzer",
    "FactorStabilityResult",
    "FactorSummary",
    "FailedFactorSummary",
    "ForwardReturnTarget",
    "InformationCoefficient",
    "InformationCoefficientResult",
    "LeaderboardSummary",
    "OverallStatistics",
    "PredictionDistributionStatistics",
    "QuantileAnalysisResult",
    "QuantileAnalyzer",
    "QuantileStatistics",
    "RankICResult",
    "RankInformationCoefficient",
    "ResearchError",
    "ResearchExperiment",
    "ResearchReport",
    "ResearchReportGenerator",
    "SignalThresholdCalibrator",
    "SkippedFactorEvaluation",
    "SkippedFactorSummary",
    "StabilityWindow",
    "SymbolSummary",
    "SymbolTimeframeCalibration",
    "TargetDefinition",
    "TargetDefinitionError",
    "TargetError",
    "ThresholdCalibrationResult",
    "ThresholdRecommendation",
    "TimeframeSummary",
    "WalkForwardResult",
    "WalkForwardValidator",
    "WalkForwardWindow",
    "find_highly_correlated",
]
