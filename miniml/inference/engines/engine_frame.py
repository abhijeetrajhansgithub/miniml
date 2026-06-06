from dataclasses import dataclass
from typing import List


@dataclass
class StrategyReason:
    strategy: str
    reason: str


@dataclass
class FeatureFlagsEngineFrame:
    is_skewed: bool
    is_sparse: bool
    is_high_cardinality: bool
    is_temporal: bool
    is_identifier: bool
    is_leakage_prone: bool


@dataclass
class QualityAssessmentEngineFrame:
    missing_severity: str
    outlier_severity: str
    distribution_type: str
    overall_quality: str


@dataclass
class RecommendationEngineFrame:
    recommended_pipeline: List[str]
    summary: str


@dataclass
class ProviderEngineFrame:
    column_name: str
    feature_type: str

    feature_flags: FeatureFlagsEngineFrame

    quality_assessment: QualityAssessmentEngineFrame

    imputation: StrategyReason

    outlier: StrategyReason

    final_recommendation: RecommendationEngineFrame



@dataclass
class ValidationEngineFrame:
    column_name: str
    feature_type: str
    llm_response_is_valid: bool
    reason: str