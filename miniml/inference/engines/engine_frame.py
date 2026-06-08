from dataclasses import dataclass

@dataclass
class GetRefsEngineFrame:
    imputation_strategy: str
    imputation_reasoning: str
    outlier_strategy: str
    outlier_reasoning: str
