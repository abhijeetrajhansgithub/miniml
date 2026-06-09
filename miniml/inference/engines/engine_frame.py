from dataclasses import dataclass

@dataclass
class GetRefsEngineFrame:
    column_name: str
    column_type: str
    imputation_strategy: str
    imputation_reasoning: str
    outlier_strategy: str
    outlier_reasoning: str
