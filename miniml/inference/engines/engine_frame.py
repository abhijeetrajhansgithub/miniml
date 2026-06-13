from typing import List
from dataclasses import dataclass

@dataclass
class GetRefsEngineFrame:
    column_name: str
    column_type: str
    imputation_strategy: str
    imputation_reasoning: str
    outlier_strategy: str
    outlier_reasoning: str

@dataclass
class FindTargetEngineFrame:
    target_column: str


@dataclass
class DatasetContext:
    n_rows: int 
    n_cols: int 
    target: str 
    problem_type: str  # regression or classification
    numeric_cols: List[str]
    categorical_cols: List[str]
    class_imbalance: int | float | None

