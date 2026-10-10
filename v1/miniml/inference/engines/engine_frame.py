from typing import List, Tuple, Dict, Any, Literal   # type: ignore
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
    transformations_applied: List[str]
    tts: str
    ml_models: List[str]


@dataclass
class MLPlannerPipelinePlan:
    models: List[str]
    iters: int 
    metrics: List[str]
    use_cross_validation: bool 


# --------------------
# Regression
# --------------------

@dataclass
class LinearRegressionModel:
    fit_intercept: bool = True


@dataclass
class RidgeRegressionModel:
    alpha: float = 1.0
    fit_intercept: bool = True


@dataclass
class LassoRegressionModel:
    alpha: float = 1.0
    fit_intercept: bool = True


@dataclass
class ElasticNetRegressionModel:
    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = True


@dataclass
class DecisionTreeRegressorModel:
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass
class RandomForestRegressorModel:
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass
class ExtraTreesRegressorModel:
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass
class GradientBoostingRegressorModel:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3


@dataclass
class XGBRegressorModel:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0


@dataclass
class KNeighborsRegressorModel:
    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"


@dataclass
class SVRModel:
    C: float = 1.0
    kernel: Literal["linear", "rbf", "poly"] = "rbf"
    gamma: str | float = "scale"


@dataclass
class MLPRegressorModel:
    hidden_layer_sizes: tuple[int, ...] = (100,)
    alpha: float = 0.0001
    learning_rate_init: float = 0.001


# --------------------
# Classification
# --------------------

@dataclass
class LogisticRegressionModel:
    C: float = 1.0
    penalty: Literal["l2"] = "l2"
    max_iter: int = 1000


@dataclass
class DecisionTreeClassifierModel:
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass
class RandomForestClassifierModel:
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass
class ExtraTreesClassifierModel:
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass
class GradientBoostingClassifierModel:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3


@dataclass
class XGBClassifierModel:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0


@dataclass
class KNeighborsClassifierModel:
    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"


@dataclass
class SVCModel:
    C: float = 1.0
    kernel: Literal["linear", "rbf", "poly"] = "rbf"
    gamma: str | float = "scale"


@dataclass
class GaussianNBModel:
    var_smoothing: float = 1e-9


@dataclass
class MLPClassifierModel:
    hidden_layer_sizes: tuple[int, ...] = (100,)
    alpha: float = 0.0001
    learning_rate_init: float = 0.001



@dataclass(slots=True)
class LLMResponse:
    content: str | None
    tool_calls: list[Any] | None = None

@dataclass(slots=True)
class LLMSingleResponse:
    content: str | Dict[str, Any] | None = None


@dataclass(slots=True)
class SubAgentParserResponse:
    instance_: str
    strategy: str | None = None
    models: List[str] | List[Any] | None = None
    error: str | None = None

@dataclass(slots=True)
class SubAgentGenericResponse:
    instance_: str
    stage: str
    data: SubAgentParserResponse | Any | None = None

@dataclass(slots=True)
class AgentParserResponse:
    instance_: str
    data: Any | None = None
    target_column: str | None = None
    valid: bool | None = None
    reasoning: str | None = None
    error: str | None = None


@dataclass(slots=True)
class AgentGetRefsIOResponse:
    imputation_strategy: str | None = None
    imputation_reasoning: str | None = None
    outlier_strategy: str | None = None
    outlier_reasoning: str | None = None


# --------------------
# Cross Validation
# --------------------

@dataclass(slots=True)
class CVConfig:
    strategy: str | Any


@dataclass(slots=True)
class KFoldConfig(CVConfig):
    strategy: Literal["kfold"] = "kfold"

    n_splits: int = 5
    shuffle: bool = True
    random_state: int | None = 42


@dataclass(slots=True)
class StratifiedKFoldConfig(CVConfig):
    strategy: Literal["stratified_kfold"] = "stratified_kfold"

    n_splits: int = 5
    shuffle: bool = True
    random_state: int | None = 42


@dataclass(slots=True)
class RepeatedKFoldConfig(CVConfig):
    strategy: Literal["repeated_kfold"] = "repeated_kfold"

    n_splits: int = 5
    n_repeats: int = 10
    random_state: int | None = 42



@dataclass(slots=True)
class RepeatedStratifiedKFoldConfig(CVConfig):
    strategy: Literal["repeated_stratified_kfold"] = "repeated_stratified_kfold"

    n_splits: int = 5
    n_repeats: int = 10
    random_state: int | None = 42


@dataclass(slots=True)
class LeaveOneOutConfig(CVConfig):
    strategy: Literal["leave_one_out"] = "leave_one_out"


@dataclass(slots=True)
class LeavePOutConfig(CVConfig):
    strategy: Literal["leave_p_out"] = "leave_p_out"

    p: int = 2


@dataclass(slots=True)
class GroupKFoldConfig(CVConfig):
    strategy: Literal["group_kfold"] = "group_kfold"

    n_splits: int = 5


@dataclass(slots=True)
class StratifiedGroupKFoldConfig(CVConfig):
    strategy: Literal["stratified_group_kfold"] = "stratified_group_kfold"

    n_splits: int = 5
    shuffle: bool = True
    random_state: int | None = 42


CrossValidationConfig = (
    KFoldConfig
    | StratifiedKFoldConfig
    | RepeatedKFoldConfig
    | RepeatedStratifiedKFoldConfig
    | LeaveOneOutConfig
    | LeavePOutConfig
    | GroupKFoldConfig
    | StratifiedGroupKFoldConfig
)