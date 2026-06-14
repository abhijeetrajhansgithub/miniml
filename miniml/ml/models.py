# models.py

# build dictionary of all best suitable models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

from sklearn.svm import (
    SVR,
    SVC,
)

from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier,
)

from sklearn.naive_bayes import (
    GaussianNB,
)

from sklearn.neural_network import (
    MLPRegressor,
    MLPClassifier,
)

from xgboost import (
    XGBRegressor,
    XGBClassifier,
)


_MODELS = {
    "regression": {
        # Linear Models
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,

        # Trees
        "decision_tree": DecisionTreeRegressor,

        # Ensembles
        "random_forest": RandomForestRegressor,
        "extra_trees": ExtraTreesRegressor,
        "gradient_boosting": GradientBoostingRegressor,

        # Boosting
        "xgboost": XGBRegressor,

        # Distance Based
        "knn": KNeighborsRegressor,

        # Kernel Based
        "svr": SVR,

        # Neural
        "mlp": MLPRegressor,
    },

    "classification": {
        # Trees
        "decision_tree": DecisionTreeClassifier,

        # Ensembles
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
        "gradient_boosting": GradientBoostingClassifier,

        # Boosting
        "xgboost": XGBClassifier,

        # Distance Based
        "knn": KNeighborsClassifier,

        # Kernel Based
        "svc": SVC,

        # Probabilistic
        "gaussian_nb": GaussianNB,

        # Neural
        "mlp": MLPClassifier,
    },
}