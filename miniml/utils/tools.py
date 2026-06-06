from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats             # type: ignore
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from miniml.tools.tool import tool  # type: ignore

# ============================================================================
# Imputation Tools
# ============================================================================

@tool
def fn_tool_mean_imputer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Impute missing numerical values using mean strategy.

        Recommended for approximately normal numerical
        distributions with low skewness.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring
            mean-based imputation.

        options:
            Optional configuration dictionary.

    Returns:
        pd.DataFrame:
            Dataframe with imputed numerical columns.
    """
    df_cpy = df.copy()

    for col in cols:
        df_cpy[col] = df_cpy[col].fillna(
            df_cpy[col].mean()
        )

    return df_cpy


@tool
def fn_tool_median_imputer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Impute missing numerical values using median strategy.

        Recommended for skewed numerical distributions
        and datasets containing significant outliers.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring
            median-based imputation.

        options:
            Optional configuration dictionary.

    Returns:
        pd.DataFrame:
            Dataframe with imputed numerical columns.
    """
    df_cpy = df.copy()

    for col in cols:
        df_cpy[col] = df_cpy[col].fillna(
            df_cpy[col].median()
        )

    return df_cpy


@tool
def fn_tool_mode_imputer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Impute missing values using mode strategy.

        Recommended for categorical or discrete features.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of categorical columns requiring
            mode-based imputation.

        options:
            Optional configuration dictionary.

    Returns:
        pd.DataFrame:
            Dataframe with imputed categorical columns.
    """
    df_cpy = df.copy()

    for col in cols:
        # mode() returns a Series; take the first value to handle
        # multimodal cases gracefully.
        mode_val = df_cpy[col].mode()
        if not mode_val.empty:
            df_cpy[col] = df_cpy[col].fillna(mode_val.iloc[0])

    return df_cpy


@tool
def fn_tool_ffill_imputer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Impute missing values using forward-fill strategy.

        Recommended for temporal or sequential datasets.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of sequential columns requiring
            forward-fill imputation.

        options:
            Optional configuration dictionary.

    Returns:
        pd.DataFrame:
            Dataframe with forward-filled columns.
    """
    df_cpy = df.copy()

    for col in cols:
        df_cpy[col] = df_cpy[col].ffill()

    return df_cpy


@tool
def fn_tool_bfill_imputer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Impute missing values using backward-fill strategy.

        Recommended for temporal or sequential datasets.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of sequential columns requiring
            backward-fill imputation.

        options:
            Optional configuration dictionary.

    Returns:
        pd.DataFrame:
            Dataframe with backward-filled columns.
    """
    df_cpy = df.copy()

    for col in cols:
        df_cpy[col] = df_cpy[col].bfill()

    return df_cpy


@tool
def fn_tool_knn_imputer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Impute missing values using K-Nearest Neighbors.

        Suitable for structured datasets with strong
        feature similarity patterns.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of columns requiring KNN-based imputation.

        options:
            Optional configuration dictionary.
            Supported keys:
                n_neighbors (int): Number of neighbors to use. Default 5.

    Returns:
        pd.DataFrame:
            Dataframe with KNN-imputed columns.
    """
    df_cpy = df.copy()
    n_neighbors: int = (options or {}).get("n_neighbors", 5)

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_cpy[cols] = imputer.fit_transform(df_cpy[cols])                          # type: ignore

    return df_cpy


@tool
def fn_tool_iterative_imputer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Impute missing values using iterative multivariate estimation.

        Recommended for datasets with strong inter-feature
        relationships.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of columns requiring iterative imputation.

        options:
            Optional configuration dictionary.
            Supported keys:
                max_iter (int): Maximum imputation iterations. Default 10.
                random_state (int): Seed for reproducibility. Default 0.

    Returns:
        pd.DataFrame:
            Dataframe with iteratively imputed columns.
    """
    df_cpy = df.copy()
    opts = options or {}
    max_iter: int = opts.get("max_iter", 10)
    random_state: int = opts.get("random_state", 0)

    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    df_cpy[cols] = imputer.fit_transform(df_cpy[cols])

    return df_cpy


@tool
def fn_tool_constant_imputer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Impute missing values using a fixed constant value.

        Useful when a semantically meaningful placeholder
        value is required.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of columns requiring constant-value imputation.

        options:
            Optional configuration dictionary.
            Supported keys:
                fill_value (any): The constant to fill with. Default "missing".

    Returns:
        pd.DataFrame:
            Dataframe with constant-imputed columns.
    """
    df_cpy = df.copy()
    fill_value = (options or {}).get("fill_value", "missing")

    for col in cols:
        df_cpy[col] = df_cpy[col].fillna(fill_value)

    return df_cpy


@tool
def fn_tool_zero_imputer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Impute missing numerical values using zero.

        Recommended only when zero is semantically valid.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring
            zero-value imputation.

        options:
            Optional configuration dictionary.

    Returns:
        pd.DataFrame:
            Dataframe with zero-imputed columns.
    """
    df_cpy = df.copy()

    for col in cols:
        df_cpy[col] = df_cpy[col].fillna(0)

    return df_cpy


@tool
def fn_tool_drop_missing(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Remove rows or columns containing missing values.

        Recommended only when missingness is severe
        or data quality is critically compromised.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of columns to evaluate for missing-value removal.

        options:
            Optional configuration dictionary.
            Supported keys:
                axis (int): 0 to drop rows (default), 1 to drop columns.

    Returns:
        pd.DataFrame:
            Dataframe with missing observations removed.
    """
    df_cpy = df.copy()
    axis: int = (options or {}).get("axis", 0)

    df_cpy = df_cpy.dropna(subset=cols, axis=axis)                              # type: ignore

    return df_cpy


# ============================================================================
# Outlier Tools
# ============================================================================

@tool
def fn_tool_zscore_outlier(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Detect and handle outliers using Z-score method.

        Recommended for approximately normal distributions.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring
            Z-score outlier handling.

        options:
            Optional configuration dictionary.
            Supported keys:
                threshold (float): Z-score cutoff. Default 3.0.
                strategy (str): "clip" to cap values at threshold bounds,
                                "remove" to drop outlier rows. Default "clip".

    Returns:
        pd.DataFrame:
            Dataframe with processed outliers.
    """
    df_cpy = df.copy()
    opts = options or {}
    threshold: float = opts.get("threshold", 3.0)
    strategy: str = opts.get("strategy", "clip")

    for col in cols:
        z_scores = np.abs(stats.zscore(df_cpy[col].dropna()))                   # type: ignore
        outlier_mask = np.abs(                                                  # type: ignore
            stats.zscore(df_cpy[col].fillna(df_cpy[col].mean()))                # type: ignore
        ) > threshold

        if strategy == "remove":
            df_cpy = df_cpy[~outlier_mask]                                      # type: ignore
        else:
            # clip: replace outliers with the threshold boundary values
            col_mean = df_cpy[col].mean()                                       # type: ignore
            col_std = df_cpy[col].std()                                         # type: ignore
            lower = col_mean - threshold * col_std                              # type: ignore
            upper = col_mean + threshold * col_std                              # type: ignore
            df_cpy[col] = df_cpy[col].clip(lower=lower, upper=upper)            # type: ignore

    return df_cpy                                                               # type: ignore


@tool
def fn_tool_iqr_outlier(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Detect and handle outliers using Interquartile Range (IQR).

        Recommended for skewed numerical distributions.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring
            IQR-based outlier handling.

        options:
            Optional configuration dictionary.
            Supported keys:
                factor (float): IQR multiplier for fence calculation. Default 1.5.
                strategy (str): "clip" to cap at fence bounds,
                                "remove" to drop outlier rows. Default "clip".

    Returns:
        pd.DataFrame:
            Dataframe with processed outliers.
    """
    df_cpy = df.copy()
    opts = options or {}
    factor: float = opts.get("factor", 1.5)
    strategy: str = opts.get("strategy", "clip")

    for col in cols:
        q1 = df_cpy[col].quantile(0.25)
        q3 = df_cpy[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        if strategy == "remove":
            df_cpy = df_cpy[df_cpy[col].between(lower, upper)]
        else:
            df_cpy[col] = df_cpy[col].clip(lower=lower, upper=upper)

    return df_cpy


@tool
def fn_tool_log_transform_outlier(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Apply logarithmic transformation to reduce skewness.

        Effective for heavily right-skewed numerical features.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring
            logarithmic transformation.

        options:
            Optional configuration dictionary.
            Supported keys:
                shift (float): Constant added before log to handle zeros/negatives.
                               Default 1.0 (i.e. log1p behaviour).

    Returns:
        pd.DataFrame:
            Dataframe with transformed numerical features.
    """
    df_cpy = df.copy()
    shift: float = (options or {}).get("shift", 1.0)

    for col in cols:
        df_cpy[col] = np.log(df_cpy[col] + shift)

    return df_cpy


@tool
def fn_tool_drop_outlier(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Remove extreme or statistically invalid outlier observations.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring
            outlier-based row removal.

        options:
            Optional configuration dictionary.
            Supported keys:
                method (str): "iqr" (default) or "zscore".
                factor (float): IQR multiplier when method="iqr". Default 1.5.
                threshold (float): Z-score cutoff when method="zscore". Default 3.0.

    Returns:
        pd.DataFrame:
            Dataframe with extreme outliers removed.
    """
    df_cpy = df.copy()
    opts = options or {}
    method: str = opts.get("method", "iqr")

    for col in cols:
        if method == "zscore":
            threshold: float = opts.get("threshold", 3.0)
            mask = (                                                                # type: ignore
                np.abs(stats.zscore(df_cpy[col].fillna(df_cpy[col].mean())))        # type: ignore
                <= threshold
            )
        else:
            factor: float = opts.get("factor", 1.5)
            q1 = df_cpy[col].quantile(0.25)
            q3 = df_cpy[col].quantile(0.75)
            iqr = q3 - q1
            mask = df_cpy[col].between(
                q1 - factor * iqr,
                q3 + factor * iqr,
            )

        df_cpy = df_cpy[mask]

    return df_cpy


# ============================================================================
# Dataset Preprocessing Tools
# ============================================================================

@tool
def fn_tool_drop_duplicates(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Remove duplicate rows from the dataset.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of columns used for duplicate validation.

        options:
            Optional configuration dictionary.
            Supported keys:
                keep (str): Which duplicate to keep —
                            "first" (default), "last", or False to drop all.

    Returns:
        pd.DataFrame:
            Dataframe with duplicate rows removed.
    """
    df_cpy = df.copy()
    keep = (options or {}).get("keep", "first")

    df_cpy = df_cpy.drop_duplicates(subset=cols, keep=keep)

    return df_cpy


@tool
def fn_tool_standardizer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Standardize numerical features using standard scaling.

        Transforms features to zero mean and unit variance.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring
            standard scaling.

        options:
            Optional configuration dictionary.

    Returns:
        pd.DataFrame:
            Dataframe with standardized numerical features.
    """
    df_cpy = df.copy()

    scaler = StandardScaler()
    df_cpy[cols] = scaler.fit_transform(df_cpy[cols])       # type: ignore

    return df_cpy


@tool
def fn_tool_normalizer(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Normalize numerical features using Min-Max scaling.

        Scales values into a bounded range,
        typically between 0 and 1.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring normalization.

        options:
            Optional configuration dictionary.
            Supported keys:
                feature_range (tuple): Target range. Default (0, 1).

    Returns:
        pd.DataFrame:
            Dataframe with normalized numerical features.
    """
    df_cpy = df.copy()
    feature_range: tuple = (options or {}).get("feature_range", (0, 1))         # type: ignore

    scaler = MinMaxScaler(feature_range=feature_range)      # type: ignore
    df_cpy[cols] = scaler.fit_transform(df_cpy[cols])       # type: ignore

    return df_cpy


@tool
def fn_tool_robust_scaler(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Scale numerical features using robust scaling.

        Recommended for datasets containing significant outliers.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of numerical columns requiring
            robust scaling.

        options:
            Optional configuration dictionary.

    Returns:
        pd.DataFrame:
            Dataframe with robust-scaled numerical features.
    """
    df_cpy = df.copy()

    scaler = RobustScaler()
    df_cpy[cols] = scaler.fit_transform(df_cpy[cols])       # type: ignore

    return df_cpy


# ============================================================================
# Encoding Tools
# ============================================================================

@tool
def fn_tool_one_hot_encoder(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Encode categorical features using one-hot encoding.

        Suitable for nominal categorical variables
        without ordinal relationships.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of categorical columns requiring
            one-hot encoding.

        options:
            Optional configuration dictionary.
            Supported keys:
                drop_first (bool): Drop the first dummy column to avoid
                                   multicollinearity. Default False.

    Returns:
        pd.DataFrame:
            Dataframe with one-hot encoded features.
    """
    df_cpy = df.copy()
    drop_first: bool = (options or {}).get("drop_first", False)

    df_cpy = pd.get_dummies(df_cpy, columns=cols, drop_first=drop_first)

    return df_cpy


@tool
def fn_tool_label_encoder(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Encode categorical labels into integer representations.

        Suitable for ordinal categorical variables.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of categorical columns requiring
            label encoding.

        options:
            Optional configuration dictionary.

    Returns:
        pd.DataFrame:
            Dataframe with label-encoded features.
    """
    df_cpy = df.copy()
    encoder = LabelEncoder()

    for col in cols:
        # fit only on non-null values; leave NaNs as NaN
        non_null_mask = df_cpy[col].notna()
        df_cpy.loc[non_null_mask, col] = encoder.fit_transform(     # type: ignore
            df_cpy.loc[non_null_mask, col]
        )

    return df_cpy


@tool
def fn_tool_target_encoder(
    df: pd.DataFrame,
    cols: List[str],
    options: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Description:
        Encode categorical variables using target statistics.

        Recommended for high-cardinality categorical features
        in supervised learning tasks.

    Args:
        df:
            Input pandas dataframe.

        cols:
            List of categorical columns requiring
            target encoding.

        options:
            Optional configuration dictionary.
            Supported keys:
                target_col (str): Name of the target column. Required.
                smoothing (float): Smoothing factor to balance category mean
                                   against global mean. Default 1.0.

    Returns:
        pd.DataFrame:
            Dataframe with target-encoded features.
    """
    df_cpy = df.copy()
    opts = options or {}
    target_col: str = opts.get("target_col", "")
    smoothing: float = opts.get("smoothing", 1.0)

    if not target_col or target_col not in df_cpy.columns:
        raise ValueError(
            f"fn_tool_target_encoder requires a valid 'target_col' in options. "
            f"Got: '{target_col}'"
        )

    global_mean = df_cpy[target_col].mean()

    for col in cols:
        # Per-category stats
        agg = df_cpy.groupby(col)[target_col].agg(["mean", "count"])
        # Smoothed target mean: blends category mean toward global mean
        # as category sample size shrinks.
        smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
            agg["count"] + smoothing
        )
        df_cpy[col] = df_cpy[col].map(smoothed)

    return df_cpy

