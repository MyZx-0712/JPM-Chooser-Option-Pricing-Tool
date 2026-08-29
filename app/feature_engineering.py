"""Feature reconstruction for the frozen Approach 1 model."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_SOURCE_COLUMNS = [
    "date",
    "jpm_volume",
    "jpm_daily_return",
    "jpm_log_return",
    "jpm_rolling_vol_5d",
    "jpm_rolling_vol_20d",
    "jpm_rolling_vol_60d",
    "jpm_dividend_amount",
    "dividend_growth_yoy",
    "dgs1_rate",
    "dgs1_rate_change",
    "dgs1_rate_momentum_20d",
    "vix_level",
    "vix_change",
    "vix_return",
    "vix_rolling_mean_20d",
    "vix_jpm_corr_20d",
    "sentiment_score_0_1",
    "market_stress_score_0_1",
    "jpm_return_outlier_iqr",
    "vix_level_outlier_iqr",
]


def build_inference_features(
    source_data,
    ordered_feature_columns,
):
    """Build the frozen model's 34 ordered features.

    Only information available on or before each date is used.
    The future target is intentionally not constructed because
    it is unavailable during real inference.
    """

    data = (
        source_data.copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    missing_source_columns = [
        column
        for column in REQUIRED_SOURCE_COLUMNS
        if column not in data.columns
    ]

    if missing_source_columns:
        raise ValueError(
            "Missing source columns: "
            + ", ".join(missing_source_columns)
        )

    if data["date"].isna().any():
        raise ValueError(
            "Source data contains invalid dates"
        )

    if data["date"].duplicated().any():
        raise ValueError(
            "Source data contains duplicate dates"
        )

    # Dividend-event and volume transformations.
    data["dividend_event_flag"] = (
        data["jpm_dividend_amount"]
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    data["jpm_dividend_amount"] = (
        data["jpm_dividend_amount"]
        .fillna(0)
    )

    data["dividend_growth_yoy"] = (
        data["dividend_growth_yoy"]
        .fillna(0)
    )

    data["jpm_log_volume"] = np.log1p(
        data["jpm_volume"]
    )

    # Historical cumulative log returns.
    data["jpm_return_5d_sum"] = (
        data["jpm_log_return"]
        .rolling(5, min_periods=5)
        .sum()
    )

    data["jpm_return_20d_sum"] = (
        data["jpm_log_return"]
        .rolling(20, min_periods=20)
        .sum()
    )

    # Lagged features use current and past data only.
    lag_source_columns = [
        "jpm_daily_return",
        "jpm_rolling_vol_20d",
        "vix_level",
        "dgs1_rate",
    ]

    for column in lag_source_columns:
        for lag in [1, 5, 20]:
            feature_name = (
                f"{column}_lag_{lag}"
            )

            data[feature_name] = (
                data[column].shift(lag)
            )

    missing_model_features = [
        column
        for column in ordered_feature_columns
        if column not in data.columns
    ]

    if missing_model_features:
        raise ValueError(
            "Unable to build model features: "
            + ", ".join(missing_model_features)
        )

    inference_data = (
        data[
            [
                "date",
                *ordered_feature_columns,
            ]
        ]
        .dropna()
        .reset_index(drop=True)
    )

    if inference_data.empty:
        raise ValueError(
            "No complete inference rows were generated"
        )

    feature_values = inference_data[
        ordered_feature_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Inference features contain non-finite values"
        )

    return inference_data


def predict_future_volatility(
    model,
    inference_data,
    ordered_feature_columns,
):
    """Predict future 20-day annualized volatility.

    The frozen Ridge model predicts log volatility, so the
    raw prediction must be transformed with exp().
    """

    if inference_data.empty:
        raise ValueError(
            "Inference data is empty"
        )

    latest_row = inference_data.tail(1)

    feature_data = latest_row[
        ordered_feature_columns
    ].copy()

    if list(feature_data.columns) != list(
        ordered_feature_columns
    ):
        raise ValueError(
            "Model feature order is incorrect"
        )

    predicted_log_volatility = float(
        model.predict(feature_data)[0]
    )

    predicted_volatility = float(
        np.exp(predicted_log_volatility)
    )

    if (
        not np.isfinite(predicted_volatility)
        or predicted_volatility <= 0
    ):
        raise ValueError(
            "Model returned invalid volatility"
        )

    return {
        "as_of_date": latest_row[
            "date"
        ].iloc[0],
        "predicted_volatility":
            predicted_volatility,
        "feature_data": feature_data,
    }
