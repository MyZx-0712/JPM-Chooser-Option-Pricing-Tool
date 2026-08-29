"""Loading and inference utilities for Approach 1."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_engineering import (
    predict_future_volatility,
)


def load_approach1_bundle(project_directory):
    """Load the frozen model and its configurations."""

    project_directory = Path(
        project_directory
    )

    model_path = (
        project_directory
        / "models"
        / "week6_approach1_final_model.joblib"
    )

    model_config_path = (
        project_directory
        / "config"
        / "week6_final_configuration.json"
    )

    error_config_path = (
        project_directory
        / "config"
        / "approach1_error_margin.json"
    )

    model = joblib.load(model_path)

    model_config = json.loads(
        model_config_path.read_text(
            encoding="utf-8"
        )
    )

    error_config = json.loads(
        error_config_path.read_text(
            encoding="utf-8"
        )
    )

    feature_columns = model_config[
        "approach1"
    ]["feature_columns"]

    if hasattr(model, "feature_names_in_"):
        if (
            list(model.feature_names_in_)
            != feature_columns
        ):
            raise ValueError(
                "Model and configuration "
                "feature order do not match"
            )

    return {
        "model": model,
        "feature_columns": feature_columns,
        "model_config": model_config,
        "error_config": error_config,
    }


def load_historical_reference(
    project_directory,
    model_bundle,
):
    """Load the historical feature cache.

    This output is for historical reference and validation.
    It must not be presented as a current market forecast.
    """

    project_directory = Path(
        project_directory
    )

    feature_cache_path = (
        project_directory
        / "data_cache"
        / "approach1_feature_cache.csv"
    )

    metadata_path = (
        project_directory
        / "data_cache"
        / "approach1_feature_cache_metadata.json"
    )

    feature_cache = pd.read_csv(
        feature_cache_path,
        parse_dates=["date"],
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    expected_columns = [
        "date",
        *model_bundle["feature_columns"],
    ]

    if list(feature_cache.columns) != expected_columns:
        raise ValueError(
            "Historical feature cache schema "
            "does not match the frozen model"
        )

    prediction = predict_future_volatility(
        model=model_bundle["model"],
        inference_data=feature_cache,
        ordered_feature_columns=(
            model_bundle["feature_columns"]
        ),
    )

    return {
        "as_of_date": prediction["as_of_date"],
        "predicted_volatility": (
            prediction[
                "predicted_volatility"
            ]
        ),
        "metadata": metadata,
        "is_current_market_forecast": False,
    }


def calculate_empirical_error_band(
    ml_price,
    error_config,
):
    """Calculate the historical empirical price-error band."""

    ml_price = float(ml_price)

    if not np.isfinite(ml_price):
        raise ValueError(
            "ML price must be finite"
        )

    if ml_price < 0:
        raise ValueError(
            "ML price cannot be negative"
        )

    error_radius = float(
        error_config[
            "empirical_absolute_error_95"
        ]
    )

    lower_bound = max(
        0.0,
        ml_price - error_radius,
    )

    upper_bound = (
        ml_price + error_radius
    )

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "error_radius": error_radius,
        "label": (
            "Historical empirical error band "
            "against the analytical benchmark"
        ),
        "is_market_confidence_interval": False,
    }
