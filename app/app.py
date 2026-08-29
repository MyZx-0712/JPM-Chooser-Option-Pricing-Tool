
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from market_data import update_market_data
from ml_engine import (
    calculate_empirical_error_band,
    load_approach1_bundle,
    load_historical_reference,
)
from pricing_engine import (
    analytical_simple_chooser_value,
    calculate_stress_prices,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
MARKET_CACHE_PATH = (
    PROJECT_DIRECTORY
    / "data_cache"
    / "latest_market_data.csv"
)


st.set_page_config(
    page_title="JPM Chooser Option Pricing Tool",
    page_icon="📈",
    layout="wide",
)

st.title("JPM Chooser Option Pricing Tool")
st.caption(
    "Traditional BSM and Approach 1 ML Volatility Forecasting"
)


@st.cache_resource(show_spinner=False)
def get_model_assets():
    bundle = load_approach1_bundle(
        PROJECT_DIRECTORY
    )
    historical = load_historical_reference(
        PROJECT_DIRECTORY,
        bundle,
    )
    return bundle, historical


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def get_market_snapshot(cache_path_text):
    cache_path = Path(cache_path_text)

    try:
        return update_market_data(cache_path)
    except Exception as error:
        if not cache_path.exists():
            raise

        cached = pd.read_csv(cache_path)
        cached["data_status"] = (
            "local_cache_fallback"
        )
        cached["update_error"] = str(error)
        return cached


model_bundle, historical_reference = (
    get_model_assets()
)

if "market_snapshot" not in st.session_state:
    with st.spinner(
        "Checking market data..."
    ):
        st.session_state["market_snapshot"] = (
            get_market_snapshot(
                str(MARKET_CACHE_PATH)
            )
        )


st.subheader("Data and Model Status")

if st.button("Refresh market data"):
    get_market_snapshot.clear()

    with st.spinner(
        "Refreshing market data..."
    ):
        st.session_state["market_snapshot"] = (
            get_market_snapshot(
                str(MARKET_CACHE_PATH)
            )
        )


market_snapshot = st.session_state[
    "market_snapshot"
]

latest_market_row = market_snapshot.iloc[0]

market_status = str(
    latest_market_row.get(
        "data_status",
        "unknown",
    )
)

market_date = pd.to_datetime(
    latest_market_row["market_date"]
).date()

historical_date = pd.to_datetime(
    historical_reference["as_of_date"]
).date()

historical_volatility = float(
    historical_reference[
        "predicted_volatility"
    ]
)


status_column_1, status_column_2, status_column_3, status_column_4 = (
    st.columns(4)
)

status_column_1.metric(
    "Market-data date",
    str(market_date),
)

status_column_2.metric(
    "Approach 1 reference date",
    str(historical_date),
)

status_column_3.metric(
    "Historical predicted volatility",
    f"{historical_volatility * 100:.2f}%",
)

status_column_4.metric(
    "Model features",
    len(model_bundle["feature_columns"]),
)


st.warning(
    "Current Approach 1 pricing is disabled because the remote "
    "sources have not supplied a complete, current 34-feature "
    "input. The cached model result below is historical reference "
    f"only (as of {historical_date}) and is not a current market forecast."
)


def safe_float(value, fallback):
    try:
        number = float(value)

        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass

    return float(fallback)


default_stock_price = safe_float(
    latest_market_row.get("jpm_close"),
    239.71,
)

default_rate = safe_float(
    latest_market_row.get("dgs1_rate_pct"),
    4.16,
)


with st.sidebar:
    st.header("Contract Inputs")

    stock_price = st.number_input(
        "Stock price ($)",
        min_value=1.0,
        max_value=2000.0,
        value=default_stock_price,
        step=1.0,
    )

    strike_price = st.number_input(
        "Strike price ($)",
        min_value=1.0,
        max_value=2000.0,
        value=240.0,
        step=1.0,
    )

    risk_free_rate_pct = st.number_input(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=default_rate,
        step=0.10,
    )

    dividend_yield_pct = st.number_input(
        "Dividend yield (%)",
        min_value=0.0,
        max_value=20.0,
        value=2.33,
        step=0.10,
    )

    traditional_volatility_pct = (
        st.number_input(
            "Traditional BSM volatility (%)",
            min_value=1.0,
            max_value=200.0,
            value=20.00,
            step=0.50,
        )
    )

    maturity_time = st.number_input(
        "Maturity time (years)",
        min_value=0.10,
        max_value=5.00,
        value=1.00,
        step=0.10,
    )

    choice_fraction = st.slider(
        "Choice time as share of maturity",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )


risk_free_rate = risk_free_rate_pct / 100
dividend_yield = dividend_yield_pct / 100
traditional_volatility = (
    traditional_volatility_pct / 100
)
choice_time = (
    maturity_time * choice_fraction
)


traditional_price = (
    analytical_simple_chooser_value(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=traditional_volatility,
        choice_time=choice_time,
        maturity_time=maturity_time,
    )
)

historical_approach1_price = (
    analytical_simple_chooser_value(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=historical_volatility,
        choice_time=choice_time,
        maturity_time=maturity_time,
    )
)

historical_error_band = (
    calculate_empirical_error_band(
        historical_approach1_price,
        model_bundle["error_config"],
    )
)


overview_tab, sensitivity_tab, performance_tab, data_tab = (
    st.tabs(
        [
            "Pricing Overview",
            "Sensitivity Analysis",
            "Model Performance",
            "Data Details",
        ]
    )
)


with overview_tab:
    st.subheader("Pricing Comparison")

    price_column_1, price_column_2 = (
        st.columns(2)
    )

    price_column_1.metric(
        "Traditional BSM price",
        f"${traditional_price:,.2f}",
    )

    price_difference = (
        historical_approach1_price
        - traditional_price
    )

    price_column_2.metric(
        "Approach 1 reference price",
        f"${historical_approach1_price:,.2f}",
    )

    price_column_2.caption(
        "Difference vs BSM: "
        f"{price_difference:+,.2f} USD"
    )

    st.info(
        "The Approach 1 value uses the model volatility recorded "
        f"as of {historical_date}. It is displayed for historical "
        "demonstration only."
    )

    st.write(
        "**Historical empirical error range:** "
        f"\\${historical_error_band['lower_bound']:,.2f} "
        "to "
        f"\\${historical_error_band['upper_bound']:,.2f}"
    )

    st.caption(
        "This range is based on the 95th percentile of historical "
        "absolute test errors. It is not a market confidence interval."
    )

    comparison_table = pd.DataFrame(
        {
            "Pricing method": [
                "Traditional BSM",
                (
                    "Approach 1 historical reference "
                    f"({historical_date})"
                ),
            ],
            "Chooser-option price ($)": [
                traditional_price,
                historical_approach1_price,
            ],
        }
    )

    st.bar_chart(
        comparison_table.set_index(
            "Pricing method"
        )
    )

    st.subheader("Required Stress Scenarios")

    scenario_prices = calculate_stress_prices(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=traditional_volatility,
        choice_time=choice_time,
        maturity_time=maturity_time,
    )

    baseline_price = scenario_prices[
        "Baseline"
    ]

    scenario_table = pd.DataFrame(
        {
            "Scenario": list(
                scenario_prices.keys()
            ),
            "Price ($)": list(
                scenario_prices.values()
            ),
        }
    )

    scenario_table["Change ($)"] = (
        scenario_table["Price ($)"]
        - baseline_price
    )

    scenario_table["Change (%)"] = (
        scenario_table["Change ($)"]
        / baseline_price
        * 100
    )

    st.dataframe(
        scenario_table.style.format(
            {
                "Price ($)": "${:,.4f}",
                "Change ($)": "${:,.4f}",
                "Change (%)": "{:,.2f}%",
            }
        ),
        use_container_width=True,
    )


with sensitivity_tab:
    st.subheader("Volatility Sensitivity")

    volatility_grid = np.linspace(
        max(0.01, traditional_volatility * 0.50),
        traditional_volatility * 1.50,
        25,
    )

    volatility_prices = [
        analytical_simple_chooser_value(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=value,
            choice_time=choice_time,
            maturity_time=maturity_time,
        )
        for value in volatility_grid
    ]

    volatility_chart = pd.DataFrame(
        {
            "Volatility (%)": (
                volatility_grid * 100
            ),
            "Chooser-option price ($)": (
                volatility_prices
            ),
        }
    ).set_index("Volatility (%)")

    st.line_chart(volatility_chart)

    st.subheader("Interest-Rate Sensitivity")

    rate_grid = np.linspace(
        max(0.0, risk_free_rate - 0.02),
        risk_free_rate + 0.02,
        25,
    )

    rate_prices = [
        analytical_simple_chooser_value(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=value,
            dividend_yield=dividend_yield,
            volatility=traditional_volatility,
            choice_time=choice_time,
            maturity_time=maturity_time,
        )
        for value in rate_grid
    ]

    rate_price_changes = (
        np.asarray(rate_prices)
        - traditional_price
    )

    rate_chart = pd.DataFrame(
        {
            "Risk-free rate (%)": (
                rate_grid * 100
            ),
            "Price change vs current rate (USD)": (
                rate_price_changes
            ),
        }
    ).set_index("Risk-free rate (%)")

    st.line_chart(rate_chart)

    st.caption(
        "The chart shows price changes relative to the "
        "current risk-free-rate input. The effect is small "
        "because the call and put components partly offset."
    )


with performance_tab:
    st.subheader(
        "Approach 1 Frozen Test Performance"
    )

    error_config = model_bundle[
        "error_config"
    ]

    metric_column_1, metric_column_2, metric_column_3 = (
        st.columns(3)
    )

    metric_column_1.metric(
        "MAE",
        f"${error_config['mae']:,.4f}",
    )

    metric_column_2.metric(
        "RMSE",
        f"${error_config['rmse']:,.4f}",
    )

    metric_column_3.metric(
        "R²",
        f"{error_config['r2']:.6f}",
    )

    st.write(
        {
            "Test observations": (
                error_config[
                    "number_of_test_observations"
                ]
            ),
            "90% empirical absolute error": (
                error_config[
                    "empirical_absolute_error_90"
                ]
            ),
            "95% empirical absolute error": (
                error_config[
                    "empirical_absolute_error_95"
                ]
            ),
            "Selected model": (
                error_config["model"]
            ),
        }
    )

    st.success(
        "Approach 1 is the selected ML method. "
        "Approach 2 is intentionally excluded from the final tool "
        "because it failed the frozen out-of-sample test."
    )


with data_tab:
    st.subheader("Market Data")

    st.write(
        {
            "market_status": market_status,
            "market_date": str(market_date),
            "data_source": latest_market_row.get(
                "data_source",
                "",
            ),
            "retrieved_at": latest_market_row.get(
                "retrieved_at",
                "",
            ),
            "update_error": latest_market_row.get(
                "update_error",
                "",
            ),
        }
    )

    st.subheader(
        "Approach 1 Historical Cache"
    )

    st.write(
        {
            "as_of_date": str(
                historical_date
            ),
            "predicted_volatility": (
                f"{historical_volatility * 100:.2f}%"
            ),
            "feature_count": len(
                model_bundle["feature_columns"]
            ),
            "is_current_market_forecast": (
                historical_reference[
                    "is_current_market_forecast"
                ]
            ),
            "warning": (
                historical_reference[
                    "metadata"
                ]["warning"]
            ),
        }
    )


st.divider()

st.caption(
    "Educational project tool only; not investment advice. "
    "Current ML pricing is displayed only when a complete and "
    "date-valid 34-feature input is available."
)
