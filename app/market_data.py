
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


def _remove_timezone(date_value):
    date_value = pd.Timestamp(date_value)

    if date_value.tzinfo is not None:
        date_value = date_value.tz_localize(None)

    return date_value


def _extract_latest_close(
    downloaded_data,
    ticker,
):
    ticker_data = downloaded_data.copy()

    if isinstance(
        ticker_data.columns,
        pd.MultiIndex,
    ):
        ticker_level = None

        for level_number in range(
            ticker_data.columns.nlevels
        ):
            level_values = (
                ticker_data.columns
                .get_level_values(level_number)
            )

            if ticker in level_values:
                ticker_level = level_number
                break

        if ticker_level is None:
            raise ValueError(
                f"Ticker {ticker} was not returned"
            )

        ticker_data = ticker_data.xs(
            ticker,
            axis=1,
            level=ticker_level,
        )

    if "Close" not in ticker_data.columns:
        raise ValueError(
            f"Close column missing for {ticker}"
        )

    close_values = ticker_data["Close"]

    if isinstance(close_values, pd.DataFrame):
        close_values = close_values.iloc[:, 0]

    close_values = pd.to_numeric(
        close_values,
        errors="coerce",
    ).dropna()

    if close_values.empty:
        raise ValueError(
            f"No valid close values for {ticker}"
        )

    latest_date = _remove_timezone(
        close_values.index[-1]
    )

    latest_close = float(
        close_values.iloc[-1]
    )

    return latest_date, latest_close


def fetch_yahoo_snapshot():
    downloaded_data = yf.download(
        tickers=["JPM", "^VIX"],
        period="1mo",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="ticker",
    )

    if downloaded_data.empty:
        raise ValueError(
            "Yahoo Finance returned no data"
        )

    jpm_date, jpm_close = (
        _extract_latest_close(
            downloaded_data,
            "JPM",
        )
    )

    vix_date, vix_level = (
        _extract_latest_close(
            downloaded_data,
            "^VIX",
        )
    )

    return {
        "jpm_date": jpm_date,
        "jpm_close": jpm_close,
        "vix_date": vix_date,
        "vix_level": vix_level,
    }


def fetch_dgs1_snapshot():
    fred_url = (
        "https://fred.stlouisfed.org/"
        "graph/fredgraph.csv?id=DGS1"
    )

    response = requests.get(
        fred_url,
        timeout=20,
        headers={
            "User-Agent": (
                "Week7-Market-Data-Prototype/1.0"
            )
        },
    )

    response.raise_for_status()

    fred_data = pd.read_csv(
        StringIO(response.text)
    )

    normalized_columns = {
        str(column).strip().lower(): column
        for column in fred_data.columns
    }

    date_column = next(
        (
            original_column
            for normalized_name, original_column
            in normalized_columns.items()
            if "date" in normalized_name
        ),
        None,
    )

    value_column = next(
        (
            original_column
            for normalized_name, original_column
            in normalized_columns.items()
            if normalized_name == "dgs1"
        ),
        None,
    )

    if date_column is None or value_column is None:
        raise ValueError(
            "Unexpected FRED DGS1 response format"
        )

    cleaned_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                fred_data[date_column],
                errors="coerce",
            ),
            "value": pd.to_numeric(
                fred_data[value_column],
                errors="coerce",
            ),
        }
    ).dropna()

    if cleaned_data.empty:
        raise ValueError(
            "FRED returned no valid DGS1 data"
        )

    latest_row = cleaned_data.iloc[-1]

    return {
        "dgs1_date": _remove_timezone(
            latest_row["date"]
        ),
        "dgs1_rate_pct": float(
            latest_row["value"]
        ),
    }


def load_cached_market_data(cache_path):
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Market-data cache not found: "
            f"{cache_path}"
        )

    cached_data = pd.read_csv(
        cache_path
    ).tail(1).copy()

    if cached_data.empty:
        raise ValueError(
            "Market-data cache is empty"
        )

    return cached_data


def update_market_data(cache_path):
    cache_path = Path(cache_path)

    try:
        yahoo_snapshot = (
            fetch_yahoo_snapshot()
        )

        rate_snapshot = (
            fetch_dgs1_snapshot()
        )

        component_dates = [
            yahoo_snapshot["jpm_date"],
            yahoo_snapshot["vix_date"],
            rate_snapshot["dgs1_date"],
        ]

        live_snapshot = pd.DataFrame(
            [
                {
                    "market_date": max(
                        component_dates
                    ),
                    "jpm_date": (
                        yahoo_snapshot[
                            "jpm_date"
                        ]
                    ),
                    "jpm_close": (
                        yahoo_snapshot[
                            "jpm_close"
                        ]
                    ),
                    "vix_date": (
                        yahoo_snapshot[
                            "vix_date"
                        ]
                    ),
                    "vix_level": (
                        yahoo_snapshot[
                            "vix_level"
                        ]
                    ),
                    "dgs1_date": (
                        rate_snapshot[
                            "dgs1_date"
                        ]
                    ),
                    "dgs1_rate_pct": (
                        rate_snapshot[
                            "dgs1_rate_pct"
                        ]
                    ),
                    "retrieved_at": (
                        pd.Timestamp.now(
                            tz="UTC"
                        ).isoformat()
                    ),
                    "data_status": (
                        "live_update"
                    ),
                    "data_source": (
                        "Yahoo Finance: JPM/VIX; "
                        "FRED: DGS1"
                    ),
                    "update_error": "",
                }
            ]
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        live_snapshot.to_csv(
            cache_path,
            index=False,
        )

        return live_snapshot

    except Exception as update_exception:
        cached_data = load_cached_market_data(
            cache_path
        )

        if "jpm_date" not in cached_data:
            cached_data["jpm_date"] = (
                cached_data["market_date"]
            )

        if "vix_date" not in cached_data:
            cached_data["vix_date"] = (
                cached_data["market_date"]
            )

        if "dgs1_date" not in cached_data:
            cached_data["dgs1_date"] = (
                cached_data["market_date"]
            )

        cached_data["checked_at"] = (
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
        )

        cached_data["data_status"] = (
            "cache_fallback"
        )

        cached_data["update_error"] = (
            f"{type(update_exception).__name__}: "
            f"{str(update_exception)[:400]}"
        )

        return cached_data
