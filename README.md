# JPM Chooser Option Pricing Tool

## Project purpose

This Streamlit application is the final interactive tool for the
completed JPM chooser option pricing research project. It compares a traditional
analytical BSM-based chooser price with the selected Approach 1
machine-learning volatility model.

The tool is for research, demonstration, and educational use only.

## Pricing methods

### Traditional BSM benchmark

The traditional benchmark uses the volatility entered manually by the
user and applies the analytical simple chooser-option pricing formula.

### Approach 1

Approach 1 uses a trained Ridge regression pipeline to predict future
20-trading-day annualized volatility from 34 market, volatility,
interest-rate, dividend, sentiment, stress, outlier, and lag features.

The model predicts log volatility. The result is converted back to
annualized volatility and entered into the analytical chooser formula.

Approach 2 is intentionally excluded because it failed the frozen
out-of-sample test and extrapolated poorly outside its training range.

## Main functions

- Traditional analytical chooser-option pricing
- Approach 1 ML-volatility reference pricing
- Historical empirical pricing-error range
- Interactive contract inputs
- Volatility and interest-rate sensitivity charts
- 50% volatility-spike scenario
- Two-percentage-point interest-rate increase scenario
- Frozen model-performance metrics
- Market-data refresh and local-cache fallback
- Explicit data date, source, status, and warning display

## Model performance

Approach 1 frozen pricing-test results:

- Test observations: 250
- MAE: 2.4317 USD
- RMSE: 3.5284 USD
- R-squared: 0.971443
- 90% empirical absolute error: 5.2689 USD
- 95% empirical absolute error: 9.2416 USD

The error range is an empirical historical error band against the
internal analytical benchmark. It is not a market confidence interval.

## Current data policy

Approach 1 requires one complete, correctly ordered row containing all
34 model features.

If current remote data cannot provide all 34 features:

- Current Approach 1 pricing is disabled.
- The data problem and date are displayed.
- Traditional BSM pricing remains available.
- Historical ML output is shown only as a dated reference.
- Missing features are not filled with zero or future information.

The available historical reference is dated 2024-12-31 and must not be
interpreted as a current market forecast.

## Data sources

- JPM and VIX: Yahoo Finance through yfinance
- One-year US Treasury rate: FRED DGS1
- Historical fallback: Week 2 data with Week 5 to Week 7 feature
  reconstruction

Remote sources may be rate-limited, delayed, or unavailable.

## Application structure

- `app/app.py`: final Streamlit interface
- `app/pricing_engine.py`: analytical pricing and stress scenarios
- `app/market_data.py`: remote update and cache-fallback logic
- `app/feature_engineering.py`: reconstruction of 34 model features
- `app/ml_engine.py`: model loading, inference, and error-band logic
- `requirements.txt`: fixed Python dependencies
- `models/week6_approach1_final_model.joblib`: selected frozen model
- `config/week6_final_configuration.json`: model configuration
- `config/approach1_error_margin.json`: test-error configuration
- `data_cache/approach1_feature_cache.csv`: historical feature cache
- `data_cache/approach1_feature_cache_metadata.json`: cache metadata
- `data_cache/latest_market_data.csv`: latest available market snapshot

## Installation

Python 3.10 is recommended for compatibility with the frozen model.

From the project directory, run:

    python -m pip install -r requirements.txt

## Run the application

From the project directory, run:

    python -m streamlit run app/app.py

Then open the local address shown by Streamlit, normally:

    http://localhost:8501

## How to use

1. Review the market-data date and warning.
2. Enter the contract and traditional-volatility inputs.
3. Compare Traditional BSM with the dated Approach 1 reference.
4. Review the historical empirical error range.
5. Review the sensitivity, performance, and data-detail tabs.
6. Do not describe a historical-cache result as a current forecast.

## Limitations

- The current remote snapshot does not provide all 34 ML features.
- The historical cache is retained for reproducible demonstration.
- Model performance is based on the frozen project test set.
- Results exclude liquidity, transaction costs, and execution risk.
- Remote sources do not guarantee commercial real-time availability.

## Disclaimer

This application is for project and educational purposes only. It does
not provide financial advice, trading recommendations, or guaranteed
market prices.
