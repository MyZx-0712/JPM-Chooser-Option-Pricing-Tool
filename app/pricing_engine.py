
import math
from scipy.stats import norm


def black_scholes_price(
    option_type,
    stock_price,
    strike_price,
    risk_free_rate,
    dividend_yield,
    volatility,
    maturity_time,
):
    """Calculate a European call or put price."""

    if stock_price <= 0:
        raise ValueError("stock_price must be positive")
    if strike_price <= 0:
        raise ValueError("strike_price must be positive")
    if volatility <= 0:
        raise ValueError("volatility must be positive")
    if maturity_time <= 0:
        raise ValueError("maturity_time must be positive")

    sqrt_time = math.sqrt(maturity_time)

    d1 = (
        math.log(stock_price / strike_price)
        + (
            risk_free_rate
            - dividend_yield
            + 0.5 * volatility ** 2
        )
        * maturity_time
    ) / (volatility * sqrt_time)

    d2 = d1 - volatility * sqrt_time

    if option_type == "call":
        return (
            stock_price
            * math.exp(-dividend_yield * maturity_time)
            * norm.cdf(d1)
            - strike_price
            * math.exp(-risk_free_rate * maturity_time)
            * norm.cdf(d2)
        )

    if option_type == "put":
        return (
            strike_price
            * math.exp(-risk_free_rate * maturity_time)
            * norm.cdf(-d2)
            - stock_price
            * math.exp(-dividend_yield * maturity_time)
            * norm.cdf(-d1)
        )

    raise ValueError("option_type must be 'call' or 'put'")


def analytical_simple_chooser_value(
    stock_price,
    strike_price,
    risk_free_rate,
    dividend_yield,
    volatility,
    choice_time,
    maturity_time,
):
    """Calculate the analytical value of a simple chooser option."""

    if not 0 < choice_time < maturity_time:
        raise ValueError(
            "choice_time must be between 0 and maturity_time"
        )

    call_value = black_scholes_price(
        option_type="call",
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity_time=maturity_time,
    )

    adjusted_strike = strike_price * math.exp(
        -(risk_free_rate - dividend_yield)
        * (maturity_time - choice_time)
    )

    choice_put_value = black_scholes_price(
        option_type="put",
        stock_price=stock_price,
        strike_price=adjusted_strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity_time=choice_time,
    )

    chooser_adjustment = math.exp(
        -dividend_yield * (maturity_time - choice_time)
    ) * choice_put_value

    return call_value + chooser_adjustment


def calculate_stress_prices(
    stock_price,
    strike_price,
    risk_free_rate,
    dividend_yield,
    volatility,
    choice_time,
    maturity_time,
):
    """Return baseline and the two required Week 7 scenarios."""

    common_inputs = {
        "stock_price": stock_price,
        "strike_price": strike_price,
        "dividend_yield": dividend_yield,
        "choice_time": choice_time,
        "maturity_time": maturity_time,
    }

    baseline_price = analytical_simple_chooser_value(
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        **common_inputs,
    )

    volatility_spike_price = analytical_simple_chooser_value(
        risk_free_rate=risk_free_rate,
        volatility=volatility * 1.50,
        **common_inputs,
    )

    rate_hike_price = analytical_simple_chooser_value(
        risk_free_rate=risk_free_rate + 0.02,
        volatility=volatility,
        **common_inputs,
    )

    return {
        "Baseline": baseline_price,
        "50% volatility spike": volatility_spike_price,
        "2% rate hike": rate_hike_price,
    }
