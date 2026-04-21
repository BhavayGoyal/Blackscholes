from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

OptionType = Literal["call", "put"]
EUROPEAN_STYLE_SYMBOLS: tuple[str, ...] = ("^SPX", "^NDX", "^RUT", "^VIX")

@dataclass
class MarketContext:
    symbol: str
    spot: float
    expiry: str
    rate: float
    sigma: float
    option_type: OptionType


def validate_symbol_is_european(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if normalized not in EUROPEAN_STYLE_SYMBOLS:
        allowed = ", ".join(EUROPEAN_STYLE_SYMBOLS)
        raise ValueError(
            f"{symbol} is not in the European-style allowlist. Use one of: {allowed}."
        )
    return normalized


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    rate: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike must be positive.")
    if time_to_expiry_years <= 0:
        raise ValueError("Time to expiry must be > 0.")
    if sigma <= 0:
        raise ValueError("Volatility must be > 0.")

    d1 = (
        log(spot / strike) + (rate + 0.5 * sigma * sigma) * time_to_expiry_years
    ) / (sigma * sqrt(time_to_expiry_years))
    d2 = d1 - sigma * sqrt(time_to_expiry_years)

    if option_type == "call":
        return spot * norm_cdf(d1) - strike * exp(-rate * time_to_expiry_years) * norm_cdf(d2)
    if option_type == "put":
        return strike * exp(-rate * time_to_expiry_years) * norm_cdf(-d2) - spot * norm_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'.")


def get_spot_price(ticker: yf.Ticker) -> float:
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info:
        value = fast_info.get("lastPrice") or fast_info.get("regularMarketPrice")
        if value and value > 0:
            return float(value)

    hist = ticker.history(period="5d", interval="1d")
    if hist.empty or "Close" not in hist:
        raise ValueError("Unable to fetch a valid spot price from yfinance.")
    return float(hist["Close"].dropna().iloc[-1])


def estimate_historical_volatility(ticker: yf.Ticker, period: str = "1y") -> float:
    hist = ticker.history(period=period, interval="1d")
    if hist.empty or "Close" not in hist:
        raise ValueError("Unable to estimate volatility: missing historical price data.")

    returns = hist["Close"].pct_change().dropna()
    if returns.empty:
        raise ValueError("Unable to estimate volatility: insufficient return samples.")
    return float(returns.std() * np.sqrt(252.0))


def get_risk_free_rate_auto() -> float:
    irx = yf.Ticker("^IRX")
    hist = irx.history(period="5d", interval="1d")
    if hist.empty or "Close" not in hist:
        raise ValueError("Unable to fetch risk-free rate proxy (^IRX).")
    latest_yield_percent = float(hist["Close"].dropna().iloc[-1])
    return latest_yield_percent / 100.0


def list_expiries(symbol: str) -> list[str]:
    symbol = validate_symbol_is_european(symbol)
    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options)
    if not expiries:
        raise ValueError(f"No listed option expiries found for {symbol}.")
    return expiries


def time_to_expiry_years(expiry: str) -> float:
    expiry_ts = pd.Timestamp(expiry).tz_localize(None)
    now = pd.Timestamp.utcnow().tz_localize(None)
    days = (expiry_ts - now).total_seconds() / (24.0 * 3600.0)
    return max(days / 365.0, 1e-6)


def _choose_market_price(df: pd.DataFrame) -> pd.Series:
    bid = df.get("bid", pd.Series(np.nan, index=df.index))
    ask = df.get("ask", pd.Series(np.nan, index=df.index))
    mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, np.nan)
    last = df.get("lastPrice", pd.Series(np.nan, index=df.index))
    chosen = np.where(np.isfinite(mid), mid, last)
    return pd.Series(chosen, index=df.index, dtype="float64")


def fetch_option_chain(symbol: str, expiry: str, option_type: OptionType) -> tuple[pd.DataFrame, float]:
    symbol = validate_symbol_is_european(symbol)
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(expiry)
    table = chain.calls if option_type == "call" else chain.puts
    if table.empty:
        raise ValueError(f"No {option_type} options found for {symbol} on {expiry}.")
    return table.copy(), get_spot_price(ticker)


def build_pricing_comparison(
    chain_df: pd.DataFrame,
    context: MarketContext,
    strike_min: float | None = None,
    strike_max: float | None = None,
    min_open_interest: int = 0,
) -> pd.DataFrame:
    df = chain_df.copy()

    if strike_min is not None:
        df = df[df["strike"] >= strike_min]
    if strike_max is not None:
        df = df[df["strike"] <= strike_max]
    if min_open_interest > 0 and "openInterest" in df:
        df = df[df["openInterest"].fillna(0) >= min_open_interest]
    if df.empty:
        raise ValueError("No options left after filters. Widen strike range or lower OI filter.")

    t = time_to_expiry_years(context.expiry)
    df["marketPrice"] = _choose_market_price(df)
    df = df[df["marketPrice"].notna() & (df["marketPrice"] > 0)].copy()
    if df.empty:
        raise ValueError("No valid market prices available (bid/ask/last).")

    df["theoreticalPrice"] = df["strike"].apply(
        lambda k: black_scholes_price(
            spot=context.spot,
            strike=float(k),
            time_to_expiry_years=t,
            rate=context.rate,
            sigma=context.sigma,
            option_type=context.option_type,
        )
    )
    df["absError"] = (df["theoreticalPrice"] - df["marketPrice"]).abs()
    df["signedError"] = df["theoreticalPrice"] - df["marketPrice"]
    df["pctError"] = np.where(
        df["marketPrice"] > 0, 100.0 * df["signedError"] / df["marketPrice"], np.nan
    )
    df["moneyness"] = context.spot / df["strike"]
    return df.sort_values("strike").reset_index(drop=True)


def summarize_errors(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        raise ValueError("Cannot summarize errors for empty dataframe.")
    return {
        "contracts_analyzed": float(len(df)),
        "mae": float(df["absError"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(df["signedError"])))),
        "mape_percent": float(np.nanmean(np.abs(df["pctError"]))),
        "max_abs_error": float(df["absError"].max()),
    }


def generate_explanation(df: pd.DataFrame, summary: dict[str, float], context: MarketContext) -> str:
    t = time_to_expiry_years(context.expiry)
    lines = [
        f"Black-Scholes was run for {int(summary['contracts_analyzed'])} {context.option_type} contracts "
        f"for {context.symbol} ({context.expiry}) using sigma={context.sigma:.2%} and r={context.rate:.2%}.",
        f"Average absolute pricing error is {summary['mae']:.4f} and MAPE is {summary['mape_percent']:.2f}%.",
    ]

    deep_itm = df[df["moneyness"] > 1.1]["absError"].mean() if not df[df["moneyness"] > 1.1].empty else np.nan
    deep_otm = df[df["moneyness"] < 0.9]["absError"].mean() if not df[df["moneyness"] < 0.9].empty else np.nan
    if np.isfinite(deep_itm) and np.isfinite(deep_otm):
        if deep_itm > deep_otm:
            lines.append("Errors are larger on deep ITM strikes, often driven by wide spreads and early-exercise effects.")
        elif deep_otm > deep_itm:
            lines.append("Errors are larger on deep OTM strikes, often where low prices amplify relative error.")
        else:
            lines.append("ITM and OTM error levels are similar across the selected range.")

    if t >= 1.0:
        lines.append(
            "Because this is a long-dated expiry, errors are usually larger: a single short-rate proxy and a single historical sigma "
            "cannot capture the full term structure and volatility skew over multiple years."
        )
    if context.option_type == "put":
        lines.append(
            "For puts, model underpricing is common when dividends and American-exercise premium are not explicitly modeled."
        )

    lines.append(
        "Observed differences are expected because Black-Scholes assumes constant volatility/rates and European exercise, "
        "while market prices embed volatility smiles, discrete dividends, liquidity effects, and American-style features."
    )
    return " ".join(lines)