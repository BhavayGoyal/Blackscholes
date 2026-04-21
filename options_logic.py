from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path
import re
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------

OptionType = Literal["call", "put"]
MarketMode = Literal["us", "india"]

# US: index options are European-style (cash-settled)
# Individual US stocks (AAPL, AMZN, etc.) are American-style — flagged in UI
US_EUROPEAN_SYMBOLS: tuple[str, ...] = ("^SPX", "^NDX", "^RUT", "^VIX")
US_AMERICAN_SYMBOLS: tuple[str, ...] = ("AAPL", "AMZN", "MSFT", "TSLA", "NVDA")
EUROPEAN_STYLE_SYMBOLS = US_EUROPEAN_SYMBOLS  # kept for backward compat

# India: NSE index options are European-style (cash-settled)
# NSE stock options are American-style — flagged in UI
INDIA_EUROPEAN_SYMBOLS: tuple[str, ...] = ("NIFTY", "BANKNIFTY")
INDIA_AMERICAN_SYMBOLS: tuple[str, ...] = (
    "DMART",
    "HYUNDAI",
    "RELIANCE",
    "HDFCBANK",
    "INFY",
    "GAIL",
    "FEDERALBNK",
    "BANKBARODA",
    "INDIGO",
)

# yfinance ticker mapping for Indian symbols (used for spot + historical vol)
INDIA_YF_MAP: dict[str, str] = {
    "NIFTY":     "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "DMART":     "DMART.NS",
    "HYUNDAI":   "HYUNDAI.NS",
    "RELIANCE":  "RELIANCE.NS",
    "HDFCBANK":  "HDFCBANK.NS",
    "INFY":      "INFY.NS",
}

# Approximate dividend yields — used to flag systematic BS error in explanation
INDIA_DIV_YIELD: dict[str, float] = {
    "NIFTY":     0.013,
    "BANKNIFTY": 0.010,
    "DMART":     0.001,
    "HYUNDAI":   0.005,
    "RELIANCE":  0.004,
    "HDFCBANK":  0.010,
    "INFY":      0.025,
}

# RBI repo rate — best available proxy for Indian risk-free rate.
# yfinance does not carry Indian T-bill yields (^IRX is US-only).
# Update this when RBI changes policy rate (~4 times/year).
RBI_REPO_RATE: float = 0.0625  # 6.25% as of Apr 2026

# Minimum trading days required to trust a historical vol estimate
MIN_HISTORY_DAYS: int = 60


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarketContext:
    symbol: str
    spot: float
    expiry: str
    rate: float
    sigma: float
    option_type: OptionType
    market: MarketMode = "us"
    is_american_style: bool = False
    div_yield: float = 0.0


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def is_american_style(symbol: str, market: MarketMode) -> bool:
    if market == "us":
        return symbol.upper() in US_AMERICAN_SYMBOLS
    return False


def get_exercise_style_note(symbol: str, market: MarketMode) -> str | None:
    if is_american_style(symbol, market):
        return (
            f"{symbol} options are American-style (early exercise allowed). "
            "Black-Scholes assumes European exercise only, so it will systematically "
            "underprice puts (early exercise premium is ignored)."
        )
    return None


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    rate: float,
    sigma: float,
    option_type: OptionType,
    div_yield: float = 0.0,
) -> float:
    """
    Black-Scholes-Merton price with continuous dividend yield q.

    For US indices / stocks pass div_yield=0 (default, matches original behaviour).
    For Indian indices pass the approximate annual dividend yield so the
    explanation can quantify that error source; the formula itself becomes BSM.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike must be positive.")
    if time_to_expiry_years <= 0:
        raise ValueError("Time to expiry must be > 0.")
    if sigma <= 0:
        raise ValueError("Volatility must be > 0.")

    q = div_yield
    d1 = (
        log(spot / strike) + (rate - q + 0.5 * sigma * sigma) * time_to_expiry_years
    ) / (sigma * sqrt(time_to_expiry_years))
    d2 = d1 - sigma * sqrt(time_to_expiry_years)

    if option_type == "call":
        return (
            spot * exp(-q * time_to_expiry_years) * norm_cdf(d1)
            - strike * exp(-rate * time_to_expiry_years) * norm_cdf(d2)
        )
    if option_type == "put":
        return (
            strike * exp(-rate * time_to_expiry_years) * norm_cdf(-d2)
            - spot * exp(-q * time_to_expiry_years) * norm_cdf(-d1)
        )
    raise ValueError("option_type must be 'call' or 'put'.")


# ---------------------------------------------------------------------------
# Spot price
# ---------------------------------------------------------------------------

def get_spot_price(ticker: yf.Ticker) -> float:
    """Works for both US and Indian yfinance tickers."""
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info:
        value = fast_info.get("lastPrice") or fast_info.get("regularMarketPrice")
        if value and value > 0:
            return float(value)
    # Fallback: use recent close (more reliable for Indian indices)
    hist = ticker.history(period="5d", interval="1d")
    if hist.empty or "Close" not in hist:
        raise ValueError("Unable to fetch a valid spot price from yfinance.")
    return float(hist["Close"].dropna().iloc[-1])


def get_spot_price_india(symbol: str) -> float:
    normalized = symbol.upper()
    yf_sym = INDIA_YF_MAP.get(normalized, f"{normalized}.NS")
    return get_spot_price(yf.Ticker(yf_sym))


# ---------------------------------------------------------------------------
# Historical volatility
# ---------------------------------------------------------------------------

def estimate_historical_volatility(ticker: yf.Ticker, period: str = "1y") -> float:
    """
    Annualised historical volatility from daily closes.
    Works for any yfinance ticker — US or Indian (.NS / ^NSEI etc.).
    """
    hist = ticker.history(period=period, interval="1d")
    if hist.empty or "Close" not in hist:
        raise ValueError("Unable to estimate volatility: missing historical price data.")

    returns = hist["Close"].pct_change().dropna()
    if len(returns) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Only {len(returns)} trading days of history found "
            f"(minimum {MIN_HISTORY_DAYS} required). "
            "Use manual volatility input or choose a shorter period."
        )
    return float(returns.std() * np.sqrt(252.0))


def estimate_historical_volatility_india(symbol: str, period: str = "1y") -> tuple[float, str | None]:
    """
    Returns (sigma, warning_or_None).
    Falls back to shorter periods for recently-listed stocks (e.g. HYUNDAI).
    """
    normalized = symbol.upper()
    yf_sym = INDIA_YF_MAP.get(normalized, f"{normalized}.NS")

    ticker = yf.Ticker(yf_sym)

    for p, label in [(period, period), ("6mo", "6 months"), ("3mo", "3 months")]:
        try:
            sigma = estimate_historical_volatility(ticker, period=p)
            warning = (
                None if p == period
                else (
                    f"Limited price history for {symbol}: vol estimated from "
                    f"{label} of data. Consider using manual vol input."
                )
            )
            return sigma, warning
        except ValueError:
            continue

    raise ValueError(
        f"Insufficient price history for {symbol}. Use manual volatility input."
    )


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

def get_risk_free_rate_auto() -> float:
    """US: 13-week T-bill via ^IRX."""
    irx = yf.Ticker("^IRX")
    hist = irx.history(period="5d", interval="1d")
    if hist.empty or "Close" not in hist:
        raise ValueError("Unable to fetch risk-free rate proxy (^IRX).")
    return float(hist["Close"].dropna().iloc[-1]) / 100.0


def get_india_risk_free_rate() -> float:
    """
    India: RBI repo rate is the standard academic proxy.
    yfinance does not carry Indian T-bill yields, so this returns a
    hardcoded constant that should be updated when RBI changes policy.
    Current value: 6.25% (Apr 2026).
    """
    return RBI_REPO_RATE


# ---------------------------------------------------------------------------
# US option chain (yfinance)
# ---------------------------------------------------------------------------

def validate_us_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    all_us = US_EUROPEAN_SYMBOLS + US_AMERICAN_SYMBOLS
    if normalized not in all_us:
        allowed = ", ".join(all_us)
        raise ValueError(f"'{symbol}' not in US symbol list. Use one of: {allowed}.")
    return normalized


def list_expiries_us(symbol: str) -> list[str]:
    symbol = validate_us_symbol(symbol)
    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options)
    if not expiries:
        raise ValueError(f"No listed option expiries found for {symbol}.")
    return expiries


def fetch_option_chain_us(
    symbol: str, expiry: str, option_type: OptionType
) -> tuple[pd.DataFrame, float]:
    symbol = validate_us_symbol(symbol)
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(expiry)
    table = chain.calls if option_type == "call" else chain.puts
    if table.empty:
        raise ValueError(f"No {option_type} options found for {symbol} on {expiry}.")
    return table.copy(), get_spot_price(ticker)


# ---------------------------------------------------------------------------
# India option chain (NSE CSV)
# ---------------------------------------------------------------------------

# NSE CSV column layout (wide format):
#   CALLS: oi, chng_oi, volume, iv, ltp, chng, bid_qty, bid, ask, ask_qty
#   STRIKE
#   PUTS:  bid_qty, bid, ask, ask_qty, chng, ltp, iv, volume, chng_oi, oi

_NSE_COLS: list[str] = [
    "_lead",                                                      # empty leading col
    "c_oi", "c_chng_oi", "c_volume", "c_iv",
    "c_ltp", "c_chng", "c_bid_qty", "c_bid", "c_ask", "c_ask_qty",
    "strike",
    "p_bid_qty", "p_bid", "p_ask", "p_ask_qty",
    "p_chng", "p_ltp", "p_iv", "p_volume", "p_chng_oi", "p_oi",
    "_trail",                                                     # empty trailing col
]
_NSE_DROP_COLS: list[str] = ["_lead", "_trail"]


def _clean_nse_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("-", np.nan)
        .pipe(pd.to_numeric, errors="coerce")
    )


def parse_nse_csv(path: str | Path) -> pd.DataFrame:
    """
    Parse an NSE option-chain CSV (wide format) into a clean DataFrame.
    NSE CSVs have:
      row 0  — 'CALLS,,PUTS' header (skipped)
      row 1  — column name row (used as pandas header, then overridden)
      row 2+ — data rows, with a leading empty col and trailing empty col
    """
    df = pd.read_csv(
        path,
        skiprows=1,       # skip the 'CALLS,,PUTS' banner row
        header=0,         # treat next row as header (discarded below)
        names=_NSE_COLS,  # force our own column names
    )
    # Drop the structural empty columns
    df = df.drop(columns=[c for c in _NSE_DROP_COLS if c in df.columns], errors="ignore")
    for col in df.columns:
        df[col] = _clean_nse_numeric(df[col])
    return df.dropna(subset=["strike"]).reset_index(drop=True)


def nse_to_chain(df: pd.DataFrame, option_type: OptionType) -> pd.DataFrame:
    """
    Convert NSE wide DataFrame to the yfinance-compatible schema used by
    build_pricing_comparison (strike, bid, ask, lastPrice, openInterest,
    volume, impliedVolatility).
    """
    if option_type == "call":
        return pd.DataFrame({
            "strike":            df["strike"],
            "bid":               df["c_bid"],
            "ask":               df["c_ask"],
            "lastPrice":         df["c_ltp"],
            "openInterest":      df["c_oi"],
            "volume":            df["c_volume"],
            "impliedVolatility": df["c_iv"] / 100.0,
        })
    return pd.DataFrame({
        "strike":            df["strike"],
        "bid":               df["p_bid"],
        "ask":               df["p_ask"],
        "lastPrice":         df["p_ltp"],
        "openInterest":      df["p_oi"],
        "volume":            df["p_volume"],
        "impliedVolatility": df["p_iv"] / 100.0,
    })


def parse_expiry_from_filename(filename: str) -> str:
    """
    Extract expiry date string from NSE CSV filename.
    Pattern: option-chain-ED-<SYMBOL>-<DD>-<Mon>-<YYYY>.csv
    Returns ISO format: YYYY-MM-DD
    """
    stem = Path(filename).stem
    match = re.search(r"(\d{2}-[A-Za-z]{3}-\d{4})$", stem)
    if not match:
        raise ValueError(f"Could not parse expiry from filename '{filename}'.")
    return pd.Timestamp(match.group(1)).strftime("%Y-%m-%d")


def parse_symbol_from_filename(filename: str) -> str:
    """Extract symbol (NIFTY, BANKNIFTY, etc.) from NSE CSV filename."""
    stem = Path(filename).stem
    # Format: option-chain-ED-<SYMBOL>-DD-Mon-YYYY
    match = re.match(r"option-chain-ED-(.+)-\d{2}-[A-Za-z]{3}-\d{4}$", stem)
    if not match:
        raise ValueError(f"Could not parse symbol from filename '{filename}'.")
    return match.group(1).upper()


def list_india_csv_files(data_dir: str | Path) -> dict[str, list[Path]]:
    """
    Scan data_dir for NSE CSV files and return a dict:
        { symbol: [sorted list of Path objects] }
    Includes all symbols found in CSV filenames.
    """
    data_dir = Path(data_dir)
    result: dict[str, list[Path]] = {}
    for f in sorted(data_dir.glob("option-chain-ED-*.csv")):
        sym = parse_symbol_from_filename(f.name)
        result.setdefault(sym, []).append(f)
    return result


def fetch_option_chain_india(
    csv_path: str | Path,
    option_type: OptionType,
    symbol: str,
) -> tuple[pd.DataFrame, float]:
    """
    Load an NSE CSV and return (chain_df, spot_price).
    Spot is fetched live from yfinance (known map, else SYMBOL.NS fallback).
    """
    raw = parse_nse_csv(csv_path)
    chain = nse_to_chain(raw, option_type)
    spot = get_spot_price_india(symbol)
    return chain, spot


# ---------------------------------------------------------------------------
# Shared: time to expiry
# ---------------------------------------------------------------------------

def time_to_expiry_years(expiry: str) -> float:
    expiry_ts = pd.Timestamp(expiry).tz_localize(None)
    now = pd.Timestamp.utcnow().tz_localize(None)
    days = (expiry_ts - now).total_seconds() / (24.0 * 3600.0)
    return max(days / 365.0, 1e-6)


# ---------------------------------------------------------------------------
# Shared: market price selection
# ---------------------------------------------------------------------------

def _choose_market_price(df: pd.DataFrame) -> pd.Series:
    bid = df.get("bid", pd.Series(np.nan, index=df.index))
    ask = df.get("ask", pd.Series(np.nan, index=df.index))
    mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, np.nan)
    last = df.get("lastPrice", pd.Series(np.nan, index=df.index))
    chosen = np.where(np.isfinite(mid), mid, last)
    return pd.Series(chosen, index=df.index, dtype="float64")


# ---------------------------------------------------------------------------
# Shared: pricing comparison & error analysis
# ---------------------------------------------------------------------------

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
            div_yield=context.div_yield,
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


def generate_explanation(
    df: pd.DataFrame,
    summary: dict[str, float],
    context: MarketContext,
) -> str:
    t = time_to_expiry_years(context.expiry)
    lines = [
        f"Black-Scholes was run for {int(summary['contracts_analyzed'])} "
        f"{context.option_type} contracts on {context.symbol} "
        f"(expiry {context.expiry}) using sigma={context.sigma:.2%} "
        f"and r={context.rate:.2%}.",
        f"Average absolute pricing error: {summary['mae']:.4f}. "
        f"MAPE: {summary['mape_percent']:.2f}%.",
    ]

    # ITM vs OTM error pattern
    deep_itm = df[df["moneyness"] > 1.1]["absError"].mean() if not df[df["moneyness"] > 1.1].empty else np.nan
    deep_otm = df[df["moneyness"] < 0.9]["absError"].mean() if not df[df["moneyness"] < 0.9].empty else np.nan
    if np.isfinite(deep_itm) and np.isfinite(deep_otm):
        if deep_itm > deep_otm:
            lines.append(
                "Errors are larger on deep ITM strikes — typically driven by wide "
                "bid-ask spreads and, for American-style options, early-exercise premium."
            )
        elif deep_otm > deep_itm:
            lines.append(
                "Errors are larger on deep OTM strikes — low absolute prices amplify "
                "relative error, and market prices embed a fear/skew premium BS ignores."
            )

    # American-style warning
    if context.is_american_style:
        lines.append(
            f"{context.symbol} options are American-style. BS assumes European exercise "
            "only, so it systematically underprices puts where early exercise has value."
        )

    # Dividend yield impact
    if context.div_yield > 0.005:
        lines.append(
            f"A dividend yield of ~{context.div_yield:.1%} is applied (BSM adjustment). "
            "Without this, BS call prices would be overstated and put prices understated."
        )

    # Long-dated expiry
    if t >= 1.0:
        lines.append(
            "This is a long-dated expiry. Errors grow because a single constant vol "
            "and a single rate cannot capture the full term structure over multiple years."
        )

    # India-specific
    if context.market == "india":
        lines.append(
            "For Indian markets, the risk-free rate used is the RBI repo rate (a proxy). "
            "Actual market participants may use MIFOR or OIS rates, introducing a "
            "small but systematic rate-level error."
        )

    # Universal caveat
    lines.append(
        "Remaining differences reflect BS model limitations: constant-vol assumption "
        "(ignores volatility smile/skew), log-normal return distribution (ignores fat tails), "
        "no transaction costs, and continuous hedging."
    )
    return " ".join(lines)
