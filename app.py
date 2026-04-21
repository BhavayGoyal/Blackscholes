from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

from options_logic import (
    # constants
    US_EUROPEAN_SYMBOLS,
    US_AMERICAN_SYMBOLS,
    INDIA_EUROPEAN_SYMBOLS,
    INDIA_AMERICAN_SYMBOLS,
    INDIA_YF_MAP,
    INDIA_DIV_YIELD,
    RBI_REPO_RATE,
    # helpers
    MarketContext,
    build_pricing_comparison,
    estimate_historical_volatility,
    estimate_historical_volatility_india,
    fetch_option_chain_us,
    fetch_option_chain_india,
    generate_explanation,
    get_exercise_style_note,
    get_india_risk_free_rate,
    get_risk_free_rate_auto,
    get_spot_price,
    is_american_style,
    list_expiries_us,
    list_india_csv_files,
    parse_expiry_from_filename,
    summarize_errors,
    time_to_expiry_years,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Black-Scholes vs Market", layout="wide")
st.title("Black-Scholes Pricing vs Market Prices")
st.caption("Compare theoretical option prices with live market data — US (Yahoo Finance) or India (NSE CSV).")

# ---------------------------------------------------------------------------
# Data directory for India CSVs
# Adjust this path to wherever your CSV files are stored.
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Market selection ────────────────────────────────────────────────────
    st.subheader("Market")
    market = st.radio(
        "Market",
        options=["🇺🇸 US Market", "🇮🇳 Indian Market (NSE)"],
        index=0,
        label_visibility="collapsed",
    )
    market_mode: str = "us" if market.startswith("🇺🇸") else "india"

    st.divider()

    # ── Symbol & expiry ─────────────────────────────────────────────────────
    st.subheader("Instrument")

    if market_mode == "us":
        all_us_symbols = list(US_EUROPEAN_SYMBOLS) + list(US_AMERICAN_SYMBOLS)
        symbol = st.selectbox("Symbol", options=all_us_symbols, index=0)
        option_type = st.selectbox("Option type", options=["call", "put"], index=0)

        # Exercise-style notice
        american = is_american_style(symbol, "us")

        expiries: list[str] = []
        expiry_error = None
        try:
            expiries = list_expiries_us(symbol)
        except Exception as exc:
            expiry_error = str(exc)

        if expiry_error:
            st.error(expiry_error)
            st.stop()

        expiry = st.selectbox("Expiry", options=expiries, index=0)

    else:
        # India ── driven by CSV files in data/
        india_files = list_india_csv_files(DATA_DIR)
        if not india_files:
            st.error(
                f"No NSE CSV files found in '{DATA_DIR}/'. "
                "Expected filenames like: option-chain-ED-NIFTY-05-May-2026.csv"
            )
            st.stop()

        india_symbols = sorted(india_files.keys())
        symbol = st.selectbox("Symbol", options=india_symbols, index=0)
        option_type = st.selectbox("Option type", options=["call", "put"], index=0)

        # Exercise-style notice
        american = is_american_style(symbol, "india")

        # Expiry derived from available CSV filenames for this symbol
        csv_files_for_symbol = india_files[symbol]
        expiry_options = {
            parse_expiry_from_filename(f.name): f
            for f in csv_files_for_symbol
        }
        expiry_labels = sorted(expiry_options.keys())
        expiry = st.selectbox("Expiry", options=expiry_labels, index=0)
        selected_csv = expiry_options[expiry]

    st.divider()

    # ── Risk-free rate ───────────────────────────────────────────────────────
    st.subheader("Risk-Free Rate")

    if market_mode == "us":
        rate_mode = st.radio("Source", options=["Auto (^IRX T-bill)", "Manual"], index=0)
        manual_rate = st.number_input(
            "Manual rate (%)", value=4.50, min_value=0.0, max_value=25.0, step=0.05,
            disabled=(rate_mode != "Manual"),
        )
    else:
        rate_mode = st.radio(
            "Source",
            options=[f"Auto (RBI Repo ≈ {RBI_REPO_RATE:.2%})", "Manual"],
            index=0,
        )
        manual_rate = st.number_input(
            "Manual rate (%)", value=RBI_REPO_RATE * 100, min_value=0.0, max_value=25.0, step=0.05,
            disabled=("Manual" not in rate_mode),
        )
        if "Auto" in rate_mode:
            st.caption(
                "RBI repo rate used as proxy. yfinance does not carry Indian T-bill "
                "yields. Update `RBI_REPO_RATE` in options_logic.py when RBI changes policy."
            )

    st.divider()

    # ── Volatility ───────────────────────────────────────────────────────────
    st.subheader("Volatility")
    vol_mode = st.radio("Source", options=["Historical (1y)", "Manual"], index=0, key="vol_radio")
    manual_vol = st.number_input(
        "Manual volatility (%)", value=25.0, min_value=1.0, max_value=300.0, step=1.0,
        disabled=(vol_mode != "Manual"),
    )
    if market_mode == "india" and vol_mode == "Historical (1y)":
        st.caption(
            "Historical vol fetched from yfinance using the NSE/BSE ticker "
            f"({INDIA_YF_MAP.get(symbol, '?')}). "
            "For recently-listed stocks, a shorter window may be used automatically."
        )

    st.divider()

    # ── Filters ──────────────────────────────────────────────────────────────
    st.subheader("Filters")
    strike_window = st.slider("Strike window around spot (%)", 5, 80, 30, 5)
    min_open_interest = st.number_input("Min open interest", 0, 10000, 0, 1)

    st.divider()
    run = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Wait for run
# ---------------------------------------------------------------------------
if not run:
    st.info("Configure parameters in the sidebar and click **▶ Run Analysis**.")

    # Show a helpful reference table while idle
    with st.expander("📋 Available symbols reference"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**US — European style (index options)**")
            st.table(pd.DataFrame({"Symbol": list(US_EUROPEAN_SYMBOLS)}))
            st.markdown("**US — American style (stock options)**")
            st.table(pd.DataFrame({"Symbol": list(US_AMERICAN_SYMBOLS)}))
        with col2:
            st.markdown("**India — European style (NSE index options)**")
            st.table(pd.DataFrame({"Symbol": list(INDIA_EUROPEAN_SYMBOLS)}))
            st.markdown("**India — American style (NSE stock options)**")
            st.table(pd.DataFrame({"Symbol": list(INDIA_AMERICAN_SYMBOLS)}))
    st.stop()

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
option_type_lit = "call" if option_type == "call" else "put"
vol_warning: str | None = None

try:
    # ── Fetch chain & spot ───────────────────────────────────────────────────
    if market_mode == "us":
        chain_df, spot = fetch_option_chain_us(symbol, expiry, option_type_lit)
        ticker = yf.Ticker(symbol)
    else:
        chain_df, spot = fetch_option_chain_india(selected_csv, option_type_lit, symbol)

    # ── Risk-free rate ───────────────────────────────────────────────────────
    if market_mode == "us":
        if "Auto" in rate_mode:
            rate = get_risk_free_rate_auto()
            rate_source = "13-week US T-bill (^IRX)"
        else:
            rate = manual_rate / 100.0
            rate_source = "Manual input"
    else:
        if "Auto" in rate_mode:
            rate = get_india_risk_free_rate()
            rate_source = f"RBI repo rate proxy ({rate:.2%})"
        else:
            rate = manual_rate / 100.0
            rate_source = "Manual input"

    # ── Volatility ───────────────────────────────────────────────────────────
    if vol_mode == "Historical (1y)":
        if market_mode == "us":
            sigma = estimate_historical_volatility(ticker, period="1y")
            vol_source = "1y historical realized volatility (yfinance)"
        else:
            sigma, vol_warning = estimate_historical_volatility_india(symbol, period="1y")
            vol_source = f"1y historical realized volatility ({INDIA_YF_MAP.get(symbol, symbol)})"
    else:
        sigma = manual_vol / 100.0
        vol_source = "Manual input"

    # ── Strike filter ────────────────────────────────────────────────────────
    lower = spot * (1.0 - strike_window / 100.0)
    upper = spot * (1.0 + strike_window / 100.0)

    # ── Dividend yield (India only) ──────────────────────────────────────────
    div_yield = INDIA_DIV_YIELD.get(symbol.upper(), 0.0) if market_mode == "india" else 0.0

    # ── Build context ────────────────────────────────────────────────────────
    context = MarketContext(
        symbol=symbol,
        spot=spot,
        expiry=expiry,
        rate=rate,
        sigma=sigma,
        option_type=option_type_lit,
        market=market_mode,
        is_american_style=is_american_style(symbol, market_mode),
        div_yield=div_yield,
    )

    # ── Pricing comparison ───────────────────────────────────────────────────
    result = build_pricing_comparison(
        chain_df=chain_df,
        context=context,
        strike_min=lower,
        strike_max=upper,
        min_open_interest=min_open_interest,
    )
    summary = summarize_errors(result)
    explanation = generate_explanation(result, summary, context)

except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Warnings & notices
# ---------------------------------------------------------------------------
if vol_warning:
    st.warning(f"⚠️ {vol_warning}")

exercise_note = get_exercise_style_note(symbol, market_mode)
if exercise_note:
    st.warning(f"⚠️ {exercise_note}")

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
st.subheader("Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Spot", f"{spot:,.2f}")
col2.metric("Contracts", int(summary["contracts_analyzed"]))
col3.metric("MAE", f"{summary['mae']:.4f}")
col4.metric("MAPE", f"{summary['mape_percent']:.2f}%")
col5.metric("Max Abs Error", f"{summary['max_abs_error']:.4f}")

st.markdown(
    f"**Model inputs:** σ = {sigma:.2%} ({vol_source}) &nbsp;|&nbsp; "
    f"r = {rate:.2%} ({rate_source})"
    + (f" &nbsp;|&nbsp; q = {div_yield:.2%} (div yield)" if div_yield > 0 else "")
)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📈 Theoretical vs Market Price by Strike")
price_df = result[["strike", "marketPrice", "theoreticalPrice"]].melt(
    id_vars="strike", var_name="Series", value_name="Price"
)
price_chart = (
    alt.Chart(price_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("strike:Q", title="Strike"),
        y=alt.Y("Price:Q", title="Option Price"),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(
                domain=["marketPrice", "theoreticalPrice"],
                range=["#f43f5e", "#0ea5e9"],
            ),
        ),
        tooltip=["strike:Q", "Series:N", alt.Tooltip("Price:Q", format=".4f")],
    )
    .properties(height=350)
)
st.altair_chart(price_chart, use_container_width=True)

st.subheader("📊 Pricing Error by Strike (Theoretical − Market)")
error_chart = (
    alt.Chart(result)
    .mark_bar()
    .encode(
        x=alt.X("strike:Q", title="Strike"),
        y=alt.Y("signedError:Q", title="Theoretical − Market"),
        color=alt.condition(
            alt.datum.signedError > 0,
            alt.value("#0ea5e9"),
            alt.value("#f43f5e"),
        ),
        tooltip=[
            "strike:Q",
            alt.Tooltip("marketPrice:Q", format=".4f"),
            alt.Tooltip("theoreticalPrice:Q", format=".4f"),
            alt.Tooltip("signedError:Q", format=".4f"),
            alt.Tooltip("pctError:Q", format=".2f"),
        ],
    )
    .properties(height=300)
)
st.altair_chart(error_chart, use_container_width=True)

st.subheader("📉 Error Distribution (% Error)")
hist_chart = (
    alt.Chart(result)
    .mark_bar(color="#6366f1")
    .encode(
        x=alt.X("pctError:Q", bin=alt.Bin(maxbins=40), title="Percent Error (%)"),
        y=alt.Y("count():Q", title="Contracts"),
        tooltip=["count():Q"],
    )
    .properties(height=250)
)
st.altair_chart(hist_chart, use_container_width=True)

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔍 Interpretation")
st.write(explanation)

# ---------------------------------------------------------------------------
# Detailed table
# ---------------------------------------------------------------------------
st.subheader("📋 Detailed Table")
display_cols = [
    "strike",
    "marketPrice",
    "theoreticalPrice",
    "signedError",
    "absError",
    "pctError",
    "impliedVolatility",
    "bid",
    "ask",
    "lastPrice",
    "openInterest",
    "volume",
]
# contractSymbol only exists in yfinance data
if "contractSymbol" in result.columns:
    display_cols = ["contractSymbol"] + display_cols

available_cols = [c for c in display_cols if c in result.columns]
st.dataframe(
    result[available_cols].sort_values("strike"),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
csv_bytes = result.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download analysis CSV",
    data=csv_bytes,
    file_name=f"{symbol}_{expiry}_{option_type}_bs_comparison.csv",
    mime="text/csv",
)