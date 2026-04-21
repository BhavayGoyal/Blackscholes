from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

from options_logic import (
    EUROPEAN_STYLE_SYMBOLS,
    MarketContext,
    build_pricing_comparison,
    estimate_historical_volatility,
    fetch_option_chain,
    generate_explanation,
    get_risk_free_rate_auto,
    list_expiries,
    summarize_errors,
)


st.set_page_config(page_title="Black-Scholes vs Market", layout="wide")
st.title("Black-Scholes Pricing vs Market Prices")
st.caption("Compare theoretical option prices with live Yahoo Finance options-chain data.")


with st.sidebar:
    st.header("Inputs")
    symbol = st.selectbox("Ticker symbol", options=list(EUROPEAN_STYLE_SYMBOLS), index=0)
    option_type = st.selectbox("Option type", options=["call", "put"], index=0)

    expiries: list[str] = []
    expiry_error = None
    if symbol:
        try:
            expiries = list_expiries(symbol)
        except Exception as exc:
            expiry_error = str(exc)

    if expiry_error:
        st.error(expiry_error)
        st.stop()

    expiry = st.selectbox("Expiry", options=expiries, index=0)

    rate_mode = st.radio("Risk-free rate", options=["Auto (^IRX)", "Manual"], index=0)
    manual_rate = st.number_input(
        "Manual rate (%)",
        value=4.50,
        min_value=0.0,
        max_value=25.0,
        step=0.05,
        disabled=(rate_mode != "Manual"),
    )

    vol_mode = st.radio("Volatility", options=["Historical (1y)", "Manual"], index=0)
    manual_vol = st.number_input(
        "Manual volatility (%)",
        value=25.0,
        min_value=1.0,
        max_value=300.0,
        step=1.0,
        disabled=(vol_mode != "Manual"),
    )

    strike_window = st.slider("Strike window around spot (%)", 5, 80, 30, 5)
    min_open_interest = st.number_input("Minimum open interest", 0, 10000, 0, 1)
    run = st.button("Run analysis", type="primary")


if not run:
    st.info("Choose parameters in the sidebar and click **Run analysis**.")
    st.stop()

option_type_lit = "call" if option_type == "call" else "put"

try:
    chain_df, spot = fetch_option_chain(symbol=symbol, expiry=expiry, option_type=option_type_lit)
    ticker = yf.Ticker(symbol)

    if rate_mode == "Auto (^IRX)":
        rate = get_risk_free_rate_auto()
        rate_source = "13-week T-bill proxy (^IRX)"
    else:
        rate = manual_rate / 100.0
        rate_source = "Manual input"

    if vol_mode == "Historical (1y)":
        sigma = estimate_historical_volatility(ticker, period="1y")
        vol_source = "1y historical realized volatility"
    else:
        sigma = manual_vol / 100.0
        vol_source = "Manual input"

    lower = spot * (1.0 - strike_window / 100.0)
    upper = spot * (1.0 + strike_window / 100.0)

    context = MarketContext(
        symbol=symbol,
        spot=spot,
        expiry=expiry,
        rate=rate,
        sigma=sigma,
        option_type=option_type_lit,
    )
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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Spot", f"{spot:.2f}")
col2.metric("MAE", f"{summary['mae']:.4f}")
col3.metric("MAPE", f"{summary['mape_percent']:.2f}%")
col4.metric("Max Abs Error", f"{summary['max_abs_error']:.4f}")

st.markdown(
    f"**Model inputs:** r={rate:.2%} ({rate_source}), sigma={sigma:.2%} ({vol_source}), "
    f"contracts={int(summary['contracts_analyzed'])}"
)

st.subheader("Theoretical vs Market Price by Strike")
price_df = result[["strike", "marketPrice", "theoreticalPrice"]].melt(
    id_vars="strike", var_name="Series", value_name="Price"
)
price_chart = (
    alt.Chart(price_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("strike:Q", title="Strike"),
        y=alt.Y("Price:Q", title="Option Price"),
        color=alt.Color("Series:N", title="Series"),
        tooltip=["strike:Q", "Series:N", "Price:Q"],
    )
    .properties(height=350)
)
st.altair_chart(price_chart, use_container_width=True)

st.subheader("Pricing Error by Strike")
error_chart = (
    alt.Chart(result)
    .mark_bar()
    .encode(
        x=alt.X("strike:Q", title="Strike"),
        y=alt.Y("signedError:Q", title="Theoretical - Market"),
        color=alt.condition(
            alt.datum.signedError > 0,
            alt.value("#0ea5e9"),
            alt.value("#f43f5e"),
        ),
        tooltip=["strike:Q", "marketPrice:Q", "theoreticalPrice:Q", "signedError:Q", "pctError:Q"],
    )
    .properties(height=300)
)
st.altair_chart(error_chart, use_container_width=True)

st.subheader("Error Distribution")
hist_chart = (
    alt.Chart(result)
    .mark_bar()
    .encode(
        x=alt.X("pctError:Q", bin=alt.Bin(maxbins=40), title="Percent Error"),
        y=alt.Y("count():Q", title="Contracts"),
        tooltip=["count():Q"],
    )
    .properties(height=250)
)
st.altair_chart(hist_chart, use_container_width=True)

st.subheader("Interpretation")
st.write(explanation)

st.subheader("Detailed Table")
display_cols = [
    "contractSymbol",
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
available_cols = [c for c in display_cols if c in result.columns]
st.dataframe(
    result[available_cols].sort_values("strike"),
    use_container_width=True,
    hide_index=True,
)

csv = result.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download analysis CSV",
    data=csv,
    file_name=f"{symbol}_{expiry}_{option_type}_black_scholes_comparison.csv",
    mime="text/csv",
)