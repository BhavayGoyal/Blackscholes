# Black-Scholes Pricing vs Market Prices

A Streamlit application that compares theoretical Black-Scholes option prices against real market data — supporting both **US markets** (via Yahoo Finance) and **Indian markets** (via NSE CSV files).

---

## Overview

This dashboard lets you:

- Access the live app: [blackscholes-grp-8.streamlit.app](https://blackscholes-grp-8.streamlit.app/)
- Price options using the **Black-Scholes-Merton formula** (with optional continuous dividend yield)
- Fetch **live US option chains** from Yahoo Finance
- Load **Indian NSE option chains** from locally stored CSV files
- Compare theoretical prices against observed market prices
- Quantify and visualise **pricing errors** across strikes
- Understand **why** errors occur through an auto-generated interpretation

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/BhavayGoyal/Blackscholes
cd <path-to-repo>
pip install -r requirements.txt
```

### Requirements
```
streamlit
yfinance
pandas
numpy
altair
```

### Running

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project Structure

```
.
├── app.py              # Streamlit UI — sidebar, charts, table, download
├── options_logic.py    # All pricing logic, data fetching, CSV parsing
├── data/               # NSE CSV files (Indian market option chains)
│   ├── option-chain-ED-NIFTY-05-May-2026.csv
│   ├── option-chain-ED-BANKNIFTY-28-Apr-2026.csv
│   ├── option-chain-ED-DMART-26-May-2026.csv
│   └── ...
└── README.md
```

> **Important:** All NSE CSV files must be placed in the `data/` folder and follow the exact naming convention: `option-chain-ED-<SYMBOL>-<DD>-<Mon>-<YYYY>.csv`. The app scans this folder automatically — no manual configuration needed.

---

## Supported Symbols

### US Market (Yahoo Finance)

| Symbol | Style | Notes |
|--------|-------|-------|
| `^SPX` | European | S&P 500 index — BS fully appropriate |
| `^NDX` | European | NASDAQ-100 index |
| `^RUT` | European | Russell 2000 index |
| `^VIX` | European | Volatility index |
| `AAPL` | American | Early exercise applies — BS underprices puts |
| `AMZN` | American | Early exercise applies |
| `MSFT` | American | Early exercise applies |
| `TSLA` | American | Early exercise applies |
| `NVDA` | American | Early exercise applies |

### Indian Market (NSE CSV)

| Symbol | Style | Notes |
|--------|-------|-------|
| `NIFTY` | European | Cash-settled — BS fully appropriate |
| `BANKNIFTY` | European | Cash-settled |
| `DMART` | American | NSE stock option — early exercise applies |
| `HYUNDAI` | American | Listed Oct 2024 — limited vol history |
| `RELIANCE` | American | NSE stock option |
| `HDFCBANK` | American | NSE stock option |
| `INFY` | American | NSE stock option |

The app warns you in the sidebar and on the main page whenever you select an American-style symbol.

---

## Configuration

### Data Directory
Change `DATA_DIR` at the top of `app.py` if your CSVs are stored elsewhere:
```python
DATA_DIR = Path("data")   # default
DATA_DIR = Path("/path/to/your/csvs")  # custom
```

### RBI Repo Rate
The Indian risk-free rate is a hardcoded constant in `options_logic.py` (see note below). Update it when RBI changes policy:
```python
RBI_REPO_RATE: float = 0.0625  # 6.25% as of Apr 2026
```

---

## Features

### 1. Market Toggle
Switch between **US Market** and **Indian Market (NSE)** in the sidebar. The entire input flow — symbols, expiries, rate source, vol source — adapts accordingly.

### 2. Risk-Free Rate

| Market | Auto Source | Notes |
|--------|------------|-------|
| US | `^IRX` (13-week T-bill via yfinance) | Fetched live |
| India | RBI repo rate (hardcoded) | yfinance has no Indian T-bill yield equivalent |

Both markets support a **Manual** override for the rate.

### 3. Volatility

| Market | Source | How |
|--------|--------|-----|
| US | 1y historical (yfinance) | Daily returns → annualised std dev × √252 |
| India | 1y historical (yfinance) | Same formula using `^NSEI`, `RELIANCE.NS` etc. |

For recently-listed Indian stocks (e.g. HYUNDAI, listed Oct 2024), the app automatically falls back to 6-month then 3-month windows and surfaces a warning. Manual vol input is always available.

### 4. Dividend Yield (India)
For Indian symbols a continuous dividend yield `q` is applied in the BSM formula. This makes the call underpricing from dividends quantifiable in the error output.

```
d₁ = [ln(S/K) + (r − q + σ²/2) × T] / (σ√T)
```

Approximate yields used:

| Symbol | q |
|--------|---|
| NIFTY | 1.3% |
| BANKNIFTY | 1.0% |
| INFY | 2.5% |
| HDFCBANK | 1.0% |
| Others | 0.4–0.5% |

### 5. Strike Filter
A percentage window around the current spot filters strikes to the liquid, near-money range. A minimum open interest filter removes illiquid contracts.

### 6. Charts

- **Theoretical vs Market Price** — line chart overlaying both series across strikes
- **Pricing Error by Strike** — bar chart of signed error (blue = BS over, red = BS under)
- **Error Distribution** — histogram of % error across all contracts

### 7. Auto-generated Interpretation
After each run, a plain-English explanation is generated covering:
- Number of contracts, sigma, and rate used
- ITM vs OTM error pattern and its likely cause
- American-style early-exercise impact (if applicable)
- Dividend yield adjustment applied
- Long-dated expiry caveat (if T ≥ 1 year)
- India-specific rate proxy note (if Indian market)
- Universal BS model limitation reminder

### 8. Detailed Table + CSV Download
Full per-strike table showing market price, theoretical price, signed/absolute/% error, IV, bid, ask, OI, and volume. Downloadable as CSV.

---

## Mathematical Foundation

### Black-Scholes-Merton Formula

**Call:**
```
C = S × e^(−qT) × N(d₁) − K × e^(−rT) × N(d₂)
```

**Put:**
```
P = K × e^(−rT) × N(−d₂) − S × e^(−qT) × N(−d₁)
```

**Where:**
```
d₁ = [ln(S/K) + (r − q + σ²/2) × T] / (σ√T)
d₂ = d₁ − σ√T
q  = continuous dividend yield (0 for US, symbol-specific for India)
```

### Historical Volatility
```
σ = std(daily log returns) × √252
```
Computed from 1 year of daily closing prices via yfinance. Minimum 60 trading days required.

### Error Metrics

| Metric | Formula |
|--------|---------|
| MAE | mean(|theoretical − market|) |
| RMSE | √mean((theoretical − market)²) |
| MAPE | mean(|theoretical − market| / market) × 100 |
| Max Abs Error | max(|theoretical − market|) |

---

## NSE CSV Format

NSE option chain CSVs use a wide format: calls on the left, strike in the middle, puts on the right. The parser handles:
- The `CALLS,,PUTS` banner row (skipped)
- Leading and trailing empty columns
- Comma-formatted numbers (e.g. `"1,400.50"`)
- Missing values represented as `-`

Expected column order per data row:
```
[empty] | c_oi | c_chng_oi | c_volume | c_iv | c_ltp | c_chng | c_bid_qty | c_bid | c_ask | c_ask_qty | STRIKE | p_bid_qty | p_bid | p_ask | p_ask_qty | p_chng | p_ltp | p_iv | p_volume | p_chng_oi | p_oi | [empty]
```

---

## Known Limitations & Pricing Error Sources

| Source | Impact | Affects |
|--------|--------|---------|
| Volatility smile/skew | High | OTM options — BS uses flat vol, market prices a skew |
| American exercise premium | Medium | All US stocks, NSE stock options |
| Dividends (partially modelled) | Medium | Long-dated options, high-yield stocks |
| Constant vol assumption | Medium | All symbols — fat tails ignored |
| Single risk-free rate | Low | Long-dated expiries |
| Rate proxy for India | Low | All Indian symbols — RBI repo ≠ market OIS rate |
| Limited history (new listings) | Medium | HYUNDAI and other recent IPOs |
| Transaction costs / liquidity | Medium | Deep OTM/ITM strikes with wide spreads |

---

## Disclaimer

This application is for **educational and research purposes only**. It is not financial advice. The Black-Scholes model assumes European exercise, constant volatility, log-normal returns, no transaction costs, and continuous hedging — assumptions real markets routinely violate.
