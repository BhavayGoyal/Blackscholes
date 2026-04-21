import yfinance as yf

ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="1y")

print(hist)
print(f"\nRows: {len(hist)}")
print(f"\nLast close: {hist['Close'].iloc[-1]:.2f}")