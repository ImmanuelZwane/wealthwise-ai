import yfinance as yf

class MarketAgent:
    """Fetches real-time and historical market data."""

    def __init__(self):
        pass

    def get_current_price(self, symbol):
        """Get the latest price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            # Try regularMarketPrice first, fallback to previousClose
            price = ticker.info.get('regularMarketPrice', ticker.info.get('previousClose'))
            return price
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def get_historical_prices(self, symbol, period="1mo"):
        """Get historical closing prices for a given period."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            print(f"Error fetching history for {symbol}: {e}")
            return None