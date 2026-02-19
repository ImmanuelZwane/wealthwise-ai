import numpy as np

class RiskAgent:
    """Monitors drawdown and volatility for each holding."""

    def __init__(self, max_drawdown=15, max_volatility=20):
        self.max_drawdown = max_drawdown      # percent
        self.max_volatility = max_volatility  # annualized percent

    def check_drawdown(self, market_agent, symbols):
        """Calculate current drawdown from 3-month peak."""
        risks = []
        for sym in symbols:
            hist = market_agent.get_historical_prices(sym, period="3mo")
            if hist is not None and not hist.empty:
                peak = hist['Close'].max()
                current = hist['Close'].iloc[-1]
                drawdown = (peak - current) / peak * 100
                risks.append({'Symbol': sym, 'Drawdown (%)': round(drawdown, 2)})
            else:
                risks.append({'Symbol': sym, 'Drawdown (%)': None})
        return risks

    def check_volatility(self, market_agent, symbols):
        """Calculate annualized volatility from 1-month daily returns."""
        risks = []
        for sym in symbols:
            hist = market_agent.get_historical_prices(sym, period="1mo")
            if hist is not None and len(hist) > 1:
                returns = hist['Close'].pct_change().dropna()
                daily_vol = returns.std()
                annual_vol = daily_vol * np.sqrt(252) * 100
                risks.append({'Symbol': sym, 'Volatility (ann. %)': round(annual_vol, 2)})
            else:
                risks.append({'Symbol': sym, 'Volatility (ann. %)': None})
        return risks