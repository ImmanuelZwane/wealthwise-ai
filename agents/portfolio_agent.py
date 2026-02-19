import pandas as pd

class PortfolioAgent:
    """Manages holdings, calculates allocations, and suggests rebalancing trades."""

    def __init__(self, holdings_df, target_allocation):
        """
        holdings_df: DataFrame with columns ['Symbol', 'Shares', 'CostBasis']
        target_allocation: dict {symbol: target_percentage}
        """
        self.holdings = holdings_df.copy()
        self.target_allocation = target_allocation
        self.total_value = 0

    def update_prices(self, market_agent):
        """Add current price and market value to holdings."""
        prices = []
        for sym in self.holdings['Symbol']:
            price = market_agent.get_current_price(sym)
            prices.append(price if price else 0)
        self.holdings['CurrentPrice'] = prices
        self.holdings['MarketValue'] = self.holdings['Shares'] * self.holdings['CurrentPrice']
        self.total_value = self.holdings['MarketValue'].sum()

    def get_allocation(self):
        """Return DataFrame with current allocation percentages."""
        alloc = self.holdings.copy()
        alloc['Allocation'] = alloc['MarketValue'] / self.total_value * 100
        return alloc[['Symbol', 'Allocation']]

    def rebalance_suggestions(self):
        """
        Suggest trades to reach target allocation.
        Returns list of dicts with Symbol, Action, Shares, Value.
        """
        current = self.get_allocation().set_index('Symbol')['Allocation']
        suggestions = []
        for sym, target_pct in self.target_allocation.items():
            if sym in current.index:
                current_pct = current[sym]
                diff = target_pct - current_pct
                # approximate value to trade
                target_value = self.total_value * target_pct / 100
                current_value = self.holdings[self.holdings['Symbol'] == sym]['MarketValue'].values[0]
                trade_value = target_value - current_value
                if abs(trade_value) > 10:  # ignore tiny differences
                    action = "BUY" if trade_value > 0 else "SELL"
                    price = self.holdings[self.holdings['Symbol'] == sym]['CurrentPrice'].values[0]
                    shares = abs(trade_value) / price
                    suggestions.append({
                        'Symbol': sym,
                        'Action': action,
                        'Shares': round(shares, 2),
                        'Value': round(abs(trade_value), 2)
                    })
        return suggestions