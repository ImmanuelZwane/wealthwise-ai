import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import yfinance as yf
import os
from dotenv import load_dotenv

from agents.market_agent import MarketAgent
from agents.portfolio_agent import PortfolioAgent
from agents.risk_agent import RiskAgent
from agents.explanation_agent import ExplanationAgent

# ---------- Helper Functions ----------
def get_exchange_rate(target_currency, base_currency='USD'):
    """Fetch live exchange rate from Yahoo Finance."""
    if target_currency == base_currency:
        return 1.0
    try:
        # Yahoo Finance symbol for forex, e.g., EURUSD=X
        pair = f"{target_currency}{base_currency}=X"
        ticker = yf.Ticker(pair)
        rate = ticker.info.get('regularMarketPrice', ticker.info.get('previousClose'))
        if rate is None:
            # Try inverse pair
            pair = f"{base_currency}{target_currency}=X"
            ticker = yf.Ticker(pair)
            inverse = ticker.info.get('regularMarketPrice', ticker.info.get('previousClose'))
            if inverse:
                rate = 1 / inverse
        return rate if rate else None
    except Exception as e:
        st.warning(f"Could not fetch exchange rate for {target_currency}: {e}")
        return None

def format_currency(amount, currency_code, symbol=True):
    """Format amount with currency symbol and thousand separators."""
    if amount is None:
        return "N/A"
    # Currency symbols mapping
    symbols = {
        'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 
        'CHF': 'CHF', 'CAD': 'C$', 'ZAR': 'R'
    }
    sym = symbols.get(currency_code, currency_code) if symbol else currency_code
    # Handle different decimal rules (JPY typically has 0 decimals)
    if currency_code == 'JPY':
        return f"{sym}{amount:,.0f}"
    else:
        return f"{sym}{amount:,.2f}"

def estimate_rmd(balance, age):
    """Simplified RMD using IRS Uniform Lifetime table factors (approx)."""
    rmd_factors = {
        70: 27.4, 71: 26.5, 72: 25.6, 73: 24.7, 74: 23.8,
        75: 22.9, 76: 22.0, 77: 21.2, 78: 20.3, 79: 19.5,
        80: 18.7, 81: 17.9, 82: 17.1, 83: 16.3, 84: 15.5,
        85: 14.8, 86: 14.1, 87: 13.4, 88: 12.7, 89: 12.0,
        90: 11.4, 91: 10.8, 92: 10.2, 93: 9.6, 94: 9.1, 95: 8.6
    }
    factor = rmd_factors.get(age, 25.0)
    return balance / factor

def monte_carlo_simulation(initial_value, withdrawal, years, returns, volatility, n_sims=1000):
    dt = 1/12
    n_steps = int(years * 12)
    successes = 0
    for _ in range(n_sims):
        value = initial_value
        for _ in range(n_steps):
            monthly_return = np.random.normal(returns/12, volatility/np.sqrt(12))
            value = value * (1 + monthly_return) - withdrawal/12
            if value <= 0:
                break
        if value > 0:
            successes += 1
    return successes / n_sims

# ---------- Page Configuration ----------
st.set_page_config(page_title="WealthWise AI", layout="wide")
st.title("🛡️ WealthWise AI – Retirement Portfolio Assistant")

# ---------- Sidebar ----------
st.sidebar.header("Configuration")

# Load environment variables
load_dotenv(dotenv_path='api.env')

# Gemini API key input (optional)
api_key = st.sidebar.text_input("Gemini API Key (optional)", type="password")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# Currency selection
currency_options = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'ZAR']
selected_currency = st.sidebar.selectbox("Display Currency", currency_options, index=0)

# Refresh data button
if st.sidebar.button("🔄 Refresh Market Data"):
    st.cache_resource.clear()
    st.rerun()

# Initialize agents (cached)
@st.cache_resource
def init_agents():
    return {
        'market': MarketAgent(),
        'explanation': ExplanationAgent()
    }

agents = init_agents()

# Portfolio input
st.sidebar.header("Your Portfolio")
uploaded_file = st.sidebar.file_uploader("Upload holdings CSV", type=['csv'])

if uploaded_file is not None:
    holdings = pd.read_csv(uploaded_file)
else:
    # Sample data (all in USD)
    holdings = pd.DataFrame({
        'Symbol': ['BND', 'VTI', 'SCHD', 'SGOV'],
        'Shares': [100, 50, 40, 200],
        'CostBasis': [75, 180, 60, 100]
    })

symbols = holdings['Symbol'].tolist()

# Target allocation sliders
st.sidebar.subheader("Target Allocation (%)")
target = {}
cols = st.sidebar.columns(2)
for i, sym in enumerate(holdings['Symbol']):
    with cols[i % 2]:
        target[sym] = st.number_input(f"{sym}", min_value=0, max_value=100, value=25, step=5)

total_target = sum(target.values())
if total_target != 100:
    st.sidebar.warning(f"Target sums to {total_target}%. Please adjust to 100%.")
    st.stop()

# Retirement planning inputs
st.sidebar.header("Retirement Settings")
age = st.sidebar.number_input("Your Age", min_value=50, max_value=95, value=65, step=1)
annual_withdrawal = st.sidebar.number_input("Annual Withdrawal ($)", min_value=0, value=20000, step=1000)
life_expectancy = st.sidebar.number_input("Life Expectancy (age)", min_value=age+1, max_value=100, value=90, step=1)

# Create portfolio agent and update prices (original prices are in USD)
portfolio_agent = PortfolioAgent(holdings.copy(), target)
with st.spinner("Fetching latest market data..."):
    portfolio_agent.update_prices(agents['market'])

# ---------- Currency Conversion ----------
# Fetch exchange rate from USD to selected currency
exchange_rate = get_exchange_rate(selected_currency, 'USD')
if exchange_rate is None:
    st.sidebar.warning(f"Using USD values – exchange rate for {selected_currency} unavailable.")
    exchange_rate = 1.0
    effective_currency = 'USD'
else:
    effective_currency = selected_currency

# Function to convert USD amounts to selected currency
def convert(usd_amount):
    return usd_amount * exchange_rate if usd_amount is not None else None

# ---------- Main Dashboard ----------
# Top metrics (converted)
st.subheader("📈 Portfolio Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_value_converted = convert(portfolio_agent.total_value)
    st.metric("Total Value", format_currency(total_value_converted, effective_currency))
with col2:
    st.metric("Holdings", len(holdings))
with col3:
    alloc = portfolio_agent.get_allocation()
    deviation = abs(alloc['Allocation'] - alloc['Symbol'].map(target)).mean()
    st.metric("Avg Deviation", f"{deviation:.1f}%")
with col4:
    st.metric("Last Updated", pd.Timestamp.now().strftime("%H:%M"))

# Two-column layout
left_col, right_col = st.columns([0.4, 0.6])

with left_col:
    st.subheader(" Current Holdings")
    df_display = portfolio_agent.holdings[['Symbol', 'Shares', 'CurrentPrice', 'MarketValue']].copy()
    # Convert USD prices and values
    df_display['CurrentPrice'] = df_display['CurrentPrice'].apply(lambda x: convert(x))
    df_display['MarketValue'] = df_display['MarketValue'].apply(lambda x: convert(x))
    df_display['CurrentPrice'] = df_display['CurrentPrice'].apply(lambda x: format_currency(x, effective_currency))
    df_display['MarketValue'] = df_display['MarketValue'].apply(lambda x: format_currency(x, effective_currency))
    st.dataframe(df_display, use_container_width=True)

    # Allocation pie chart (unchanged)
    fig = px.pie(alloc, values='Allocation', names='Symbol',
                 title='Current Allocation', hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    suggestions = portfolio_agent.rebalance_suggestions()
    if suggestions:
        st.subheader("⚖️ Rebalancing Suggestions")
        df_sugg = pd.DataFrame(suggestions)
        # Convert suggestion values
        df_sugg['Value'] = df_sugg['Value'].apply(lambda x: convert(x))
        df_sugg['Value'] = df_sugg['Value'].apply(lambda x: format_currency(x, effective_currency))

        def color_action(val):
            color = 'green' if val == 'BUY' else 'red' if val == 'SELL' else 'black'
            return f'color: {color}; font-weight: bold'

        styled_sugg = df_sugg.style.applymap(color_action, subset=['Action'])
        st.dataframe(styled_sugg, use_container_width=True)

        csv = df_sugg.to_csv(index=False)
        st.download_button(" Download Suggestions", csv, "trades.csv", "text/csv")

        # Risk indicators
        st.subheader("⚠️ Risk Indicators")
        with st.expander("ℹ️ What do these numbers mean?"):
            st.write("""
            - **Drawdown (%)**: How far the price has fallen from its recent peak (lower is better).
            - **Volatility (ann. %)**: Annual price fluctuation; lower means more stable.
            """)

        risk_agent = RiskAgent()
        drawdowns = risk_agent.check_drawdown(agents['market'], symbols)
        volatilities = risk_agent.check_volatility(agents['market'], symbols)
        risk_df = pd.DataFrame(drawdowns)
        if volatilities:
            risk_df = risk_df.merge(pd.DataFrame(volatilities), on='Symbol')

        risk_df['Drawdown (%)'] = risk_df['Drawdown (%)'].fillna(0).round(2)
        risk_df['Volatility (ann. %)'] = risk_df['Volatility (ann. %)'].fillna(0).round(2)

        def color_risk(val):
            if val > 15:
                return 'background-color: #ffcccc'
            elif val > 10:
                return 'background-color: #fff3cd'
            return ''

        styled_risk = risk_df.style.applymap(color_risk, subset=['Drawdown (%)', 'Volatility (ann. %)'])
        st.dataframe(styled_risk, use_container_width=True)

        # AI Explanation
        st.subheader("💬 AI Advisor")
        if st.button(" Explain Suggestions", type="primary"):
            with st.spinner("Generating explanation..."):
                summary = f"Total portfolio value: {format_currency(total_value_converted, effective_currency)}\n"
                summary += f"Your age: {age}, Annual withdrawal: {format_currency(convert(annual_withdrawal), effective_currency)}\n"
                summary += "Current allocation:\n"
                for _, row in alloc.iterrows():
                    summary += f"  {row['Symbol']}: {row['Allocation']:.1f}%\n"
                market_note = "Current market conditions are stable."
                explanation = agents['explanation'].explain_rebalance(
                    summary, df_sugg.to_string(), market_note
                )
                st.info(explanation)
    else:
        st.success("✅ Portfolio is perfectly balanced! No trades needed.")
        st.balloons()

# ---------- Retirement & Tax Features ----------
st.markdown("---")
st.header(" Retirement & Tax Insights")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader(" Required Minimum Distribution (RMD)")
    if age >= 70:
        rmd_usd = estimate_rmd(portfolio_agent.total_value, age)
        rmd_converted = convert(rmd_usd)
        st.metric("Estimated RMD for this year", format_currency(rmd_converted, effective_currency))
        st.caption("Based on IRS Uniform Lifetime Table (simplified).")
    else:
        st.info("RMDs start at age 70. You're not yet required to withdraw.")

    # Tax-loss harvesting suggestions (values in USD, then convert)
    st.subheader("Tax-Loss Harvesting Opportunities")
    tax_loss = []
    for _, row in portfolio_agent.holdings.iterrows():
        if row['CurrentPrice'] < row['CostBasis']:
            loss_per_share = row['CostBasis'] - row['CurrentPrice']
            total_loss_usd = loss_per_share * row['Shares']
            total_loss_converted = convert(total_loss_usd)
            tax_loss.append({
                'Symbol': row['Symbol'],
                'Unrealized Loss': format_currency(total_loss_converted, effective_currency),
                'Action': 'Consider selling to harvest loss'
            })
    if tax_loss:
        st.dataframe(pd.DataFrame(tax_loss), use_container_width=True)
    else:
        st.success("No tax-loss harvesting opportunities at this time.")

with col_b:
    st.subheader(" Longevity Stress Test")
    years_to_run = life_expectancy - age
    if years_to_run > 0:
        # Estimate portfolio return and volatility (same as before)
        hist_data = pd.DataFrame()
        for sym in symbols:
            hist = agents['market'].get_historical_prices(sym, period="1mo")
            if hist is not None and not hist.empty:
                hist_data[sym] = hist['Close']
        if not hist_data.empty:
            pct_changes = hist_data.pct_change().dropna()
            avg_daily_return = pct_changes.mean().mean()
            avg_volatility = pct_changes.std().mean() * np.sqrt(252)
        else:
            avg_daily_return = 0.0003
            avg_volatility = 0.15

        prob = monte_carlo_simulation(
            portfolio_agent.total_value,
            annual_withdrawal,   # this is in USD but simulation uses same unit; we keep in USD for consistency
            years_to_run,
            avg_daily_return * 252,
            avg_volatility
        )
        st.metric("Probability portfolio lasts", f"{prob:.1%}")
        if prob < 0.5:
            st.warning("Your withdrawal rate may be too high.")
        elif prob < 0.7:
            st.info("Consider adjusting withdrawal or allocation.")
        else:
            st.success("Your portfolio appears sustainable.")
    else:
        st.warning("Life expectancy must be greater than current age.")

# ---------- Target vs Current Comparison ----------
st.subheader(" Target vs Current Allocation")
compare_df = alloc.copy()
compare_df['Target'] = compare_df['Symbol'].map(target)
fig = px.bar(compare_df, x='Symbol', y=['Allocation', 'Target'],
             barmode='group', title='Allocation Comparison',
             labels={'value': 'Percentage', 'variable': 'Type'})
st.plotly_chart(fig, use_container_width=True)

# ---------- Historical Performance ----------
st.subheader(" Price History (Last Month)")
hist_data = pd.DataFrame()
for sym in symbols:
    hist = agents['market'].get_historical_prices(sym, period="1mo")
    if hist is not None and not hist.empty:
        hist_data[sym] = hist['Close']
if not hist_data.empty:
    # Convert historical prices if needed? For consistency, we convert all series.
    hist_data_converted = hist_data * exchange_rate
    st.line_chart(hist_data_converted)
else:
    st.info("No historical data available.")