# ==========================================
# LLM PROVIDER TOGGLE ("gemini" or "openai")
# ==========================================
ACTIVE_LLM_PROVIDER = "gemini"

# ==========================================
# CUSTOM 28-TICKER WATCHLIST (Equities & ETFs)
# ==========================================
WATCHLIST = [
    # Tech, AI & Cloud
    "NVDA", "GOOGL", "PLTR", "NBIS", "AMZN", "AAPL", "AMD", "TSLA", "CRWD", "MSFT",
    # Hardware, Space & High-Beta Growth
    "CLS", "HOOD", "ASTS", "WDC", "RKLB", "SPCX",
    # Market Index ETFs
    "QQQ", "SPY",
    # Financials & Payments
    "JPM", "GS", "V", "MA",
    # Consumer, Healthcare, Industrial & Energy
    "SBUX", "LLY", "UNH", "CAT", "GE", "XOM"
]

# ==========================================
# ALPACA FREE TIER SETTINGS
# ==========================================
ALPACA_FEED = "iex"  # Required on Alpaca Free Tier to avoid 15-min REST delay

# ==========================================
# RISK & PROFIT PARAMETERS (USD)
# ==========================================
DAILY_PROFIT_TARGET = 150.0   # Sweet spot for $100-$200 daily goal
MAX_DAILY_LOSS = 100.0        # Hard stop for the day
MAX_RISK_PER_TRADE = 50.0     # Allows 2 losses before daily stop
MIN_REWARD_TO_RISK = 2.0      # 1:2 R:R ($50 risk -> $100 target)

# ==========================================
# SCREENER THRESHOLDS
# ==========================================
PREMARKET_MIN_GAP_PCT = 1.5   # Minimum overnight gap (%) to qualify
INTRADAY_MIN_RVOL = 2.0       # Minimum Relative Volume spike
MAX_ACTIVE_TICKERS = 5        # Narrow 28 tickers down to top 3-5 movers

# ==========================================
# MARKET SCHEDULE (America/Chicago - CST/CDT)
# ==========================================
TIMEZONE = "America/Chicago"
PREMARKET_START_HOUR = 7
MARKET_OPEN_HOUR = 8
MARKET_OPEN_MINUTE = 30
