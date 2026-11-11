# ---------------------------------------------------------------------------
# Master Configuration: Watchlist, Risk Guardrails, Market Schedule & LLM
# ---------------------------------------------------------------------------

# 28-Ticker Liquid & High-Beta Watchlist (26 Stocks + SPY & QQQ Macro ETFs)
WATCHLIST_28 = [
    # Tech, AI & Cloud
    "NVDA", "GOOGL", "PLTR", "NBIS", "AMZN", "AAPL", "AMD", "TSLA", "CRWD", "MSFT",
    # Hardware, Space & High-Beta
    "CLS", "HOOD", "ASTS", "WDC", "RKLB", "SPCX",
    # Index ETFs (Macro Regime Benchmarks)
    "QQQ", "SPY",
    # Financials & Payments
    "JPM", "GS", "V", "MA",
    # Consumer, Healthcare, Industrial & Energy
    "SBUX", "LLY", "UNH", "CAT", "GE", "XOM",
]

MACRO_SYMBOLS = ["SPY", "QQQ"]

# Alpaca Free Tier Data Settings
ALPACA_FEED = "iex"

# Risk & Profit Parameters (USD)
DAILY_PROFIT_TARGET = 150.0   # Sweet spot for $100-$200 goal
MAX_DAILY_LOSS = 100.0        # Hard stop for the day
MAX_RISK_PER_TRADE = 50.0     # Allows 2 losses before daily stop
MIN_REWARD_TO_RISK = 2.0      # 1:2 R:R ($50 risk -> $100 target)

# Screener & Anomaly Thresholds
PREMARKET_MIN_GAP_PCT = 1.5   # Minimum overnight gap % to qualify
INTRADAY_MIN_RVOL = 2.0       # Minimum Relative Volume spike
INTRADAY_MIN_MOVE_PCT = 0.75  # Minimum intraday bar/move % extension
INTRADAY_MIN_ATR_MULT = 0.35  # Minimum move as fraction of 14-day ATR
MAX_ACTIVE_TICKERS = 5        # Narrow 28 down to top 3-5 movers
ALERT_COOLDOWN_SECONDS = 900  # 15-min per-ticker LLM alert cooldown

# Market Schedule (America/Chicago - CST/CDT)
TIMEZONE = "America/Chicago"
PREMARKET_START_HOUR = 7
MARKET_OPEN_HOUR = 8
MARKET_OPEN_MINUTE = 30

# LLM Provider Toggle: "gemini" or "openai"
ACTIVE_LLM_PROVIDER = "gemini"

# Conservative Free-Tier Ceilings to Prevent 429 Errors
LLM_LIMITS = {
    "gemini": {
        "model": "gemini-3.8-flash",
        "max_rpm": 10,
        "max_tpm": 200_000,
        "max_rpd": 200,
    },
    "openai": {
        "model": "gpt-5.4-mini",
        "max_rpm": 3,
        "max_tpm": 40_000,
        "max_rpd": 150,
    },
}
