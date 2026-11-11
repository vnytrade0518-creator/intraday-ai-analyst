from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st

from src.config import (
    ALPACA_FEED,
    DAILY_PROFIT_TARGET,
    MACRO_SYMBOLS,
    MAX_ACTIVE_TICKERS,
    MAX_RISK_PER_TRADE,
    MIN_REWARD_TO_RISK,
    PREMARKET_MIN_GAP_PCT,
    WATCHLIST_28,
)
from src.llm_client import generate_analysis, get_quota_status
from src.rate_limiter import execute_with_backoff


def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": st.secrets["broker"]["API_KEY"],
        "APCA-API-SECRET-KEY": st.secrets["broker"]["SECRET_KEY"],
        "Accept": "application/json",
    }


def fetch_snapshots(symbols: List[str]) -> Dict[str, Any]:
    """Batch-fetches snapshots for all watchlist symbols in 1 REST call."""
    base_url = st.secrets["broker"]["DATA_URL"].rstrip("/")
    url = f"{base_url}/v2/stocks/snapshots"
    params = {
        "symbols": ",".join(symbols),
        "feed": ALPACA_FEED,
    }

    def _get() -> Dict[str, Any]:
        resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    return execute_with_backoff(_get)


def screen_premarket_movers(
    snapshots: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Separates SPY/QQQ macro benchmarks and ranks stock gap movers."""
    macro_context: Dict[str, Any] = {}
    candidates: List[Dict[str, Any]] = []

    for sym, snap in snapshots.items():
        if not snap:
            continue

        prev_bar = snap.get("prevDailyBar") or {}
        latest_trade = snap.get("latestTrade") or {}
        minute_bar = snap.get("minuteBar") or {}
        daily_bar = snap.get("dailyBar") or {}

        prev_close = float(prev_bar.get("c") or 0.0)
        current_price = float(
            latest_trade.get("p") or minute_bar.get("c") or daily_bar.get("c") or prev_close
        )
        volume = int(daily_bar.get("v") or minute_bar.get("v") or 0)

        gap_pct = ((current_price - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0

        row = {
            "ticker": sym,
            "prev_close": round(prev_close, 2),
            "current_price": round(current_price, 2),
            "gap_pct": round(gap_pct, 2),
            "abs_gap_pct": round(abs(gap_pct), 2),
            "volume": volume,
        }

        if sym in MACRO_SYMBOLS:
            macro_context[sym] = row
        else:
            candidates.append(row)

    qualified = [c for c in candidates if c["abs_gap_pct"] >= PREMARKET_MIN_GAP_PCT]
    qualified.sort(key=lambda x: (x["abs_gap_pct"], x["volume"]), reverse=True)

    if len(qualified) < 3:
        candidates.sort(key=lambda x: (x["abs_gap_pct"], x["volume"]), reverse=True)
        top_movers = candidates[: max(3, MAX_ACTIVE_TICKERS)]
    else:
        top_movers = qualified[:MAX_ACTIVE_TICKERS]

    return top_movers, macro_context


def fetch_recent_news(symbols: List[str], hours_back: int = 16) -> Dict[str, List[Dict[str, Any]]]:
    """Fetches real-time Alpaca news headlines only for top screened symbols."""
    if not symbols:
        return {}

    base_url = st.secrets["broker"]["DATA_URL"].rstrip("/")
    url = f"{base_url}/v1beta1/news"
    start_iso = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()

    params = {
        "symbols": ",".join(symbols),
        "start": start_iso,
        "limit": 25,
        "sort": "desc",
    }

    def _get() -> Dict[str, Any]:
        resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    data = execute_with_backoff(_get)
    grouped: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}

    for article in data.get("news", []):
        item = {
            "headline": article.get("headline", ""),
            "summary": (article.get("summary") or "")[:300],
            "source": article.get("source", ""),
            "created_at": article.get("created_at", ""),
        }
        for sym in article.get("symbols", []):
            if sym in grouped and len(grouped[sym]) < 5:
                grouped[sym].append(item)

    return grouped


def analyze_premarket_with_llm(
    movers: List[Dict[str, Any]],
    macro_context: Dict[str, Any],
    news_by_symbol: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Synthesizes pre-market gaps + news into a structured 1:2 R:R game plan."""
    system_prompt = (
        "You are an institutional intraday equity strategist. Analyze pre-market gap movers "
        "and overnight news catalysts. Enforce strict risk discipline: maximum risk per trade "
        f"is ${MAX_RISK_PER_TRADE:.0f}, minimum reward-to-risk ratio is 1:{MIN_REWARD_TO_RISK:.1f} "
        f"(target +${MAX_RISK_PER_TRADE * MIN_REWARD_TO_RISK:.0f} per trade toward a "
        f"${DAILY_PROFIT_TARGET:.0f} daily profit goal). "
        "Reject weak rumors or low-quality gaps with verdict 'NO-GO'. "
        "Return ONLY valid JSON matching this exact schema:
"
        "{
"
        '  "market_regime_summary": "1-2 sentence macro read from SPY/QQQ",
'
        '  "trade_setups": [
'
        "    {
"
        '      "ticker": "SYMBOL",
'
        '      "verdict": "GO - LONG" | "GO - SHORT" | "NO-GO",
'
        '      "catalyst_quality": "High (Earnings/Contract)" | "Medium" | "Low (Noise/Rumor)",
'
        '      "entry_price": 0.0,
'
        '      "stop_loss": 0.0,
'
        '      "take_profit": 0.0,
'
        '      "suggested_shares": 0,
'
        '      "confidence": 1,
'
        '      "thesis": "Concise 2-sentence catalyst + technical explanation",
'
        '      "invalidation_rule": "Specific price or condition that voids the trade"
'
        "    }
"
        "  ]
"
        "}"
    )

    user_prompt = (
        f"MACRO BENCHMARKS (SPY & QQQ):
{macro_context}

"
        f"TOP SCREENED GAP MOVERS:
{movers}

"
        f"OVERNIGHT NEWS HEADLINES BY TICKER:
{news_by_symbol}"
    )

    return generate_analysis(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)


def run_premarket_pipeline() -> Dict[str, Any]:
    """Orchestrates Phase 2 end-to-end in a single button-triggered batch."""
    snapshots = fetch_snapshots(WATCHLIST_28)
    screened_movers, macro_context = screen_premarket_movers(snapshots)
    active_symbols = [m["ticker"] for m in screened_movers]
    news_by_symbol = fetch_recent_news(active_symbols, hours_back=16)
    llm_analysis = analyze_premarket_with_llm(screened_movers, macro_context, news_by_symbol)

    return {
        "macro_context": macro_context,
        "screened_movers": screened_movers,
        "news_by_symbol": news_by_symbol,
        "llm_analysis": llm_analysis,
        "llm_quota_usage": get_quota_status(),
    }
