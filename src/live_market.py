import asyncio
import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

import pytz
import requests
import streamlit as st
import websockets

from src.config import (
    ALERT_COOLDOWN_SECONDS,
    ALPACA_FEED,
    DAILY_PROFIT_TARGET,
    INTRADAY_MIN_ATR_MULT,
    INTRADAY_MIN_MOVE_PCT,
    INTRADAY_MIN_RVOL,
    MACRO_SYMBOLS,
    MAX_RISK_PER_TRADE,
    MIN_REWARD_TO_RISK,
    TIMEZONE,
)
from src.llm_client import generate_analysis
from src.premarket import fetch_recent_news
from src.rate_limiter import execute_with_backoff


@dataclass
class Bar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class SymbolTracker:
    symbol: str
    atr_14: float = 1.0
    avg_minute_vol: float = 1000.0
    session_open: float = 0.0
    session_high: float = 0.0
    session_low: float = 0.0
    last_price: float = 0.0
    cumulative_volume: int = 0
    current_rvol: float = 0.0
    bars_1m: Deque[Bar] = field(default_factory=lambda: deque(maxlen=240))
    bars_5m: Deque[Bar] = field(default_factory=lambda: deque(maxlen=78))
    last_alert_ts: float = 0.0
    _current_1m: Optional[Bar] = None
    _current_1m_bucket: str = ""


class LiveMarketState:
    """Thread-safe in-memory tick-to-bar aggregator & anomaly screener."""

    def __init__(self, symbols: List[str], baselines: Dict[str, Dict[str, float]]) -> None:
        self._lock = threading.Lock()
        self.symbols = symbols
        self.trackers: Dict[str, SymbolTracker] = {}
        for sym in set(symbols + MACRO_SYMBOLS):
            base = baselines.get(sym, {})
            self.trackers[sym] = SymbolTracker(
                symbol=sym,
                atr_14=float(base.get("atr_14", 1.0)),
                avg_minute_vol=float(base.get("avg_minute_vol", 1000.0)),
                session_open=float(base.get("prev_close", 0.0)),
            )

    def ingest_trade(self, symbol: str, price: float, size: int, ts_iso: str) -> None:
        with self._lock:
            tr = self.trackers.get(symbol)
            if tr is None:
                return

            if tr.session_open <= 0:
                tr.session_open = price
            tr.last_price = price
            tr.session_high = max(tr.session_high, price) if tr.session_high > 0 else price
            tr.session_low = min(tr.session_low, price) if tr.session_low > 0 else price
            tr.cumulative_volume += size

            minute_bucket = ts_iso[:16]
            if tr._current_1m is None or tr._current_1m_bucket != minute_bucket:
                if tr._current_1m is not None:
                    tr.bars_1m.append(tr._current_1m)
                    self._maybe_rollup_5m(tr)
                tr._current_1m_bucket = minute_bucket
                tr._current_1m = Bar(
                    timestamp=minute_bucket,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=size,
                )
            else:
                b = tr._current_1m
                b.high = max(b.high, price)
                b.low = min(b.low, price)
                b.close = price
                b.volume += size

            if tr._current_1m and tr.avg_minute_vol > 0:
                tr.current_rvol = round(tr._current_1m.volume / tr.avg_minute_vol, 2)

    def ingest_bar(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = msg.get("S", "")
        with self._lock:
            tr = self.trackers.get(symbol)
            if tr is None:
                return None

            bar = Bar(
                timestamp=msg.get("t", ""),
                open=float(msg.get("o", 0.0)),
                high=float(msg.get("h", 0.0)),
                low=float(msg.get("l", 0.0)),
                close=float(msg.get("c", 0.0)),
                volume=int(msg.get("v", 0)),
            )
            if tr.session_open <= 0:
                tr.session_open = bar.open
            tr.last_price = bar.close
            tr.session_high = max(tr.session_high, bar.high) if tr.session_high > 0 else bar.high
            tr.session_low = min(tr.session_low, bar.low) if tr.session_low > 0 else bar.low
            tr.bars_1m.append(bar)
            self._maybe_rollup_5m(tr)

            if tr.avg_minute_vol > 0:
                tr.current_rvol = round(bar.volume / tr.avg_minute_vol, 2)

            if symbol in MACRO_SYMBOLS:
                return None

            return self._check_anomaly_locked(tr, bar)

    @staticmethod
    def _maybe_rollup_5m(tr: SymbolTracker) -> None:
        if len(tr.bars_1m) >= 5 and len(tr.bars_1m) % 5 == 0:
            chunk = list(tr.bars_1m)[-5:]
            tr.bars_5m.append(
                Bar(
                    timestamp=chunk[-1].timestamp,
                    open=chunk[0].open,
                    high=max(b.high for b in chunk),
                    low=min(b.low for b in chunk),
                    close=chunk[-1].close,
                    volume=sum(b.volume for b in chunk),
                )
            )

    def _check_anomaly_locked(self, tr: SymbolTracker, bar: Bar) -> Optional[Dict[str, Any]]:
        now = time.monotonic()
        if (now - tr.last_alert_ts) < ALERT_COOLDOWN_SECONDS:
            return None

        if tr.current_rvol < INTRADAY_MIN_RVOL:
            return None

        bar_move = abs(bar.close - bar.open)
        bar_move_pct = (bar_move / bar.open * 100.0) if bar.open > 0 else 0.0
        atr_ratio = (bar_move / tr.atr_14) if tr.atr_14 > 0 else 0.0

        if bar_move_pct < INTRADAY_MIN_MOVE_PCT and atr_ratio < INTRADAY_MIN_ATR_MULT:
            return None

        tr.last_alert_ts = now
        direction = "BULLISH SPIKE" if bar.close >= bar.open else "BEARISH DROP"
        return {
            "ticker": tr.symbol,
            "direction": direction,
            "price": round(bar.close, 2),
            "bar_open": round(bar.open, 2),
            "bar_move_pct": round(bar_move_pct, 2),
            "rvol": tr.current_rvol,
            "atr_14": round(tr.atr_14, 2),
            "session_high": round(tr.session_high, 2),
            "session_low": round(tr.session_low, 2),
            "trigger_reason": (
                f"{direction}: RVOL {tr.current_rvol:.1f}x | "
                f"1m Move {bar_move_pct:.2f}% ({atr_ratio:.2f}x ATR)"
            ),
        }

    def get_ui_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            macro_out: Dict[str, Any] = {}
            ticker_rows: List[Dict[str, Any]] = []

            for sym in MACRO_SYMBOLS:
                tr = self.trackers.get(sym)
                if tr:
                    chg = (
                        ((tr.last_price - tr.session_open) / tr.session_open * 100.0)
                        if tr.session_open > 0 and tr.last_price > 0
                        else 0.0
                    )
                    macro_out[sym] = {
                        "price": round(tr.last_price, 2),
                        "change_pct": round(chg, 2),
                    }

            for sym in self.symbols:
                tr = self.trackers.get(sym)
                if not tr:
                    continue
                chg = (
                    ((tr.last_price - tr.session_open) / tr.session_open * 100.0)
                    if tr.session_open > 0 and tr.last_price > 0
                    else 0.0
                )
                ticker_rows.append(
                    {
                        "Ticker": sym,
                        "Last Price": round(tr.last_price, 2),
                        "Session Chg %": round(chg, 2),
                        "RVOL (1m)": round(tr.current_rvol, 2),
                        "14d ATR": round(tr.atr_14, 2),
                        "Session High": round(tr.session_high, 2),
                        "Session Low": round(tr.session_low, 2),
                        "Stream Vol": tr.cumulative_volume,
                    }
                )

            return {"macro": macro_out, "tickers": ticker_rows}


def fetch_intraday_baselines(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetches 20 daily bars in 1 REST call to compute 14-day ATR & avg 1-min volume."""
    all_syms = list(set(symbols + MACRO_SYMBOLS))
    base_url = st.secrets["broker"]["DATA_URL"].rstrip("/")
    url = f"{base_url}/v2/stocks/bars"
    start_iso = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()

    headers = {
        "APCA-API-KEY-ID": st.secrets["broker"]["API_KEY"],
        "APCA-API-SECRET-KEY": st.secrets["broker"]["SECRET_KEY"],
        "Accept": "application/json",
    }
    params = {
        "symbols": ",".join(all_syms),
        "timeframe": "1Day",
        "start": start_iso,
        "limit": 25,
        "feed": ALPACA_FEED,
    }

    def _get() -> Dict[str, Any]:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    data = execute_with_backoff(_get)
    bars_map = data.get("bars", {})
    baselines: Dict[str, Dict[str, float]] = {}

    for sym in all_syms:
        daily = bars_map.get(sym, [])[-20:]
        if not daily:
            baselines[sym] = {"atr_14": 1.0, "avg_minute_vol": 1000.0, "prev_close": 0.0}
            continue

        trs: List[float] = []
        for idx, bar in enumerate(daily):
            high = float(bar.get("h", 0.0))
            low = float(bar.get("l", 0.0))
            prev_c = float(daily[idx - 1].get("c", low)) if idx > 0 else low
            tr = max(high - low, abs(high - prev_c), abs(low - prev_c))
            trs.append(tr)

        atr_14 = sum(trs[-14:]) / max(len(trs[-14:]), 1)
        avg_daily_vol = sum(float(b.get("v", 0.0)) for b in daily) / len(daily)
        avg_min_vol = max(100.0, avg_daily_vol / 390.0)
        prev_close = float(daily[-1].get("c", 0.0))

        baselines[sym] = {
            "atr_14": round(atr_14, 4),
            "avg_minute_vol": round(avg_min_vol, 2),
            "prev_close": round(prev_close, 2),
        }

    return baselines


def evaluate_anomaly_with_llm(
    anomaly: Dict[str, Any],
    macro_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Fetches last 2h news for the triggered ticker and gets a 1:2 R:R LLM decision."""
    ticker = anomaly["ticker"]
    news = fetch_recent_news([ticker], hours_back=2)

    target_profit = MAX_RISK_PER_TRADE * MIN_REWARD_TO_RISK
    system_prompt = f"""You are an elite intraday proprietary risk officer. Evaluate this live volume/price anomaly against real-time news catalysts. Enforce strict risk discipline: maximum risk per trade is ${MAX_RISK_PER_TRADE:.0f}, minimum reward-to-risk ratio is 1:{MIN_REWARD_TO_RISK:.1f} (target +${target_profit:.0f} toward a ${DAILY_PROFIT_TARGET:.0f} daily profit goal). Reject fake breakouts or empty news catalysts with 'NO-GO'.

Return ONLY valid JSON matching this exact schema:
{{
  "ticker": "SYMBOL",
  "verdict": "GO - LONG" | "GO - SHORT" | "NO-GO",
  "catalyst_quality": "High (Earnings/Contract)" | "Medium" | "Low (Noise/Rumor)",
  "entry_price": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "suggested_shares": 0,
  "confidence": 1,
  "thesis": "Concise 2-sentence intraday momentum + catalyst explanation",
  "invalidation_rule": "Specific price or condition that voids the trade"
}}"""


    user_prompt = (
        f"LIVE ANOMALY METRICS:
{anomaly}

"
        f"LIVE MACRO CONTEXT (SPY/QQQ):
{macro_snapshot}

"
        f"BREAKING NEWS (LAST 2 HOURS):
{news}"
    )

    return generate_analysis(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)


async def _process_anomaly_async(
    anomaly: Dict[str, Any],
    state: LiveMarketState,
    on_alert: Optional[Callable[[Dict[str, Any]], None]],
) -> None:
    macro_snap = state.get_ui_snapshot().get("macro", {})
    decision = await asyncio.to_thread(evaluate_anomaly_with_llm, anomaly, macro_snap)
    now_cst = datetime.now(pytz.timezone(TIMEZONE)).strftime("%I:%M:%S %p CST")

    if on_alert is not None:
        on_alert(
            {
                "triggered_at": now_cst,
                "trigger_reason": anomaly["trigger_reason"],
                "decision": decision,
            }
        )


async def stream_live_market(
    active_symbols: List[str],
    state: LiveMarketState,
    stop_event: asyncio.Event,
    on_alert: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Connects to Alpaca IEX WebSocket, aggregates bars, and triggers LLM alerts."""
    ws_url = st.secrets["broker"]["WS_URL"]
    api_key = st.secrets["broker"]["API_KEY"]
    secret_key = st.secrets["broker"]["SECRET_KEY"]
    sub_symbols = list(set(active_symbols + MACRO_SYMBOLS))

    reconnect_attempt = 0

    while not stop_event.is_set():
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                # 1. Welcome handshake
                await asyncio.wait_for(ws.recv(), timeout=10)

                # 2. Authenticate
                await ws.send(
                    json.dumps({"action": "auth", "key": api_key, "secret": secret_key})
                )
                auth_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                auth_msgs = json.loads(auth_raw)
                if not any(m.get("msg") == "authenticated" for m in auth_msgs):
                    raise RuntimeError(f"Alpaca WebSocket authentication failed: {auth_msgs}")

                # 3. Subscribe to trades & 1-minute bars
                await ws.send(
                    json.dumps(
                        {
                            "action": "subscribe",
                            "trades": sub_symbols,
                            "bars": sub_symbols,
                        }
                    )
                )
                reconnect_attempt = 0

                # 4. Stream loop
                while not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    events = json.loads(raw)
                    for ev in events:
                        ev_type = ev.get("T")
                        if ev_type == "t":
                            state.ingest_trade(
                                symbol=ev.get("S", ""),
                                price=float(ev.get("p", 0.0)),
                                size=int(ev.get("s", 0)),
                                ts_iso=ev.get("t", ""),
                            )
                        elif ev_type == "b":
                            anomaly = state.ingest_bar(ev)
                            if anomaly is not None:
                                asyncio.create_task(
                                    _process_anomaly_async(anomaly, state, on_alert)
                                )

        except Exception:
            if stop_event.is_set():
                break
            reconnect_attempt += 1
            delay = min(30.0, (2.0 ** min(reconnect_attempt, 4)) + random.uniform(0.2, 1.0))
            await asyncio.sleep(delay)
