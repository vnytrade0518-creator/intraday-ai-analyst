import asyncio
import threading
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import pytz
import streamlit as st

from src.config import (
    ACTIVE_LLM_PROVIDER,
    DAILY_PROFIT_TARGET,
    LLM_LIMITS,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MAX_ACTIVE_TICKERS,
    MAX_DAILY_LOSS,
    MAX_RISK_PER_TRADE,
    MIN_REWARD_TO_RISK,
    TIMEZONE,
    WATCHLIST_28,
)
from src.llm_client import get_quota_status
from src.premarket import (
    fetch_snapshots,
    run_premarket_pipeline,
    screen_premarket_movers,
)
from src.live_market import (
    LiveMarketState,
    fetch_intraday_baselines,
    stream_live_market,
)

# ---------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intraday AI Analyst | S&P & High-Beta Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .verdict-go-long {
        background-color: #0d3b26;
        border-left: 5px solid #00e676;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .verdict-go-short {
        background-color: #3b1219;
        border-left: 5px solid #ff5252;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .verdict-nogo {
        background-color: #262930;
        border-left: 5px solid #90a4ae;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 2. Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "premarket_result": None,
        "active_symbols": [],
        "live_state": None,
        "live_alerts": [],
        "ws_stop_event": None,
        "ws_thread": None,
        "ws_running": False,
        "last_status_msg": "Idle — Ready to run on your command.",
        "last_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ---------------------------------------------------------------------------
# 3. Time-Based Routing Helpers (America/Chicago)
# ---------------------------------------------------------------------------
def get_chicago_now() -> datetime:
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz)


def detect_market_phase(now_cst: datetime) -> str:
    """Returns 'PREMARKET' if strictly before 8:30 AM CST, else 'LIVE'."""
    cutoff = now_cst.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )
    return "PREMARKET" if now_cst < cutoff else "LIVE"


# ---------------------------------------------------------------------------
# 4. Background WebSocket Worker (Isolated from Streamlit Reruns)
# ---------------------------------------------------------------------------
def _ws_background_runner(
    symbols: List[str],
    state: LiveMarketState,
    stop_event: threading.Event,
    alerts_sink: List[Dict[str, Any]],
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async_stop = asyncio.Event()

    async def _watch_stop() -> None:
        while not stop_event.is_set():
            await asyncio.sleep(0.5)
        async_stop.set()

    def _on_alert(alert_payload: Dict[str, Any]) -> None:
        alerts_sink.insert(0, alert_payload)

    async def _main() -> None:
        watcher = asyncio.create_task(_watch_stop())
        try:
            await stream_live_market(
                active_symbols=symbols,
                state=state,
                stop_event=async_stop,
                on_alert=_on_alert,
            )
        finally:
            watcher.cancel()

    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()


def start_live_stream(symbols: List[str]) -> None:
    if st.session_state.ws_running:
        return

    with st.spinner(f"Loading 20-day ATR & volume baselines for {', '.join(symbols)}..."):
        baselines = fetch_intraday_baselines(symbols)

    state = LiveMarketState(symbols=symbols, baselines=baselines)
    stop_event = threading.Event()
    alerts_sink: List[Dict[str, Any]] = st.session_state.live_alerts

    thread = threading.Thread(
        target=_ws_background_runner,
        args=(symbols, state, stop_event, alerts_sink),
        daemon=True,
        name="alpaca-iex-ws-worker",
    )
    thread.start()

    st.session_state.live_state = state
    st.session_state.ws_stop_event = stop_event
    st.session_state.ws_thread = thread
    st.session_state.ws_running = True
    st.session_state.last_status_msg = (
        f"LIVE STREAM ACTIVE for {', '.join(symbols)} (+ SPY/QQQ macro)."
    )
    st.rerun()


def stop_live_stream() -> None:
    if st.session_state.ws_stop_event is not None:
        st.session_state.ws_stop_event.set()
    st.session_state.ws_running = False
    st.session_state.last_status_msg = "Live WebSocket stream stopped."


# ---------------------------------------------------------------------------
# 5. Master Button Handler (Time-Based Auto-Router)
# ---------------------------------------------------------------------------
def handle_run_session(force_mode: str = "AUTO") -> None:
    st.session_state.last_error = None
    now_cst = get_chicago_now()
    phase = detect_market_phase(now_cst) if force_mode == "AUTO" else force_mode

    try:
        if phase == "PREMARKET":
            if st.session_state.ws_running:
                stop_live_stream()

            with st.spinner("Scanning 28 watchlist tickers & auditing overnight news with LLM..."):
                result = run_premarket_pipeline()

            st.session_state.premarket_result = result
            setups = result.get("llm_analysis", {}).get("trade_setups", [])
            screened = result.get("screened_movers", [])

            go_tickers = [
                s["ticker"]
                for s in setups
                if str(s.get("verdict", "")).upper().startswith("GO")
            ]
            fallback_tickers = [m["ticker"] for m in screened[:MAX_ACTIVE_TICKERS]]
            chosen = (go_tickers or fallback_tickers)[:MAX_ACTIVE_TICKERS]

            st.session_state.active_symbols = chosen
            st.session_state.last_status_msg = (
                f"Pre-Market Prep completed at {now_cst.strftime('%I:%M:%S %p CST')}. "
                f"Active Core: {', '.join(chosen) if chosen else 'None'}."
            )

        else:
            symbols = st.session_state.active_symbols
            if not symbols:
                with st.spinner("No pre-market list found — running 1-call snapshot screen..."):
                    snaps = fetch_snapshots(WATCHLIST_28)
                    screened, _ = screen_premarket_movers(snaps)
                    symbols = [m["ticker"] for m in screened[:MAX_ACTIVE_TICKERS]]
                    st.session_state.active_symbols = symbols

            if not symbols:
                st.session_state.last_error = (
                    "Could not resolve active symbols from snapshot feed. Check Alpaca keys."
                )
                return

            start_live_stream(symbols)

    except Exception as exc:
        st.session_state.last_error = f"Execution error: {exc}"


# ---------------------------------------------------------------------------
# 6. Sidebar: Risk Rules, Controls & Free-Tier Quota Telemetry
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control & Risk Desk")

    now_cst = get_chicago_now()
    detected_phase = detect_market_phase(now_cst)
    st.metric(
        label="Chicago Time (CST/CDT)",
        value=now_cst.strftime("%I:%M:%S %p"),
        delta=f"Auto-Route: {detected_phase}",
    )

    st.divider()
    st.subheader("🎯 Daily Risk Guardrails")
    c1, c2 = st.columns(2)
    c1.metric("Daily Goal", f"${DAILY_PROFIT_TARGET:.0f}")
    c2.metric("Max Daily Loss", f"-${MAX_DAILY_LOSS:.0f}")
    c3, c4 = st.columns(2)
    c3.metric("Risk / Trade", f"${MAX_RISK_PER_TRADE:.0f}")
    c4.metric("Min R:R", f"1 : {MIN_REWARD_TO_RISK:.1f}")

    st.divider()
    st.subheader("🛡️ Free-Tier LLM Quota")
    provider_cfg = LLM_LIMITS[ACTIVE_LLM_PROVIDER]
    quota = get_quota_status()

    st.caption(
        f"Provider: **{ACTIVE_LLM_PROVIDER.upper()}** (`{provider_cfg['model']}`)"
    )
    st.progress(
        min(quota["rpm_used"] / max(provider_cfg["max_rpm"], 1), 1.0),
        text=f"RPM: {quota['rpm_used']} / {provider_cfg['max_rpm']}",
    )
    st.progress(
        min(quota["tpm_used"] / max(provider_cfg["max_tpm"], 1), 1.0),
        text=f"TPM: {quota['tpm_used']:,} / {provider_cfg['max_tpm']:,}",
    )
    st.progress(
        min(quota["rpd_used"] / max(provider_cfg["max_rpd"], 1), 1.0),
        text=f"Daily Calls: {quota['rpd_used']} / {provider_cfg['max_rpd']}",
    )

    st.divider()
    with st.expander("📋 View 28-Ticker Universe"):
        st.write(", ".join(WATCHLIST_28))


# ---------------------------------------------------------------------------
# 7. Top Command Bar
# ---------------------------------------------------------------------------
st.title("📡 Intraday AI Trading Analyst")
st.caption(
    "On-Demand Decision Engine • Alpaca Free IEX Data + News • Rate-Limited LLM Analyst"
)

btn_col1, btn_col2, btn_col3, status_col = st.columns([1.4, 1.2, 1.0, 3.4])

with btn_col1:
    if st.button("🚀 RUN SESSION (Auto)", type="primary", width="stretch"):
        handle_run_session("AUTO")

with btn_col2:
    if st.button("🌅 Force Pre-Market", width="stretch"):
        handle_run_session("PREMARKET")

with btn_col3:
    if st.session_state.ws_running:
        if st.button("🛑 Stop Stream", width="stretch"):
            stop_live_stream()
            st.rerun()
    else:
        if st.button("⚡ Force Live WS", width="stretch"):
            handle_run_session("LIVE")

with status_col:
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    else:
        st.info(st.session_state.last_status_msg)


# ---------------------------------------------------------------------------
# 8. Helper Renderer for Trade Verdict Cards
# ---------------------------------------------------------------------------
def render_trade_card(setup: Dict[str, Any]) -> None:
    verdict = str(setup.get("verdict", "NO-GO")).upper()
    ticker = setup.get("ticker", "N/A")
    catalyst_q = setup.get("catalyst_quality", "Unknown")
    thesis = setup.get("thesis", "No thesis provided.")
    invalidation = setup.get("invalidation_rule", "N/A")

    if "LONG" in verdict:
        css_class = "verdict-go-long"
        badge = "🟢 GO - LONG"
    elif "SHORT" in verdict:
        css_class = "verdict-go-short"
        badge = "🔴 GO - SHORT"
    else:
        css_class = "verdict-nogo"
        badge = "⚪ NO-GO / AVOID"

    st.markdown(
        f"""
        <div class="{css_class}">
            <strong>{ticker} — {badge}</strong> &nbsp;|&nbsp;
            <span>Catalyst Quality: <b>{catalyst_q}</b></span><br/>
            <small>{thesis}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Entry", f"${setup.get('entry_price', 0):.2f}")
    m2.metric("Stop-Loss (-$50)", f"${setup.get('stop_loss', 0):.2f}")
    m3.metric("Target (+$100)", f"${setup.get('take_profit', 0):.2f}")
    m4.metric("Position Size", f"{setup.get('suggested_shares', 0)} shs")
    m5.metric("Confidence", f"{setup.get('confidence', 0)}/10")
    st.caption(f"**Invalidation Rule:** {invalidation}")
    st.divider()


# ---------------------------------------------------------------------------
# 9. Main Workspace Tabs (Pre-Market Blueprint vs. Live Radar)
# ---------------------------------------------------------------------------
tab_pre, tab_live = st.tabs(["🌅 Pre-Market Game Plan", "⚡ Live Intraday Radar"])

with tab_pre:
    pm = st.session_state.premarket_result
    if not pm:
        st.info(
            "No pre-market report generated yet. Click **🚀 RUN SESSION** before 8:30 AM CST "
            "(or click **🌅 Force Pre-Market**) to scan your 28 tickers."
        )
    else:
        macro = pm.get("macro_context", {})
        llm_out = pm.get("llm_analysis", {})

        st.subheader("🌐 Macro Index Benchmarks")
        mc1, mc2, mc3 = st.columns(3)
        spy_info = macro.get("SPY", {})
        qqq_info = macro.get("QQQ", {})
        mc1.metric(
            "SPY Pre-Market",
            f"${spy_info.get('current_price', 0):.2f}",
            f"{spy_info.get('gap_pct', 0):+.2f}%",
        )
        mc2.metric(
            "QQQ Pre-Market",
            f"${qqq_info.get('current_price', 0):.2f}",
            f"{qqq_info.get('gap_pct', 0):+.2f}%",
        )
        mc3.info(f"**Market Regime:** {llm_out.get('market_regime_summary', 'N/A')}")

        st.subheader("🎯 LLM Pre-Market Trade Blueprints")
        setups = llm_out.get("trade_setups", [])
        if setups:
            for item in setups:
                render_trade_card(item)
        else:
            st.warning("LLM did not return structured setups for this scan.")

        with st.expander("📊 Raw Screened Gap & Volume Table"):
            movers_df = pd.DataFrame(pm.get("screened_movers", []))
            if not movers_df.empty:
                st.dataframe(movers_df, width="stretch")


@st.fragment(run_every=2)
def render_live_radar_fragment() -> None:
    state: LiveMarketState | None = st.session_state.live_state

    if not st.session_state.ws_running or state is None:
        st.info(
            "Live WebSocket stream is currently **OFF**. Click **🚀 RUN SESSION** at or after "
            "8:30 AM CST (or click **⚡ Force Live WS**) to start streaming real-time IEX ticks."
        )
        if st.session_state.active_symbols:
            st.write(
                f"**Queued Active Core Watchlist:** {', '.join(st.session_state.active_symbols)}"
            )
        return

    snapshot = state.get_ui_snapshot()
    macro_live = snapshot.get("macro", {})

    st.subheader("📡 Live IEX Stream & Anomaly Radar")
    lm1, lm2, lm3 = st.columns(3)
    spy_l = macro_live.get("SPY", {})
    qqq_l = macro_live.get("QQQ", {})
    lm1.metric(
        "SPY Live",
        f"${spy_l.get('price', 0):.2f}",
        f"{spy_l.get('change_pct', 0):+.2f}%",
    )
    lm2.metric(
        "QQQ Live",
        f"${qqq_l.get('price', 0):.2f}",
        f"{qqq_l.get('change_pct', 0):+.2f}%",
    )
    lm3.metric(
        "Stream Status",
        "CONNECTED (IEX)",
        f"Tracking {len(snapshot.get('tickers', []))} Core Symbols",
    )

    rows = snapshot.get("tickers", [])
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.caption("Waiting for first IEX trades/bars to arrive...")

    st.subheader("🚨 Real-Time LLM Anomaly Alerts")
    alerts = st.session_state.live_alerts
    if not alerts:
        st.caption(
            "Monitoring 1-min & 5-min bars for RVOL >= 2.0x and ATR/price breakouts..."
        )
    else:
        for alert in alerts:
            st.caption(
                f"Triggered at **{alert.get('triggered_at', '')}** • "
                f"Reason: `{alert.get('trigger_reason', '')}`"
            )
            render_trade_card(alert.get("decision", {}))


with tab_live:
    render_live_radar_fragment()
