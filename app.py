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
# 1. Page Configuration (Collapsed Sidebar for Instant Mobile View)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intraday AI Analyst",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Responsive Mobile-First CSS
st.markdown(
    """
    <style>
    /* Reduce top padding so content fits above the fold on mobile */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Thumb-friendly touch targets for buttons */
    div.stButton > button {
        min-height: 48px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 8px !important;
    }

    /* Verdict Banner Cards */
    .verdict-go-long {
        background-color: #0d3b26;
        border-left: 5px solid #00e676;
        padding: 14px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        line-height: 1.45;
    }
    .verdict-go-short {
        background-color: #3b1219;
        border-left: 5px solid #ff5252;
        padding: 14px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        line-height: 1.45;
    }
    .verdict-nogo {
        background-color: #262930;
        border-left: 5px solid #90a4ae;
        padding: 14px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        line-height: 1.45;
    }

    /* Auto-Wrapping Mobile Metric Grid (2 cols on phone, 5 cols on desktop) */
    .trade-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
        gap: 8px;
        margin-bottom: 10px;
    }
    .trade-tile {
        background-color: #161a22;
        border: 1px solid #2a303c;
        border-radius: 8px;
        padding: 10px 12px;
    }
    .trade-tile-label {
        font-size: 0.72rem;
        color: #9aa4b2;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .trade-tile-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .val-risk { color: #ff6b6b !important; }
    .val-reward { color: #00e676 !important; }

    /* Mobile Live Ticker Strip Cards */
    .ticker-card-mobile {
        background-color: #161a22;
        border: 1px solid #2a303c;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
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
        "last_status_msg": "Idle — Tap RUN SESSION to fetch latest data.",
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
    cutoff = now_cst.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )
    return "PREMARKET" if now_cst < cutoff else "LIVE"


# ---------------------------------------------------------------------------
# 4. Background WebSocket Worker
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

    with st.spinner(f"Loading 20d baselines for {', '.join(symbols)}..."):
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
        f"LIVE STREAM ACTIVE: {', '.join(symbols)} (+ SPY/QQQ)"
    )


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

            with st.spinner("Scanning 28 stocks & auditing overnight news with LLM..."):
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
                f"Pre-Market updated ({now_cst.strftime('%I:%M %p CST')}) • "
                f"Core: {', '.join(chosen) if chosen else 'None'}"
            )
            st.rerun()

        else:
            symbols = st.session_state.active_symbols
            if not symbols:
                with st.spinner("Screening top 5 active movers..."):
                    snaps = fetch_snapshots(WATCHLIST_28)
                    screened, _ = screen_premarket_movers(snaps)
                    symbols = [m["ticker"] for m in screened[:MAX_ACTIVE_TICKERS]]
                    st.session_state.active_symbols = symbols

            if not symbols:
                st.session_state.last_error = (
                    "Could not resolve active symbols from snapshot feed."
                )
                return

            start_live_stream(symbols)

    except Exception as exc:
        st.session_state.last_error = f"Execution error: {exc}"


# ---------------------------------------------------------------------------
# 6. Sidebar: Risk Rules, Controls & Free-Tier Quota Telemetry
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Risk & Quota Desk")

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
    c2.metric("Max Loss", f"-${MAX_DAILY_LOSS:.0f}")
    c3, c4 = st.columns(2)
    c3.metric("Risk/Trade", f"${MAX_RISK_PER_TRADE:.0f}")
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
    with st.expander("📋 28-Ticker Universe"):
        st.write(", ".join(WATCHLIST_28))


# ---------------------------------------------------------------------------
# 7. Compact Mobile-Friendly Header & Touch Command Bar
# ---------------------------------------------------------------------------
now_header = get_chicago_now()
auto_tag = detect_market_phase(now_header)
quota_mini = get_quota_status()
provider_mini = LLM_LIMITS[ACTIVE_LLM_PROVIDER]

st.markdown("### 📡 Intraday AI Analyst")
st.caption(
    f"🕒 {now_header.strftime('%I:%M %p CST')} (`{auto_tag}`) • "
    f"🛡️ Quota: `{quota_mini['rpd_used']}/{provider_mini['max_rpd']}` calls"
)

btn_col1, btn_col2, btn_col3 = st.columns([1.4, 1.1, 1.1])

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

if st.session_state.last_error:
    st.error(st.session_state.last_error)
else:
    st.info(st.session_state.last_status_msg)


# ---------------------------------------------------------------------------
# 8. Mobile-Responsive Trade Verdict Card Renderer
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

    entry = float(setup.get("entry_price", 0) or 0)
    stop = float(setup.get("stop_loss", 0) or 0)
    target = float(setup.get("take_profit", 0) or 0)
    shares = int(setup.get("suggested_shares", 0) or 0)
    conf = int(setup.get("confidence", 0) or 0)

    st.markdown(
        f"""
        <div class="{css_class}">
            <strong>{ticker} — {badge}</strong><br/>
            <small><b>Catalyst:</b> {catalyst_q}</small><br/>
            <small>{thesis}</small>
        </div>
        <div class="trade-grid">
            <div class="trade-tile">
                <div class="trade-tile-label">Entry</div>
                <div class="trade-tile-value">${entry:.2f}</div>
            </div>
            <div class="trade-tile">
                <div class="trade-tile-label">Stop (-$50)</div>
                <div class="trade-tile-value val-risk">${stop:.2f}</div>
            </div>
            <div class="trade-tile">
                <div class="trade-tile-label">Target (+$100)</div>
                <div class="trade-tile-value val-reward">${target:.2f}</div>
            </div>
            <div class="trade-tile">
                <div class="trade-tile-label">Size</div>
                <div class="trade-tile-value">{shares} shs</div>
            </div>
            <div class="trade-tile">
                <div class="trade-tile-label">Confidence</div>
                <div class="trade-tile-value">{conf}/10</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"**Invalidation:** {invalidation}")
    st.divider()


# ---------------------------------------------------------------------------
# 9. Main Workspace Tabs (Pre-Market Blueprint vs. Live Radar)
# ---------------------------------------------------------------------------
tab_pre, tab_live = st.tabs(["🌅 Pre-Market Plan", "⚡ Live Radar"])

# --- TAB 1: PRE-MARKET PREP ---
with tab_pre:
    pm = st.session_state.premarket_result
    if not pm:
        st.info(
            "No pre-market report yet. Tap **🚀 RUN SESSION** before 8:30 AM CST "
            "(or **🌅 Force Pre-Market**) to scan your watchlist."
        )
    else:
        macro = pm.get("macro_context", {})
        llm_out = pm.get("llm_analysis", {})

        st.subheader("🌐 Macro Benchmarks")
        mc1, mc2 = st.columns(2)
        spy_info = macro.get("SPY", {})
        qqq_info = macro.get("QQQ", {})
        mc1.metric(
            "SPY",
            f"${spy_info.get('current_price', 0):.2f}",
            f"{spy_info.get('gap_pct', 0):+.2f}%",
        )
        mc2.metric(
            "QQQ",
            f"${qqq_info.get('current_price', 0):.2f}",
            f"{qqq_info.get('gap_pct', 0):+.2f}%",
        )
        st.info(f"**Market Regime:** {llm_out.get('market_regime_summary', 'N/A')}")

        st.subheader("🎯 LLM Trade Blueprints")
        setups = llm_out.get("trade_setups", [])
        if setups:
            for item in setups:
                render_trade_card(item)
        else:
            st.warning("LLM did not return structured setups for this scan.")

        with st.expander("📊 Raw Screened Gap & Volume Table"):
            movers_df = pd.DataFrame(pm.get("screened_movers", []))
            if not movers_df.empty:
                st.dataframe(movers_df, width="stretch", hide_index=True)


# --- TAB 2: LIVE INTRADAY RADAR ---
@st.fragment(run_every=2)
def render_live_radar_fragment() -> None:
    state: LiveMarketState | None = st.session_state.live_state

    if not st.session_state.ws_running or state is None:
        st.info(
            "Live stream is **OFF**. Tap **🚀 RUN SESSION** at or after 8:30 AM CST "
            "(or **⚡ Force Live WS**) to start streaming."
        )
        if st.session_state.active_symbols:
            st.write(
                f"**Queued Core:** {', '.join(st.session_state.active_symbols)}"
            )
        return

    snapshot = state.get_ui_snapshot()
    macro_live = snapshot.get("macro", {})

    st.subheader("📡 Live IEX Stream")
    lm1, lm2 = st.columns(2)
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

    rows = snapshot.get("tickers", [])
    if rows:
        for r in rows:
            chg = float(r.get("Session Chg %", 0) or 0)
            chg_color = "#00e676" if chg > 0 else ("#ff6b6b" if chg < 0 else "#9aa4b2")
            st.markdown(
                f"""
                <div class="ticker-card-mobile">
                    <div>
                        <strong style="font-size:1.05rem;">{r.get('Ticker')}</strong><br/>
                        <small style="color:#9aa4b2;">RVOL: <b>{r.get('RVOL (1m)', 0)}x</b> • ATR: ${r.get('14d ATR', 0)}</small>
                    </div>
                    <div style="text-align:right;">
                        <strong style="font-size:1.05rem;">${float(r.get('Last Price', 0)):.2f}</strong><br/>
                        <small style="color:{chg_color}; font-weight:600;">{chg:+.2f}%</small>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📋 Full Desktop Metrics Table"):
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.caption("Waiting for first IEX trades/bars to arrive...")

    st.subheader("🚨 Real-Time LLM Alerts")
    alerts = st.session_state.live_alerts
    if not alerts:
        st.caption(
            "Monitoring 1-min & 5-min bars for RVOL >= 2.0x and ATR/price breakouts..."
        )
    else:
        for alert in alerts:
            st.caption(
                f"Triggered at **{alert.get('triggered_at', '')}** • "
                f"`{alert.get('trigger_reason', '')}`"
            )
            render_trade_card(alert.get("decision", {}))


with tab_live:
    render_live_radar_fragment()
