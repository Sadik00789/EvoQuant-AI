import os
import time
import pandas as pd
import sqlalchemy
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. PAGE CONFIG & REFINED DARK THEME CSS
# ==========================================

st.set_page_config(
    page_title="EvoQuant-AI | Swarm Analytics Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Design tokens ---
BG_APP = "#0B0E14"
BG_PANEL = "#141925"
BG_PANEL_ALT = "#171D2B"
BORDER = "#242C3D"
TEXT_PRIMARY = "#F3F6FC"
TEXT_MUTED = "#8B96AC"
ACCENT = "#5EEAD4"       # teal
ACCENT_2 = "#818CF8"     # indigo
ACCENT_WARN = "#FBBF24"
ACCENT_DOWN = "#FB7185"
ACCENT_UP = "#34D399"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
    }}

    /* Top header bar */
    header[data-testid="stHeader"] {{
        background-color: {BG_APP} !important;
    }}
    div[data-testid="stDecoration"] {{
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2}) !important;
        height: 3px !important;
    }}
    header[data-testid="stHeader"] * {{ color: {TEXT_PRIMARY} !important; }}

    /* App background */
    .stApp {{
        background:
            radial-gradient(1200px 500px at 15% -5%, rgba(94,234,212,0.06), transparent),
            radial-gradient(1200px 500px at 100% 0%, rgba(129,140,248,0.07), transparent),
            {BG_APP} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* Labels */
    label, .stMultiSelect label, .stSelectbox label {{
        color: {ACCENT} !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }}
    p {{ color: {TEXT_PRIMARY}; }}

    /* Selectbox & multiselect inputs */
    div[data-baseweb="select"] > div {{
        background-color: {BG_PANEL_ALT} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT_PRIMARY} !important;
        border-radius: 8px !important;
        transition: border-color 0.15s ease;
    }}
    div[data-baseweb="select"]:hover > div {{
        border-color: {ACCENT} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 500 !important;
    }}
    div[data-baseweb="tag"] {{
        background: linear-gradient(135deg, {ACCENT_2}, #6366F1) !important;
        border: 1px solid rgba(129,140,248,0.6) !important;
        box-shadow: 0 2px 6px rgba(99,102,241,0.35);
    }}
    div[data-baseweb="tag"] span {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}
    div[data-baseweb="tag"] svg {{
        fill: #FFFFFF !important;
        opacity: 0.85;
    }}
    div[data-baseweb="tag"] svg:hover {{
        opacity: 1;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, {BG_PANEL_ALT} 0%, {BG_PANEL} 100%) !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        border-color: rgba(94,234,212,0.4) !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: clamp(1.05rem, 1.7vw, 1.5rem) !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
    }}
    div[data-testid="stMetricValue"] > div {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        white-space: normal !important;
        overflow: visible !important;
    }}
    div[data-testid="stMetric"] {{
        min-width: 0 !important;
    }}
    div[data-testid="stMetricDelta"] svg {{ display: none; }}

    /* DataFrames / tables */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
        background-color: {BG_PANEL} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 6px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }}

    /* Info / alert banner */
    .stAlert {{
        background: linear-gradient(90deg, rgba(94,234,212,0.10), rgba(129,140,248,0.06)) !important;
        border: 1px solid rgba(94,234,212,0.35) !important;
        border-left: 3px solid {ACCENT} !important;
        color: {TEXT_PRIMARY} !important;
        border-radius: 10px !important;
    }}
    .stAlert div {{ color: {TEXT_PRIMARY} !important; font-weight: 500 !important; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {BG_PANEL} !important;
        border-right: 1px solid {BORDER} !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] span {{ color: {TEXT_PRIMARY} !important; }}

    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stCheckbox p {{
        color: {TEXT_PRIMARY} !important;
        text-transform: none !important;
        font-weight: 500 !important;
    }}

    /* Sidebar buttons */
    [data-testid="stSidebar"] button {{
        background: linear-gradient(135deg, {ACCENT}, {ACCENT_2}) !important;
        color: #0B0E14 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        transition: opacity 0.15s ease;
    }}
    [data-testid="stSidebar"] button:hover {{ opacity: 0.85; }}

    /* Guardrails card */
    .guardrail-card {{
        background: {BG_PANEL_ALT};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
    }}
    .guardrail-item {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        padding: 6px 0;
        display: flex;
        justify-content: space-between;
        border-bottom: 1px dashed {BORDER};
    }}
    .guardrail-item:last-child {{ border-bottom: none; }}
    .guardrail-value {{
        font-weight: 700;
        color: {ACCENT};
        font-family: 'JetBrains Mono', monospace;
    }}

    /* Section headers */
    .section-title {{
        display: flex;
        align-items: center;
        gap: 10px;
        color: {TEXT_PRIMARY} !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        margin-bottom: 4px;
    }}
    .section-subtitle {{
        color: {TEXT_MUTED} !important;
        font-size: 0.85rem !important;
        margin-top: -2px;
        margin-bottom: 18px;
    }}

    .app-divider {{
        height: 1px;
        border: none;
        margin: 30px 0;
        background: linear-gradient(90deg, transparent, {BORDER}, transparent);
    }}

    /* Hero banner */
    .hero-banner {{
        background: linear-gradient(120deg, rgba(94,234,212,0.10), rgba(129,140,248,0.08));
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 26px;
    }}
    .hero-title {{
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: {TEXT_PRIMARY} !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
    }}
    .hero-sub {{
        color: {TEXT_MUTED} !important;
        font-size: 0.95rem !important;
        margin-top: 6px !important;
    }}

    .pill {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        background: rgba(52,211,153,0.15);
        color: {ACCENT_UP};
        border: 1px solid rgba(52,211,153,0.35);
    }}
</style>
""", unsafe_allow_html=True)

AGENT_COLORS = {
    "Agent_Alpha": ACCENT,       # Teal
    "Agent_Beta": ACCENT_2,      # Indigo
    "Agent_Gamma": ACCENT_WARN,  # Amber
    "Agent_Delta": "#F472B6",    # Pink
    "Agent_Epsilon": ACCENT_UP   # Green
}

PLOT_FONT = dict(color=TEXT_MUTED, family="Inter, sans-serif")

# ==========================================
# 2. POSTGRESQL / TIMESCALEDB READERS
# ==========================================

POSTGRES_USER = os.getenv("POSTGRES_USER", "evoquant")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "evoquant_secret_pass")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "evoquant_db")

POSTGRES_URL = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

@st.cache_resource
def get_db_engine():
    return sqlalchemy.create_engine(POSTGRES_URL, pool_size=5, max_overflow=10)

@st.cache_data(ttl=2)
def load_snapshots() -> pd.DataFrame:
    try:
        engine = get_db_engine()
        df = pd.read_sql_query("SELECT * FROM agent_snapshots ORDER BY timestamp ASC", engine)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=2)
def load_trades() -> pd.DataFrame:
    try:
        engine = get_db_engine()
        df = pd.read_sql_query("SELECT * FROM trade_logs ORDER BY id DESC", engine)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=2)
def load_holdings() -> pd.DataFrame:
    try:
        engine = get_db_engine()
        df = pd.read_sql_query("SELECT * FROM agent_holdings WHERE amount > 0", engine)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_macro_regime() -> pd.DataFrame:
    try:
        engine = get_db_engine()
        return pd.read_sql_query("SELECT * FROM macro_regime ORDER BY id DESC LIMIT 1", engine)
    except Exception:
        return pd.DataFrame()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================

st.sidebar.markdown(
    "<h1 style='color:#FFFFFF !important; font-size:1.7rem; margin-bottom:0; "
    "background: linear-gradient(90deg, #5EEAD4, #818CF8); -webkit-background-clip: text; "
    "-webkit-text-fill-color: transparent;'>⚡ EvoQuant-AI</h1>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"<span style='color:{TEXT_MUTED} !important; font-size:0.85rem;'>"
    f"<b style='color:{ACCENT} !important;'>Engine:</b> Llama-70B + Risk Parity</span>",
    unsafe_allow_html=True
)
st.sidebar.markdown("<hr class='app-divider' style='margin:16px 0;'>", unsafe_allow_html=True)

auto_refresh = st.sidebar.checkbox("Auto-Refresh Terminal (5s)", value=True)
if auto_refresh:
    st.sidebar.markdown(
        f"<span class='pill'>● Live</span> "
        f"<span style='color:{TEXT_MUTED}; font-size:0.8rem;'>Syncing with <code>TimescaleDB</code></span>",
        unsafe_allow_html=True
    )

if st.sidebar.button("🔄 Manual Refresh", use_container_width=True):
    st.rerun()

st.sidebar.markdown("<hr class='app-divider' style='margin:20px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h3 style='color:{ACCENT} !important; font-size:1rem;'>🛡️ Active Guardrails</h3>", unsafe_allow_html=True)

# Updated Max Position Cap display to 5.0% to match 100-stock universe configuration
st.sidebar.markdown(f"""
<div class="guardrail-card">
    <div class="guardrail-item"><span>Hard Stop-Loss</span> <span class="guardrail-value">-2.5%</span></div>
    <div class="guardrail-item"><span>Hard Take-Profit</span> <span class="guardrail-value">+5.0%</span></div>
    <div class="guardrail-item"><span>Slippage Friction</span> <span class="guardrail-value">ADV Impact</span></div>
    <div class="guardrail-item"><span>Max Single Pos Cap</span> <span class="guardrail-value">5.0%</span></div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. MAIN DASHBOARD LAYOUT
# ==========================================

st.markdown("""
<div class="hero-banner">
    <p class="hero-title">📊 Evolutionary Swarm Analytics</p>
    <p class="hero-sub">Real-time Darwinian performance tracking, convex risk parity allocations, and execution risk audits.</p>
</div>
""", unsafe_allow_html=True)

df_snapshots = load_snapshots()
df_trades = load_trades()
df_holdings = load_holdings()
df_macro = load_macro_regime()

if df_snapshots.empty:
    st.warning("⏳ Waiting for `TimescaleDB` telemetry. Ensure `swarm_consumer.py` has processed at least one market tick!")
    st.stop()

# --- TOP KPI METRICS ---
latest_snapshots = df_snapshots.sort_values('timestamp').groupby('agent_id').last().reset_index()
total_capital = latest_snapshots['equity'].sum()
top_agent = latest_snapshots.sort_values('equity', ascending=False).iloc[0]

risk_events_count = 0
if not df_trades.empty and "reason" in df_trades.columns:
    risk_events_count = len(df_trades[df_trades["reason"].str.contains("HARD_STOP|HARD_TAKE", na=False)])

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Swarm Capital", f"${total_capital:,.2f}")
kpi2.metric("Leader Agent", top_agent['agent_id'])
kpi3.metric("Leader Equity", f"${top_agent['equity']:,.2f}", f"{top_agent['pnl_pct']:+.2f}%")
kpi4.metric("Risk Guard Triggers", f"{risk_events_count} Events", delta_color="inverse")

if not df_macro.empty:
    macro_mult = df_macro['risk_multiplier'].iloc[0]
    kpi5.metric("Macro Risk Scale", f"{macro_mult:.2f}x")
else:
    kpi5.metric("Macro Risk Scale", "1.00x")

if not df_macro.empty:
    st.info(f"📰 **Latest News RAG Reasoning:** {df_macro['summary_reasoning'].iloc[0]}")

st.markdown("<hr class='app-divider'>", unsafe_allow_html=True)

# --- ROW 1: EQUITY CURVES & LIVE LEADERBOARD ---
col_chart, col_leaderboard = st.columns([3, 2])

with col_chart:
    st.markdown("<p class='section-title'>📈 Real-Time Agent Equity Growth</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Equity trajectory across all live swarm agents</p>", unsafe_allow_html=True)

    fig_equity = px.line(
        df_snapshots,
        x="timestamp",
        y="equity",
        color="agent_id",
        color_discrete_map=AGENT_COLORS,
        labels={"equity": "Equity ($)", "timestamp": "Timestamp", "agent_id": "Agent"}
    )

    fig_equity.update_traces(line=dict(width=2.75), marker=dict(size=6))
    fig_equity.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_PANEL_ALT,
        font=PLOT_FONT,
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        xaxis=dict(gridcolor=BORDER, showline=True, linecolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor=BORDER, showline=True, linecolor=BORDER, tickprefix="$", zeroline=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_PRIMARY, size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(bgcolor=BG_PANEL, font_color=TEXT_PRIMARY, bordercolor=BORDER)
    )
    st.plotly_chart(fig_equity, use_container_width=True)

with col_leaderboard:
    st.markdown("<p class='section-title'>🏆 Swarm Leaderboard</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Ranked by total equity</p>", unsafe_allow_html=True)

    leaderboard_df = latest_snapshots[['agent_id', 'equity', 'cash', 'pnl_pct']].sort_values('equity', ascending=False)

    st.dataframe(
        leaderboard_df,
        column_config={
            "agent_id": st.column_config.TextColumn("Agent ID"),
            "equity": st.column_config.NumberColumn("Total Equity", format="$%.2f"),
            "cash": st.column_config.NumberColumn("Available Cash", format="$%.2f"),
            "pnl_pct": st.column_config.NumberColumn("PnL %", format="%+.2f%%"),
        },
        use_container_width=True,
        hide_index=True,
        height=350
    )

st.markdown("<hr class='app-divider'>", unsafe_allow_html=True)

# --- ROW 2: AGENT PORTFOLIO INSPECTOR ---
st.markdown("<p class='section-title'>🔍 Agent Position & Allocation Inspector</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Drill into any agent's live book</p>", unsafe_allow_html=True)

agents_list = latest_snapshots['agent_id'].unique().tolist()
selected_agent = st.selectbox("Select Agent Persona to Inspect:", agents_list)

col_positions, col_pie = st.columns([3, 2])

if not df_holdings.empty:
    agent_pos = df_holdings[df_holdings['agent_id'] == selected_agent].copy()

    if not agent_pos.empty:
        if 'entry_price' in agent_pos.columns and 'amount' in agent_pos.columns:
            agent_pos['pos_value'] = agent_pos['amount'] * agent_pos['entry_price']
        else:
            agent_pos['pos_value'] = agent_pos['amount']

    with col_positions:
        if not agent_pos.empty:
            st.markdown(f"**Active Stock Positions for `{selected_agent}`:**")
            st.dataframe(
                agent_pos[['ticker', 'amount', 'entry_price', 'pos_value']],
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker"),
                    "amount": st.column_config.NumberColumn("Shares Held", format="%.2f"),
                    "entry_price": st.column_config.NumberColumn("Avg Entry Price", format="$%.2f"),
                    "pos_value": st.column_config.NumberColumn("Position Value", format="$%.2f"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info(f"💡 `{selected_agent}` is currently holding **100% Cash**.")

    with col_pie:
        if not agent_pos.empty and (agent_pos['pos_value'] > 0).any():
            pie_df = agent_pos[agent_pos['pos_value'] > 0].copy()

            fig_donut = px.pie(
                pie_df,
                names="ticker",
                values="pos_value",
                hole=0.55,
                color_discrete_sequence=[ACCENT, ACCENT_2, ACCENT_WARN, "#F472B6", ACCENT_UP, "#60A5FA"]
            )

            fig_donut.update_traces(
                textposition='inside',
                textinfo='percent+label',
                insidetextfont=dict(color='#0B0E14', size=11, family="Inter, sans-serif"),
                marker=dict(line=dict(color=BG_PANEL_ALT, width=3))
            )

            fig_donut.update_layout(
                title=dict(
                    text=f"Allocation Breakdown ({selected_agent})",
                    font=dict(color=TEXT_PRIMARY, size=14, family="Inter, sans-serif")
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(
                    font=dict(color=TEXT_PRIMARY, size=12),
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                ),
                height=360,
                margin=dict(l=10, r=10, t=40, b=10),
                hoverlabel=dict(bgcolor=BG_PANEL, font_color=TEXT_PRIMARY, bordercolor=BORDER)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
else:
    st.info(f"💡 `{selected_agent}` has no active stock positions registered across the swarm.")

st.markdown("<hr class='app-divider'>", unsafe_allow_html=True)

# --- ROW 3: MASTER TRADE AUDIT LOG ---
st.markdown("<p class='section-title'>📜 Master Execution Trade Audit Log</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Full execution history across the swarm, filterable by agent, action, and reason</p>", unsafe_allow_html=True)

if not df_trades.empty:
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        selected_agents_filter = st.multiselect(
            "Filter Agents:", options=df_trades['agent_id'].unique(), default=df_trades['agent_id'].unique()
        )
    with f_col2:
        selected_actions_filter = st.multiselect(
            "Filter Order Type:", options=df_trades['action'].unique(), default=df_trades['action'].unique()
        )
    with f_col3:
        reasons_list = df_trades['reason'].unique() if "reason" in df_trades.columns else ["ALLOCATION"]
        selected_reasons_filter = st.multiselect(
            "Filter Execution Reason:", options=reasons_list, default=reasons_list
        )

    filtered_df = df_trades[
        (df_trades['agent_id'].isin(selected_agents_filter)) &
        (df_trades['action'].isin(selected_actions_filter))
    ].copy()

    if "reason" in df_trades.columns:
        filtered_df = filtered_df[filtered_df['reason'].isin(selected_reasons_filter)]

    cols_to_show = ['timestamp', 'agent_id', 'action', 'ticker', 'shares', 'price', 'allocation_pct']
    if "reason" in df_trades.columns:
        cols_to_show.append('reason')

    display_df = filtered_df[cols_to_show].copy()
    display_df['allocation_pct'] = display_df['allocation_pct'] * 100.0

    st.dataframe(
        display_df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
            "agent_id": st.column_config.TextColumn("Agent"),
            "action": st.column_config.TextColumn("Action"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "shares": st.column_config.NumberColumn("Shares", format="%.2f"),
            "price": st.column_config.NumberColumn("Exec Price", format="$%.2f"),
            "allocation_pct": st.column_config.NumberColumn("Alloc %", format="%.1f%%"),
            "reason": st.column_config.TextColumn("Execution Reason"),
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No trade logs recorded in `TimescaleDB` yet.")

# --- 5-SECOND AUTO-REFRESH EXECUTION LOOP ---
if auto_refresh:
    time.sleep(5)
    st.rerun()
