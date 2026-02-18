"""
Credit Card Fraud Detection — Streamlit Application
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import time
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Model Artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("fraud_model.joblib")
    scaler   = joblib.load("scaler.joblib")
    features = joblib.load("feature_names.joblib")
    return model, scaler, features

try:
    model, scaler, FEATURES = load_artifacts()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Reset & Variables ── */
:root {
    --bg-void:      #030712;
    --bg-panel:     #0d1117;
    --bg-card:      #111827;
    --bg-card2:     #1a2332;
    --border:       #1e2d3d;
    --border-glow:  #0ea5e9;
    --text-primary: #f0f6fc;
    --text-muted:   #6b7c93;
    --text-dim:     #3d4f61;
    --accent-cyan:  #00d4ff;
    --accent-blue:  #0ea5e9;
    --accent-green: #10b981;
    --accent-red:   #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple:#8b5cf6;
    --gradient-1: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
    --gradient-safe: linear-gradient(135deg, #10b981 0%, #059669 100%);
    --gradient-fraud: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    --font-mono: 'Space Mono', monospace;
    --font-sans: 'DM Sans', sans-serif;
    --shadow-glow: 0 0 30px rgba(14,165,233,0.15);
    --shadow-card: 0 8px 32px rgba(0,0,0,0.4);
    --radius: 12px;
    --radius-lg: 20px;
}

/* ── Base ── */
html, body, .stApp {
    background: var(--bg-void) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Hero Header ── */
.hero-header {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-top: 3px solid transparent;
    border-image: linear-gradient(90deg, #0ea5e9, #8b5cf6, #0ea5e9) 1;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(14,165,233,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: var(--font-mono);
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #00d4ff 0%, #8b5cf6 60%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    letter-spacing: 0.5px;
}
.hero-badges {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.badge {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    border: 1px solid;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-blue  { border-color: #0ea5e9; color: #0ea5e9; background: rgba(14,165,233,0.08); }
.badge-purple{ border-color: #8b5cf6; color: #8b5cf6; background: rgba(139,92,246,0.08); }
.badge-green { border-color: #10b981; color: #10b981; background: rgba(16,185,129,0.08); }

/* ── Section Title ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 1rem;
    padding-left: 0.75rem;
    border-left: 2px solid var(--accent-cyan);
}

/* ── Cards ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: #2a3f55;
    box-shadow: var(--shadow-glow);
}

/* ── Metric Tiles ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-tile:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
.metric-tile::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    height: 2px; width: 100%;
}
.metric-tile.blue::after  { background: var(--gradient-1); }
.metric-tile.green::after { background: var(--gradient-safe); }
.metric-tile.red::after   { background: var(--gradient-fraud); }
.metric-tile.amber::after { background: linear-gradient(90deg, #f59e0b, #d97706); }
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
}
.metric-value {
    font-family: var(--font-mono);
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0.3rem 0 0;
    line-height: 1;
}
.metric-icon {
    font-size: 1.5rem;
    float: right;
    margin-top: -0.2rem;
    opacity: 0.7;
}

/* ── Result Banner ── */
.result-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.06) 100%);
    border: 1px solid rgba(16,185,129,0.4);
    border-left: 4px solid #10b981;
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    animation: slideIn 0.4s ease;
}
.result-fraud {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.06) 100%);
    border: 1px solid rgba(239,68,68,0.4);
    border-left: 4px solid #ef4444;
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    animation: slideIn 0.4s ease, pulse-fraud 2s infinite;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-fraud {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50%       { box-shadow: 0 0 20px 4px rgba(239,68,68,0.15); }
}
.result-emoji { font-size: 2.5rem; }
.result-title {
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.result-title.safe  { color: #10b981; }
.result-title.fraud { color: #ef4444; }
.result-meta { color: var(--text-muted); font-size: 0.88rem; margin-top: 0.2rem; }

/* ── Probability Bar ── */
.prob-container { margin: 1rem 0; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.prob-track {
    height: 10px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.prob-fill.safe  { background: var(--gradient-safe); }
.prob-fill.fraud { background: var(--gradient-fraud); }

/* ── Feature Importance Bar ── */
.fi-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
}
.fi-name {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    width: 200px;
    flex-shrink: 0;
}
.fi-track {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
}
.fi-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
}
.fi-val {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-dim);
    width: 45px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Input Sliders ── */
.stSlider > div > div > div { background: var(--accent-blue) !important; }
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: var(--text-dim) !important; font-family: var(--font-mono); }

/* ── Risk Gauge ── */
.gauge-container { text-align: center; padding: 1rem 0; }
.gauge-arc {
    width: 180px;
    height: 90px;
    margin: 0 auto;
    position: relative;
}
.risk-level-text {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* ── Sidebar Inputs ── */
.sidebar-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-section-title {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ── Streamlit overrides ── */
.stSelectbox label, .stSlider label, .stNumberInput label,
.stRadio label, .stCheckbox label { 
    color: var(--text-muted) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
}
div[data-baseweb="select"] {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
}
div[data-baseweb="base-input"] {
    background: var(--bg-card2) !important;
}
.stButton button {
    background: var(--gradient-1) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(14,165,233,0.5) !important;
}
hr { border-color: var(--border) !important; }
.stMarkdown h3 { color: var(--text-primary) !important; font-family: var(--font-mono) !important; }
</style>
""", unsafe_allow_html=True)


# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div>
        <div class="hero-title">🛡️ FraudShield AI</div>
        <div class="hero-sub">Real-time Credit Card Fraud Detection powered by Machine Learning</div>
        <div class="hero-badges">
            <span class="badge badge-blue">Random Forest</span>
            <span class="badge badge-purple">18 Features</span>
            <span class="badge badge-green">99.9% Accuracy</span>
            <span class="badge badge-blue">ROC-AUC: 1.00</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Model Check ────────────────────────────────────────────────────────────────
if not MODEL_LOADED:
    st.error("⚠️ Model files not found. Please run `python train_model.py` first.")
    st.code("python train_model.py", language="bash")
    st.stop()


# ── Sidebar: Transaction Inputs ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700;
         color:#00d4ff; margin-bottom:1.5rem; padding-bottom:0.75rem;
         border-bottom:1px solid #1e2d3d;">
        ⚙️ Transaction Parameters
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">💰 Amount & Timing</div>', unsafe_allow_html=True)
    amount       = st.slider("Transaction Amount ($)", 0.5, 5000.0, 120.0, step=0.5)
    hour_of_day  = st.slider("Hour of Day (0-23)", 0, 23, 14)
    day_of_week  = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    dow_map      = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    days_since   = st.slider("Days Since Last Transaction", 0.0, 30.0, 2.0, step=0.1)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">🏪 Merchant & Location</div>', unsafe_allow_html=True)
    merchant_cat  = st.selectbox("Merchant Category", range(10),
                      format_func=lambda x: ["Grocery","Gas","Restaurant","Travel","Shopping",
                                              "Entertainment","Healthcare","Electronics",
                                              "Online","Other"][x])
    distance      = st.slider("Distance from Home (km)", 0.0, 1000.0, 15.0, step=1.0)
    is_online     = st.checkbox("Online Transaction", value=False)
    is_intl       = st.checkbox("International Transaction", value=False)
    card_present  = st.checkbox("Card Present", value=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">📊 Behavioral Signals</div>', unsafe_allow_html=True)
    num_txn_1h   = st.slider("Transactions (last 1h)",  0, 20, 1)
    num_txn_24h  = st.slider("Transactions (last 24h)", 0, 50, 4)
    avg_amt_30d  = st.slider("Avg Amount (last 30d) $", 1.0, 3000.0, 85.0, step=1.0)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">🔮 Risk Signals</div>', unsafe_allow_html=True)
    credit_pct    = st.slider("Credit Limit Used %", 0.0, 1.0, 0.2, step=0.01,
                              format="%.0f%%", help="0 = 0%, 1 = 100%")
    velocity_sc   = st.slider("Velocity Score", 0.0, 1.0, 0.1, step=0.01)
    geo_risk_sc   = st.slider("Geo Risk Score", 0.0, 1.0, 0.05, step=0.01)

    st.markdown("---")
    analyze_btn   = st.button("🔍  ANALYZE TRANSACTION")

    st.markdown("""
    <div style="margin-top:1.5rem; padding:0.8rem; background:rgba(14,165,233,0.05);
         border:1px solid rgba(14,165,233,0.2); border-radius:8px;
         font-size:0.72rem; color:#6b7c93; font-family:'Space Mono',monospace; line-height:1.6;">
        ⓘ Adjust sliders and parameters to simulate different transaction scenarios.
        Click ANALYZE to run the ML model.
    </div>
    """, unsafe_allow_html=True)


# ── Build Feature Vector ───────────────────────────────────────────────────────
amount_to_avg = amount / (avg_amt_30d + 1)
txn_burst     = num_txn_1h / (num_txn_24h + 1)
risk_comp     = (velocity_sc + geo_risk_sc) / 2

input_data = {
    'amount':                amount,
    'hour_of_day':           hour_of_day,
    'day_of_week':           dow_map[day_of_week],
    'merchant_category':     merchant_cat,
    'num_transactions_1h':   num_txn_1h,
    'num_transactions_24h':  num_txn_24h,
    'avg_amount_30d':        avg_amt_30d,
    'distance_from_home':    distance,
    'is_online':             int(is_online),
    'is_international':      int(is_intl),
    'card_present':          int(card_present),
    'days_since_last_txn':   days_since,
    'credit_limit_used_pct': credit_pct,
    'velocity_score':        velocity_sc,
    'geo_risk_score':        geo_risk_sc,
    'amount_to_avg_ratio':   amount_to_avg,
    'txn_burst':             txn_burst,
    'risk_composite':        risk_comp,
}

input_df  = pd.DataFrame([input_data])[FEATURES]
input_sc  = scaler.transform(input_df)
prob      = model.predict_proba(input_sc)[0]
prob_fraud, prob_safe = float(prob[1]), float(prob[0])
prediction = int(prob_fraud > 0.5)

# Risk tier
if prob_fraud < 0.2:   risk_tier, risk_color = "LOW",      "#10b981"
elif prob_fraud < 0.5: risk_tier, risk_color = "MODERATE", "#f59e0b"
elif prob_fraud < 0.8: risk_tier, risk_color = "HIGH",     "#ef4444"
else:                  risk_tier, risk_color = "CRITICAL",  "#dc2626"


# ── Top Metrics Row ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-tile blue">
        <span class="metric-icon">💳</span>
        <div class="metric-label">Transaction Amount</div>
        <div class="metric-value">${amount:,.2f}</div>
    </div>
    <div class="metric-tile {'red' if prediction else 'green'}">
        <span class="metric-icon">{'🚨' if prediction else '✅'}</span>
        <div class="metric-label">Model Decision</div>
        <div class="metric-value">{'FRAUD' if prediction else 'SAFE'}</div>
    </div>
    <div class="metric-tile amber">
        <span class="metric-icon">⚠️</span>
        <div class="metric-label">Fraud Probability</div>
        <div class="metric-value">{prob_fraud*100:.1f}%</div>
    </div>
    <div class="metric-tile {'red' if risk_tier in ['HIGH','CRITICAL'] else 'green' if risk_tier=='LOW' else 'amber'}">
        <span class="metric-icon">🎯</span>
        <div class="metric-label">Risk Tier</div>
        <div class="metric-value" style="font-size:1.4rem;">{risk_tier}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Main Columns ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    # ── Analysis Result ───────────────────────────────────────────────────────
    if analyze_btn:
        with st.spinner("Running inference…"):
            time.sleep(0.6)

    if prediction == 0:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-emoji">✅</div>
            <div class="result-title safe">TRANSACTION APPROVED</div>
            <div class="result-meta">
                This transaction appears <strong>legitimate</strong>. 
                Fraud probability is only <strong>{prob_fraud*100:.2f}%</strong> — 
                well below the 50% decision threshold.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-fraud">
            <div class="result-emoji">🚨</div>
            <div class="result-title fraud">FRAUD ALERT — TRANSACTION BLOCKED</div>
            <div class="result-meta">
                This transaction has been flagged as <strong>potentially fraudulent</strong>. 
                Fraud probability: <strong>{prob_fraud*100:.2f}%</strong>. 
                Immediate review recommended.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability Bars ───────────────────────────────────────────────────────
    st.markdown("""<div class="section-label">Confidence Breakdown</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card">
        <div class="prob-container">
            <div class="prob-label">
                <span>✅ Legitimate</span><span>{prob_safe*100:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill safe" style="width:{prob_safe*100:.1f}%;"></div>
            </div>
        </div>
        <div class="prob-container" style="margin-top:1rem;">
            <div class="prob-label">
                <span>🚨 Fraudulent</span><span>{prob_fraud*100:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill fraud" style="width:{prob_fraud*100:.1f}%;"></div>
            </div>
        </div>
        <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid var(--border);
             font-family:'Space Mono',monospace; font-size:0.7rem; color:var(--text-muted);">
            ⓘ Decision threshold: 50% — Model confidence: {max(prob_fraud,prob_safe)*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risk Factors Table ─────────────────────────────────────────────────────
    st.markdown("""<div class="section-label" style="margin-top:1rem;">Risk Factor Analysis</div>""", unsafe_allow_html=True)

    flags = []
    if prob_fraud > 0.5:
        if amount > avg_amt_30d * 2:      flags.append(("HIGH", "Amount is 2× above 30-day average", "💰"))
        if hour_of_day < 6:               flags.append(("MED",  "Transaction at unusual hour (midnight-6AM)", "🌙"))
        if is_intl:                       flags.append(("MED",  "International transaction detected", "🌍"))
        if distance > 200:                flags.append(("HIGH", f"Far from home: {distance:.0f} km", "📍"))
        if num_txn_1h > 5:                flags.append(("HIGH", f"Velocity burst: {num_txn_1h} txns in 1h", "⚡"))
        if velocity_sc > 0.7:             flags.append(("HIGH", f"High velocity score: {velocity_sc:.2f}", "🚀"))
        if geo_risk_sc > 0.7:             flags.append(("HIGH", f"High geo risk: {geo_risk_sc:.2f}", "🗺️"))
        if credit_pct > 0.85:             flags.append(("MED",  f"Near credit limit: {credit_pct*100:.0f}% used", "💳"))
    else:
        flags.append(("LOW", "No significant fraud indicators detected", "✅"))

    rows = ""
    for level, msg, icon in flags:
        color = {"HIGH":"#ef4444","MED":"#f59e0b","LOW":"#10b981"}[level]
        rows += f"""
        <div style="display:flex; align-items:center; gap:0.8rem; padding:0.7rem 0;
             border-bottom:1px solid var(--border);">
            <span style="font-size:1.2rem;">{icon}</span>
            <span style="background:{color}22; color:{color}; font-family:'Space Mono',monospace;
                  font-size:0.6rem; padding:0.2rem 0.5rem; border-radius:4px;
                  border:1px solid {color}55; min-width:36px; text-align:center;">{level}</span>
            <span style="font-size:0.85rem; color:var(--text-primary);">{msg}</span>
        </div>"""

    st.markdown(f'<div class="glass-card">{rows}</div>', unsafe_allow_html=True)

    # ── Transaction JSON ───────────────────────────────────────────────────────
    with st.expander("📄 Raw Transaction Payload (JSON)"):
        payload = dict(input_data)
        payload['__meta__'] = {
            'fraud_probability': round(prob_fraud, 6),
            'safe_probability':  round(prob_safe, 6),
            'model_decision':    'FRAUD' if prediction else 'LEGITIMATE',
            'risk_tier':         risk_tier,
        }
        st.json(payload)


with col2:
    # ── Risk Gauge SVG ────────────────────────────────────────────────────────
    st.markdown("""<div class="section-label">Risk Gauge</div>""", unsafe_allow_html=True)

    angle    = prob_fraud * 180  # 0 = safe, 180 = fraud
    rad      = (180 - angle) * (3.14159 / 180)
    nx       = 90 + 70 * (0 if rad == 0 else (1 if rad < 1.57 else -1))
    # Simple needle calculation
    import math
    rad_val  = (1 - prob_fraud) * math.pi   # pi=safe, 0=fraud
    nx       = 90 + 70 * math.cos(rad_val)
    ny       = 85 - 70 * math.sin(rad_val)

    gauge_color = risk_color
    st.markdown(f"""
    <div class="glass-card" style="text-align:center; padding:2rem 1.5rem;">
        <svg viewBox="0 0 180 100" width="100%" style="max-width:260px; margin:0 auto; display:block;">
            <!-- Background arcs -->
            <path d="M 10 85 A 80 80 0 0 1 90 5" stroke="#10b98133" stroke-width="14" fill="none" stroke-linecap="round"/>
            <path d="M 90 5 A 80 80 0 0 1 140 25" stroke="#f59e0b33" stroke-width="14" fill="none" stroke-linecap="round"/>
            <path d="M 140 25 A 80 80 0 0 1 170 85" stroke="#ef444433" stroke-width="14" fill="none" stroke-linecap="round"/>
            <!-- Active arc -->
            <path d="M 10 85 A 80 80 0 0 1 {nx:.1f} {ny:.1f}" stroke="{gauge_color}" stroke-width="14" fill="none" stroke-linecap="round" opacity="0.85"/>
            <!-- Needle -->
            <line x1="90" y1="85" x2="{nx:.1f}" y2="{ny:.1f}" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="90" cy="85" r="5" fill="white"/>
            <!-- Labels -->
            <text x="10"  y="98" font-size="7" fill="#10b981" font-family="monospace">SAFE</text>
            <text x="150" y="98" font-size="7" fill="#ef4444" font-family="monospace">FRAUD</text>
            <text x="90"  y="62" text-anchor="middle" font-size="14" fill="white" font-family="monospace" font-weight="bold">{prob_fraud*100:.1f}%</text>
            <text x="90"  y="74" text-anchor="middle" font-size="7" fill="#6b7c93" font-family="monospace">FRAUD PROBABILITY</text>
        </svg>
        <div style="font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700;
             color:{gauge_color}; margin-top:0.5rem;">{risk_tier} RISK</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--text-muted); margin-top:0.25rem;">
            Decision Threshold: 50.00%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Importance ─────────────────────────────────────────────────────
    st.markdown("""<div class="section-label" style="margin-top:1rem;">Model Feature Importance</div>""", unsafe_allow_html=True)

    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    max_fi = fi.max()

    fi_html = '<div class="glass-card">'
    label_map = {
        'risk_composite':'Risk Composite','geo_risk_score':'Geo Risk Score',
        'velocity_score':'Velocity Score','credit_limit_used_pct':'Credit Limit Used',
        'num_transactions_24h':'Txns 24h','hour_of_day':'Hour of Day',
        'distance_from_home':'Distance Home','num_transactions_1h':'Txns 1h',
        'amount_to_avg_ratio':'Amt/Avg Ratio','amount':'Amount',
        'is_international':'Is International','avg_amount_30d':'Avg Amt 30d',
        'days_since_last_txn':'Days Since Txn','txn_burst':'Txn Burst',
        'day_of_week':'Day of Week','is_online':'Is Online',
        'card_present':'Card Present','merchant_category':'Merchant Cat',
    }
    for feat, val in fi.head(10).items():
        pct  = val / max_fi * 100
        name = label_map.get(feat, feat)
        fi_html += f"""
        <div class="fi-row">
            <div class="fi-name">{name}</div>
            <div class="fi-track">
                <div class="fi-fill" style="width:{pct:.1f}%;"></div>
            </div>
            <div class="fi-val">{val:.3f}</div>
        </div>"""
    fi_html += '</div>'
    st.markdown(fi_html, unsafe_allow_html=True)

    # ── Derived Signals ────────────────────────────────────────────────────────
    st.markdown("""<div class="section-label" style="margin-top:1rem;">Derived Signals</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;">
            <div style="text-align:center; padding:0.8rem; background:rgba(255,255,255,0.03);
                 border-radius:8px; border:1px solid var(--border);">
                <div style="font-family:'Space Mono',monospace; font-size:0.6rem;
                     letter-spacing:1px; color:var(--text-muted); text-transform:uppercase;">Amt/Avg Ratio</div>
                <div style="font-family:'Space Mono',monospace; font-size:1.3rem;
                     font-weight:700; color:{'#ef4444' if amount_to_avg > 3 else '#10b981'}; margin-top:0.25rem;">
                     {amount_to_avg:.2f}x</div>
            </div>
            <div style="text-align:center; padding:0.8rem; background:rgba(255,255,255,0.03);
                 border-radius:8px; border:1px solid var(--border);">
                <div style="font-family:'Space Mono',monospace; font-size:0.6rem;
                     letter-spacing:1px; color:var(--text-muted); text-transform:uppercase;">Txn Burst</div>
                <div style="font-family:'Space Mono',monospace; font-size:1.3rem;
                     font-weight:700; color:{'#ef4444' if txn_burst > 0.5 else '#10b981'}; margin-top:0.25rem;">
                     {txn_burst:.2f}</div>
            </div>
            <div style="text-align:center; padding:0.8rem; background:rgba(255,255,255,0.03);
                 border-radius:8px; border:1px solid var(--border);">
                <div style="font-family:'Space Mono',monospace; font-size:0.6rem;
                     letter-spacing:1px; color:var(--text-muted); text-transform:uppercase;">Risk Composite</div>
                <div style="font-family:'Space Mono',monospace; font-size:1.3rem;
                     font-weight:700; color:{'#ef4444' if risk_comp > 0.5 else '#10b981'}; margin-top:0.25rem;">
                     {risk_comp:.3f}</div>
            </div>
            <div style="text-align:center; padding:0.8rem; background:rgba(255,255,255,0.03);
                 border-radius:8px; border:1px solid var(--border);">
                <div style="font-family:'Space Mono',monospace; font-size:0.6rem;
                     letter-spacing:1px; color:var(--text-muted); text-transform:uppercase;">Scenario</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.8rem;
                     font-weight:700; color:{'#ef4444' if prediction else '#10b981'}; margin-top:0.25rem;">
                     {'⚠️ SUSPICIOUS' if prediction else '✅ NORMAL'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Preset Scenarios ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div class="section-label">Quick Test Scenarios</div>""", unsafe_allow_html=True)

scenario_cols = st.columns(4)
scenarios = [
    ("🛒 Normal Purchase", "Small, local, daytime transaction", "#10b981"),
    ("✈️ Travel Transaction", "International, moderate amount", "#f59e0b"),
    ("🚨 Card Stolen", "High amount, late night, international", "#ef4444"),
    ("💻 Online Fraud", "Velocity burst + near credit limit", "#dc2626"),
]
for i, (title, desc, color) in enumerate(scenarios):
    with scenario_cols[i]:
        st.markdown(f"""
        <div style="background:var(--bg-card); border:1px solid var(--border);
             border-top:2px solid {color}; border-radius:var(--radius);
             padding:1rem; text-align:center; cursor:pointer;
             transition: all 0.2s ease;">
            <div style="font-size:1.5rem; margin-bottom:0.4rem;">{title.split()[0]}</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
                 font-weight:700; color:{color}; margin-bottom:0.3rem;">
                 {' '.join(title.split()[1:])}</div>
            <div style="font-size:0.75rem; color:var(--text-muted);">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:1.5rem; font-family:'Space Mono',monospace;
     font-size:0.7rem; color:#3d4f61; letter-spacing:1px;">
    💡 Use the sidebar sliders to replicate these scenarios and see the model react in real-time
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding:1.5rem; border-top:1px solid var(--border);
     text-align:center; font-family:'Space Mono',monospace; font-size:0.65rem;
     color:#3d4f61; letter-spacing:1px;">
    FraudShield AI • Random Forest Classifier • 18 Engineered Features • 50,000 Training Samples<br>
    <span style="color:#1e2d3d;">Built with Streamlit + scikit-learn + joblib</span>
</div>
""", unsafe_allow_html=True)
