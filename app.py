"""
Salary Bankruptcy Prediction — A Humanoid ML Model Powered by AI
Streamlit app.

Run with:
    streamlit run app.py

Expects model.pkl (sklearn VotingClassifier) and columns.pkl (list of the
95 training feature names) in the same folder as this file.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Page config — set this before anything else renders
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Salary Bankruptcy Prediction",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theming — inject CSS on top of the .streamlit/config.toml dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.01em; }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #C9A94A;
    }
    .mono { font-family: 'IBM Plex Mono', monospace; }

    div[data-testid="stMetric"] {
        background: #121F36;
        border: 1px solid #22304A;
        border-radius: 6px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace; font-size: 11px; }

    .chip {
        display:inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        border: 1px solid #22304A;
        background: #121F36;
        padding: 6px 12px;
        border-radius: 999px;
        color: #B7C0D1;
        margin: 3px 4px 3px 0;
    }

    .verdict-risk {
        display:inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 9px 18px;
        border-radius: 999px;
        background: #3A1E1B;
        color: #E8897C;
        border: 1px solid #C1483A;
    }
    .verdict-safe {
        display:inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 9px 18px;
        border-radius: 999px;
        background: #16261F;
        color: #7FCBA8;
        border: 1px solid #3C8368;
    }

    .disclaimer-box {
        margin-top: 18px;
        padding: 16px 20px;
        border-left: 2px solid #C9A94A;
        background: #121F36;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        color: #7C8AA0;
        line-height: 1.6;
        border-radius: 0 6px 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model + schema (cached — only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "columns.pkl", "rb") as f:
        raw_cols = pickle.load(f)
    columns = [c.strip() for c in raw_cols]
    return model, columns


MODEL, COLUMNS = load_model()
ESTIMATOR_NAMES = list(MODEL.named_estimators_.keys()) if hasattr(MODEL, "named_estimators_") else []

# ---------------------------------------------------------------------------
# Feature groups — every one of the 95 model features, organised by
# financial category (profitability, liquidity, leverage, turnover, cash flow…)
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "Profitability": [
        " ROA(C) before interest and depreciation before interest",
        " ROA(A) before interest and % after tax",
        " ROA(B) before interest and depreciation after tax",
        " Operating Gross Margin",
        " Realized Sales Gross Margin",
        " Operating Profit Rate",
        " Pre-tax net Interest Rate",
        " After-tax net Interest Rate",
        " Non-industry income and expenditure/revenue",
        " Continuous interest rate (after tax)",
        " Operating Expense Rate",
        " Research and development expense rate",
        " Tax rate (A)",
        " Gross Profit to Sales",
        " Net Income to Total Assets",
        " Net Income to Stockholder's Equity",
        " Operating profit/Paid-in capital",
        " Net profit before tax/Paid-in capital",
        " Interest-bearing debt interest rate",
        " Retained Earnings to Total Assets",
    ],
    "Per-Share Metrics": [
        " Net Value Per Share (B)",
        " Net Value Per Share (A)",
        " Net Value Per Share (C)",
        " Persistent EPS in the Last Four Seasons",
        " Cash Flow Per Share",
        " Revenue Per Share (Yuan ¥)",
        " Operating Profit Per Share (Yuan ¥)",
        " Per Share Net profit before tax (Yuan ¥)",
    ],
    "Growth": [
        " Realized Sales Gross Profit Growth Rate",
        " Operating Profit Growth Rate",
        " After-tax Net Profit Growth Rate",
        " Regular Net Profit Growth Rate",
        " Continuous Net Profit Growth Rate",
        " Total Asset Growth Rate",
        " Net Value Growth Rate",
        " Total Asset Return Growth Rate Ratio",
    ],
    "Liquidity": [
        " Current Ratio",
        " Quick Ratio",
        " Cash Reinvestment %",
        " Working Capital to Total Assets",
        " Quick Assets/Total Assets",
        " Current Assets/Total Assets",
        " Cash/Total Assets",
        " Quick Assets/Current Liability",
        " Cash/Current Liability",
        " Current Liability to Assets",
        " No-credit Interval",
    ],
    "Leverage & Solvency": [
        " Total debt/Total net worth",
        " Debt ratio %",
        " Net worth/Assets",
        " Long-term fund suitability ratio (A)",
        " Borrowing dependency",
        " Contingent liabilities/Net worth",
        " Interest Expense Ratio",
        " Current Liabilities/Liability",
        " Current Liability to Liability",
        " Working Capital/Equity",
        " Current Liabilities/Equity",
        " Long-term Liability to Current Assets",
        " Liability to Equity",
        " Equity to Long-term Liability",
        " Equity to Liability",
        " Current Liability to Equity",
        " Current Liability to Current Assets",
        " Liability-Assets Flag",
        " Degree of Financial Leverage (DFL)",
        " Interest Coverage Ratio (Interest expense to EBIT)",
        " Inventory and accounts receivable/Net value",
        " Total assets to GNP price",
    ],
    "Efficiency & Turnover": [
        " Total Asset Turnover",
        " Accounts Receivable Turnover",
        " Average Collection Days",
        " Inventory Turnover Rate (times)",
        " Fixed Assets Turnover Frequency",
        " Net Worth Turnover Rate (times)",
        " Current Asset Turnover Rate",
        " Quick Asset Turnover Rate",
        " Working capitcal Turnover Rate",
        " Cash Turnover Rate",
        " Inventory/Working Capital",
        " Inventory/Current Liability",
        " Fixed Assets to Assets",
    ],
    "Cash Flow": [
        " Cash flow rate",
        " Cash Flow to Sales",
        " Cash Flow to Total Assets",
        " Cash Flow to Liability",
        " CFO to Assets",
        " Cash Flow to Equity",
        " Operating Funds to Liability",
        " Total income/Total expense",
        " Total expense/Assets",
    ],
    "Other Indicators": [
        " Revenue per person",
        " Operating profit per person",
        " Allocation rate per person",
        " Net Income Flag",
    ],
}

# Sanity check: every model column must appear exactly once across groups.
_flat = [c for grp in FEATURE_GROUPS.values() for c in grp]
if sorted(_flat) != sorted(COLUMNS):
    st.error("Feature group config is out of sync with columns.pkl — check FEATURE_GROUPS in app.py.")
    st.stop()

PRESETS = {
    "Healthy profile": {c: 0.55 for c in COLUMNS},
    "Distressed profile": {c: 0.15 for c in COLUMNS},
}
PRESETS["Healthy profile"][" Current Ratio"] = 2.1
PRESETS["Healthy profile"][" Debt ratio %"] = 0.28
PRESETS["Distressed profile"][" Current Ratio"] = 0.6
PRESETS["Distressed profile"][" Debt ratio %"] = 0.81


# ---------------------------------------------------------------------------
# Prediction helper — runs the ensemble + collects each member's individual vote
# ---------------------------------------------------------------------------
def score(values: dict):
    X = pd.DataFrame([values], columns=COLUMNS).astype(float)
    prediction = int(MODEL.predict(X)[0])

    votes = {}
    positive = 0
    for name, est in MODEL.named_estimators_.items():
        try:
            v = int(est.predict(X)[0])
        except Exception:
            v = None
        votes[name] = v
        if v == 1:
            positive += 1

    confidence = round(100 * positive / max(len(votes), 1), 1)
    return prediction, confidence, votes, positive, len(votes)


def gauge_figure(value: float, danger: bool) -> go.Figure:
    accent = "#C1483A" if danger else "#3C8368"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " / 100", "font": {"size": 34, "color": "#F4F1E8", "family": "Fraunces"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#7C8AA0", "tickfont": {"color": "#7C8AA0", "size": 10}},
                "bar": {"color": accent, "thickness": 0.28},
                "bgcolor": "#121F36",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#16261F"},
                    {"range": [40, 70], "color": "#241E14"},
                    {"range": [70, 100], "color": "#3A1E1B"},
                ],
                "threshold": {"line": {"color": "#E4C877", "width": 3}, "thickness": 0.9, "value": value},
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F4F1E8"},
    )
    return fig


def render_result(prediction, confidence, votes, positive, total):
    risky = prediction == 1
    badge_class = "verdict-risk" if risky else "verdict-safe"
    badge_text = "Elevated Bankruptcy Risk" if risky else "Low Bankruptcy Risk"
    st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(gauge_figure(confidence, risky), use_container_width=True)
        st.caption("Gauge reflects the **share of the 5-model panel that voted 'at risk.'**")
    with col2:
        st.markdown("**Panel vote breakdown**")
        for name, v in votes.items():
            tag = "🔴 At risk" if v == 1 else ("🟢 Not at risk" if v == 0 else "⚪️ N/A")
            st.markdown(f"<span class='mono'>{name}</span> — {tag}", unsafe_allow_html=True)
        st.metric("Consensus", f"{positive} / {total} models flagged risk")

    st.markdown(
        """
        <div class="disclaimer-box">
        This tool is an educational / analytical instrument trained on historical financial
        statement ratios. It does not constitute investment, credit, or audit advice, and
        should not be the sole basis for any financial decision.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ◆ SALARY\n**BANKRUPTCY PREDICTION**")
st.sidebar.caption("A Humanoid ML Model Powered by AI")
page = st.sidebar.radio("Navigate", ["Home", "Assess a Company", "Model & Methodology"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(f"<span class='mono' style='font-size:12px;color:#7C8AA0;'>{len(COLUMNS)} ratios · {len(ESTIMATOR_NAMES)} voting models</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------------
if page == "Home":
    st.markdown('<span class="eyebrow">Ensemble Risk Scoring · 5-Model Vote</span>', unsafe_allow_html=True)
    st.title("Read a balance sheet the way five analysts would.")
    st.write(
        f"Salary Bankruptcy Prediction runs **{len(COLUMNS)} audited financial ratios** — "
        "profitability, liquidity, leverage, turnover, and cash flow — through a voting "
        "ensemble of five independent classifiers, and returns a single, defensible read "
        "on solvency risk."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Financial ratios", len(COLUMNS))
    c2.metric("Voting classifiers", len(ESTIMATOR_NAMES))
    c3.metric("Outcomes tracked", 2)

    if st.button("Assess a Company →", type="primary"):
        st.session_state["_nav"] = "Assess a Company"
        st.rerun()

    st.markdown("---")
    st.subheader("How it works")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("**01 — Input**")
        st.write(f"Enter the {len(COLUMNS)} financial indicators by category, upload a CSV of a filing, or load a sample profile.")
    with h2:
        st.markdown("**02 — Vote**")
        st.write("A Random Forest, Naive Bayes, SVC, k-Nearest Neighbors, and Logistic Regression model each cast an independent vote.")
    with h3:
        st.markdown("**03 — Verdict**")
        st.write("The majority vote becomes the headline call, with the full per-model breakdown shown alongside it.")

    st.markdown("---")
    st.subheader("The panel")
    st.write("Voting across model families guards against any one algorithm's blind spot.")
    st.markdown("".join(f'<span class="chip">{e}</span>' for e in ESTIMATOR_NAMES), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: Assess a Company
# ---------------------------------------------------------------------------
elif page == "Assess a Company":
    st.markdown('<span class="eyebrow">Step 1 of 2</span>', unsafe_allow_html=True)
    st.title("Enter the filing.")
    st.write(f"All {len(COLUMNS)} fields accept the raw ratio value used in training. Leave a field at 0 to default it.")

    tab_manual, tab_sample, tab_csv = st.tabs(["✍️ Manual entry", "⚡ Sample profiles", "📄 Upload CSV"])

    # ---- Manual entry ----
    with tab_manual:
        with st.form("manual_form"):
            values = {}
            for group_name, fields in FEATURE_GROUPS.items():
                with st.expander(f"{group_name} ({len(fields)} fields)", expanded=(group_name == "Profitability")):
                    cols = st.columns(3)
                    for i, feat in enumerate(fields):
                        with cols[i % 3]:
                            values[feat] = st.number_input(feat.strip(), value=0.0, format="%.4f", key=f"manual_{feat}")
            submitted = st.form_submit_button("Run Assessment →", type="primary")
        if submitted:
            prediction, confidence, votes, positive, total = score(values)
            st.markdown("### Result")
            render_result(prediction, confidence, votes, positive, total)

    # ---- Sample profiles ----
    with tab_sample:
        st.write("Load an illustrative profile to see the instrument in action without typing all fields.")
        pc1, pc2 = st.columns(2)
        with pc1:
            if st.button("Load 'Healthy' profile"):
                prediction, confidence, votes, positive, total = score(PRESETS["Healthy profile"])
                st.markdown("### Result")
                render_result(prediction, confidence, votes, positive, total)
        with pc2:
            if st.button("Load 'Distressed' profile"):
                prediction, confidence, votes, positive, total = score(PRESETS["Distressed profile"])
                st.markdown("### Result")
                render_result(prediction, confidence, votes, positive, total)

    # ---- CSV upload ----
    with tab_csv:
        st.write(f"Upload a CSV whose header row includes all {len(COLUMNS)} ratio names exactly as used in training. Only the first data row is scored.")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                df.columns = [c.strip() for c in df.columns]
                missing = [c for c in COLUMNS if c not in df.columns]
                if missing:
                    st.error(f"CSV is missing {len(missing)} required column(s), e.g. '{missing[0]}'.")
                else:
                    row = df[COLUMNS].iloc[0].to_dict()
                    st.success("File loaded. Preview of the row being scored:")
                    st.dataframe(df[COLUMNS].iloc[[0]], use_container_width=True)
                    if st.button("Run Assessment on this row →", type="primary"):
                        prediction, confidence, votes, positive, total = score(row)
                        st.markdown("### Result")
                        render_result(prediction, confidence, votes, positive, total)
            except Exception as exc:
                st.error(f"Could not read that CSV: {exc}")

# ---------------------------------------------------------------------------
# PAGE: Model & Methodology
# ---------------------------------------------------------------------------
else:
    st.markdown('<span class="eyebrow">Under the hood</span>', unsafe_allow_html=True)
    st.title("Model & Methodology")

    st.write(
        "The scoring engine is a **scikit-learn `VotingClassifier`** combining five model "
        "families trained on historical corporate financial-statement ratios:"
    )
    for e in ESTIMATOR_NAMES:
        st.markdown(f"- `{e}`")

    st.markdown("---")
    st.subheader("Feature categories")
    rows = [{"Category": g, "# Features": len(f)} for g, f in FEATURE_GROUPS.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("See all feature names by category"):
        for g, fields in FEATURE_GROUPS.items():
            st.markdown(f"**{g}**")
            st.write(", ".join(f.strip() for f in fields))

    st.markdown(
        """
        <div class="disclaimer-box">
        This tool is an educational / analytical instrument trained on historical financial
        statement ratios. It does not constitute investment, credit, or audit advice, and
        should not be the sole basis for any financial decision.
        </div>
        """,
        unsafe_allow_html=True,
    )
