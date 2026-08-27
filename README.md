# Salary Bankruptcy Prediction — Streamlit App

A ready-to-run Streamlit app for the trained `VotingClassifier` bankruptcy model.

## Files
- `app.py` — the entire app (single file: Home / Assess a Company / Model & Methodology pages)
- `model.pkl` — your trained VotingClassifier (RandomForest + GaussianNB + SVC + KNN + LogisticRegression)
- `columns.pkl` — the 95 financial-ratio feature names, in training order
- `requirements.txt` — dependencies
- `.streamlit/config.toml` — dark navy/gold theme

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## What's inside

- **Home** — overview, stats, and the 5-model ensemble panel.
- **Assess a Company** — three ways to score a filing:
  - **Manual entry** — all 95 ratios grouped into 8 collapsible categories (Profitability, Liquidity, Leverage & Solvency, Efficiency & Turnover, Cash Flow, Growth, Per-Share Metrics, Other).
  - **Sample profiles** — one click "Healthy" / "Distressed" demo runs.
  - **Upload CSV** — upload a one-row CSV whose header matches the 95 training column names; the app validates columns before scoring.
- **Model & Methodology** — lists the 5 voting estimators and every feature by category.

Every prediction shows the ensemble's majority verdict, a risk gauge (share of the panel that flagged risk), and the individual vote from each of the 5 models — nothing is a black box.

## Notes
- `columns.pkl` entries have a leading space in several names (e.g. `" ROA(C) before interest and depreciation before interest"`) — this is preserved intentionally because the model was trained with that exact column naming; do not strip it if you edit `FEATURE_GROUPS` in `app.py`.
- If you deploy this on Streamlit Community Cloud, just push this whole folder to a GitHub repo and point Streamlit Cloud at `app.py`.
- Title says "Salary Bankruptcy Prediction" per your naming choice — feel free to swap the string in `app.py`'s sidebar/page titles if you settle on different wording later.
