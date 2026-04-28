"""
Synthetic Twin City Finder
──────────────────────────
Purpose : Identify the best "control" country/market for a geo-lift /
          incrementality test by comparing Google Trends demand profiles.

How it works:
  1. Enter a Target Market and up to 8 Comparison Markets in the sidebar.
  2. Choose up to 5 keywords that represent your category.
  3. The app fetches Google Trends "interest by region" scores for each keyword
     (one request per keyword, with a polite delay between requests).
  4. It builds a keyword × market score matrix, normalises it, and computes
     the Pearson correlation between the target and every comparison market.
  5. The market with the highest correlation is declared the "Synthetic Twin."

Why this method?
  • Comparing demand profiles across multiple keywords is more robust than
    a single time-series comparison — it captures broader consumer behaviour.
  • "Interest by region" is the most reliable Google Trends endpoint available
    without an API key.

Tips:
  • Use 3–5 keywords that represent your product category (e.g. for luxury:
    "Luxury Watches", "Designer Bags", "Premium Skincare").
  • If a keyword returns no data, it is skipped — at least 2 keywords must
    succeed for a correlation to be computed.
  • Google Trends throttles automated requests; the app adds a 12 s delay
    between keyword fetches and retries once on failure.
"""

import time
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pytrends.request import TrendReq
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synthetic Twin City Finder",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; }
.stApp { background-color: #F0F2F5; }
[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E4E6EB; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #1877F2; }
[data-testid="metric-container"] {
    background-color: #FFFFFF; border: 1px solid #E4E6EB;
    border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #1C1E21;
    margin-top: 1.5rem; margin-bottom: 0.5rem;
    border-left: 4px solid #1877F2; padding-left: 10px;
}
.twin-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1877F2, #42A5F5);
    color: white; font-size: 1.4rem; font-weight: 700;
    padding: 14px 28px; border-radius: 12px;
    box-shadow: 0 4px 12px rgba(24,119,242,0.35); letter-spacing: 0.5px;
}
.info-box {
    background-color: #E7F3FF; border-left: 4px solid #1877F2;
    border-radius: 6px; padding: 12px 16px;
    font-size: 0.9rem; color: #1C1E21; margin-bottom: 1rem;
}
.warn-box {
    background-color: #FFF3CD; border-left: 4px solid #FFC107;
    border-radius: 6px; padding: 12px 16px;
    font-size: 0.9rem; color: #1C1E21; margin-bottom: 1rem;
}
.method-box {
    background-color: #F0F7FF; border: 1px solid #BDD7FF;
    border-radius: 8px; padding: 14px 18px;
    font-size: 0.88rem; color: #1C1E21; margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TIMEFRAME_OPTIONS = {
    "Last 3 Months": "today 3-m",
    "Last 6 Months": "today 6-m",
    "Last 12 Months": "today 12-m",
    "Last 5 Years": "today 5-y",
}

MARKET_LIST = sorted([
    "Singapore", "Australia", "Japan", "South Korea", "India",
    "Indonesia", "Thailand", "Malaysia", "Philippines", "Vietnam",
    "Hong Kong", "Taiwan", "China", "New Zealand",
    "United Kingdom", "Germany", "France", "Netherlands", "Spain", "Italy",
    "Switzerland", "Sweden", "Norway", "Denmark",
    "United States", "Canada", "Brazil", "Mexico", "Argentina",
    "United Arab Emirates", "Saudi Arabia", "Turkey", "Israel",
    "South Africa", "Nigeria", "Kenya", "Egypt",
])

KEYWORD_PRESETS = {
    "Luxury / Premium": ["Luxury Watches", "Designer Bags", "Premium Skincare", "Fine Dining", "Luxury Cars"],
    "E-Commerce / Retail": ["Online Shopping", "Flash Sale", "Free Shipping", "Buy Online", "Discount Code"],
    "Travel": ["Flight Booking", "Hotel Deals", "Travel Insurance", "Visa Application", "Holiday Package"],
    "Technology": ["Smartphones", "Laptop", "Wireless Earbuds", "Smart TV", "Gaming"],
    "Finance": ["Credit Card", "Investment", "Cryptocurrency", "Insurance", "Personal Loan"],
    "Food & Beverage": ["Coffee", "Food Delivery", "Restaurant", "Bubble Tea", "Fast Food"],
    "Health & Fitness": ["Gym Membership", "Protein Supplement", "Yoga", "Running Shoes", "Diet"],
    "Custom": [],
}

# ── Helper functions ──────────────────────────────────────────────────────────

def normalize_row(row: pd.Series) -> pd.Series:
    mn, mx = row.min(), row.max()
    if mx == mn:
        return pd.Series(np.zeros(len(row)), index=row.index)
    return (row - mn) / (mx - mn)


def fetch_interest_by_region(keyword: str, timeframe: str) -> pd.Series | None:
    """Fetch interest_by_region for a single keyword. Returns a Series (country → score)."""
    pt = TrendReq(hl="en-US", tz=0, timeout=(15, 35), retries=0, backoff_factor=0)
    try:
        pt.build_payload([keyword], timeframe=timeframe, geo="")
        df = pt.interest_by_region(resolution="COUNTRY", inc_low_vol=True, inc_geo_code=False)
        if df.empty or keyword not in df.columns:
            return None
        return df[keyword].rename(keyword)
    except Exception:
        return None


def fetch_all_keywords(
    keywords: list[str],
    timeframe: str,
    progress_bar,
    status_text,
) -> pd.DataFrame:
    """
    Fetch interest_by_region for each keyword.
    Returns a DataFrame: rows = keywords, columns = countries.
    """
    results: dict[str, pd.Series] = {}
    n = len(keywords)

    for i, kw in enumerate(keywords):
        status_text.markdown(f"📡 Fetching keyword **{i+1}/{n}**: `{kw}`…")
        series = fetch_interest_by_region(kw, timeframe)

        if series is None:
            # Retry once after a longer wait
            status_text.markdown(f"⏳ Rate-limited on `{kw}` — waiting 20 s then retrying…")
            time.sleep(20)
            series = fetch_interest_by_region(kw, timeframe)

        if series is not None:
            results[kw] = series
        else:
            status_text.markdown(f"⚠️ Skipping `{kw}` — no data after retry.")

        progress_bar.progress((i + 1) / n)
        if i < n - 1:
            time.sleep(12)

    status_text.markdown("✅ Fetch complete.")
    if not results:
        return pd.DataFrame()

    # Build matrix: rows = keywords, columns = countries
    return pd.DataFrame(results).T  # shape: (keywords × countries)


def compute_correlations(
    score_matrix: pd.DataFrame,
    target: str,
    comparisons: list[str],
) -> pd.DataFrame:
    """
    Correlate the target market's keyword-score vector against each comparison.
    score_matrix: rows = keywords, columns = markets.
    """
    if target not in score_matrix.columns:
        return pd.DataFrame()

    norm = score_matrix.apply(normalize_row, axis=1)  # normalise per keyword
    target_vec = norm[target].dropna()

    rows = []
    for market in comparisons:
        if market not in norm.columns:
            rows.append({"Market": market, "Correlation (r)": None,
                         "Match Score (%)": None, "Keywords Used": 0,
                         "Status": "⚠️ Not in data"})
            continue
        comp_vec = norm[market].dropna()
        common = target_vec.index.intersection(comp_vec.index)
        if len(common) < 2:
            rows.append({"Market": market, "Correlation (r)": None,
                         "Match Score (%)": None, "Keywords Used": len(common),
                         "Status": "⚠️ Insufficient data"})
            continue
        try:
            r, _ = pearsonr(target_vec[common], comp_vec[common])
        except Exception:
            r = float("nan")
        score = round(max(r, 0) * 100, 1) if not np.isnan(r) else None
        rows.append({
            "Market": market,
            "Correlation (r)": round(r, 4) if not np.isnan(r) else None,
            "Match Score (%)": score,
            "Keywords Used": len(common),
            "Status": (
                "✅ Strong" if (score or 0) >= 80 else
                ("⚠️ Moderate" if (score or 0) >= 50 else "❌ Weak")
            ) if score is not None else "⚠️ No correlation",
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Correlation (r)", ascending=False, na_position="last")
        .reset_index(drop=True)
        .pipe(lambda d: d.assign(Rank=range(1, len(d) + 1)).set_index("Rank"))
    )


def score_color(val):
    if pd.isna(val):
        return "background-color: #F0F2F5"
    if val >= 80:
        return "background-color: #D4EDDA; color: #155724; font-weight:700"
    if val >= 50:
        return "background-color: #FFF3CD; color: #856404; font-weight:700"
    return "background-color: #F8D7DA; color: #721C24; font-weight:700"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌏 Synthetic Twin\nCity Finder")
    st.caption("Geo-Lift Test Control Market Selector")
    st.divider()

    st.subheader("🎯 Target Market")
    target_market = st.selectbox(
        "Target Market", MARKET_LIST,
        index=MARKET_LIST.index("Singapore"),
        label_visibility="collapsed",
    )

    st.subheader("🗺️ Comparison Markets")
    st.caption("Select up to 8 markets to compare against.")
    default_comp = ["Australia", "Japan", "South Korea", "India", "Thailand"]
    comparison_markets = st.multiselect(
        "Comparison Markets",
        options=[m for m in MARKET_LIST if m != target_market],
        default=[m for m in default_comp if m != target_market],
        label_visibility="collapsed",
        max_selections=8,
    )

    st.subheader("🔍 Keywords / Category")
    preset_choice = st.selectbox("Category preset", list(KEYWORD_PRESETS.keys()), index=0)

    if preset_choice == "Custom":
        kw_raw = st.text_area(
            "Enter keywords (one per line, max 5)",
            value="Luxury Watches\nDesigner Bags\nPremium Skincare",
            height=130,
        )
        keywords = [k.strip() for k in kw_raw.strip().splitlines() if k.strip()][:5]
    else:
        preset_kws = KEYWORD_PRESETS[preset_choice]
        keywords = st.multiselect(
            "Select keywords (max 5)",
            options=preset_kws,
            default=preset_kws[:3],
            max_selections=5,
        )

    st.subheader("📅 Timeframe")
    timeframe_label = st.selectbox("Period", list(TIMEFRAME_OPTIONS.keys()), index=2)
    timeframe = TIMEFRAME_OPTIONS[timeframe_label]

    st.divider()
    n_kw = len(keywords)
    est_wait = n_kw * 12 + n_kw * 5
    st.caption(
        f"ℹ️ {n_kw} keyword(s) × ~12 s delay = **~{est_wait} s** estimated wait."
    )
    run_btn = st.button("🚀 Find Synthetic Twin", use_container_width=True, type="primary")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("## 🌏 Synthetic Twin City Finder")

st.markdown(
    '<div class="info-box">'
    "<b>What is a Synthetic Twin?</b> A market whose consumer demand profile closely "
    "mirrors your target market — making it an ideal <em>control group</em> for a "
    "geo-lift / incrementality test. High correlation (r ≥ 0.80) = strong candidate."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="method-box">'
    "<b>📐 Methodology:</b> This tool fetches Google Trends <em>interest by region</em> "
    "scores for each keyword you select, builds a keyword × market demand matrix, "
    "normalises it, and computes <b>Pearson correlation</b> between your target market "
    "and each comparison market. Comparing across multiple keywords captures broader "
    "consumer behaviour patterns — more robust than a single time-series comparison."
    "</div>",
    unsafe_allow_html=True,
)

if not run_btn:
    st.markdown(
        '<div class="warn-box">👈 Configure your inputs in the sidebar and click '
        "<b>Find Synthetic Twin</b> to begin.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Validation ────────────────────────────────────────────────────────────────
if not comparison_markets:
    st.error("Please select at least one Comparison Market.")
    st.stop()
if not keywords:
    st.error("Please select or enter at least one keyword.")
    st.stop()

# ── Fetch ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📡 Fetching Google Trends Data</p>', unsafe_allow_html=True)
progress_bar = st.progress(0)
status_text  = st.empty()

score_matrix = fetch_all_keywords(keywords, timeframe, progress_bar, status_text)

if score_matrix.empty:
    st.error(
        "No data returned from Google Trends.\n\n"
        "**Possible causes:**\n"
        "- All keywords returned empty responses (try broader terms)\n"
        "- Google Trends rate limit — wait 60 s and retry\n"
    )
    st.stop()

# ── Check markets ─────────────────────────────────────────────────────────────
available_markets = [m for m in score_matrix.columns if m in [target_market] + comparison_markets]
missing_markets   = [m for m in [target_market] + comparison_markets if m not in score_matrix.columns]

if missing_markets:
    st.warning(f"⚠️ No search data for: **{', '.join(missing_markets)}** — excluded from analysis.")

if target_market not in score_matrix.columns:
    st.error(
        f"Target market **{target_market}** has no Google Trends data for the selected keywords. "
        "Try different keywords or a broader timeframe."
    )
    st.stop()

comp_available = [m for m in comparison_markets if m in score_matrix.columns]
if not comp_available:
    st.error("None of the comparison markets have data. Try different markets or keywords.")
    st.stop()

# ── Correlations ──────────────────────────────────────────────────────────────
corr_df  = compute_correlations(score_matrix, target_market, comp_available)
valid    = corr_df.dropna(subset=["Correlation (r)"])

if valid.empty:
    st.error("Could not compute correlations — insufficient data. Try adding more keywords.")
    st.stop()

best_row   = valid.iloc[0]
twin       = best_row["Market"]
twin_r     = best_row["Correlation (r)"]
twin_score = best_row["Match Score (%)"]

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Summary</p>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🎯 Target Market",   target_market)
c2.metric("🏆 Synthetic Twin",  twin)
c3.metric("📈 Correlation (r)", f"{twin_r:.4f}")
c4.metric("🎯 Match Score",     f"{twin_score}%")
c5.metric("🔍 Keywords Used",   f"{len(score_matrix)}")

st.markdown(
    f'<div style="text-align:center;margin:1.5rem 0;">'
    f'<div class="twin-badge">🏆 Synthetic Twin: {twin} &nbsp;|&nbsp; Match Score: {twin_score}%</div>'
    f"</div>",
    unsafe_allow_html=True,
)

# ── Radar / Spider chart: demand profile comparison ───────────────────────────
st.markdown('<p class="section-header">🕸️ Demand Profile — Target vs Synthetic Twin</p>', unsafe_allow_html=True)

norm_matrix = score_matrix.apply(normalize_row, axis=1)
kw_labels   = list(norm_matrix.index)

if target_market in norm_matrix.columns and twin in norm_matrix.columns:
    target_vals = norm_matrix[target_market].fillna(0).tolist()
    twin_vals   = norm_matrix[twin].fillna(0).tolist()

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=target_vals + [target_vals[0]],
        theta=kw_labels + [kw_labels[0]],
        fill="toself",
        name=f"🎯 {target_market} (Target)",
        line=dict(color="#1877F2", width=2.5),
        fillcolor="rgba(24,119,242,0.15)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=twin_vals + [twin_vals[0]],
        theta=kw_labels + [kw_labels[0]],
        fill="toself",
        name=f"🏆 {twin} (Twin)",
        line=dict(color="#F5A623", width=2.5, dash="dot"),
        fillcolor="rgba(245,166,35,0.15)",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=420,
        margin=dict(l=40, r=40, t=40, b=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ── Heatmap: all markets × keywords ──────────────────────────────────────────
with st.expander("🗺️ Full Demand Heatmap — All Markets × Keywords"):
    all_markets_in_matrix = [target_market] + [m for m in comp_available if m != target_market]
    heat_data = norm_matrix[
        [m for m in all_markets_in_matrix if m in norm_matrix.columns]
    ].fillna(0)

    fig_heat = go.Figure(go.Heatmap(
        z=heat_data.values,
        x=list(heat_data.columns),
        y=list(heat_data.index),
        colorscale="Blues",
        text=np.round(heat_data.values, 2),
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> in <b>%{x}</b><br>Normalised score: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Score"),
    ))
    fig_heat.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Helvetica, Arial", size=12),
        xaxis=dict(title="Market", tickangle=-30),
        yaxis=dict(title="Keyword"),
        height=max(300, len(kw_labels) * 60 + 100),
        margin=dict(l=20, r=20, t=30, b=60),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Bar chart: Match Scores ───────────────────────────────────────────────────
st.markdown('<p class="section-header">🏅 Match Score by Market</p>', unsafe_allow_html=True)
valid_corr = corr_df.dropna(subset=["Match Score (%)"])
if not valid_corr.empty:
    bar_colors = [
        "#1877F2" if m == twin else
        ("#42A5F5" if s >= 80 else ("#FFC107" if s >= 50 else "#EF5350"))
        for m, s in zip(valid_corr["Market"], valid_corr["Match Score (%)"])
    ]
    fig_bar = go.Figure(go.Bar(
        x=valid_corr["Market"], y=valid_corr["Match Score (%)"],
        marker_color=bar_colors,
        text=[f"{s}%" for s in valid_corr["Match Score (%)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Match Score: %{y}%<extra></extra>",
    ))
    fig_bar.add_hline(y=80, line_dash="dash", line_color="#28A745",
                      annotation_text="Strong (80%)", annotation_position="top right")
    fig_bar.add_hline(y=50, line_dash="dash", line_color="#FFC107",
                      annotation_text="Moderate (50%)", annotation_position="top right")
    fig_bar.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Helvetica, Arial", size=13, color="#1C1E21"),
        xaxis=dict(title="Market", showgrid=False),
        yaxis=dict(title="Match Score (%)", range=[0, 115],
                   showgrid=True, gridcolor="#E4E6EB"),
        height=380, margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Ranking table ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Control Market Ranking — Geo-Lift Suitability</p>', unsafe_allow_html=True)

styled = (
    corr_df.style
    .map(score_color, subset=["Match Score (%)"])
    .format({
        "Correlation (r)": lambda x: f"{x:.4f}" if pd.notna(x) else "—",
        "Match Score (%)": lambda x: f"{x}%" if pd.notna(x) else "—",
    })
    .set_properties(**{"text-align": "center"})
    .set_table_styles([
        {"selector": "thead th",
         "props": [("background-color", "#1877F2"), ("color", "white"),
                   ("font-weight", "bold"), ("text-align", "center")]},
        {"selector": "tbody tr:nth-child(1)",
         "props": [("background-color", "#E7F3FF")]},
    ])
)
st.dataframe(styled, use_container_width=True)

# ── Keyword score table ───────────────────────────────────────────────────────
with st.expander("📊 Raw Keyword Interest Scores by Market"):
    all_mkt_cols = [target_market] + comp_available
    raw_sub = score_matrix[[m for m in all_mkt_cols if m in score_matrix.columns]]
    st.dataframe(raw_sub.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)

# ── Interpretation guide ──────────────────────────────────────────────────────
st.markdown('<p class="section-header">📖 How to Interpret Results</p>', unsafe_allow_html=True)
st.markdown("""
| Match Score | Correlation (r) | Recommendation |
|---|---|---|
| ≥ 80% | ≥ 0.80 | ✅ **Strong control candidate** — proceed with geo-lift test |
| 50–79% | 0.50–0.79 | ⚠️ **Moderate** — acceptable; validate with longer pre-period |
| < 50% | < 0.50 | ❌ **Weak** — avoid; demand profiles diverge too much |

**Next steps after identifying your Synthetic Twin:**
1. **Validate** with 4–8 weeks of pre-period campaign data to confirm parallel trends.
2. **Design** the test: hold-out the Twin market while activating media in the Target.
3. **Measure lift** = (Target conversion rate − Twin conversion rate) / Twin conversion rate.
4. **Report** incremental reach, conversions, and cost per incremental result.

**Improving match quality:**
- Add more keywords (3–5) that represent your product category.
- Use a longer timeframe (12 months or 5 years) for more stable correlations.
- If no strong match exists, consider a synthetic control using a weighted blend of 2–3 markets.
""")

# ── Downloads ─────────────────────────────────────────────────────────────────
with st.expander("⬇️ Download Data"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "📥 Raw Score Matrix CSV",
            data=score_matrix.to_csv().encode(),
            file_name="raw_score_matrix.csv", mime="text/csv",
        )
    with col_b:
        st.download_button(
            "📥 Correlation Rankings CSV",
            data=corr_df.to_csv().encode(),
            file_name="correlation_rankings.csv", mime="text/csv",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data source: Google Trends via pytrends (public scraper — interest_by_region) · "
    "Correlation: Pearson r · Normalisation: Min-Max per keyword · "
    "Built for Media Strategy geo-lift test planning"
)
