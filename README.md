# 🌏 Synthetic Twin City Finder

A Streamlit web app that identifies the best **control market** for geo-lift / incrementality tests using Google Trends demand profiles.

## What it does

- Fetches Google Trends `interest_by_region` scores for up to 5 keywords
- Builds a keyword × market demand matrix
- Computes **Pearson correlation** between your target market and each comparison market
- Identifies the **Synthetic Twin** — the market with the highest demand profile similarity
- Outputs a ranked control market table, radar chart, heatmap, and CSV downloads

## How to use

1. Select your **Target Market** (the one receiving media)
2. Select up to 8 **Comparison Markets**
3. Pick a **keyword category** (or enter custom keywords)
4. Set the **timeframe** (12 months recommended)
5. Click **🚀 Find Synthetic Twin**

## Stack

- Python · Streamlit · pytrends · Pandas · Plotly · SciPy

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).
