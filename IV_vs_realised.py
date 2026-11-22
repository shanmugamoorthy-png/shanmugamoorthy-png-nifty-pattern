"""oi-dash extra page: options-implied vs realised afternoon range.

Upload this file to the oi-dash GitHub repo as   pages/IV_vs_realised.py   (type the folder name "pages/" in the file name box).
Streamlit then shows it in the left menu as "IV vs realised". It needs iv_compare.py (already uploaded) in the repo root.
"""
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IV vs realised range", layout="wide")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, where iv_compare.py lives
import iv_compare as ivc  # noqa: E402

st.title("Options IV vs what the market really did")
st.caption("For each saved day: the range the options implied from the cutoff to the close, against the range that actually happened. "
           "Uses oi-dash's own saved snapshots, so it needs no extra data.")


def default_db():
    p = os.getenv("DB_PATH", "")
    for cand in (p, "/data/oi.db", "oi.db"):
        if cand and os.path.exists(cand):
            return cand
    return p or "/data/oi.db"


with st.sidebar:
    db = st.text_input("Database file", default_db())
    symbol = st.selectbox("Index", ["NIFTY", "SENSEX", "BANKNIFTY", "FINNIFTY"], 0)
    cut_s = st.text_input("Cutoff (HH:MM)", "10:20")
    incl = st.checkbox("Include expiry-day sessions (IV is distorted)", False)
    dpy = st.number_input("Trading days per year", 200, 365, 252)
    st.button("Refresh")


@st.cache_data(ttl=300, show_spinner="Reading saved snapshots…")
def compute(db, symbol, cut, dpy):
    return ivc.per_day(ivc.load(db, symbol), cut, int(dpy))


if not os.path.exists(db):
    st.error(f"Database not found at {db}. Set the path in the sidebar (try /data/oi.db).")
    st.stop()

h, m = map(int, cut_s.split(":"))
try:
    r = compute(db, symbol, h * 60 + m, dpy)
except Exception as e:                                   # e.g. no snap table yet
    st.error(f"Could not read the database: {e}")
    st.stop()

if r.empty:
    st.warning("No complete days found yet for this index and cutoff. The collector needs to have saved a full session "
               "(roughly 09:15 to 15:30) with option-chain data.")
    st.stop()

use = r if incl else r[~r.expiry_day]
st.write(f"**{len(r)}** complete days saved ({int(r.expiry_day.sum())} were expiry days). "
         f"Using **{len(use)}** days{'' if incl else ' (expiry days excluded)'}.")
if len(use) < 3:
    st.info("Too few days to summarise yet. Come back after more trading days are saved.")
    st.dataframe(r.round(3), hide_index=True, use_container_width=True)
    st.stop()

s = ivc.summarise(use)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Options implied (avg range)", f"{s['avg_implied']:.2f}%")
c2.metric("Market delivered (avg range)", f"{s['avg_realised']:.2f}%")
c3.metric("Realised ÷ implied (median day)", f"{s['median_ratio']:.2f}")
c4.metric("Days realised > implied", f"{s['share_realised_above_implied']:.0f}%")
d1, d2, d3 = st.columns(3)
d1.metric("IV → realised range (rank corr)", f"{s['corr_iv_realised']:+.2f}")
d2.metric("Morning range → realised (rank corr)", f"{s['corr_morning_realised']:+.2f}")
d3.metric("Morning range's extra info beyond IV", f"{s['morning_extra_beyond_iv']:+.2f}")

if s["days"] < 60:
    st.warning(f"Only {s['days']} days. Results this early are mostly noise; trust them at about 100+ days. Check again monthly.")
ratio = s["median_ratio"]
st.write("**Reading:** " + (
    "options were **expensive** versus what the market delivered (selling premium paid on average, before costs)." if ratio < 0.9 else
    "options were **cheap** versus what the market delivered (buying paid on average, before costs)." if ratio > 1.1 else
    "options were **roughly fairly priced** on this sample (no clear edge for buyers or sellers)."))

x = pd.to_datetime(use["date"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=use["implied_range_%"], name="Implied range %", mode="lines+markers"))
fig.add_trace(go.Scatter(x=x, y=use["realised_range_%"], name="Realised range %", mode="lines+markers"))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="cutoff → close high–low, %", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.dataframe(r.round(3).iloc[::-1], hide_index=True, use_container_width=True)
st.download_button("Download per-day table (CSV)", r.round(3).to_csv(index=False), "iv_compare.csv", "text/csv")
st.caption("Realised range uses 1-minute spot samples, so it reads slightly lower than the true 5-minute high–low. "
           "Implied range = 1.596 × IV × √(time left). Not trading advice; compare with brokerage and slippage.")
