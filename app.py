import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

st.set_page_config(page_title="Nifty pattern match", layout="wide")
LIVE = bool(os.getenv("DHAN_CLIENT_ID") and os.getenv("DHAN_ACCESS_TOKEN"))
IST = ZoneInfo("Asia/Kolkata")

st.title("Nifty 5m — similar-day pattern match")
st.caption("Mode: " + ("🟢 Live (Dhan)" if LIVE else "🟡 Demo (random data — set DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN)"))

with st.sidebar:
    cut_s = st.text_input("Predict at (HH:MM)", "10:20")
    top_k = st.slider("Top matching days", 5, 50, 20)
    weekday = st.checkbox("Same weekday only", False)
    w_gap = st.slider("Weight: gap vs prev close", 0.0, 1.0, 0.4, 0.1)
    w_extra = st.slider("Weight: morning high/low range", 0.0, 1.0, 0.5, 0.1)
    go_btn = st.button("Refresh data + predict", type="primary")


@st.cache_data(ttl=3600, show_spinner="Loading 5-year history…")
def history():
    if LIVE:
        import dhan_history as D
        return D.update_history()
    import demo
    return demo.make_demo()


@st.cache_data(ttl=60, show_spinner="Fetching today's candles…")
def today_bars(cut):
    if LIVE:
        import dhan_history as D
        return D.fetch_today()
    import demo
    return demo.make_today(demo.make_demo(), cut)


if go_btn:
    st.cache_data.clear()

h, m = map(int, cut_s.split(":"))
cut = h * 60 + m
hist, today = history(), today_bars(cut)

if today.empty:
    st.warning("No candles for today yet (market closed / holiday). Showing nothing to match.")
    st.stop()

try:
    S = E.predict(hist, today, cut, top_k, weekday, w_gap=w_gap, w_extra=w_extra)
except ValueError as e:
    st.info(str(e))
    st.stop()

ref, top = S["ref_price"], S["top"]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"NIFTY @ {cut_s}", f"{ref:,.2f}")
c2.metric(f"Closed higher (top {S['k']})", f"{S['up']} / {S['k']}")
c3.metric("Median move to close", f"{S['median_ret']:+.2f}%", f"{ref * S['median_ret'] / 100:+.0f} pts")
c4.metric("Library", f"{S['n_hist_days']} days")

st.subheader("Scenarios (days out of the top matches)")
bc = st.columns(5)
for col, (name, cnt) in zip(bc, S["buckets"]):
    col.metric(name.split("(")[0].strip(), f"{cnt} / {S['k']}", name[name.index("("):])

st.write(
    f"**Likely zone:** close ≈ **{S['exp_close']:,.0f}** (middle half of days: {S['p25_close']:,.0f} – {S['p75_close']:,.0f}). "
    f"Typical high ≈ **{S['exp_high']:,.0f}** around {S['median_high_time']}, typical low ≈ **{S['exp_low']:,.0f}** around {S['median_low_time']}."
)
st.caption("Stability check — same question with 10 / 20 / 50 neighbours: " +
           " | ".join(f"top {k}: {u} up, median {r:+.2f}%" for k, u, r in S["by_k"]) +
           ".  If these disagree, the pattern has no real edge today.")

# ---- chart: today + forecast fan + each matched day ----
t0 = E.OPEN_MIN
fig = go.Figure()
tb = S["today"]
fig.add_trace(go.Scatter(x=[E._hm(t0 + i * 5) for i in range(len(tb))], y=tb.c, name="Today", line=dict(color="black", width=3)))
fx = [E._hm(t) for t in S["path_times"]]
P = S["path"]
fig.add_trace(go.Scatter(x=fx, y=ref * (1 + P["p75"] / 100), line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fx, y=ref * (1 + P["p25"] / 100), fill="tonexty", fillcolor="rgba(30,120,255,0.18)",
                         line=dict(width=0), name="25–75% band"))
fig.add_trace(go.Scatter(x=fx, y=ref * (1 + P["median"] / 100), name="Median path", line=dict(color="royalblue", width=3, dash="dash")))
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="time (IST)", yaxis_title="NIFTY")
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Top {S['k']} most similar days")
show = top[["date", "weekday", "similarity", "morning_ret", "ret_close", "mfe", "mae"]].copy()
show["high_at"] = top.high_min.map(E._hm)
show["low_at"] = top.low_min.map(E._hm)
show.columns = ["Date", "Day", "Similarity %", "Morning % (open→cutoff)", "Cutoff→Close %", "Max up %", "Max down %", "High at", "Low at"]
st.dataframe(show.round(2), use_container_width=True, hide_index=True)
st.download_button("Download matches (CSV)", show.to_csv(index=False), "nifty_top_matches.csv", "text/csv")
st.caption("Pattern matching is a base-rate tool, not a guarantee. History repeating 17 of 20 times can still fail today; use it with your own levels and stop-loss.")
