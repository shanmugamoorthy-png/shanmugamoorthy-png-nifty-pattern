import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

st.set_page_config(page_title="Nifty volatility readout", layout="wide")
LIVE = bool(os.getenv("DHAN_CLIENT_ID") and os.getenv("DHAN_ACCESS_TOKEN"))

st.title("Nifty 5m — afternoon volatility readout")
st.caption("Mode: " + ("🟢 Live (Dhan)" if LIVE else "🟡 Demo (random data — set DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN)"))

with st.sidebar:
    cut_s = st.text_input("Read at (HH:MM)", "10:20")
    k_vol = st.slider("Past days used for the range forecast", 50, 300, 150, 25)
    prev_days = st.slider("Previous days shown on chart", 0, 5, 2)
    go_btn = st.button("Refresh data", type="primary")


@st.cache_data(ttl=3600, show_spinner="Loading history…")
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


@st.cache_data(ttl=300, show_spinner=False)
def forecast(cut, key, _hist, _today):
    """Cached so moving the slider / switching tabs does not rebuild the library. `key` changes with the data."""
    return E.predict(_hist, _today, cut)


@st.cache_data(ttl=3600, show_spinner="Running walk-forward backtest…")
def run_backtest(cut, k, test_days):
    return E.backtest(history(), cut, k, test_days=test_days)


if go_btn:
    st.cache_data.clear()

try:
    h, m = map(int, cut_s.split(":"))
except ValueError:
    st.error("Use HH:MM, e.g. 10:20")
    st.stop()
cut = h * 60 + m
hist = history()
today = today_bars(cut)

# ---------- closed market -> replay the last completed day ----------
replay = False
if today.empty:
    hh = hist.copy()
    hh["d"] = hh.ts.dt.date
    last = hh.d.max()
    today = hh[hh.d == last].drop(columns="d").reset_index(drop=True)
    replay = True
    st.info(f"Market closed / no candles yet. Replaying the last trading day ({last:%a %d %b %Y}) as if it were "
            f"{cut_s} — its real outcome is shown for comparison. Live mode starts automatically when today's candles exist.")

tab_pred, tab_bt = st.tabs(["Today", "Backtest"])

# =====================================================================
with tab_pred:
    try:
        S = forecast(cut, f"{len(hist)}|{today.ts.iloc[-1]}", hist, today)
    except ValueError as e:
        st.info(str(e))
        st.stop()
    V = E.vol_readout(*S["_vol_args"], k=k_vol)
    ref, tb, tdate = S["ref_price"], S["today"], S["today_date"]

    st.subheader(f"How big a move to expect from {cut_s} to the close")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"NIFTY @ {cut_s}", f"{ref:,.2f}")
    c2.metric("Morning range (09:15→cutoff)", f"{V['morning_range']:.2f}%", f"{V['bucket']} · {V['pct_rank']:.0f}th pct", delta_color="off")
    c3.metric("Expected afternoon range", f"{V['range_med']:.2f}%", f"≈ {V['range_pts']:.0f} pts", delta_color="off")
    c4.metric("Middle half of days", f"{V['range_p25']:.2f}–{V['range_p75']:.2f}%")
    d1, d2, d3 = st.columns(3)
    d1.metric("Bigger-than-usual afternoon", f"{V['p_big']:.0f}%", f"usual = {V['lib_median_range']:.2f}% range", delta_color="off")
    d2.metric("Range above 1%", f"{V['p_gt1']:.0f}%")
    d3.metric("Quiet (range under 0.5%)", f"{V['p_lt05']:.0f}%")
    st.write(
        f"**Typical zone (no direction implied):** up to ≈ **{V['up_med']:,.0f}** (stretch {V['up_p75']:,.0f}) and "
        f"down to ≈ **{V['dn_med']:,.0f}** (stretch {V['dn_p25']:,.0f}). Based on the {V['n']} past days whose morning range was closest to today's.")
    st.caption("Backtested on real days: afternoon move size was predictable (rank corr ≈ +0.46), direction was not (≈ coin flip). "
               "Out-of-sample it explains only about 10–15% of the day-to-day variation, so a quiet morning can still end in a big day. "
               "Compare with option IV before trading.")

    # ---------------- chart ----------------
    lab = lambda ts: ts.strftime("%d %b %H:%M")
    mins = lambda s: s.dt.hour * 60 + s.dt.minute
    hh = hist[hist.ts.dt.date < tdate].copy()
    hh["d"] = hh.ts.dt.date
    pdates = sorted(hh.d.unique())[-prev_days:] if prev_days else []
    prev = hh[hh.d.isin(pdates)].sort_values("ts")
    prev = prev[(mins(prev.ts) >= E.OPEN_MIN) & (mins(prev.ts) < E.CLOSE_MIN)]

    fig = go.Figure()
    cd = lambda df, nm, up, dn: go.Candlestick(x=df.ts.map(lab), open=df.o, high=df.h, low=df.l, close=df.c, name=nm,
                                               increasing_line_color=up, decreasing_line_color=dn)
    if len(prev):
        fig.add_trace(cd(prev, "Previous days", "#9aa5b1", "#5f6b76"))
    fig.add_trace(cd(tb, "Today (to cutoff)", "#16a34a", "#dc2626"))
    last_lab = lab(tb.ts.iloc[-1])
    end_lab = f"{tdate:%d %b} {E._hm(E.CLOSE_MIN - E.TF)}"
    for lvl, nm, col, dash in [(V["up_med"], "Typical high", "#16a34a", "dash"), (V["dn_med"], "Typical low", "#dc2626", "dash"),
                               (V["up_p75"], "Stretch high", "#16a34a", "dot"), (V["dn_p25"], "Stretch low", "#dc2626", "dot")]:
        fig.add_trace(go.Scatter(x=[last_lab, end_lab], y=[lvl, lvl], mode="lines", name=nm, line=dict(color=col, width=2, dash=dash)))
    act_range = actual_ret = None
    if replay:
        full = hist[hist.ts.dt.date == tdate].sort_values("ts")
        fut = full[mins(full.ts) >= cut]
        if len(fut):
            fig.add_trace(go.Scatter(x=[last_lab] + list(fut.ts.map(lab)), y=np.r_[ref, fut.c.to_numpy()],
                                     name="Actual (replay)", line=dict(color="orange", width=3)))
            actual_ret = (fut.c.iloc[-1] / ref - 1) * 100
            act_range = (fut.h.max() / ref - 1) * 100 - (fut.l.min() / ref - 1) * 100
    for xv in [lab(tb.ts.iloc[0]), last_lab]:
        fig.add_shape(type="line", x0=xv, x1=xv, yref="paper", y0=0, y1=1, line=dict(color="gray", width=1, dash="dot"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      xaxis=dict(type="category", nticks=14), yaxis_title="NIFTY", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dotted vertical lines: today's open and the cutoff. Grey candles: previous days. Horizontal lines: typical / stretch high and low for the rest of the day.")
    if replay and actual_ret is not None:
        st.write(f"**Replay result:** expected range {V['range_med']:.2f}% → actual range **{act_range:.2f}%** "
                 f"({'inside the middle half' if V['range_p25'] <= act_range <= V['range_p75'] else 'outside the middle half'}); "
                 f"net move {actual_ret:+.2f}%.")

# =====================================================================
with tab_bt:
    st.write(f"Replays the method on past days: for each day it forecasts the **{cut_s} → close** range using **only earlier days**, "
             "then compares with what really happened.")
    test_days = st.slider("Days to test (most recent)", 100, 1000, 500, 50)
    if st.button("Run backtest"):
        st.session_state["bt"] = run_backtest(cut, k_vol, test_days)
    if "bt" in st.session_state:
        bt = st.session_state["bt"]
        R = E.range_summary(bt)
        if R:
            st.subheader("Does the morning range predict the afternoon range?")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Days tested", R["days"])
            r2.metric("Rank correlation with actual range", f"{R['spearman']:+.2f}")
            r3.metric("Avg forecast vs actual range", f"{R['avg_pred_range']:.2f}% / {R['avg_actual_range']:.2f}%")
            r4.metric("Actual inside forecast middle half", f"{R['inside_middle_half']:.0f}%", "ideal ≈ 50%", delta_color="off")
            qm = pd.DataFrame(R["quartiles_morning"], columns=["Morning-range group", "Days", "Morning range from %", "to %",
                                                               "Avg afternoon range %", "Median afternoon range %",
                                                               "Days with above-median range %", "Avg |move to close| %"]).round(2)
            st.dataframe(qm, hide_index=True, use_container_width=True)
            st.caption("Afternoon range = cutoff→close high minus low, % of price. Rank correlation ~0 means no skill; ~0.4–0.5 is useful but far from perfect.")

            ex = E.range_extras(bt)
            if ex:
                st.subheader("Do other inputs add anything? (fitted on first half, scored on second half)")
                exd = pd.DataFrame(ex, columns=["Model for afternoon range", "R² first half (fit)", "R² second half (honest)"]).round(3)
                st.dataframe(exd, hide_index=True, use_container_width=True)
                base_oos = ex[0][2]
                best = max(ex[1:], key=lambda e: e[2])
                st.caption(f"Honest score of the baseline: {base_oos:.3f}. Best add-on: **{best[0]}** at {best[2]:.3f} "
                           f"({'worth keeping' if best[2] - base_oos > 0.02 else 'no meaningful improvement — keep it simple'}). "
                           "Differences under ~0.02 are noise with this many days.")

            out = bt.round(3)
            st.dataframe(out.iloc[::-1], hide_index=True, use_container_width=True, height=300)
            st.download_button("Download backtest (CSV)", out.to_csv(index=False), "nifty_backtest.csv", "text/csv")
    st.caption("Past results do not guarantee future results. Percent = Nifty index move; no brokerage, slippage or option premium.")
