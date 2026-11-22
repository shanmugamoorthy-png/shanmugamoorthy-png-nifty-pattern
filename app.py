import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

st.set_page_config(page_title="Nifty pattern match", layout="wide")
LIVE = bool(os.getenv("DHAN_CLIENT_ID") and os.getenv("DHAN_ACCESS_TOKEN"))

st.title("Nifty 5m — similar-day pattern match")
st.caption("Mode: " + ("🟢 Live (Dhan)" if LIVE else "🟡 Demo (random data — set DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN)"))

with st.sidebar:
    cut_s = st.text_input("Predict at (HH:MM)", "10:20")
    top_k = st.slider("Top matching days", 5, 50, 20)
    weekday = st.checkbox("Same weekday only", False)
    prev_days = st.slider("Previous days shown on chart", 0, 5, 2)
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


@st.cache_data(ttl=3600, show_spinner="Running walk-forward backtest…")
def run_backtest(cut, k, wd, test_days, wg, we):
    return E.backtest(history(), cut, k, wd, test_days=test_days, w_gap=wg, w_extra=we)


if go_btn:
    st.cache_data.clear()

h, m = map(int, cut_s.split(":"))
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

tab_pred, tab_bt = st.tabs(["Prediction", "Backtest"])

# =====================================================================
with tab_pred:
    try:
        S = E.predict(hist, today, cut, top_k, weekday, w_gap=w_gap, w_extra=w_extra)
    except ValueError as e:
        st.info(str(e))
        st.stop()

    ref, top, tb, tdate = S["ref_price"], S["top"], S["today"], S["today_date"]
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
        f"Typical high ≈ **{S['exp_high']:,.0f}** around {S['median_high_time']}, typical low ≈ **{S['exp_low']:,.0f}** around {S['median_low_time']}.")
    st.caption("Stability check — 10 / 20 / 50 neighbours: " +
               " | ".join(f"top {k}: {u} up, median {r:+.2f}%" for k, u, r in S["by_k"]) +
               ".  If these disagree, the pattern has no real edge today.")

    # ---------- chart: previous days + today + forecast (+ actual when replaying) ----------
    lab = lambda ts: ts.strftime("%d %b %H:%M")
    hh = hist.copy()
    hh["d"] = hh.ts.dt.date
    pdates = sorted(d for d in hh.d.unique() if d < tdate)[-prev_days:] if prev_days else []
    prev = hh[hh.d.isin(pdates)].sort_values("ts")
    prev = prev[(prev.ts.dt.hour * 60 + prev.ts.dt.minute >= E.OPEN_MIN) & (prev.ts.dt.hour * 60 + prev.ts.dt.minute < E.CLOSE_MIN)]

    fig = go.Figure()
    cd = lambda df, nm, up, dn: go.Candlestick(x=df.ts.map(lab), open=df.o, high=df.h, low=df.l, close=df.c, name=nm,
                                               increasing_line_color=up, decreasing_line_color=dn)
    if len(prev):
        fig.add_trace(cd(prev, "Previous days", "#9aa5b1", "#5f6b76"))
    fig.add_trace(cd(tb, "Today (to cutoff)", "#16a34a", "#dc2626"))
    last_lab = lab(tb.ts.iloc[-1])
    dstr = f"{tdate:%d %b}"
    fx = [last_lab] + [f"{dstr} {E._hm(t)}" for t in S["path_times"]]
    P = S["path"]
    up_y = np.r_[ref, ref * (1 + P["p75"] / 100)]
    lo_y = np.r_[ref, ref * (1 + P["p25"] / 100)]
    md_y = np.r_[ref, ref * (1 + P["median"] / 100)]
    fig.add_trace(go.Scatter(x=fx, y=up_y, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=lo_y, fill="tonexty", fillcolor="rgba(30,120,255,0.18)", line=dict(width=0), name="25–75% band"))
    fig.add_trace(go.Scatter(x=fx, y=md_y, name="Median forecast", line=dict(color="royalblue", width=3, dash="dash")))
    actual_ret = None
    if replay:
        full = hist[hist.ts.dt.date == tdate].sort_values("ts")
        fut = full[(full.ts.dt.hour * 60 + full.ts.dt.minute) >= cut]
        if len(fut):
            fig.add_trace(go.Scatter(x=[last_lab] + list(fut.ts.map(lab)), y=np.r_[ref, fut.c.to_numpy()],
                                     name="Actual (replay)", line=dict(color="orange", width=3)))
            actual_ret = (fut.c.iloc[-1] / ref - 1) * 100
    for xv in [lab(tb.ts.iloc[0]), last_lab]:
        fig.add_shape(type="line", x0=xv, x1=xv, yref="paper", y0=0, y1=1, line=dict(color="gray", width=1, dash="dot"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      xaxis=dict(type="category", nticks=14), yaxis_title="NIFTY", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dotted lines: today's open and the cutoff. Grey candles: previous days leading into today's open.")
    if actual_ret is not None:
        st.write(f"**Replay result:** predicted median {S['median_ret']:+.2f}% → actual {actual_ret:+.2f}% "
                 f"({'✅ same direction' if np.sign(actual_ret) == np.sign(S['median_ret']) else '❌ opposite direction'}).")

    st.subheader(f"Top {S['k']} most similar days")
    show = top[["date", "weekday", "similarity", "morning_ret", "ret_close", "mfe", "mae"]].copy()
    show["high_at"] = top.high_min.map(E._hm)
    show["low_at"] = top.low_min.map(E._hm)
    show.columns = ["Date", "Day", "Similarity %", "Morning % (open→cutoff)", "Cutoff→Close %", "Max up %", "Max down %", "High at", "Low at"]
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)
    st.download_button("Download matches (CSV)", show.to_csv(index=False), "nifty_top_matches.csv", "text/csv")

# =====================================================================
with tab_bt:
    st.write(f"Replays the method on past days: for each day it predicts **{cut_s} → close** using **only earlier days** "
             "(no future data), then compares with what really happened.")
    b1, b2 = st.columns(2)
    test_days = b1.slider("Days to test (most recent)", 100, 1000, 500, 50)
    thr = b2.slider("Signal threshold (share of top matches that agree)", 0.55, 0.85, 0.65, 0.05)
    if st.button("Run backtest"):
        st.session_state["bt"] = (run_backtest(cut, top_k, weekday, test_days, w_gap, w_extra), thr)
    if "bt" in st.session_state:
        bt, thr0 = st.session_state["bt"]
        B = E.backtest_summary(bt, thr)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Days tested", B["days"])
        m2.metric("Market up-rate (baseline)", f"{B['base_up_rate']:.1f}%")
        m3.metric("Direction accuracy", f"{B['dir_acc']:.1f}%", f"{B['dir_acc'] - 50:+.1f} vs coin flip")
        m4.metric("Corr(predicted, actual)", f"{B['corr']:+.2f}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Signal days", f"{B['trades']} ({B['longs']}L / {B['shorts']}S)")
        m6.metric("Win rate", f"{B['win_rate']:.1f}%" if B["trades"] else "–")
        m7.metric("Avg per signal day", f"{B['avg_trade']:+.2f}%" if B["trades"] else "–")
        m8.metric("Total: signals vs buy&hold", f"{B['total']:+.0f}% / {B['buy_hold']:+.0f}%")
        st.caption("Percent = Nifty index move from the cutoff to the close, summed over days. No brokerage, slippage or option "
                   "premium effects — real option P&L will be lower. Accuracy near 50% or correlation near 0 means the pattern has no edge.")

        st.subheader("When the matches agree, how often was the day right?")
        bd = pd.DataFrame(B["bands"], columns=["Match agreement", "Days", "Actual up-rate %", "Avg actual move %"]).round(2)
        st.dataframe(bd, hide_index=True, use_container_width=True)

        eq = go.Figure()
        d = pd.to_datetime(bt.date)
        eq.add_trace(go.Scatter(x=d, y=B["equity"], name="Follow signal (long/short/flat)", line=dict(width=3)))
        eq.add_trace(go.Scatter(x=d, y=B["equity_bh"], name="Always long (buy & hold)", line=dict(dash="dot")))
        eq.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="cumulative % (cutoff→close)")
        st.plotly_chart(eq, use_container_width=True)

        out = bt.copy()
        out["signal"] = np.where(out.up_frac >= thr, "LONG", np.where(out.up_frac <= 1 - thr, "SHORT", "-"))
        out["correct"] = np.where(out.signal == "-", "", np.where((out.signal == "LONG") == (out.actual > 0), "✓", "✗"))
        out = out.round(3).rename(columns={"up_frac": "share_up", "pred_median": "pred_median_%", "pred_mean": "pred_mean_%", "actual": "actual_%"})
        st.dataframe(out.iloc[::-1], hide_index=True, use_container_width=True, height=300)
        st.download_button("Download backtest (CSV)", out.to_csv(index=False), "nifty_backtest.csv", "text/csv")
    st.caption("Past results do not guarantee future results.")
