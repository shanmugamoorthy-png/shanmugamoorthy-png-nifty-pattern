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
    show_path = st.checkbox("Show 20-day pattern path on chart (reference)", False)
    st.markdown("**Pattern-match settings (reference only)**")
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

tab_pred, tab_bt = st.tabs(["Today", "Backtest"])

# =====================================================================
with tab_pred:
    try:
        S = E.predict(hist, today, cut, top_k, weekday, w_gap=w_gap, w_extra=w_extra)
    except ValueError as e:
        st.info(str(e))
        st.stop()
    V = E.vol_readout(*S["_vol_args"], k=k_vol)

    ref, top, tb, tdate = S["ref_price"], S["top"], S["today"], S["today_date"]

    # ---------------- 1) volatility readout (the part that tested as useful) ----------------
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
    st.caption("Backtest on 500 real days: afternoon move size was predictable (rank corr ≈ +0.48), direction was not (≈ coin flip). "
               "It explains only ~17% of the variation, so a quiet morning can still end in a big day. Compare with option IV before trading.")

    # ---------------- chart ----------------
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
    end_lab = f"{dstr} {E._hm(E.CLOSE_MIN - E.TF)}"
    # expected-zone lines from the volatility readout
    for lvl, nm, col, dash in [(V["up_med"], "Typical high", "#16a34a", "dash"), (V["dn_med"], "Typical low", "#dc2626", "dash"),
                               (V["up_p75"], "Stretch high", "#16a34a", "dot"), (V["dn_p25"], "Stretch low", "#dc2626", "dot")]:
        fig.add_trace(go.Scatter(x=[last_lab, end_lab], y=[lvl, lvl], mode="lines", name=nm, line=dict(color=col, width=2, dash=dash)))
    if show_path:
        fx = [last_lab] + [f"{dstr} {E._hm(t)}" for t in S["path_times"]]
        P = S["path"]
        fig.add_trace(go.Scatter(x=fx, y=np.r_[ref, ref * (1 + P["median"] / 100)], name="Pattern median path (reference)",
                                 line=dict(color="royalblue", width=2, dash="dashdot")))
    actual_ret = act_range = None
    if replay:
        full = hist[hist.ts.dt.date == tdate].sort_values("ts")
        fut = full[(full.ts.dt.hour * 60 + full.ts.dt.minute) >= cut]
        if len(fut):
            fig.add_trace(go.Scatter(x=[last_lab] + list(fut.ts.map(lab)), y=np.r_[ref, fut.c.to_numpy()],
                                     name="Actual (replay)", line=dict(color="orange", width=3)))
            actual_ret = (fut.c.iloc[-1] / ref - 1) * 100
            act_range = (fut.h.max() / fut.l.min() - 1) * 100
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

    # ---------------- 2) pattern match, demoted to reference ----------------
    with st.expander("Reference: top similar days by 5-minute shape (direction is NOT reliable — backtest ≈ coin flip)"):
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Closed higher (top {S['k']})", f"{S['up']} / {S['k']}")
        c2.metric("Median move to close", f"{S['median_ret']:+.2f}%")
        c3.metric("Library", f"{S['n_hist_days']} days")
        bc = st.columns(5)
        for col, (name, cnt) in zip(bc, S["buckets"]):
            col.metric(name.split("(")[0].strip(), f"{cnt} / {S['k']}", name[name.index("("):])
        st.caption("Stability — 10 / 20 / 50 neighbours: " +
                   " | ".join(f"top {k}: {u} up, median {r:+.2f}%" for k, u, r in S["by_k"]))
        show = top[["date", "weekday", "similarity", "morning_ret", "ret_close", "mfe", "mae"]].copy()
        show["high_at"] = top.high_min.map(E._hm)
        show["low_at"] = top.low_min.map(E._hm)
        show.columns = ["Date", "Day", "Similarity %", "Morning % (open→cutoff)", "Cutoff→Close %", "Max up %", "Max down %", "High at", "Low at"]
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)
        st.download_button("Download matches (CSV)", show.to_csv(index=False), "nifty_top_matches.csv", "text/csv")

# =====================================================================
with tab_bt:
    st.write(f"Replays the method on past days: for each day it predicts **{cut_s} → close** using **only earlier days**, "
             "then compares with what really happened.")
    b1, b2 = st.columns(2)
    test_days = b1.slider("Days to test (most recent)", 100, 1000, 500, 50)
    thr = b2.slider("Direction-signal threshold (share of top matches that agree)", 0.55, 0.85, 0.65, 0.05)
    if st.button("Run backtest"):
        st.session_state["bt"] = (run_backtest(cut, top_k, weekday, test_days, w_gap, w_extra), thr)
    if "bt" in st.session_state:
        bt, thr0 = st.session_state["bt"]
        R = E.range_summary(bt)

        # ---- volatility first ----
        if R:
            st.subheader("Volatility: does the morning range predict the afternoon range?")
            r1, r2, r3 = st.columns(3)
            r1.metric("Morning range → actual range (rank corr)", f"{R['spearman_morning']:+.2f}")
            r2.metric("20-day pattern → actual range", f"{R['spearman_pattern']:+.2f}")
            r3.metric("Pattern's extra skill beyond morning range", f"{R['spearman_pattern_extra']:+.2f}")
            qm = pd.DataFrame(R["quartiles_morning"], columns=["Morning-range group", "Days", "Morning range from %", "to %",
                                                               "Avg afternoon range %", "Median afternoon range %",
                                                               "Days with above-median range %"]).round(2)
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

        # ---- direction, kept for reference ----
        B = E.backtest_summary(bt, thr)
        st.subheader("Direction (reference — has not shown a reliable edge)")
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
        bd = pd.DataFrame(B["bands"], columns=["Match agreement", "Days", "Actual up-rate %", "Avg actual move %"]).round(2)
        st.dataframe(bd, hide_index=True, use_container_width=True)
        eq = go.Figure()
        d = pd.to_datetime(bt.date)
        eq.add_trace(go.Scatter(x=d, y=B["equity"], name="Follow signal (long/short/flat)", line=dict(width=3)))
        eq.add_trace(go.Scatter(x=d, y=B["equity_bh"], name="Always long (buy & hold)", line=dict(dash="dot")))
        eq.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="cumulative % (cutoff→close)")
        st.plotly_chart(eq, use_container_width=True)

        out = bt.copy()
        out["signal"] = np.where(out.up_frac >= thr, "LONG", np.where(out.up_frac <= 1 - thr, "SHORT", "-"))
        out["correct"] = np.where(out.signal == "-", "", np.where((out.signal == "LONG") == (out.actual > 0), "✓", "✗"))
        out = out.round(3).rename(columns={"up_frac": "share_up", "pred_median": "pred_median_%", "pred_mean": "pred_mean_%", "actual": "actual_%"})
        st.dataframe(out.iloc[::-1], hide_index=True, use_container_width=True, height=300)
        st.download_button("Download backtest (CSV)", out.to_csv(index=False), "nifty_backtest.csv", "text/csv")
    st.caption("Past results do not guarantee future results. Percent = Nifty index move, no brokerage/slippage/option premium.")
