"""Volatility engine: how big a move to expect from the cutoff (e.g. 10:20) to the close,
judged from the size of this morning's range (09:15 -> cutoff high minus low).
Direction is NOT predicted (tested on ~1000 days: no edge).

Input DataFrame columns: ts (naive IST datetime, bar START time), o, h, l, c
"""
import numpy as np
import pandas as pd

OPEN_MIN = 555            # 09:15
CLOSE_MIN = 930           # 15:30
TF = 5                    # minutes per bar
MIN_BARS = 72             # skip half days / broken days


def split_days(df):
    """-> {date: DataFrame of that day's bars (09:15-15:30 only, sorted)}"""
    d = df.copy()
    d["date"] = d.ts.dt.date
    d["m"] = d.ts.dt.hour * 60 + d.ts.dt.minute
    d = d[(d.m >= OPEN_MIN) & (d.m < CLOSE_MIN)]
    return {k: g.sort_values("ts").reset_index(drop=True) for k, g in d.groupby("date")}


def n_bars(cutoff_min):
    return (cutoff_min - OPEN_MIN) // TF


def _hm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def morning_range(bars, n):
    """High minus low of the first n bars, as % of the day's open."""
    o0 = bars.o.iloc[0]
    return float((bars.h.iloc[:n].max() - bars.l.iloc[:n].min()) / o0 * 100)


def outcome(bars, n):
    """What happened after bar n (cutoff -> close), % from the cutoff-bar close."""
    ref = bars.c.iloc[n - 1]
    fut = bars.iloc[n:]
    hi_i, lo_i = int(fut.h.to_numpy().argmax()), int(fut.l.to_numpy().argmin())
    t = lambda i: int(fut.ts.iloc[i].hour * 60 + fut.ts.iloc[i].minute)
    return {
        "ref": ref,
        "ret_close": (fut.c.iloc[-1] / ref - 1) * 100,
        "mfe": (fut.h.max() / ref - 1) * 100,       # best excursion up
        "mae": (fut.l.min() / ref - 1) * 100,       # worst excursion down
        "high_min": t(hi_i), "low_min": t(lo_i),
    }


def build_library(days, cutoff_min):
    """One row per complete historical day: morning range, gap, afternoon outcome."""
    n = n_bars(cutoff_min)
    rows = []
    prev_close = None
    for d in sorted(days):
        b = days[d]
        if len(b) >= MIN_BARS and len(b) > n:
            o0 = b.o.iloc[0]
            rows.append({"date": d, **outcome(b, n),
                         "morning_range": morning_range(b, n),
                         "gap_abs": abs((o0 / prev_close - 1) * 100) if prev_close else 0.0,
                         "day_range": (b.h.max() / b.l.min() - 1) * 100})
        prev_close = b.c.iloc[-1]
    meta = pd.DataFrame(rows)
    meta["range"] = meta.mfe - meta.mae               # cutoff -> close high-low range, %
    return meta


def vol_readout(meta, t, ref, k=150):
    """Expected size of the move from the cutoff to the close, from the k past days whose morning range
    is closest to today's t. Direction-neutral."""
    mr = meta.morning_range.to_numpy()
    rng = meta["range"].to_numpy()
    idx = np.argsort(np.abs(mr - t))[:min(k, len(mr))]
    r = rng[idx]
    q = np.percentile(mr, [25, 50, 75])
    bucket = "Q1 quiet" if t <= q[0] else "Q2" if t <= q[1] else "Q3" if t <= q[2] else "Q4 wide"
    mfe, mae = meta.mfe.to_numpy()[idx], meta.mae.to_numpy()[idx]
    px = lambda pct: ref * (1 + pct / 100)
    return {
        "morning_range": float(t), "bucket": bucket, "pct_rank": float((mr < t).mean() * 100),
        "n": len(idx), "lib_median_range": float(np.median(rng)),
        "range_med": float(np.median(r)), "range_p25": float(np.percentile(r, 25)), "range_p75": float(np.percentile(r, 75)),
        "range_pts": ref * float(np.median(r)) / 100,
        "p_big": float((r > np.median(rng)).mean() * 100),     # above-median afternoon range
        "p_gt1": float((r > 1.0).mean() * 100),                # afternoon range above 1%
        "p_lt05": float((r < 0.5).mean() * 100),
        "up_med": px(float(np.median(mfe))), "up_p75": px(float(np.percentile(mfe, 75))),
        "dn_med": px(float(np.median(mae))), "dn_p25": px(float(np.percentile(mae, 25))),
    }


def predict(df_hist, df_today, cutoff_min=620, k=150):
    """Main entry. df_hist: history (today's partial bars are excluded). df_today: today's bars so far."""
    n = n_bars(cutoff_min)
    today = df_today.sort_values("ts").reset_index(drop=True)
    today_date = today.ts.iloc[0].date()
    today = today[(today.ts.dt.hour * 60 + today.ts.dt.minute) < cutoff_min]
    if len(today) < n:
        raise ValueError(f"Need {n} bars up to {_hm(cutoff_min)}, have {len(today)} (wait for the {_hm(cutoff_min)} bar).")
    days = split_days(df_hist[df_hist.ts.dt.date != today_date])
    meta = build_library(days, cutoff_min)
    t = morning_range(today.iloc[:n].reset_index(drop=True), n)
    ref = float(today.c.iloc[n - 1])
    return {"vol": vol_readout(meta, t, ref, k), "ref_price": ref, "today": today, "today_date": today_date,
            "n_hist_days": len(meta), "cutoff_min": cutoff_min,
            "_vol_args": (meta, t, ref)}      # lets the UI re-run the readout with another neighbour count


# ======================= Walk-forward backtest =======================
def backtest(df_hist, cutoff_min=620, k=150, min_lib=250, test_days=500):
    """For each past day D (last `test_days` days), forecast the cutoff->close range using ONLY days before D,
    then compare with what D really did. No lookahead."""
    n = n_bars(cutoff_min)
    meta = build_library(split_days(df_hist), cutoff_min)
    mr = meta.morning_range.to_numpy()
    rng = meta["range"].to_numpy()
    ret = meta.ret_close.to_numpy()
    day_rng = meta.day_range.to_numpy()
    gap = meta.gap_abs.to_numpy()
    start = max(min_lib, len(meta) - test_days)
    rows = []
    for i in range(start, len(meta)):
        idx = np.argsort(np.abs(mr[:i] - mr[i]))[:min(k, i)]
        r = rng[idx]
        rows.append({"date": meta.date.iloc[i], "morning_range": float(mr[i]),
                     "pred_range": float(np.median(r)),
                     "pred_p25": float(np.percentile(r, 25)), "pred_p75": float(np.percentile(r, 75)),
                     "base_longrun_range": float(np.median(rng[:i])),
                     "base_prev_range": float(day_rng[i - 1]),
                     "base_gap_abs": float(gap[i]),
                     "actual_range": float(rng[i]), "actual_abs": float(abs(ret[i]))})
    return pd.DataFrame(rows)


def range_summary(bt):
    """How well does the morning range forecast the afternoon range?"""
    if bt.empty:
        return {}
    sp = lambda x, y: float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    y = bt.actual_range.to_numpy()
    inside = (y >= bt.pred_p25) & (y <= bt.pred_p75)
    out = {"days": len(bt),
           "spearman": sp(bt.morning_range, y),
           "spearman_abs": sp(bt.morning_range, bt.actual_abs),
           "avg_actual_range": float(y.mean()),
           "avg_pred_range": float(bt.pred_range.mean()),
           "inside_middle_half": float(inside.mean() * 100)}
    big = bt.actual_range > np.median(y)
    qm = pd.qcut(bt.morning_range.rank(method="first"), 4, labels=["Q1 quiet", "Q2", "Q3", "Q4 wide"])
    out["quartiles_morning"] = [(str(k), int(len(g)), float(g.morning_range.min()), float(g.morning_range.max()),
                                 float(g.actual_range.mean()), float(g.actual_range.median()),
                                 float(big[g.index].mean() * 100), float(g.actual_abs.mean()))
                                for k, g in bt.groupby(qm, observed=True)]
    return out


def range_extras(bt):
    """Do yesterday's range / opening gap add anything to the morning range?
    Linear models fitted on the first half of the test days, scored on the second half (out-of-sample R2)."""
    if bt.empty or len(bt) < 20:
        return []
    y = bt.actual_range.to_numpy()
    h = len(bt) // 2
    cols = {"morning": bt.morning_range.to_numpy(), "prev": bt.base_prev_range.to_numpy(),
            "gap": bt.base_gap_abs.to_numpy()}

    def fit(names):
        X = np.column_stack([np.ones(len(y))] + [cols[c] for c in names])
        b = np.linalg.lstsq(X[:h], y[:h], rcond=None)[0]
        p = X @ b
        r2 = lambda sl: 1 - ((y[sl] - p[sl]) ** 2).sum() / ((y[sl] - y[sl].mean()) ** 2).sum()
        return float(r2(slice(0, h))), float(r2(slice(h, None)))
    models = [("Morning range only (baseline)", ["morning"]), ("+ yesterday's full-day range", ["morning", "prev"]),
              ("+ opening gap size", ["morning", "gap"]), ("+ yesterday's range + gap", ["morning", "prev", "gap"])]
    return [(nm, *fit(c)) for nm, c in models]
