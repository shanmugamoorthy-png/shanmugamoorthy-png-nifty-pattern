"""Pattern-match engine: find the past days whose morning 5-minute shape (09:15 -> cutoff)
looks most like today's, then summarise what those days did from the cutoff to the close.

Input DataFrame columns: ts (naive IST datetime, bar START time), o, h, l, c
"""
import numpy as np
import pandas as pd

OPEN_MIN = 555            # 09:15
CLOSE_MIN = 930           # 15:30
TF = 5                    # minutes per bar
FULL_DAY_BARS = 75
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


def morning_features(bars, prev_close, n):
    """Feature vector from the first n bars: cumulative close path, high/low excursion, gap."""
    o0 = bars.o.iloc[0]
    path = (bars.c.iloc[:n].to_numpy() / o0 - 1) * 100
    hi = (bars.h.iloc[:n].max() / o0 - 1) * 100
    lo = (bars.l.iloc[:n].min() / o0 - 1) * 100
    gap = (o0 / prev_close - 1) * 100 if prev_close else 0.0
    return np.concatenate([path, [hi, lo, gap]])


def outcome(bars, n):
    """What happened after bar n (i.e. from the cutoff to the close). Returns dict + future path (%)."""
    ref = bars.c.iloc[n - 1]
    fut = bars.iloc[n:]
    path = (fut.c.to_numpy() / ref - 1) * 100
    hi_i, lo_i = int(fut.h.to_numpy().argmax()), int(fut.l.to_numpy().argmin())
    t = lambda i: int(fut.ts.iloc[i].hour * 60 + fut.ts.iloc[i].minute)
    return {
        "ref": ref,
        "ret_close": path[-1],
        "mfe": (fut.h.max() / ref - 1) * 100,       # best excursion up
        "mae": (fut.l.min() / ref - 1) * 100,       # worst excursion down
        "high_min": t(hi_i), "low_min": t(lo_i),
    }, path


def build_library(days, cutoff_min):
    """Features + outcomes for every complete historical day."""
    n = n_bars(cutoff_min)
    dates = sorted(days)
    rows, feats, paths = [], [], []
    prev_close = None
    for d in dates:
        b = days[d]
        ok = len(b) >= MIN_BARS and len(b) > n
        if ok:
            f = morning_features(b, prev_close, n)
            oc, p = outcome(b, n)
            rows.append({"date": d, **oc, "morning_ret": f[n - 1]})
            feats.append(f)
            full = np.full(FULL_DAY_BARS - n, np.nan)
            full[:len(p)] = p
            paths.append(pd.Series(full).ffill().to_numpy())
        prev_close = b.c.iloc[-1]
    meta = pd.DataFrame(rows)
    return meta, np.array(feats), np.array(paths)


def _weights(n, w_path, w_extra, w_gap):
    return np.concatenate([np.full(n, np.sqrt(w_path / n)), [np.sqrt(w_extra)] * 2, [np.sqrt(w_gap)]])


def find_matches(meta, feats, paths, today_feat, n, top_k=20, same_weekday=False,
                 today_date=None, w_path=1.0, w_extra=0.5, w_gap=0.4):
    """Rank history by weighted distance to today's morning. Returns (top DataFrame, top future paths)."""
    sd = feats.std(axis=0)
    sd[sd == 0] = 1.0
    w = _weights(n, w_path, w_extra, w_gap)
    dist = np.sqrt((((feats - today_feat) / sd * w) ** 2).sum(axis=1))
    m = meta.copy()
    m["distance"] = dist
    mask = np.ones(len(m), bool)
    if today_date is not None:
        mask &= (m.date != today_date).to_numpy()
    if same_weekday and today_date is not None:
        mask &= (pd.to_datetime(m.date).dt.dayofweek == pd.Timestamp(today_date).dayofweek).to_numpy()
    idx = np.where(mask)[0]
    idx = idx[np.argsort(dist[idx])][:top_k]
    top = m.iloc[idx].copy()
    ref_d = np.percentile(dist[mask], 90)
    top["similarity"] = (100 * (1 - top.distance / ref_d)).clip(0, 100).round(1)
    top["weekday"] = pd.to_datetime(top.date).dt.day_name().str[:3]
    return top.reset_index(drop=True), paths[idx]


BUCKETS = [("Strong UP  (> +0.50%)", lambda r: r > 0.5),
           ("Mild UP    (+0.15..+0.50%)", lambda r: (r > 0.15) & (r <= 0.5)),
           ("Sideways   (-0.15..+0.15%)", lambda r: (r >= -0.15) & (r <= 0.15)),
           ("Mild DOWN  (-0.50..-0.15%)", lambda r: (r < -0.15) & (r >= -0.5)),
           ("Strong DOWN (< -0.50%)", lambda r: r < -0.5)]


def summarise(top, top_paths, ref_price, n):
    """Counts like '17 of 20 closed higher', scenario buckets, price targets and expected path."""
    r = top.ret_close.to_numpy()
    k = len(top)
    out = {
        "k": k, "up": int((r > 0).sum()), "down": int((r < 0).sum()),
        "mean_ret": float(r.mean()), "median_ret": float(np.median(r)),
        "avg_mfe": float(top.mfe.mean()), "avg_mae": float(top.mae.mean()),
        "median_high_time": _hm(int(top.high_min.median())), "median_low_time": _hm(int(top.low_min.median())),
        "buckets": [(name, int(f(r).sum())) for name, f in BUCKETS],
        "exp_close": ref_price * (1 + np.median(r) / 100),
        "exp_high": ref_price * (1 + top.mfe.median() / 100),
        "exp_low": ref_price * (1 + top.mae.median() / 100),
        "p25_close": ref_price * (1 + np.percentile(r, 25) / 100),
        "p75_close": ref_price * (1 + np.percentile(r, 75) / 100),
    }
    pp = top_paths
    out["path"] = {"median": np.nanmedian(pp, axis=0), "p25": np.nanpercentile(pp, 25, axis=0),
                   "p75": np.nanpercentile(pp, 75, axis=0)}
    out["path_times"] = [OPEN_MIN + (n + i) * TF for i in range(pp.shape[1])]
    return out


def _hm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def predict(df_hist, df_today, cutoff_min=620, top_k=20, same_weekday=False, **kw):
    """Main entry. df_hist: 5y history (may include today's partial bars - they are excluded).
    df_today: today's bars so far. Returns dict with everything the UI needs."""
    n = n_bars(cutoff_min)
    today = df_today.sort_values("ts").reset_index(drop=True)
    today_date = today.ts.iloc[0].date()
    today = today[(today.ts.dt.hour * 60 + today.ts.dt.minute) < cutoff_min]
    if len(today) < n:
        raise ValueError(f"Need {n} bars up to {_hm(cutoff_min)}, have {len(today)} (wait for the {_hm(cutoff_min)} bar).")
    hist = df_hist[df_hist.ts.dt.date != today_date]
    days = split_days(hist)
    prev_dates = [d for d in sorted(days) if d < today_date]
    prev_close = days[prev_dates[-1]].c.iloc[-1] if prev_dates else None
    meta, feats, paths = build_library(days, cutoff_min)
    tf = morning_features(today.iloc[:n].reset_index(drop=True), prev_close, n)
    top, top_paths = find_matches(meta, feats, paths, tf, n, top_k, same_weekday, today_date, **kw)
    ref = float(today.c.iloc[n - 1])
    s = summarise(top, top_paths, ref, n)
    s.update(top=top, ref_price=ref, n_hist_days=len(meta), today=today, today_date=today_date, cutoff_min=cutoff_min)
    # stability: how does the answer change with 10 / 20 / 50 neighbours?
    s["by_k"] = []
    for kk in (10, 20, 50):
        t2, _ = find_matches(meta, feats, paths, tf, n, kk, same_weekday, today_date, **kw)
        s["by_k"].append((kk, int((t2.ret_close > 0).sum()), float(t2.ret_close.median())))
    return s


# ======================= Walk-forward backtest =======================
def backtest(df_hist, cutoff_min=620, top_k=20, same_weekday=False, min_lib=250, test_days=500,
             w_path=1.0, w_extra=0.5, w_gap=0.4):
    """For each past day D (last `test_days` days), predict cutoff->close using ONLY days before D,
    exactly as the live tool would, then compare with what D really did. No lookahead:
    features of D use only D's own morning; the library and its scaling use days < D."""
    n = n_bars(cutoff_min)
    days = split_days(df_hist)
    meta, feats, _ = build_library(days, cutoff_min)
    W = _weights(n, w_path, w_extra, w_gap)
    dow = pd.to_datetime(meta.date).dt.dayofweek.to_numpy()
    ret = meta.ret_close.to_numpy()
    start = max(min_lib, len(meta) - test_days)
    rows = []
    for i in range(start, len(meta)):
        lib = feats[:i]
        sd = lib.std(axis=0)
        sd[sd == 0] = 1.0
        dist = np.sqrt((((lib - feats[i]) / sd * W) ** 2).sum(axis=1))
        idx = np.arange(i)
        if same_weekday:
            idx = idx[dow[:i] == dow[i]]
        if len(idx) < top_k:
            continue
        sel = idx[np.argsort(dist[idx])][:top_k]
        r = ret[sel]
        rows.append({"date": meta.date.iloc[i], "up_frac": float((r > 0).mean()),
                     "pred_median": float(np.median(r)), "pred_mean": float(r.mean()),
                     "actual": float(ret[i])})
    return pd.DataFrame(rows)


def backtest_summary(bt, thr=0.65):
    """Turn per-day backtest rows into headline numbers. Signal: long if up_frac >= thr, short if <= 1-thr."""
    if bt.empty:
        return {}
    a = bt.actual.to_numpy()
    pos = np.where(bt.up_frac >= thr, 1, np.where(bt.up_frac <= 1 - thr, -1, 0))
    pnl = pos * a
    tr = pos != 0
    s = {
        "days": len(bt), "thr": thr,
        "base_up_rate": float((a > 0).mean() * 100),
        "dir_acc": float((np.sign(bt.pred_median) == np.sign(a)).mean() * 100),
        "corr": float(np.corrcoef(bt.pred_median, a)[0, 1]) if len(bt) > 2 else float("nan"),
        "trades": int(tr.sum()), "longs": int((pos == 1).sum()), "shorts": int((pos == -1).sum()),
        "win_rate": float((pnl[tr] > 0).mean() * 100) if tr.any() else float("nan"),
        "avg_trade": float(pnl[tr].mean()) if tr.any() else float("nan"),
        "total": float(pnl.sum()), "buy_hold": float(a.sum()),
        "worst_day": float(pnl.min()),
    }
    bands = [("Very bullish  (>=70% of matches up)", bt.up_frac >= 0.70),
             ("Bullish       (60-70%)", (bt.up_frac >= 0.60) & (bt.up_frac < 0.70)),
             ("Neutral       (40-60%)", (bt.up_frac > 0.40) & (bt.up_frac < 0.60)),
             ("Bearish       (30-40%)", (bt.up_frac > 0.30) & (bt.up_frac <= 0.40)),
             ("Very bearish  (<=30%)", bt.up_frac <= 0.30)]
    s["bands"] = [(nm, int(m.sum()), float((bt.actual[m] > 0).mean() * 100) if m.any() else float("nan"),
                   float(bt.actual[m].mean()) if m.any() else float("nan")) for nm, m in bands]
    s["equity"] = np.cumsum(pnl)
    s["equity_bh"] = np.cumsum(a)
    s["pos"] = pos
    return s
