"""Compare what options IMPLIED for the rest of the day with what the market REALISED, using only oi-dash's own database.

Run where oi.db lives (Render shell of the oi-dash service, or any copy of the file):
    python iv_compare.py                      # uses $DB_PATH or /data/oi.db, NIFTY, 10:20
    python iv_compare.py --db /data/oi.db --cutoff 10:20 --symbol NIFTY --csv iv_compare.csv

Table used: snap(symbol, expiry, ts 'YYYY-MM-DD HH:MM', price, iv, ...) with iv = average ATM call/put IV in %.

Per day:
  implied range  = 1.596 * IV * sqrt(time left / trading year)   (expected high-low of a random walk with that vol)
  realised range = (high - low) of 1-minute spot prices from the cutoff to the close, % of the cutoff price
  morning range  = same measure from the open to the cutoff, % of the open price
Only needs pandas + numpy.
"""
import argparse
import math
import os
import sqlite3

import numpy as np
import pandas as pd

OPEN_MIN, CLOSE_MIN, SESSION_MIN = 555, 930, 375
RANGE_FACTOR = 2 * math.sqrt(2 / math.pi)        # E[max - min] of Brownian motion = 1.596 * sigma


def load(db, symbol):
    c = sqlite3.connect(db)
    d = pd.read_sql_query("SELECT expiry, ts, price, iv FROM snap WHERE symbol=? AND price>0", c, params=(symbol,))
    c.close()
    d["ts"] = pd.to_datetime(d.ts)
    d["day"] = d.ts.dt.date
    d["m"] = d.ts.dt.hour * 60 + d.ts.dt.minute
    return d


def per_day(d, cutoff_min, days_per_year=252, min_rows=250):
    rows = []
    for day, g in d.groupby("day"):
        exps = sorted(e for e in g.expiry.unique() if e >= str(day))
        if not exps:
            continue
        near = exps[0]                                         # nearest not-yet-expired contract
        g = g[g.expiry == near].sort_values("ts").drop_duplicates("ts", keep="last")
        g = g[(g.m >= OPEN_MIN) & (g.m < CLOSE_MIN)]
        am, pm = g[g.m < cutoff_min], g[g.m >= cutoff_min]
        if len(g) < min_rows or am.empty or pm.empty or g.m.max() < CLOSE_MIN - 12 or am.m.max() < cutoff_min - 6:
            continue                                           # incomplete day: skip rather than guess
        ref = float(am.price.iloc[-1])
        iv = float(am.iv.iloc[-1])
        if iv <= 0:
            continue
        t_left = (CLOSE_MIN - cutoff_min) / SESSION_MIN / days_per_year
        sigma_pct = iv * math.sqrt(t_left)                     # implied 1-sigma move to the close, %
        rows.append({
            "date": day, "expiry": near, "expiry_day": str(day) == near,
            "iv": iv, "implied_sigma_%": sigma_pct, "implied_range_%": RANGE_FACTOR * sigma_pct,
            "realised_range_%": (pm.price.max() - pm.price.min()) / ref * 100,
            "realised_move_%": (pm.price.iloc[-1] / ref - 1) * 100,
            "morning_range_%": (am.price.max() - am.price.min()) / float(am.price.iloc[0]) * 100,
        })
    return pd.DataFrame(rows)


def rank_corr(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def summarise(r):
    n = len(r)
    s = {"days": n, "avg_implied": r["implied_range_%"].mean(), "avg_realised": r["realised_range_%"].mean(),
         "median_ratio": float((r["realised_range_%"] / r["implied_range_%"]).median()),
         "share_realised_above_implied": float((r["realised_range_%"] > r["implied_range_%"]).mean() * 100),
         "corr_iv_realised": rank_corr(r["iv"], r["realised_range_%"]),
         "corr_morning_realised": rank_corr(r["morning_range_%"], r["realised_range_%"]),
         "corr_morning_iv": rank_corr(r["morning_range_%"], r["iv"])}
    # does the morning range tell you anything the IV did not? (rank partial correlation)
    ri, rm, rr = (r[c].rank() for c in ("iv", "morning_range_%", "realised_range_%"))
    resid = rr - ri.corr(rr) * ri
    s["morning_extra_beyond_iv"] = float(rm.corr(resid))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("DB_PATH", "/data/oi.db"))
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--cutoff", default="10:20")
    ap.add_argument("--csv", default="iv_compare.csv")
    ap.add_argument("--include-expiry-day", action="store_true",
                    help="keep days whose nearest expiry is that same day (0-DTE IV is distorted, so they are dropped by default)")
    ap.add_argument("--days-per-year", type=int, default=252)
    a = ap.parse_args()
    h, m = map(int, a.cutoff.split(":"))
    r = per_day(load(a.db, a.symbol), h * 60 + m, a.days_per_year)
    if r.empty:
        print("No complete days found. Check --db / --symbol and that the collector has run a full session.")
        return
    r.round(3).to_csv(a.csv, index=False)
    print(f"{len(r)} complete days found ({r.expiry_day.sum()} on expiry day). Per-day rows saved to {a.csv}")
    use = r if a.include_expiry_day else r[~r.expiry_day]
    if len(use) < 3:
        print("Too few non-expiry days to summarise.")
        return
    s = summarise(use)
    print(f"\nUsing {s['days']} days{'' if a.include_expiry_day else ' (expiry days excluded)'}; cutoff {a.cutoff}")
    print(f"  Options implied a high-low range of  {s['avg_implied']:.2f}% on average")
    print(f"  The market actually delivered        {s['avg_realised']:.2f}% on average")
    print(f"  Realised / implied (median day)      {s['median_ratio']:.2f}   |  days realised > implied: {s['share_realised_above_implied']:.0f}%")
    print(f"  Rank corr with realised range:  IV {s['corr_iv_realised']:+.2f} | morning range {s['corr_morning_realised']:+.2f}")
    print(f"  Morning range's extra info beyond IV: {s['morning_extra_beyond_iv']:+.2f}")
    if s["days"] < 60:
        print(f"\n  CAUTION: only {s['days']} days. Differences of this size are mostly noise; keep the collector running "
              "and re-run monthly. Conclusions need roughly 100+ days.")
    ratio = s["median_ratio"]
    print("  Reading: ratio < 1 means options were rich (selling premium paid on average); > 1 means options were cheap "
          "(buying paid). Compare after costs.")


if __name__ == "__main__":
    main()
