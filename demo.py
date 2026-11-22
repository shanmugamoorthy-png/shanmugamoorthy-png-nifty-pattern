"""Synthetic 5-minute Nifty-like data so the app/engine can be tried without Dhan credentials."""
import numpy as np
import pandas as pd


def make_demo(days=1250, seed=7, end="2026-11-20"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=days)
    rows, px = [], 15000.0
    for d in dates:
        gap = rng.normal(0, 0.4) / 100
        px *= 1 + gap
        drift = rng.normal(0, 0.02) / 100
        vol = rng.uniform(0.04, 0.10) / 100
        t0 = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15)
        for i in range(75):
            o = px
            c = o * (1 + drift + rng.normal(0, vol))
            h = max(o, c) * (1 + abs(rng.normal(0, vol / 2)))
            l = min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
            rows.append((t0 + pd.Timedelta(minutes=5 * i), o, h, l, c))
            px = c
    return pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"])


def make_today(hist, up_to_min=620, seed=3, date="2026-11-23"):
    rng = np.random.default_rng(seed)
    px = hist.c.iloc[-1] * 1.002
    t0 = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=15)
    rows = []
    for i in range((up_to_min - 555) // 5):
        o = px
        c = o * (1 + 0.0002 + rng.normal(0, 0.0006))
        rows.append((t0 + pd.Timedelta(minutes=5 * i), o, max(o, c) * 1.0003, min(o, c) * 0.9997, c))
        px = c
    return pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"])
