"""Download Nifty 5-minute candles from Dhan (5 years) and keep an incremental CSV cache.

Env: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN (same ones oi-dash uses), DATA_DIR (default ./data)
Dhan intraday endpoint: POST /v2/charts/intraday, max ~90 days per request.
"""
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

BASE = "https://api.dhan.co/v2"
CID = os.getenv("DHAN_CLIENT_ID")
TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DATA_DIR = os.getenv("DATA_DIR", "data")
CACHE = os.path.join(DATA_DIR, "nifty_5m.csv")
NIFTY_ID = "13"
CHUNK_DAYS = 80


def _post(body):
    for _ in range(4):
        r = requests.post(BASE + "/charts/intraday", json=body, timeout=30,
                          headers={"access-token": TOKEN, "client-id": CID,
                                   "Content-Type": "application/json", "Accept": "application/json"})
        if r.status_code == 429:
            time.sleep(3)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Dhan rate limit (429) kept repeating")


def _to_ist(ts):
    """Dhan epoch -> naive IST bar-start time. Auto-detects 1980-epoch and UTC-vs-IST encoding."""
    ts = np.asarray(ts, dtype="float64")
    t = pd.to_datetime(ts, unit="s")
    if t.min().year < 2000:                      # legacy 1980 epoch
        t = t + pd.Timedelta(seconds=315532800)
    first = (t.hour * 60 + t.minute)
    # NSE opens 09:15 IST = 03:45 UTC. If the most common first-of-day time is 03:45, it is true UTC.
    s = pd.Series(first, index=t)
    day_first = s.groupby(t.date).min()
    if (day_first == 225).mean() > 0.5:
        t = t + pd.Timedelta(hours=5, minutes=30)
    return t


def fetch_range(d0, d1, security_id=NIFTY_ID):
    body = {"securityId": security_id, "exchangeSegment": "IDX_I", "instrument": "INDEX", "interval": "5",
            "fromDate": f"{d0:%Y-%m-%d} 09:15:00", "toDate": f"{d1:%Y-%m-%d} 15:30:00"}
    j = _post(body)
    if not j.get("timestamp"):
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c"])
    return pd.DataFrame({"ts": _to_ist(j["timestamp"]), "o": j["open"], "h": j["high"],
                         "l": j["low"], "c": j["close"]})


def load_cache():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE, parse_dates=["ts"])
    return pd.DataFrame(columns=["ts", "o", "h", "l", "c"])


def update_history(years=5, progress=None):
    """Fetch only what is missing, append to cache, return full DataFrame (excludes nothing)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    hist = load_cache()
    today = date.today() - timedelta(days=1)   # cache holds completed days only; today comes live
    start = (hist.ts.max().date() + timedelta(days=1)) if len(hist) else today - timedelta(days=365 * years)
    parts = [hist]
    d = start
    while d <= today:
        e = min(d + timedelta(days=CHUNK_DAYS), today)
        if progress:
            progress(f"Downloading {d} -> {e}")
        parts.append(fetch_range(d, e))
        d = e + timedelta(days=1)
        time.sleep(0.4)
    full = pd.concat(parts).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    full.to_csv(CACHE, index=False)
    return full


def fetch_today():
    t = date.today()
    return fetch_range(t, t)


if __name__ == "__main__":
    h = update_history(progress=print)
    print(len(h), "bars", h.ts.min(), "->", h.ts.max())
