# Nifty 5m afternoon volatility readout

At 10:20 (configurable) it measures today's 09:15-10:20 high-low range and finds the past days with the closest
morning range. From those it shows the expected size of the move from 10:20 to the close (typical and stretch
high/low, odds of a bigger-than-usual or quiet afternoon). It does not predict direction: that tested as a coin flip.

## Deploy on Render (new Web Service, same style as oi-dash)
1. Create a new GitHub repo (e.g. `nifty-pattern`) and upload these files.
2. Render -> New -> Web Service -> that repo.
   - Build: `pip install -r requirements.txt`
   - Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Environment: `DHAN_CLIENT_ID`, `DHAN_ACCESS_TOKEN` (same values as oi-dash). Optional `DATA_DIR=/data`.
4. Optional: add a 1 GB disk mounted at `/data` so the 5-year download is cached (otherwise it re-downloads
   about 16 requests after each restart). A Render disk belongs to one service, so oi-dash's disk cannot be shared.
5. Open the app after 10:20 and press "Refresh data". Without the Dhan variables it runs in demo mode.

Tabs: **Today** (readout, expected high/low lines on the chart with previous-day candles; replays the last trading day when the market is closed) and **Backtest** (does the morning range predict the afternoon range, do yesterday's range / gap add anything).

Command line: `python cli.py --demo` or `python cli.py --cutoff 10:20 --k 150`

## Notes
- oi-dash only stores option-chain snapshots since 24 Oct 2026, so 5-year candles come from Dhan's
  `/v2/charts/intraday` (NIFTY id 13, IDX_I, interval 5, about 80 days per request).
- Dhan access tokens expire (about 24 h): refresh the Render env var each day or the live fetch returns 401.
- Morning range explains only about 10-15% of afternoon range variation out-of-sample: a quiet morning can still end in a big day.

## IV check (runs inside oi-dash, not this app)
`iv_compare.py` reads oi-dash's own `oi.db` (snap table: 1-minute spot + ATM implied volatility) and compares the range the options
implied for 10:20 -> close with the range that really happened. Copy it into the oi-dash repo, then in the oi-dash Render Shell:
`python iv_compare.py --db $DB_PATH`  (add `--cutoff 10:20`, `--csv out.csv`). Needs ~100+ complete days before conclusions mean anything.
