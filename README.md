# Nifty 5m similar-day pattern match

At 10:20 (configurable) it compares today's 09:15-10:20 5-minute shape (13 bars + gap + morning high/low)
with every day in the last 5 years and shows the top 20 closest days and what they did until 15:30:
up/down count, scenario buckets (strong up / mild up / sideways / mild down / strong down),
expected high/low/close and time of high/low, and a median path with 25-75% band.

## Deploy on Render (new Web Service, same style as oi-dash)
1. Create a new GitHub repo (e.g. `nifty-pattern`) and upload these files.
2. Render -> New -> Web Service -> that repo.
   - Build: `pip install -r requirements.txt`
   - Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Environment: `DHAN_CLIENT_ID`, `DHAN_ACCESS_TOKEN` (same values as oi-dash). Optional `DATA_DIR=/data`.
4. Optional: add a 1 GB disk mounted at `/data` so the 5-year download is cached (otherwise it re-downloads
   about 16 requests after each restart). A Render disk belongs to one service, so oi-dash's disk cannot be shared.
5. Open the app after 10:20 and press "Refresh data + predict". Without the Dhan variables it runs in demo mode.

Tabs: **Today** (volatility readout first: expected afternoon range, typical/stretch high-low, previous-day candles; the 20-day similar-day table is kept as a reference; replays the last trading day when the market is closed) and **Backtest** (volatility test, does yesterday-range/gap add anything, direction reference).

Command line: `python cli.py --demo` or `python cli.py --cutoff 10:20 --k 20 --weekday`

## Notes
- oi-dash only stores option-chain snapshots since 24 Oct 2026, so 5-year candles come from Dhan's
  `/v2/charts/intraday` (NIFTY id 13, IDX_I, interval 5, about 80 days per request).
- Dhan access tokens expire (about 24 h): refresh the Render env var each day or the live fetch returns 401.
- This is a base-rate tool. Check the "Stability check" line: if top 10/20/50 disagree, there is no edge today.
