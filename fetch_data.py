"""
fetch_data.py -- Download and update SPX + VIX data
====================================================
Calls the Yahoo Finance v8 chart API directly (no yfinance dependency needed
for the default source) so it works on Windows systems with SSL cert issues.

USAGE
-----
    python fetch_data.py              # update existing CSV (append new rows)
    python fetch_data.py --full       # re-download full history from 2020
    python fetch_data.py --source av  # use Alpha Vantage instead
    python fetch_data.py --source polygon  # use Polygon.io

SOURCES
-------
1. Yahoo Finance direct API (default, free, no API key)
   pip install curl_cffi pandas

2. Alpha Vantage (free, 25 calls/day)
   pip install alpha-vantage
   Set env var: AV_API_KEY=your_key

3. Polygon.io (free tier, 5 calls/min)
   pip install polygon-api-client
   Set env var: POLYGON_KEY=your_key
"""

import os
import ssl
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

DATA_FILE  = 'spx_data.csv'
START_DATE = '2020-01-01'

YF_CHART_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
YF_HEADERS   = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'application/json',
}


# ── Yahoo Finance direct API ───────────────────────────────────────────────────
def _yf_chart(ticker: str, start: str, end: str, session) -> dict:
    """Fetch OHLCV from Yahoo Finance v8 chart endpoint."""
    start_ts = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
    end_ts   = int(datetime.strptime(end,   '%Y-%m-%d').timestamp()) + 86400

    url = YF_CHART_URL.format(ticker=ticker.replace('^', '%5E'))
    params = {
        'interval':  '1d',
        'period1':   start_ts,
        'period2':   end_ts,
        'events':    'div,splits',
        'includeAdjustedClose': 'true',
    }
    resp = session.get(url, params=params, headers=YF_HEADERS)
    resp.raise_for_status()
    return resp.json()


def _parse_yf_chart(data: dict) -> pd.DataFrame:
    result = data['chart']['result'][0]
    ts     = result['timestamp']
    q      = result['indicators']['quote'][0]
    adj    = result['indicators'].get('adjclose', [{}])[0]

    df = pd.DataFrame({
        'Date':   pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York').normalize().tz_localize(None),
        'Open':   q['open'],
        'High':   q['high'],
        'Low':    q['low'],
        'Close':  adj.get('adjclose', q['close']),
        'Volume': q['volume'],
    }).set_index('Date').dropna()
    df.index.name = 'Date'
    return df.round(2)


def fetch_yahoo(start: str, end: str = None) -> pd.DataFrame:
    try:
        from curl_cffi import requests as creq
        session = creq.Session(verify=False)
    except ImportError:
        import requests, urllib3
        urllib3.disable_warnings()
        session = requests.Session()
        session.verify = False

    end = end or (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Downloading SPX {start} -> {end} via Yahoo Finance API...")

    spx_data = _yf_chart('^GSPC', start, end, session)
    time.sleep(0.5)
    vix_data = _yf_chart('^VIX',  start, end, session)

    spx = _parse_yf_chart(spx_data)
    vix = _parse_yf_chart(vix_data)[['Close']].rename(columns={'Close': 'VIX'})

    df = spx[['Open', 'High', 'Low', 'Close', 'Volume']].join(vix, how='inner')
    df = df.dropna(subset=['Close', 'VIX'])
    print(f"  Downloaded {len(df)} rows.")
    return df


# ── Alpha Vantage source ───────────────────────────────────────────────────────
def fetch_alpha_vantage(start: str, end: str = None) -> pd.DataFrame:
    api_key = os.getenv('AV_API_KEY')
    if not api_key:
        sys.exit("Set AV_API_KEY environment variable.")

    try:
        from alpha_vantage.timeseries import TimeSeries
    except ImportError:
        sys.exit("Install alpha-vantage: pip install alpha-vantage")

    ts = TimeSeries(key=api_key, output_format='pandas')
    print("Downloading SPY via Alpha Vantage...")
    data, _ = ts.get_daily_adjusted('SPY', outputsize='full')
    data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
    data.index   = pd.to_datetime(data.index)
    data.index.name = 'Date'
    data = data[data.index >= start].sort_index()
    print("  Note: VIX not available on AV free tier -- using static 18.")
    data['VIX'] = 18.0
    return data[['Open', 'High', 'Low', 'Close', 'Volume', 'VIX']].round(2)


# ── Polygon.io source ──────────────────────────────────────────────────────────
def fetch_polygon(start: str, end: str = None) -> pd.DataFrame:
    api_key = os.getenv('POLYGON_KEY')
    if not api_key:
        sys.exit("Set POLYGON_KEY environment variable.")

    try:
        from polygon import RESTClient
    except ImportError:
        sys.exit("Install polygon: pip install polygon-api-client")

    client = RESTClient(api_key)
    end    = end or datetime.today().strftime('%Y-%m-%d')
    print(f"Downloading SPX via Polygon {start} -> {end}...")

    bars = []
    for agg in client.list_aggs('SPX', 1, 'day', start, end, limit=50000):
        bars.append({
            'Date':   pd.Timestamp(agg.timestamp, unit='ms'),
            'Open':   agg.open,
            'High':   agg.high,
            'Low':    agg.low,
            'Close':  agg.close,
            'Volume': agg.volume,
            'VIX':    18.0,
        })

    df = pd.DataFrame(bars).set_index('Date').sort_index()
    print(f"  Downloaded {len(df)} rows.")
    return df.round(2)


# ── Incremental update ─────────────────────────────────────────────────────────
def update_existing(source_fn) -> pd.DataFrame:
    if not Path(DATA_FILE).exists():
        print(f"{DATA_FILE} not found. Downloading full history...")
        return source_fn(START_DATE)

    existing  = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    last_date = existing.index[-1]
    new_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    today     = datetime.today().strftime('%Y-%m-%d')

    if new_start >= today:
        print("Data already up to date.")
        return existing

    print(f"Updating from {new_start}...")
    new_data = source_fn(new_start)
    combined = pd.concat([existing, new_data])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    print(f"  Added {len(new_data)} new rows. Total: {len(combined)}")
    return combined


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Fetch SPX data')
    parser.add_argument('--full',   action='store_true', help='Full re-download from 2020')
    parser.add_argument('--source', choices=['yf', 'av', 'polygon'], default='yf',
                        help='Data source (default: yf = Yahoo Finance direct API)')
    args = parser.parse_args()

    source_map = {'yf': fetch_yahoo, 'av': fetch_alpha_vantage, 'polygon': fetch_polygon}
    source_fn  = source_map[args.source]

    if args.full:
        df = source_fn(START_DATE)
    else:
        df = update_existing(source_fn)

    if df.empty:
        print("ERROR: No data downloaded. Check your internet connection.")
        sys.exit(1)

    df.to_csv(DATA_FILE)
    print(f"\nSaved {len(df)} rows to {DATA_FILE}")
    print(f"  Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Last close: {df['Close'].iloc[-1]:.2f}")
    print(f"  Last VIX:   {df['VIX'].iloc[-1]:.2f}")


if __name__ == '__main__':
    main()
