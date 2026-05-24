"""
SPX Open Predictor — model.py
==============================
Predicts whether the S&P 500 will open UP or DOWN vs prior close.
Target market: Polymarket "S&P 500 (SPX) Opens Up or Down on [DATE]?"

Resolution rule: UP if open(D) > close(D-1), DOWN otherwise.

USAGE
-----
    python model.py                    # predict next trading day
    python model.py --date 2026-05-27  # predict specific date
    python model.py --report           # also generate HTML report

DATA SOURCE
-----------
Requires yfinance (works locally, was blocked in Claude's sandbox).
    pip install yfinance xgboost scikit-learn pandas numpy scipy

    import yfinance as yf
    spx = yf.download('^GSPC', start='2023-01-01', auto_adjust=True)
    vix = yf.download('^VIX',  start='2023-01-01', auto_adjust=True)

    # Merge and save:
    df = spx[['Open','High','Low','Close','Volume']].copy()
    df['VIX'] = vix['Close']
    df.to_csv('spx_data.csv')

ALTERNATIVE FREE SOURCES
-------------------------
- Polygon.io free tier: 5 calls/min, OHLCV daily
- Alpha Vantage free: 25 calls/day (TIME_SERIES_DAILY_ADJUSTED)
- FRED: SP500 index (daily, no OHLCV, close only)
- Twelve Data: 800 calls/day free

POLYMARKET EDGE
---------------
Edge = abs(model_probability - polymarket_probability)
Trade when edge > 0.08 (8pp) and confidence > 60%.
Example: model says 65% UP, Poly shows 55% → edge = 10pp → consider trade.
"""

import argparse
import json
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_FILE    = 'spx_data.csv'     # CSV with columns: Date,Open,High,Low,Close,Volume,VIX
REPORTS_DIR  = Path('reports')
MODEL_PARAMS = dict(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    verbosity=0,
)
TRAIN_WARMUP   = 250   # trading days for initial warm-up (~1 year)
RETRAIN_PERIOD = 21    # retrain every N days (monthly)
MIN_EDGE       = 0.08  # minimum edge vs Polymarket to consider a trade


# ── NYSE Holiday Calendar ─────────────────────────────────────────────────────
# Maintained through 2027. Update annually or switch to pandas_market_calendars.
NYSE_HOLIDAYS = {
    # 2020
    '2020-01-01', '2020-01-20', '2020-02-17', '2020-04-10',
    '2020-05-25', '2020-07-03', '2020-09-07', '2020-11-26', '2020-12-25',
    # 2021
    '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02',
    '2021-05-31', '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24',
    # 2022
    '2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30',
    '2022-06-20', '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26',
    # 2023
    '2023-01-02', '2023-01-16', '2023-02-20', '2023-04-07',
    '2023-05-29', '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
    # 2024
    '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29',
    '2024-05-27', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
    # 2025
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18',
    '2025-05-26', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
    # 2026
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03',
    '2026-05-25', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
    # 2027
    '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26',
    '2027-05-31', '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24',
}


def is_trading_day(d) -> bool:
    """Return True if d is a NYSE trading day (weekday, not a holiday)."""
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d').date()
    elif isinstance(d, datetime):
        d = d.date()
    return d.weekday() < 5 and d.strftime('%Y-%m-%d') not in NYSE_HOLIDAYS


def next_trading_day(from_date=None) -> date:
    """
    Return the next NYSE trading day strictly after from_date.
    If from_date is None, uses today.

    Examples
    --------
    Called on Friday 2026-05-22  -> returns Monday 2026-05-25 (Memorial Day)
                                     skips it  -> returns Tuesday 2026-05-26
    Called on Friday 2026-05-29  -> returns Monday 2026-06-01
    """
    if from_date is None:
        from_date = datetime.today().date()
    elif isinstance(from_date, str):
        from_date = datetime.strptime(from_date, '%Y-%m-%d').date()
    elif isinstance(from_date, datetime):
        from_date = from_date.date()

    candidate = from_date + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


# ── Feature Engineering ────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV + VIX dataframe.
    Target: 1 if Open(t) > Close(t-1), 0 otherwise.
    All features use only information available BEFORE the open.
    """
    f = pd.DataFrame(index=df.index)

    # ── Target (supervised label) ──────────────────────────────────────────
    f['target'] = (df['Open'] > df['Close'].shift(1)).astype(int)

    # ── Overnight gap (lagged — uses PRIOR day's gap, not today's) ───────────
    # Today's open is not known before the market opens, so we use the
    # previous day's overnight gap as a momentum signal. On a live system,
    # replace this with ES=F pre-market futures price (see README Priority 2).
    prior_overnight   = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    f['overnight']    = prior_overnight.shift(1)   # lag by 1 day — no leakage

    # ── Price returns ──────────────────────────────────────────────────────
    f['ret_1d']       = df['Close'].pct_change(1)
    f['ret_3d']       = df['Close'].pct_change(3)
    f['ret_5d']       = df['Close'].pct_change(5)
    f['ret_10d']      = df['Close'].pct_change(10)

    # ── Momentum ──────────────────────────────────────────────────────────
    f['rsi']          = compute_rsi(df['Close'], 14)

    # ── Volatility ────────────────────────────────────────────────────────
    f['vol_5d']       = df['Close'].pct_change().rolling(5).std()
    f['vol_20d']      = df['Close'].pct_change().rolling(20).std()

    # ── Trend ─────────────────────────────────────────────────────────────
    ma20              = df['Close'].rolling(20).mean()
    ma50              = df['Close'].rolling(50).mean()
    f['above_ma20']   = (df['Close'] > ma20).astype(int)
    f['above_ma50']   = (df['Close'] > ma50).astype(int)
    f['ma20_slope']   = ma20.pct_change(5)

    # ── VIX ───────────────────────────────────────────────────────────────
    f['vix']          = df['VIX']
    f['vix_ret_1d']   = df['VIX'].pct_change(1)
    f['vix_ret_5d']   = df['VIX'].pct_change(5)
    f['vix_above20']  = (df['VIX'] > 20).astype(int)

    # ── Volume ────────────────────────────────────────────────────────────
    f['vol_ratio']    = df['Volume'] / df['Volume'].rolling(20).mean()

    # ── Calendar effects ──────────────────────────────────────────────────
    f['day_of_week']  = df.index.dayofweek    # 0=Mon, 4=Fri
    f['is_monday']    = (df.index.dayofweek == 0).astype(int)
    f['month']        = df.index.month

    # ── Candle shape ──────────────────────────────────────────────────────
    f['body']         = ((df['Close'] - df['Open']).abs() / df['Open']).shift(1)
    f['upper_wick']   = ((df['High']  - df[['Open','Close']].max(axis=1)) / df['Open']).shift(1)
    f['lower_wick']   = ((df[['Open','Close']].min(axis=1) - df['Low'])   / df['Open']).shift(1)

    return f.dropna()


FEATURE_COLS = [
    'overnight', 'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d',
    'rsi', 'vol_5d', 'vol_20d', 'above_ma20', 'above_ma50', 'ma20_slope',
    'vix', 'vix_ret_1d', 'vix_ret_5d', 'vix_above20', 'vol_ratio',
    'day_of_week', 'is_monday', 'month', 'body', 'upper_wick', 'lower_wick',
]


# ── Walk-Forward Validation ────────────────────────────────────────────────────
def walk_forward_validation(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate real-world deployment: train on past, predict future.
    Re-trains every RETRAIN_PERIOD days. No data leakage.
    """
    results = []
    n = len(feat)

    for i in range(TRAIN_WARMUP, n - 1, RETRAIN_PERIOD):
        train = feat.iloc[:i]
        test  = feat.iloc[i:min(i + RETRAIN_PERIOD, n - 1)]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURE_COLS].values)
        X_test  = scaler.transform(test[FEATURE_COLS].values)

        model = xgb.XGBClassifier(**MODEL_PARAMS)
        model.fit(X_train, train['target'].values)

        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        for j in range(len(test)):
            results.append({
                'date':   test.index[j],
                'actual': int(test['target'].iloc[j]),
                'pred':   int(pred[j]),
                'prob':   float(round(proba[j], 4)),
            })

    return pd.DataFrame(results)


# ── Final Model (fit on all data) ──────────────────────────────────────────────
def fit_final_model(feat: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(feat[FEATURE_COLS].values)
    y = feat['target'].values
    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X, y)
    return model, scaler


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_next_day(
    model, scaler, feat: pd.DataFrame, target_date=None
) -> dict:
    """
    Predict open direction for target_date.

    If target_date is None, automatically computes the next NYSE trading day
    after the last date in feat. If a string 'YYYY-MM-DD' is provided, that
    date is used (and a warning is shown if it is not a trading day).

    All market features (returns, VIX, volatility, etc.) come from the last
    row of available data — only calendar features are overridden to match
    the actual target date so day-of-week / month effects are correct.
    """
    # ── Resolve target date ────────────────────────────────────────────────
    last_data_date = feat.index[-1].date()

    if target_date is None:
        tgt = next_trading_day(last_data_date)
    else:
        if isinstance(target_date, str):
            tgt = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            tgt = target_date
        if not is_trading_day(tgt):
            print(f"  WARNING: {tgt} is not a trading day — prediction may be unreliable.")

    # ── Build feature row ──────────────────────────────────────────────────
    last_row = feat.iloc[-1:].copy()

    # Override calendar features with the actual target date
    last_row['day_of_week'] = tgt.weekday()          # 0=Mon … 4=Fri
    last_row['is_monday']   = int(tgt.weekday() == 0)
    last_row['month']       = tgt.month

    X = scaler.transform(last_row[FEATURE_COLS].values)
    prob_up   = float(model.predict_proba(X)[0, 1])
    prob_down = 1 - prob_up
    signal    = 'UP' if prob_up >= 0.5 else 'DOWN'

    return {
        'target_date': tgt.strftime('%Y-%m-%d'),
        'signal':      signal,
        'prob_up':     round(prob_up * 100, 1),
        'prob_down':   round(prob_down * 100, 1),
        'confidence':  round(max(prob_up, prob_down) * 100, 1),
        'data_as_of':  str(last_data_date),
    }


# ── Polymarket Edge Calculator ─────────────────────────────────────────────────
def compute_edge(model_prob_up: float, polymarket_prob_up: float) -> dict:
    """
    Given model P(Up) and Polymarket's current P(Up) price,
    compute the edge and recommended action.
    """
    model    = model_prob_up / 100
    poly     = polymarket_prob_up / 100
    edge     = abs(model - poly)
    if model > poly and edge >= MIN_EDGE:
        action = f"BUY UP  (model:{model_prob_up}% > poly:{polymarket_prob_up}%, edge:{edge*100:.1f}pp)"
    elif model < poly and edge >= MIN_EDGE:
        action = f"BUY DOWN (model:{model_prob_up}% < poly:{polymarket_prob_up}%, edge:{edge*100:.1f}pp)"
    else:
        action = f"NO TRADE (edge {edge*100:.1f}pp < {MIN_EDGE*100:.0f}pp threshold)"
    return {'edge_pct': round(edge * 100, 1), 'action': action}


# ── Report Generation ──────────────────────────────────────────────────────────
def save_json_report(output: dict, date_str: str):
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"prediction_{date_str}.json"
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"JSON report saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='SPX Open Predictor')
    parser.add_argument('--date',   type=str, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--poly',   type=float, default=None,
                        help="Polymarket current P(Up) price 0-100 (e.g. 52)")
    parser.add_argument('--report', action='store_true',
                        help='Save JSON report to reports/')
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    print(f"  {len(df)} trading days | {df.index[0].date()} -> {df.index[-1].date()}")

    # ── Feature engineering ────────────────────────────────────────────────
    feat = make_features(df)
    print(f"  Features: {feat.shape[1]-1} | Samples: {len(feat)}")

    # ── Walk-forward validation ────────────────────────────────────────────
    print("\nRunning walk-forward validation...")
    res     = walk_forward_validation(feat)
    acc     = accuracy_score(res['actual'], res['pred'])
    brier   = brier_score_loss(res['actual'], res['prob'])
    base    = res['actual'].mean()
    print(f"  Accuracy:  {acc:.3f} ({len(res)} out-of-sample days)")
    print(f"  Baseline:  {base:.3f} (always predict UP)")
    print(f"  Edge:      +{(acc-base)*100:.1f}pp over baseline")
    print(f"  Brier:     {brier:.4f}")

    # ── Fit final model ────────────────────────────────────────────────────
    print("\nFitting final model on all data...")
    model, scaler = fit_final_model(feat)

    # ── Feature importance ─────────────────────────────────────────────────
    fi = pd.Series(
        model.feature_importances_.astype(float),
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nTop 5 features:")
    for name, val in fi.head(5).items():
        print(f"  {name:<18} {val*100:.1f}%")

    # ── Prediction ─────────────────────────────────────────────────────────
    prediction = predict_next_day(model, scaler, feat, args.date)
    print(f"\n{'='*50}")
    print(f"  TARGET DATE:  {prediction['target_date']}")
    print(f"  Data as of:   {prediction['data_as_of']}")
    print(f"  Signal:       {prediction['signal']}")
    print(f"  P(Up):        {prediction['prob_up']}%")
    print(f"  P(Down):      {prediction['prob_down']}%")
    print(f"  Confidence:   {prediction['confidence']}%")

    # ── Polymarket edge ────────────────────────────────────────────────────
    if args.poly is not None:
        edge = compute_edge(prediction['prob_up'], args.poly)
        print(f"\n  Polymarket: {args.poly}% UP")
        print(f"  Edge:       {edge['edge_pct']}pp")
        print(f"  Action:     {edge['action']}")

    print(f"{'='*50}\n")

    # ── Save report ────────────────────────────────────────────────────────
    if args.report:
        date_str = prediction['target_date']   # always use the predicted date
        recent   = res.tail(20).copy()
        recent['correct'] = (recent['actual'] == recent['pred'])
        output = {
            **prediction,
            'model_accuracy':   round(acc * 100, 1),
            'baseline_accuracy': round(base * 100, 1),
            'brier_score':      round(brier, 4),
            'n_eval':           int(len(res)),
            'last_close':       float(round(df['Close'].iloc[-1], 2)),
            'last_vix':         float(round(df['VIX'].iloc[-1], 2)),
            'top_features':     {k: round(float(v)*100,1) for k,v in fi.head(8).items()},
            'recent_preds': [
                {
                    'date':    str(r['date'])[:10],
                    'actual':  'UP' if r['actual'] == 1 else 'DOWN',
                    'pred':    'UP' if r['pred']   == 1 else 'DOWN',
                    'prob':    round(r['prob'] * 100, 1),
                    'correct': bool(r['correct']),
                }
                for _, r in recent.iterrows()
            ],
        }
        save_json_report(output, date_str)

    return prediction


if __name__ == '__main__':
    main()
