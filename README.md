# S&P 500 Open Predictor

> Binary classification model that predicts whether the S&P 500 will open **UP or DOWN** relative to the prior close.  
> Built to generate edge against Polymarket's daily *"S&P 500 Opens Up or Down"* markets.

---

## Results (walk-forward out-of-sample, 2021–2026)

| Metric | Value |
|---|---|
| **Accuracy** | **70.2%** |
| Baseline (always predict UP) | 56.5% |
| Edge over baseline | **+13.7 pp** |
| AUC-ROC | 0.751 |
| Brier Score | 0.201 |
| McFadden R² | 0.133 |
| Nagelkerke R² | 0.224 |
| LR chi² (F-analog) | 354.04 (df=22, p<0.001) |
| Evaluation window | 1,331 out-of-sample trading days |

**Per-year accuracy** — consistent across regimes:

| Year | N | Accuracy | % UP days | AUC |
|---|---|---|---|---|
| 2021 | 231 | 71.4% | 62.3% | — |
| 2022 | 251 | 67.7% | 45.0% | — |
| 2023 | 250 | 70.0% | 52.0% | — |
| 2024 | 252 | 71.0% | 64.3% | — |
| 2025 | 250 | 70.8% | 59.6% | — |
| 2026 | 97  | 71.1% | 55.7% | — |

> **On real SPX data**, the 70%+ accuracy is driven primarily by short-term return momentum (`ret_1d`, OR=3.17, p<0.001) and volatility regime signals (VIX level, RSI).  
> The model is **not** overfit — accuracy is stable in bear markets (2022), bull markets (2021, 2024), and high-volatility regimes.

---

## Repository structure

```
spx-predictor/
├── fetch_data.py       # Download SPX + VIX data from Yahoo Finance
├── model.py            # Feature engineering, XGBoost, walk-forward validation
├── generate_stats.py   # Full econometric report (p-values, pseudo-R², calibration)
├── run_daily.py        # Daily orchestrator: fetch → predict → report → Telegram
├── requirements.txt    # Python dependencies
└── README.md
```

Generated files (excluded from git via `.gitignore`):

```
spx_data.csv                        # Historical OHLCV + VIX (1,606 rows)
reports/prediction_YYYY-MM-DD.json  # Daily prediction outputs
spx_stats_report_YYYY-MM-DD.html   # Statistical summary report
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/your-username/spx-predictor.git
cd spx-predictor
pip install -r requirements.txt

# 2. Download historical data (free, no API key needed)
python fetch_data.py --full

# 3. Run prediction for the next trading day
python model.py

# 4. Generate the full statistical report
python generate_stats.py

# 5. Compare against Polymarket (e.g. Poly shows 55% UP)
python model.py --poly 55

# 6. Run everything daily (fetch → predict → report → Telegram alert)
python run_daily.py
```

---

## How it works

### Resolution rule

> **UP** if `open(D) > close(D-1)` &nbsp;|&nbsp; **DOWN** if `open(D) ≤ close(D-1)`

This mirrors the Polymarket resolution rule for *"S&P 500 Opens Up or Down on [DATE]?"*.

---

### Feature engineering

All 22 features use only information available **before** the market opens — no lookahead bias.

| Group | Feature(s) | Notes |
|---|---|---|
| Overnight gap | `overnight` | Prior day's open-vs-close gap (lagged 1d — not today's open) |
| Price returns | `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d` | Log-price momentum |
| Momentum | `rsi` | 14-period RSI on close |
| Volatility | `vol_5d`, `vol_20d` | Realized vol (rolling std of daily returns) |
| Trend | `above_ma20`, `above_ma50`, `ma20_slope` | Price position and MA slope |
| VIX | `vix`, `vix_ret_1d`, `vix_ret_5d`, `vix_above20` | Fear gauge level and changes |
| Volume | `vol_ratio` | Volume vs 20-day average |
| Calendar | `day_of_week`, `is_monday`, `month` | Seasonality effects |
| Candle shape | `body`, `upper_wick`, `lower_wick` | Prior day's candle (lagged 1d) |

> **Leakage-free design**: `overnight`, `body`, `upper_wick`, and `lower_wick` are all shifted by one day because they depend on the current day's open/close which is not available before market open. On a live system, replace `overnight` with the ES=F (S&P 500 e-mini futures) pre-market price for a stronger signal.

---

### Model

**Algorithm:** XGBoost binary classifier

```python
XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

**Validation:** Expanding-window walk-forward (no data leakage)
- Warm-up period: 250 trading days (~1 year)
- Retrain frequency: every 21 trading days (~monthly)
- Each prediction is made on data the model has never seen

---

### Significant features (multivariate logistic regression)

| Feature | Coeff | Odds Ratio | p-value | Sig |
|---|---|---|---|---|
| 1-day return | +1.153 | 3.167 | <0.001 | *** |
| Realized vol 20d | −0.355 | 0.701 | 0.025 | * |
| RSI (14) | −0.229 | 0.795 | 0.035 | * |
| VIX > 20 | −0.201 | 0.818 | 0.035 | * |
| VIX level | +0.380 | 1.462 | 0.042 | * |
| Price > MA20 | +0.186 | 1.205 | 0.076 | . |

> Significance: `*** p<0.001` `** p<0.01` `* p<0.05` `. p<0.10`

The dominant signal is **short-term return momentum** (`ret_1d`, OR=3.17) — the market tends to open in the same direction it closed. Volatility-regime features (VIX, RSI, realized vol) add incremental discriminatory power.

Features not significant at 5% in the multivariate model: `ret_3d`, `ret_5d`, `ret_10d`, `vol_ratio`, `above_ma50`, `body`, candle wicks, calendar effects. These are candidates for pruning in a v2.

---

### Polymarket edge strategy

```
edge = |model_P(Up) - polymarket_P(Up)|
```

Consider a position when **edge > 8 pp** and model confidence > 60%.

```
Example
-------
Model:      65% UP
Polymarket: 53% UP
Edge:       12 pp  →  BUY UP
```

```bash
# Pass the current Polymarket price to get the edge calculation
python model.py --poly 53
```

---

## Data sources

### Default — Yahoo Finance (no API key)

```bash
python fetch_data.py --full          # full history from 2020
python fetch_data.py                 # incremental update (append new rows only)
```

Fetches `^GSPC` (SPX) and `^VIX` directly from the Yahoo Finance v8 chart API using `curl_cffi`. Works on Windows without SSL certificate issues.

### Alternative — Alpha Vantage

```bash
export AV_API_KEY=your_key
python fetch_data.py --source av
```

Uses SPY as a proxy (free tier does not provide the SPX index). VIX is a static placeholder.

### Alternative — Polygon.io

```bash
export POLYGON_KEY=your_key
python fetch_data.py --source polygon
```

Free tier: 5 calls/min, 2 years of history.

---

## Daily automation

### Linux / macOS (cron)

```bash
# Run at 9:00 AM ET Monday–Friday
0 14 * * 1-5 cd /path/to/spx-predictor && python run_daily.py >> logs/daily.log 2>&1
```

### Windows (Task Scheduler)

Create a basic task:
- **Trigger:** Daily, 9:00 AM (repeat Mon–Fri)
- **Action:** `python C:\path\to\spx-predictor\run_daily.py`

### Telegram alerts (optional)

```bash
export TELEGRAM_TOKEN=your_bot_token
export TELEGRAM_CHAT_ID=your_chat_id
python run_daily.py
```

Get a bot token from [@BotFather](https://t.me/BotFather) on Telegram.

---

## Statistical report

```bash
python generate_stats.py
# → spx_stats_report_YYYY-MM-DD.html
```

The HTML report includes:

- **Model-level stats**: McFadden R², Nagelkerke R², Cox-Snell R², LR chi² (F-analog), AIC, BIC
- **Goodness-of-fit**: Hosmer-Lemeshow test
- **Per-feature**: Wald Z-statistic, p-value, odds ratio, 95% confidence interval
- **Univariate tests**: point-biserial correlation, Mann-Whitney U, univariate logistic OR — for each of the 22 features
- **XGBoost importances**: gain, cover, weight
- **Walk-forward metrics**: accuracy, precision, recall, F1, AUC-ROC, Brier, log-loss, average precision
- **Confusion matrix** and per-year accuracy breakdown
- **Calibration curve**: predicted vs actual UP rate (interactive Chart.js)

---

## Requirements

```
yfinance>=0.2.40        # used internally; direct API call handles SSL
xgboost>=2.0.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
statsmodels>=0.14.0     # logistic regression + Hosmer-Lemeshow
curl_cffi>=0.6.0        # SSL-tolerant HTTP client for Yahoo Finance API

# Optional
alpha-vantage>=2.3.1
polygon-api-client>=1.12.0
pandas-market-calendars>=4.3.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Known limitations & roadmap

| Priority | Item | Status |
|---|---|---|
| 1 | ES=F pre-market futures as the `overnight` feature | Planned |
| 2 | Probability calibration (`CalibratedClassifierCV`) | Planned |
| 3 | Macro calendar flags (FOMC, CPI, NFP) | Planned |
| 4 | VIX term structure (VIX9D vs VIX3M spread) | Planned |
| 5 | Put/Call ratio, DXY | Planned |
| 6 | Feature selection (drop non-significant features) | Planned |

**Current `overnight` feature caveat**: today's open price is not available before the market opens, so `overnight` is lagged by one day (uses yesterday's open-vs-close gap as a momentum proxy). Replacing it with ES=F futures at 8:30 AM ET would be the biggest single accuracy improvement.

---

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past predictive accuracy does not guarantee future results. Prediction markets carry risk of total loss of capital.

---

## License

MIT — see [LICENSE](LICENSE).
