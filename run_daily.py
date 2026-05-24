"""
run_daily.py — Daily automation script
=======================================
Fetches data, runs the model, saves report, optionally sends Telegram alert.

Automatically predicts for the next NYSE trading day — handles weekends,
holidays, and arbitrary run times (run on Friday -> predicts Monday).

SCHEDULE (cron / Task Scheduler)
---------------------------------
Run at 9:00 AM ET on weekdays:
    Linux/macOS:  0 14 * * 1-5 cd /path/to/spx-predictor && python run_daily.py
    Windows:      Task Scheduler -> Daily 9:00 AM, Mon-Fri

TELEGRAM ALERTS (optional)
---------------------------
    export TELEGRAM_TOKEN=your_bot_token
    export TELEGRAM_CHAT_ID=your_chat_id
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Import shared trading-day logic from model.py
from model import next_trading_day, is_trading_day


def send_telegram(message: str):
    """Send a message via Telegram bot."""
    token   = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return
    import urllib.request, urllib.parse
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'
    }).encode()
    try:
        urllib.request.urlopen(url, data, timeout=10)
        print("Telegram notification sent.")
    except Exception as e:
        print(f"Telegram failed: {e}")


def format_telegram_message(output: dict) -> str:
    signal_emoji = 'UP' if output['signal'] == 'UP' else 'DOWN'
    return (
        f"<b>SPX Open Predictor</b>\n"
        f"Predicting: {output.get('target_date', 'N/A')} "
        f"(data as of {output.get('data_as_of', 'N/A')})\n\n"
        f"Signal: <b>{signal_emoji}</b>\n"
        f"P(Up):      {output['prob_up']}%\n"
        f"P(Down):    {output['prob_down']}%\n"
        f"Confidence: {output['confidence']}%\n\n"
        f"Last Close: {output.get('last_close', 'N/A')}\n"
        f"VIX:        {output.get('last_vix', 'N/A')}\n\n"
        f"Model Accuracy (walk-forward): {output.get('model_accuracy', 'N/A')}%"
    )


def main():
    today      = datetime.today().date()
    target     = next_trading_day(today)   # e.g. Friday -> Monday, pre-holiday -> day after

    print(f"\n{'='*52}")
    print(f"  SPX Predictor — run date: {today}")
    print(f"  Predicting for:           {target}")
    print(f"{'='*52}\n")

    # Skip if today itself is not a trading day (e.g. Task Scheduler fires on a holiday)
    if not is_trading_day(today):
        print(f"{today} is not a trading day. Skipping.")
        return

    # ── Step 1: Fetch latest data ──────────────────────────────────────────
    print("Step 1: Fetching latest market data...")
    result = subprocess.run(
        [sys.executable, 'fetch_data.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR fetching data:", result.stderr)
        return

    # ── Step 2: Run model and save JSON report for target date ─────────────
    print(f"Step 2: Running model (target: {target})...")
    result = subprocess.run(
        [sys.executable, 'model.py', '--date', str(target), '--report'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR running model:", result.stderr)
        return

    # ── Step 3: Load report and send notification ──────────────────────────
    report_path = Path('reports') / f"prediction_{target}.json"
    if report_path.exists():
        with open(report_path) as f:
            output = json.load(f)
        send_telegram(format_telegram_message(output))
        print(f"\nReport saved: {report_path}")
        print(f"Signal: {output['signal']} | P(Up): {output['prob_up']}%")
    else:
        print(f"Report not found at {report_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
