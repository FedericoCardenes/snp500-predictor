"""
run_daily.py — Daily automation script
=======================================
Fetches data, runs the model, saves report, optionally sends Telegram alert.

SCHEDULE (cron)
---------------
Run at 9:00 AM ET on trading days (Mon-Fri):
    0 9 * * 1-5 cd /path/to/spx-predictor && python run_daily.py

Or with Windows Task Scheduler / macOS launchd.

TELEGRAM ALERTS (optional)
---------------------------
Set these env vars for Telegram notifications:
    TELEGRAM_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id

Get a bot token from @BotFather on Telegram.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path


def is_trading_day() -> bool:
    """Simple check: Mon-Fri, not a US market holiday."""
    import pandas as pd
    today = pd.Timestamp.today()
    if today.weekday() >= 5:  # Saturday or Sunday
        return False
    # Common US market holidays (approximate — use pandas_market_calendars for precision)
    holidays = [
        '2026-01-01',  # New Year's
        '2026-01-19',  # MLK Day
        '2026-02-16',  # Presidents Day
        '2026-04-03',  # Good Friday
        '2026-05-25',  # Memorial Day
        '2026-07-04',  # Independence Day
        '2026-09-07',  # Labor Day
        '2026-11-26',  # Thanksgiving
        '2026-12-25',  # Christmas
    ]
    return today.strftime('%Y-%m-%d') not in holidays


def send_telegram(message: str):
    """Send a message via Telegram bot."""
    token   = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return
    import urllib.request, urllib.parse
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}).encode()
    try:
        urllib.request.urlopen(url, data, timeout=10)
        print("Telegram notification sent.")
    except Exception as e:
        print(f"Telegram failed: {e}")


def format_telegram_message(output: dict) -> str:
    signal_emoji = '🟢' if output['signal'] == 'UP' else '🔴'
    return (
        f"<b>SPX Open Predictor</b>\n"
        f"📅 {output.get('target_date', 'N/A')}\n\n"
        f"{signal_emoji} Signal: <b>{output['signal']}</b>\n"
        f"P(Up):   {output['prob_up']}%\n"
        f"P(Down): {output['prob_down']}%\n"
        f"Confidence: {output['confidence']}%\n\n"
        f"Last Close: {output.get('last_close', 'N/A')}\n"
        f"VIX: {output.get('last_vix', 'N/A')}\n\n"
        f"Model Accuracy (WF): {output.get('model_accuracy', 'N/A')}%\n"
        f"⚠ Synthetic data — replace with live feed for real accuracy"
    )


def main():
    today = datetime.today().strftime('%Y-%m-%d')
    print(f"\n{'='*50}")
    print(f"SPX Predictor Daily Run — {today}")
    print(f"{'='*50}\n")

    if not is_trading_day():
        print("Not a trading day. Skipping.")
        return

    # Step 1: Fetch latest data
    print("Step 1: Fetching data...")
    result = subprocess.run(
        [sys.executable, 'fetch_data.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR fetching data:", result.stderr)
        return

    # Step 2: Run model and save report
    print("Step 2: Running model...")
    result = subprocess.run(
        [sys.executable, 'model.py', '--report'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR running model:", result.stderr)
        return

    # Step 3: Load report and send notification
    report_path = Path('reports') / f"prediction_{today}.json"
    if report_path.exists():
        with open(report_path) as f:
            output = json.load(f)
        msg = format_telegram_message(output)
        send_telegram(msg)
        print(f"\nReport: {report_path}")
        print(f"Signal: {output['signal']} | P(Up): {output['prob_up']}%")
    else:
        print("Report not found.")

    print("\nDone.")


if __name__ == '__main__':
    main()
