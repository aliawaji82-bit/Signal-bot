"""
Signal Scanner - بوت تحليل فني مجاني للسكالبينج (M5/M15)
يرسل إشارات عبر تيليجرام - تحليل فقط بدون تنفيذ تلقائي
يشتغل عبر GitHub Actions كل 15 دقيقة تقريباً
"""

import os
import json
import time
from datetime import datetime, timezone

import requests
import pandas as pd

# ============================ الإعدادات ============================
# عدّل هذي القائمة حسب الرموز اللي تبي تراقبها
# صيغة الفوركس/المعادن: "EUR/USD"  |  صيغة الأسهم: "AAPL"  |  صيغة الكريبتو: "BTC/USD"
SYMBOLS = ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD"]

TREND_TF = "15min"
ENTRY_TF = "5min"

RSI_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 50
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14

MIN_ATR_PCT = 0.08          # أقل نسبة تقلب مقبولة (ATR/السعر %)
COOLDOWN_MINUTES = 30        # أقل فترة بين تنبيهين لنفس الرمز ونفس الاتجاه
SL_ATR_MULT = 1.2
TP_ATR_MULT = 2.0

AVOID_NEWS = True
NEWS_BUFFER_MINUTES = 30
ALSO_MEDIUM_IMPACT = False

API_SLEEP_SECONDS = 8         # مهلة بين طلبات API (حد Twelve Data: 8/دقيقة على الخطة المجانية)

TWELVE_DATA_KEY = os.environ["TWELVE_DATA_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

STATE_FILE = "state.json"

# ============================ أدوات مساعدة ============================

def fetch_candles(symbol: str, interval: str, outputsize: int = 100):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": TWELVE_DATA_KEY}
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if "values" not in data:
        print(f"[تحذير] فشل جلب بيانات {symbol} ({interval}): {data}")
        return None
    df = pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]

    df["ema_fast"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=EMA_SLOW, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    ema_f = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_f - ema_s
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    return df


def get_currencies(symbol: str):
    if "/" in symbol:
        base, quote = symbol.split("/")
        return [base, quote]
    return ["USD"]  # افتراض للأسهم الأمريكية


def fetch_news_events():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=20)
        return r.json()
    except Exception as e:
        print("[تحذير] فشل جلب التقويم الاقتصادي:", e)
        return []


def is_news_near(symbol: str, events: list, now_utc: datetime):
    currencies = get_currencies(symbol)
    for ev in events:
        impact = ev.get("impact", "")
        if impact != "High" and not (ALSO_MEDIUM_IMPACT and impact == "Medium"):
            continue
        if ev.get("country", "") not in currencies:
            continue
        try:
            ev_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        except Exception:
            continue
        if abs((ev_time - now_utc).total_seconds()) / 60 <= NEWS_BUFFER_MINUTES:
            return True, ev.get("title", "")
    return False, None


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)
    if r.status_code != 200:
        print("[خطأ] فشل إرسال تيليجرام:", r.text)


# ============================ المنطق الرئيسي ============================

def scan_symbol(symbol: str, news_events: list, state: dict, now_utc: datetime):
    trend_df = fetch_candles(symbol, TREND_TF, outputsize=80)
    time.sleep(API_SLEEP_SECONDS)
    entry_df = fetch_candles(symbol, ENTRY_TF, outputsize=80)
    time.sleep(API_SLEEP_SECONDS)

    if trend_df is None or entry_df is None or len(trend_df) < 55 or len(entry_df) < 30:
        return

    trend_df = add_indicators(trend_df)
    entry_df = add_indicators(entry_df)

    t = trend_df.iloc[-2]   # آخر شمعة مغلقة على فريم M15
    e1 = entry_df.iloc[-2]  # آخر شمعة مغلقة على فريم M5
    e0 = entry_df.iloc[-1]  # الشمعة الحالية (لسه ما قفلت) - نستخدمها لتأكيد التقاطع

    price = e1["close"]
    if price <= 0 or pd.isna(t["ema_fast"]) or pd.isna(e1["rsi"]):
        return

    atr_pct = (e1["atr"] / price) * 100
    if atr_pct < MIN_ATR_PCT:
        return

    if AVOID_NEWS:
        near, title = is_news_near(symbol, news_events, now_utc)
        if near:
            print(f"[تخطي] {symbol}: خبر مهم قريب ({title})")
            return

    trend_up = t["ema_fast"] > t["ema_slow"]
    trend_down = t["ema_fast"] < t["ema_slow"]

    entry_up = e1["ema_fast"] > e1["ema_slow"]
    entry_down = e1["ema_fast"] < e1["ema_slow"]

    macd_cross_up = (e1["macd"] > e1["macd_signal"]) and (e0["macd"] <= e0["macd_signal"])
    macd_cross_down = (e1["macd"] < e1["macd_signal"]) and (e0["macd"] >= e0["macd_signal"])

    rsi_recovering_up = 30 < e1["rsi"] < 55 and e0["rsi"] <= e1["rsi"]
    rsi_recovering_down = 45 < e1["rsi"] < 70 and e0["rsi"] >= e1["rsi"]

    buy_signal = trend_up and entry_up and macd_cross_up and rsi_recovering_up
    sell_signal = trend_down and entry_down and macd_cross_down and rsi_recovering_down

    for direction, active in (("BUY", buy_signal), ("SELL", sell_signal)):
        if not active:
            continue
        key = f"{symbol}_{direction}"
        last = state.get(key)
        if last:
            last_dt = datetime.fromisoformat(last)
            if (now_utc - last_dt).total_seconds() < COOLDOWN_MINUTES * 60:
                continue

        atr = e1["atr"]
        if direction == "BUY":
            sl, tp = price - atr * SL_ATR_MULT, price + atr * TP_ATR_MULT
            label = "شراء محتمل 🟢"
        else:
            sl, tp = price + atr * SL_ATR_MULT, price - atr * TP_ATR_MULT
            label = "بيع محتمل 🔴"

        msg = (f"{label} - {symbol}\n"
               f"السعر: {price:.5f}\n"
               f"وقف الخسارة المقترح: {sl:.5f}\n"
               f"الهدف المقترح: {tp:.5f}\n"
               f"اتجاه M15 + دخول M5 (مضاربة قصيرة)")
        send_telegram(msg)
        print(msg)
        state[key] = now_utc.isoformat()


def main():
    now_utc = datetime.now(timezone.utc)
    news_events = fetch_news_events() if AVOID_NEWS else []
    state = load_state()

    for symbol in SYMBOLS:
        try:
            scan_symbol(symbol, news_events, state, now_utc)
        except Exception as e:
            print(f"[خطأ] فشل تحليل {symbol}: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
