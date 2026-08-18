"""
Signal Scanner v3 - بوت تحليل فني مجاني للسكالبينج (M5/M15)
إضافات هذي النسخة:
  - فلتر جلسات التداول (بديل عملي لفلتر السبريد - السيولة العالية = سبريد أقل عادة)
  - فلتر دعوم/مقاومة (يتجنب الدخول قريب من حاجز فني)
  - تأكيد نموذج شمعة (ابتلاع صاعد/هابط) كشرط إضافي
  - حجم مركز مقترح بناءً على نسبة مخاطرة من رأس المال
  - سجل أداء تلقائي (يتتبع كل إشارة ويحسب نسبة النجاح) + ملخص أسبوعي عبر تيليجرام
يرسل عبر تيليجرام - تحليل فقط بدون تنفيذ تلقائي
"""

import os
import json
import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

# ============================ الرموز المراقبة ============================
# صيغة الفوركس/المعادن: "EUR/USD"  |  الأسهم: "AAPL"  |  الكريبتو: "BTC/USD"
FOREX_COMMODITY_SYMBOLS = ["XAU/USD", "XAG/USD", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
CRYPTO_SYMBOLS = ["BTC/USD", "ETH/USD"]
STOCK_SYMBOLS = ["AAPL", "TSLA"]

SYMBOLS = FOREX_COMMODITY_SYMBOLS + CRYPTO_SYMBOLS + STOCK_SYMBOLS

# ============================ إعدادات المؤشرات ============================
TREND_TF = "15min"
ENTRY_TF = "5min"

RSI_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 50
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14

MIN_ATR_PCT = 0.08
COOLDOWN_MINUTES = 30
SL_ATR_MULT = 1.2
TP_ATR_MULT = 2.0

# ============================ فلتر الأخبار ============================
AVOID_NEWS = True
NEWS_BUFFER_MINUTES = 30
ALSO_MEDIUM_IMPACT = False

# ============================ فلتر جلسات التداول ============================
# ملاحظة: خطة Twelve Data المجانية ما توفر بيانات سبريد حية.
# كبديل عملي، نتجنب التداول بأوقات السيولة الضعيفة (سبريد أعلى غالباً) عبر تحديد نوافذ الجلسات.
ENABLE_SESSION_FILTER = True
FOREX_SESSION_UTC = (7, 21)   # لندن + نيويورك مجتمعتين (تقريبي)
STOCK_SESSION_UTC = (13, 20)  # جلسة السوق الأمريكي تقريباً
# الكريبتو يشتغل 24/7 بدون فلتر جلسة

# ============================ فلتر الدعوم والمقاومة ============================
ENABLE_SR_FILTER = True
SR_LOOKBACK = 30            # عدد الشموع للبحث عن أقرب دعم/مقاومة
SR_MIN_DISTANCE_ATR = 1.0   # أقل مسافة مطلوبة (بوحدات ATR) بين السعر وأقرب حاجز

# ============================ تأكيد نموذج الشمعة ============================
ENABLE_CANDLE_CONFIRM = True   # لو True، لازم يتأكد نموذج ابتلاع مع باقي الشروط

# ============================ حجم المركز المقترح ============================
ENABLE_POSITION_SIZING = True
ACCOUNT_BALANCE = 1000.0        # عدّل هذا الرقم لرصيد حسابك التقريبي
RISK_PERCENT_PER_TRADE = 1.0    # نسبة المخاطرة المقترحة من الرصيد لكل صفقة (%)

# ============================ إعدادات عامة ============================
API_SLEEP_SECONDS = 8
TWELVE_DATA_KEY = os.environ["TWELVE_DATA_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

STATE_FILE = "state.json"
LOG_FILE = "signals_log.json"
SUMMARY_EVERY_DAYS = 7

# ============================ أدوات جلب البيانات ============================

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
    return ["USD"]


def symbol_category(symbol: str) -> str:
    if symbol in CRYPTO_SYMBOLS:
        return "crypto"
    if symbol in STOCK_SYMBOLS:
        return "stock"
    return "forex"


def is_session_active(symbol: str, now_utc: datetime) -> bool:
    if not ENABLE_SESSION_FILTER:
        return True
    cat = symbol_category(symbol)
    if cat == "crypto":
        return True
    if now_utc.weekday() >= 5:  # السبت/الأحد
        return cat != "stock"  # الفوركس شبه متوقف نهاية الأسبوع أصلاً، الأسهم مغلقة أكيد
    hour = now_utc.hour
    if cat == "stock":
        start, end = STOCK_SESSION_UTC
    else:
        start, end = FOREX_SESSION_UTC
    return start <= hour < end


# ============================ التقويم الاقتصادي ============================

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


# ============================ نموذج الشمعة ============================

def bullish_engulfing(prev_row, cur_row) -> bool:
    return (prev_row["close"] < prev_row["open"] and
            cur_row["close"] > cur_row["open"] and
            cur_row["close"] >= prev_row["open"] and
            cur_row["open"] <= prev_row["close"])


def bearish_engulfing(prev_row, cur_row) -> bool:
    return (prev_row["close"] > prev_row["open"] and
            cur_row["close"] < cur_row["open"] and
            cur_row["close"] <= prev_row["open"] and
            cur_row["open"] >= prev_row["close"])


# ============================ الدعم والمقاومة ============================

def sr_ok_for_buy(entry_df, price, atr) -> bool:
    if not ENABLE_SR_FILTER:
        return True
    window = entry_df.iloc[-(SR_LOOKBACK + 1):-1]
    resistance = window["high"].max()
    return (resistance - price) >= (SR_MIN_DISTANCE_ATR * atr)


def sr_ok_for_sell(entry_df, price, atr) -> bool:
    if not ENABLE_SR_FILTER:
        return True
    window = entry_df.iloc[-(SR_LOOKBACK + 1):-1]
    support = window["low"].min()
    return (price - support) >= (SR_MIN_DISTANCE_ATR * atr)


# ============================ ملفات الحالة والسجل ============================

def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)
    if r.status_code != 200:
        print("[خطأ] فشل إرسال تيليجرام:", r.text)


# ============================ تتبع الأداء ============================

def update_open_signals(log: list, symbol: str, latest_row):
    """يتحقق من الإشارات المفتوحة لهذا الرمز هل ضربت الهدف أو الوقف"""
    high, low = latest_row["high"], latest_row["low"]
    for sig in log:
        if sig["symbol"] != symbol or sig["status"] != "open":
            continue
        if sig["direction"] == "BUY":
            if high >= sig["tp"]:
                sig["status"], sig["closed_at"] = "win", datetime.now(timezone.utc).isoformat()
            elif low <= sig["sl"]:
                sig["status"], sig["closed_at"] = "loss", datetime.now(timezone.utc).isoformat()
        else:
            if low <= sig["tp"]:
                sig["status"], sig["closed_at"] = "win", datetime.now(timezone.utc).isoformat()
            elif high >= sig["sl"]:
                sig["status"], sig["closed_at"] = "loss", datetime.now(timezone.utc).isoformat()


def maybe_send_weekly_summary(log: list, state: dict, now_utc: datetime):
    last_str = state.get("last_summary_sent")
    if last_str:
        last_dt = datetime.fromisoformat(last_str)
        if (now_utc - last_dt).days < SUMMARY_EVERY_DAYS:
            return

    closed = [s for s in log if s["status"] in ("win", "loss")]
    if not closed:
        state["last_summary_sent"] = now_utc.isoformat()
        return

    wins = sum(1 for s in closed if s["status"] == "win")
    total = len(closed)
    win_rate = (wins / total) * 100

    msg = (f"📊 ملخص أداء آخر {SUMMARY_EVERY_DAYS} أيام\n"
           f"عدد الصفقات المغلقة: {total}\n"
           f"رابحة: {wins} | خاسرة: {total - wins}\n"
           f"نسبة النجاح: {win_rate:.1f}%\n"
           f"(هذا تحليل احتمالي - راقب الأداء الحقيقي بحسابك بنفسك)")
    send_telegram(msg)
    print(msg)
    state["last_summary_sent"] = now_utc.isoformat()


# ============================ المنطق الرئيسي ============================

def scan_symbol(symbol: str, news_events: list, state: dict, log: list, now_utc: datetime):
    if not is_session_active(symbol, now_utc):
        return

    trend_df = fetch_candles(symbol, TREND_TF, outputsize=80)
    time.sleep(API_SLEEP_SECONDS)
    entry_df = fetch_candles(symbol, ENTRY_TF, outputsize=80)
    time.sleep(API_SLEEP_SECONDS)

    if trend_df is None or entry_df is None or len(trend_df) < 55 or len(entry_df) < max(30, SR_LOOKBACK + 2):
        return

    trend_df = add_indicators(trend_df)
    entry_df = add_indicators(entry_df)

    # تحديث حالة أي إشارات مفتوحة سابقة لنفس الرمز بناءً على آخر شمعة
    update_open_signals(log, symbol, entry_df.iloc[-1])

    t = trend_df.iloc[-2]
    e1 = entry_df.iloc[-2]
    e0 = entry_df.iloc[-1]

    price = e1["close"]
    if price <= 0 or pd.isna(t["ema_fast"]) or pd.isna(e1["rsi"]):
        return

    atr = e1["atr"]
    atr_pct = (atr / price) * 100
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

    if buy_signal and ENABLE_CANDLE_CONFIRM:
        buy_signal = bullish_engulfing(e1, e0)
    if sell_signal and ENABLE_CANDLE_CONFIRM:
        sell_signal = bearish_engulfing(e1, e0)

    if buy_signal and not sr_ok_for_buy(entry_df, price, atr):
        buy_signal = False
    if sell_signal and not sr_ok_for_sell(entry_df, price, atr):
        sell_signal = False

    for direction, active in (("BUY", buy_signal), ("SELL", sell_signal)):
        if not active:
            continue
        key = f"{symbol}_{direction}"
        last = state.get(key)
        if last:
            last_dt = datetime.fromisoformat(last)
            if (now_utc - last_dt).total_seconds() < COOLDOWN_MINUTES * 60:
                continue

        if direction == "BUY":
            sl, tp = price - atr * SL_ATR_MULT, price + atr * TP_ATR_MULT
            label = "شراء محتمل 🟢"
        else:
            sl, tp = price + atr * SL_ATR_MULT, price - atr * TP_ATR_MULT
            label = "بيع محتمل 🔴"

        msg_lines = [
            f"{label} - {symbol}",
            f"السعر: {price:.5f}",
            f"وقف الخسارة المقترح: {sl:.5f}",
            f"الهدف المقترح: {tp:.5f}",
            "اتجاه M15 + دخول M5 + تأكيد شمعة ودعم/مقاومة",
        ]

        if ENABLE_POSITION_SIZING:
            sl_distance = abs(price - sl)
            risk_amount = ACCOUNT_BALANCE * (RISK_PERCENT_PER_TRADE / 100)
            if sl_distance > 0:
                suggested_units = risk_amount / sl_distance
                msg_lines.append(f"حجم مقترح تقريبي: {suggested_units:.2f} وحدة (بناءً على مخاطرة {RISK_PERCENT_PER_TRADE}% من {ACCOUNT_BALANCE:.0f})")
                msg_lines.append("⚠️ هذا تقدير عام، حوّله للوت الصحيح حسب حجم العقد بمنصتك قبل التنفيذ")

        msg_lines.append("⚠️ السبريد الفعلي غير متاح من مصدر البيانات المجاني - تأكد منه داخل MT5 قبل الدخول")

        msg = "\n".join(msg_lines)
        send_telegram(msg)
        print(msg)
        state[key] = now_utc.isoformat()

        log.append({
            "symbol": symbol, "direction": direction, "entry_price": price,
            "sl": sl, "tp": tp, "opened_at": now_utc.isoformat(), "status": "open",
        })


def main():
    now_utc = datetime.now(timezone.utc)
    news_events = fetch_news_events() if AVOID_NEWS else []
    state = load_json(STATE_FILE, {})
    log = load_json(LOG_FILE, [])

    for symbol in SYMBOLS:
        try:
            scan_symbol(symbol, news_events, state, log, now_utc)
        except Exception as e:
            print(f"[خطأ] فشل تحليل {symbol}: {e}")

    maybe_send_weekly_summary(log, state, now_utc)

    save_json(STATE_FILE, state)
    save_json(LOG_FILE, log)


if __name__ == "__main__":
    main()
