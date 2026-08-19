"""
Signal Scanner v4 - بوت تحليل فني مجاني للسكالبينج (M5/M15)
مصدرين للبيانات:
  - Yahoo Finance (رئيسي) - تغطية واسعة جداً، بدون سقف يومي رسمي، لكنه غير رسمي وممكن يتغير بدون إنذار
  - Twelve Data (احتياطي) - يشتغل تلقائياً لو Yahoo فشل بأي رمز، لكنه محدود بـ800 طلب/يوم بالخطة المجانية
يرسل عبر تيليجرام - تحليل فقط بدون تنفيذ تلقائي
"""

import os
import json
import time
from datetime import datetime, timezone

import requests
import pandas as pd
import yfinance as yfin

# ============================ الرموز المراقبة ============================
# كل رمز: label = الاسم المعروض | yf = رمز Yahoo Finance | td = رمز Twelve Data (احتياطي، أو None لو غير مدعوم)
# category تتحكم بفلتر الجلسة | news_currencies تتحكم بفلتر تجنب الأخبار

def fx(label, yf_sym, td_sym, base, quote):
    return {"label": label, "yf": yf_sym, "td": td_sym, "category": "forex", "news_currencies": [base, quote]}

SYMBOLS = [
    # ---- فوركس (10) ----
    fx("EUR/USD", "EURUSD=X", "EUR/USD", "EUR", "USD"),
    fx("GBP/USD", "GBPUSD=X", "GBP/USD", "GBP", "USD"),
    fx("USD/JPY", "USDJPY=X", "USD/JPY", "USD", "JPY"),
    fx("USD/CHF", "USDCHF=X", "USD/CHF", "USD", "CHF"),
    fx("AUD/USD", "AUDUSD=X", "AUD/USD", "AUD", "USD"),
    fx("USD/CAD", "USDCAD=X", "USD/CAD", "USD", "CAD"),
    fx("NZD/USD", "NZDUSD=X", "NZD/USD", "NZD", "USD"),
    fx("EUR/JPY", "EURJPY=X", "EUR/JPY", "EUR", "JPY"),
    fx("GBP/JPY", "GBPJPY=X", "GBP/JPY", "GBP", "JPY"),
    fx("EUR/GBP", "EURGBP=X", "EUR/GBP", "EUR", "GBP"),

    # ---- معادن وطاقة (3) ----
    {"label": "XAU/USD (ذهب)", "yf": "GC=F", "td": "XAU/USD", "category": "commodity", "news_currencies": ["USD"]},
{"label": "XAG/USD (فضة)", "yf": "SI=F", "td": None, "category": "commodity", "news_currencies": ["USD"]},
    {"label": "WTI (نفط)", "yf": "CL=F", "td": None, "category": "commodity", "news_currencies": ["USD"]},

    # ---- كريبتو (6) ----
    {"label": "BTC/USD", "yf": "BTC-USD", "td": "BTC/USD", "category": "crypto", "news_currencies": ["USD"]},
    {"label": "ETH/USD", "yf": "ETH-USD", "td": "ETH/USD", "category": "crypto", "news_currencies": ["USD"]},
    {"label": "SOL/USD", "yf": "SOL-USD", "td": "SOL/USD", "category": "crypto", "news_currencies": ["USD"]},
    {"label": "BNB/USD", "yf": "BNB-USD", "td": "BNB/USD", "category": "crypto", "news_currencies": ["USD"]},
    {"label": "XRP/USD", "yf": "XRP-USD", "td": "XRP/USD", "category": "crypto", "news_currencies": ["USD"]},
    {"label": "ADA/USD", "yf": "ADA-USD", "td": "ADA/USD", "category": "crypto", "news_currencies": ["USD"]},

    # ---- مؤشرات (4) ----
    {"label": "S&P 500", "yf": "^GSPC", "td": None, "category": "stock", "news_currencies": ["USD"]},
    {"label": "Nasdaq", "yf": "^IXIC", "td": None, "category": "stock", "news_currencies": ["USD"]},
    {"label": "Dow Jones", "yf": "^DJI", "td": None, "category": "stock", "news_currencies": ["USD"]},
    {"label": "DAX", "yf": "^GDAXI", "td": None, "category": "stock", "news_currencies": ["EUR"]},

    # ---- أسهم كبرى (20) ----
    *[{"label": s, "yf": s, "td": s, "category": "stock", "news_currencies": ["USD"]} for s in [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC",
        "JPM", "V", "DIS", "KO", "PEP", "WMT", "BA", "XOM", "PYPL", "ORCL",
    ]],
]

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
TP_ATR_MULT = 2.4   # نسبة TP:SL = 2.0 → عند استخدام الحجم المقترح ونسبة مخاطرة 5%، الهدف ≈ 10% من رأس المال

# ============================ فلتر الأخبار ============================
AVOID_NEWS = True
NEWS_BUFFER_MINUTES = 30
ALSO_MEDIUM_IMPACT = False

# ============================ فلتر جلسات التداول ============================
ENABLE_SESSION_FILTER = True
FOREX_SESSION_UTC = (7, 21)   # لندن + نيويورك مجتمعتين (تقريبي) - يشمل الفوركس والمعادن
STOCK_SESSION_UTC = (13, 20)  # جلسة السوق الأمريكي تقريباً - يشمل الأسهم والمؤشرات
# الكريبتو يشتغل 24/7 بدون فلتر جلسة

# ============================ فلتر الدعوم والمقاومة ============================
ENABLE_SR_FILTER = True
SR_LOOKBACK = 30
SR_MIN_DISTANCE_ATR = 0.5

# ============================ تأكيد نموذج الشمعة ============================
ENABLE_CANDLE_CONFIRM = False

# ============================ حجم المركز المقترح ============================
ENABLE_POSITION_SIZING = True
ACCOUNT_BALANCE = 1000.0
RISK_PERCENT_PER_TRADE = 5.0

# ============================ إعدادات عامة ============================
TD_API_SLEEP_SECONDS = 8     # مهلة فقط عند استخدام Twelve Data كاحتياطي
YF_API_SLEEP_SECONDS = 1     # مهلة خفيفة بين طلبات Yahoo لتجنب أي تقييد

TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

STATE_FILE = "state.json"
LOG_FILE = "signals_log.json"
SUMMARY_EVERY_DAYS = 7

# ============================ أدوات جلب البيانات ============================

def fetch_candles_yf(yf_symbol: str, interval: str):
    try:
        df = yfin.Ticker(yf_symbol).history(period="7d", interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        return df[["open", "high", "low", "close"]].astype(float)
    except Exception as e:
        print(f"[تحذير] فشل جلب بيانات Yahoo لـ {yf_symbol} ({interval}): {e}")
        return None


def fetch_candles_td(td_symbol: str, interval: str, outputsize: int = 100):
    if not TWELVE_DATA_KEY:
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": td_symbol, "interval": interval, "outputsize": outputsize, "apikey": TWELVE_DATA_KEY}
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if "values" not in data:
        print(f"[تحذير] فشل جلب بيانات Twelve Data لـ {td_symbol} ({interval}): {data}")
        return None
    df = pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def fetch_candles(cfg: dict, tf_key: str):
    """tf_key: 'trend' أو 'entry' - يجرب Yahoo أول، ولو فشل يرجع لـ Twelve Data"""
    yf_symbol = cfg.get("yf")
    if yf_symbol:
        yf_interval = "15m" if tf_key == "trend" else "5m"
        df = fetch_candles_yf(yf_symbol, yf_interval)
        time.sleep(YF_API_SLEEP_SECONDS)
        if df is not None and len(df) >= 30:
            return df

    td_symbol = cfg.get("td")
    if td_symbol:
        td_interval = TREND_TF if tf_key == "trend" else ENTRY_TF
        df = fetch_candles_td(td_symbol, td_interval)
        time.sleep(TD_API_SLEEP_SECONDS)
        return df

    return None


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


def is_session_active(cfg: dict, now_utc: datetime) -> bool:
    if not ENABLE_SESSION_FILTER:
        return True
    cat = cfg["category"]
    if cat == "crypto":
        return True
    if now_utc.weekday() >= 5:
        return False
    hour = now_utc.hour
    start, end = STOCK_SESSION_UTC if cat == "stock" else FOREX_SESSION_UTC
    return start <= hour < end


# ============================ التقويم الاقتصادي ============================

def fetch_news_events():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=20)
        return r.json()
    except Exception as e:
        print("[تحذير] فشل جلب التقويم الاقتصادي:", e)
        return []


def is_news_near(cfg: dict, events: list, now_utc: datetime):
    currencies = cfg["news_currencies"]
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


def bullish_engulfing(prev_row, cur_row) -> bool:
    return (prev_row["close"] < prev_row["open"] and cur_row["close"] > cur_row["open"] and
            cur_row["close"] >= prev_row["open"] and cur_row["open"] <= prev_row["close"])


def bearish_engulfing(prev_row, cur_row) -> bool:
    return (prev_row["close"] > prev_row["open"] and cur_row["close"] < cur_row["open"] and
            cur_row["close"] <= prev_row["open"] and cur_row["open"] >= prev_row["close"])


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

def update_open_signals(log: list, label: str, latest_row):
    high, low = latest_row["high"], latest_row["low"]
    for sig in log:
        if sig["symbol"] != label or sig["status"] != "open":
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

def scan_symbol(cfg: dict, news_events: list, state: dict, log: list, now_utc: datetime):
    label = cfg["label"]

    if not is_session_active(cfg, now_utc):
        return

    trend_df = fetch_candles(cfg, "trend")
    entry_df = fetch_candles(cfg, "entry")

    if trend_df is None or entry_df is None or len(trend_df) < 55 or len(entry_df) < max(30, SR_LOOKBACK + 2):
        print(f"[تخطي] {label}: بيانات غير كافية")
        return

    trend_df = add_indicators(trend_df)
    entry_df = add_indicators(entry_df)

    update_open_signals(log, label, entry_df.iloc[-1])

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
        near, title = is_news_near(cfg, news_events, now_utc)
        if near:
            print(f"[تخطي] {label}: خبر مهم قريب ({title})")
            return

    trend_up = t["ema_fast"] > t["ema_slow"]
    trend_down = t["ema_fast"] < t["ema_slow"]

    entry_up = e1["ema_fast"] > e1["ema_slow"]
    entry_down = e1["ema_fast"] < e1["ema_slow"]

    macd_cross_up = (e1["macd"] > e1["macd_signal"]) and (e0["macd"] <= e0["macd_signal"])
    macd_cross_down = (e1["macd"] < e1["macd_signal"]) and (e0["macd"] >= e0["macd_signal"])

    rsi_recovering_up = 25 < e1["rsi"] < 60 and e0["rsi"] <= e1["rsi"]
    rsi_recovering_down = 40 < e1["rsi"] < 75 and e0["rsi"] >= e1["rsi"]

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
        key = f"{label}_{direction}"
        last = state.get(key)
        if last:
            last_dt = datetime.fromisoformat(last)
            if (now_utc - last_dt).total_seconds() < COOLDOWN_MINUTES * 60:
                continue

        if direction == "BUY":
            sl, tp = price - atr * SL_ATR_MULT, price + atr * TP_ATR_MULT
            head = "شراء محتمل 🟢"
        else:
            sl, tp = price + atr * SL_ATR_MULT, price - atr * TP_ATR_MULT
            head = "بيع محتمل 🔴"

        lines = [
            f"{head} - {label}",
            f"السعر: {price:.5f}",
            f"وقف الخسارة المقترح: {sl:.5f}",
            f"الهدف المقترح: {tp:.5f}",
            "اتجاه M15 + دخول M5",
        ]

        if ENABLE_POSITION_SIZING:
            sl_distance = abs(price - sl)
            risk_amount = ACCOUNT_BALANCE * (RISK_PERCENT_PER_TRADE / 100)
            if sl_distance > 0:
                units = risk_amount / sl_distance
                lines.append(f"حجم مقترح تقريبي: {units:.2f} وحدة (مخاطرة {RISK_PERCENT_PER_TRADE}% من {ACCOUNT_BALANCE:.0f})")
                lines.append("⚠️ حوّله للوت الصحيح حسب حجم العقد بمنصتك قبل التنفيذ")

        lines.append("⚠️ تأكد من السبريد الفعلي داخل MT5 قبل الدخول")

        msg = "\n".join(lines)
        send_telegram(msg)
        print(msg)
        state[key] = now_utc.isoformat()

        log.append({
            "symbol": label, "direction": direction, "entry_price": price,
            "sl": sl, "tp": tp, "opened_at": now_utc.isoformat(), "status": "open",
        })


def main():
    now_utc = datetime.now(timezone.utc)
    news_events = fetch_news_events() if AVOID_NEWS else []
    state = load_json(STATE_FILE, {})
    log = load_json(LOG_FILE, [])

    for cfg in SYMBOLS:
        try:
            scan_symbol(cfg, news_events, state, log, now_utc)
        except Exception as e:
            print(f"[خطأ] فشل تحليل {cfg['label']}: {e}")

    maybe_send_weekly_summary(log, state, now_utc)

    save_json(STATE_FILE, state)
    save_json(LOG_FILE, log)


if __name__ == "__main__":
    main()
